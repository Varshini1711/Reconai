"""
ReconAI - Layer 4: Grouped Reconciliation.

Handles relationships a 1:1 layer structurally cannot see:
  - many transactions -> one settlement (batched payout, fee deducted)
  - one transaction -> many settlements (split payout)
  - transaction minus refund -> settlement
  - transaction minus fee -> settlement
  - partial settlement (only part of a transaction paid out so far)

Search is bounded (max group size MAX_GROUP) and restricted to
same-merchant, date-window candidates from Layer 2's settlement-side
pool, so this stays a small combinatorial search, not an explosion.

Fees/refunds are read directly (this layer DOES touch those files,
unlike Layers 1-3) - never ground truth.
"""

import itertools
from reconciliation.models import Transaction, Settlement

MAX_GROUP = 3
SUM_TOLERANCE_ABS = 1.0   # absolute rupee tolerance for sum-matching (covers rounding)
PARTIAL_MIN_RATIO = 0.3   # a settlement must be at least 30% of the transaction to count as "partial", not noise
PARTIAL_MAX_RATIO = 0.95  # and less than 95% (otherwise it'd already be a near-exact match)


def _fees_for(txn_ids: set, fees: list) -> list:
    matched = []
    for f in fees:
        related = set(f["related_txn_ids"].split(";"))
        if related and related.issubset(txn_ids):
            matched.append(f)
    return matched


def _refund_for(txn_id: str, refunds: list):
    for r in refunds:
        if r["original_txn_id"] == txn_id:
            return r
    return None


def try_many_to_one(unresolved_txns: list, unresolved_settlements: list, fees: list) -> list:
    """
    For each unresolved settlement, search small combinations of
    same-merchant unresolved transactions whose amounts (minus any
    applicable batched fee) sum to the settlement amount.

    Returns a list of dicts: {settlement_id, txn_ids, fee_id or None, evidence}
    """
    found = []
    by_merchant = {}
    for t in unresolved_txns:
        by_merchant.setdefault(t.merchant_name.lower(), []).append(t)

    for s in unresolved_settlements:
        pool = by_merchant.get(s.merchant_name.lower(), [])
        if len(pool) < 2:
            continue
        for size in range(2, min(MAX_GROUP, len(pool)) + 1):
            for combo in itertools.combinations(pool, size):
                total = sum(t.amount for t in combo)
                txn_ids = {t.txn_id for t in combo}

                # try with no fee
                if abs(total - s.amount) <= SUM_TOLERANCE_ABS:
                    found.append(dict(settlement_id=s.settlement_id, txn_ids=[t.txn_id for t in combo],
                                       fee_id=None,
                                       evidence=[f"{size} transactions sum to settlement amount exactly"]))
                    break

                # try with a batched fee that references exactly these txns
                applicable_fees = _fees_for(txn_ids, fees)
                for fee in applicable_fees:
                    if abs((total - fee["amount"]) - s.amount) <= SUM_TOLERANCE_ABS:
                        found.append(dict(settlement_id=s.settlement_id, txn_ids=[t.txn_id for t in combo],
                                           fee_id=fee["fee_id"],
                                           evidence=[f"{size} transactions minus fee {fee['fee_id']} "
                                                     f"({fee['amount']}) sum to settlement amount"]))
                        break
            else:
                continue
            break
    return found


def try_one_to_many(unresolved_txns: list, unresolved_settlements: list) -> list:
    """For each unresolved transaction, search small combinations of
    same-merchant unresolved settlements whose amounts sum to the
    transaction amount (split payout)."""
    found = []
    by_merchant = {}
    for s in unresolved_settlements:
        by_merchant.setdefault(s.merchant_name.lower(), []).append(s)

    for t in unresolved_txns:
        pool = by_merchant.get(t.merchant_name.lower(), [])
        if len(pool) < 2:
            continue
        for size in range(2, min(MAX_GROUP, len(pool)) + 1):
            for combo in itertools.combinations(pool, size):
                total = sum(s.amount for s in combo)
                if abs(total - t.amount) <= SUM_TOLERANCE_ABS:
                    found.append(dict(txn_id=t.txn_id, settlement_ids=[s.settlement_id for s in combo],
                                       evidence=[f"Settlement group of {size} sums to transaction amount"]))
                    break
            else:
                continue
            break
    return found


def try_refund_adjusted(unresolved_txns: list, unresolved_settlements: list, refunds: list) -> list:
    """Transaction minus a refund referencing it equals a same-merchant
    unresolved settlement."""
    found = []
    unresolved_set_by_merchant = {}
    for s in unresolved_settlements:
        unresolved_set_by_merchant.setdefault(s.merchant_name.lower(), []).append(s)

    for t in unresolved_txns:
        refund = _refund_for(t.txn_id, refunds)
        if not refund:
            continue
        expected = t.amount - refund["amount"]
        for s in unresolved_set_by_merchant.get(t.merchant_name.lower(), []):
            if abs(s.amount - expected) <= SUM_TOLERANCE_ABS:
                found.append(dict(txn_id=t.txn_id, settlement_id=s.settlement_id,
                                   refund_id=refund["refund_id"],
                                   evidence=[f"Transaction minus refund {refund['refund_id']} "
                                             f"({refund['amount']}) equals settlement amount"]))
                break
    return found


def try_fee_adjusted_single(unresolved_txns: list, unresolved_settlements: list, fees: list) -> list:
    """Single transaction minus a single-txn fee equals a settlement
    (handles the fee-rounding-near-miss case type)."""
    found = []
    unresolved_set_by_merchant = {}
    for s in unresolved_settlements:
        unresolved_set_by_merchant.setdefault(s.merchant_name.lower(), []).append(s)

    for t in unresolved_txns:
        applicable = _fees_for({t.txn_id}, fees)
        for fee in applicable:
            expected = t.amount - fee["amount"]
            for s in unresolved_set_by_merchant.get(t.merchant_name.lower(), []):
                if abs(s.amount - expected) <= SUM_TOLERANCE_ABS:
                    found.append(dict(txn_id=t.txn_id, settlement_id=s.settlement_id,
                                       fee_id=fee["fee_id"],
                                       evidence=[f"Transaction minus fee {fee['fee_id']} "
                                                 f"({fee['amount']}) equals settlement amount"]))
                    break
    return found


PARTIAL_DATE_WINDOW_DAYS = 10


def try_partial_settlement(unresolved_txns: list, unresolved_settlements: list) -> list:
    """
    Flags (does NOT auto-resolve) transactions where a same-merchant
    settlement covers a plausible partial fraction of the amount, with no
    grouping/fee/refund explanation. This layer only DETECTS the pattern;
    the confidence/decision layer is what actually assigns NEEDS_REVIEW.

    Requires the settlement to occur ON or AFTER the transaction (money
    cannot be paid out before the transaction happened) and within a
    reasonable window, and picks the BEST (highest-ratio) candidate
    rather than the first one found - otherwise a coincidental unrelated
    settlement elsewhere in the dataset can get picked up as a false
    "partial" match ahead of the genuine one.

    Also excludes any settlement that is an EXACT amount match (within a
    tight tolerance) for a DIFFERENT still-unresolved transaction - that
    other transaction has a far stronger claim on it, and letting a
    same-merchant partial-ratio coincidence steal it away breaks the
    genuine match elsewhere in the pipeline (found via real diagnostic
    evidence: a 92%-ratio coincidence between two unrelated transactions
    consumed a settlement that was an exact match for a third).
    """
    found = []
    unresolved_set_by_merchant = {}
    for s in unresolved_settlements:
        unresolved_set_by_merchant.setdefault(s.merchant_name.lower(), []).append(s)

    # Settlements that are an exact-amount match for some OTHER unresolved
    # transaction - these are "spoken for" and must not be taken as a
    # partial match for anyone else.
    exact_claim_for = {}
    for other_t in unresolved_txns:
        for s in unresolved_set_by_merchant.get(other_t.merchant_name.lower(), []):
            if abs(s.amount - other_t.amount) <= SUM_TOLERANCE_ABS:
                exact_claim_for.setdefault(s.settlement_id, set()).add(other_t.txn_id)

    for t in unresolved_txns:
        best = None
        for s in unresolved_set_by_merchant.get(t.merchant_name.lower(), []):
            if s.amount >= t.amount:
                continue
            claimants = exact_claim_for.get(s.settlement_id, set())
            if claimants - {t.txn_id}:
                continue   # this settlement exactly matches a DIFFERENT transaction - not available as a partial
            delta_days = (s.date - t.date).days
            if not (0 <= delta_days <= PARTIAL_DATE_WINDOW_DAYS):
                continue
            ratio = s.amount / t.amount
            if PARTIAL_MIN_RATIO <= ratio <= PARTIAL_MAX_RATIO:
                if best is None or ratio > best[1]:
                    best = (s, ratio)
        if best:
            s, ratio = best
            found.append(dict(txn_id=t.txn_id, settlement_id=s.settlement_id, ratio=ratio,
                               evidence=[f"Settlement covers {ratio*100:.0f}% of transaction amount; "
                                         f"remainder {t.amount - s.amount:.2f} not yet settled"]))
    return found
