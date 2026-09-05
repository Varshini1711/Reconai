"""
ReconAI - Layer 1: Deterministic Exact Matching.

Only attempts a match when it can be SAFE and UNIQUE:
  - exact amount
  - exact merchant name (case-insensitive)
  - settlement date within SETTLEMENT_WINDOW_DAYS after transaction date
  - the candidate is unique in both directions (no other transaction
    could equally claim that settlement, and vice versa)

Anything not meeting all of the above is left UNMATCHED_LAYER1 and
passed forward - this layer must never guess. Does not attempt
merchant-name fuzziness, grouping, fees, or refunds; those are later
layers.
"""

from datetime import datetime, UTC
from reconciliation.models import Transaction, Settlement, MatchResult

SETTLEMENT_WINDOW_DAYS = 3


class Layer1ExactMatcher:
    def __init__(self, transactions: list[Transaction], settlements: list[Settlement],
                 window_days: int = SETTLEMENT_WINDOW_DAYS):
        self.transactions = transactions
        self.settlements = settlements
        self.window_days = window_days

    def _same_amount(self, t: Transaction, s: Settlement) -> bool:
        return abs(t.amount - s.amount) < 0.001

    def _same_merchant(self, t: Transaction, s: Settlement) -> bool:
        return t.merchant_name.strip().lower() == s.merchant_name.strip().lower()

    def _within_window(self, t: Transaction, s: Settlement) -> bool:
        delta_days = (s.date - t.date).days
        return 0 <= delta_days <= self.window_days

    def _find_candidates(self, txn: Transaction, used_settlements: set) -> list[Settlement]:
        candidates = []
        for s in self.settlements:
            if s.settlement_id in used_settlements:
                continue
            if self._same_amount(txn, s) and self._same_merchant(txn, s) and self._within_window(txn, s):
                candidates.append(s)
        return candidates

    def run(self) -> list[MatchResult]:
        used_settlements = set()
        results = []

        # Build every transaction's candidate pool BEFORE committing any
        # match, so cross-transaction ambiguity (two txns both wanting
        # the same settlement) is detected rather than resolved by
        # processing order.
        txn_candidates = {
            t.txn_id: self._find_candidates(t, used_settlements)
            for t in self.transactions
        }

        now = datetime.now(UTC).isoformat()

        for t in self.transactions:
            candidates = txn_candidates[t.txn_id]

            if len(candidates) == 0:
                results.append(MatchResult(
                    txn_id=t.txn_id, settlement_id=None, status="UNMATCHED_LAYER1",
                    match_method="exact_amount_merchant_date",
                    evidence=["No exact amount+merchant match within date window"],
                    reason="No unique deterministic candidate found.",
                    timestamp=now,
                ))
                continue

            if len(candidates) > 1:
                results.append(MatchResult(
                    txn_id=t.txn_id, settlement_id=None, status="UNMATCHED_LAYER1",
                    match_method="exact_amount_merchant_date",
                    evidence=[f"Multiple exact candidates: {[c.settlement_id for c in candidates]}"],
                    reason="No unique deterministic candidate found.",
                    timestamp=now,
                ))
                continue

            # exactly one candidate - but confirm no OTHER transaction
            # equally claims it (mutual uniqueness, not just this txn's view)
            sid = candidates[0].settlement_id
            competing_txns = [
                tid for tid, cands in txn_candidates.items()
                if tid != t.txn_id and any(c.settlement_id == sid for c in cands)
            ]
            if competing_txns:
                results.append(MatchResult(
                    txn_id=t.txn_id, settlement_id=None, status="UNMATCHED_LAYER1",
                    match_method="exact_amount_merchant_date",
                    evidence=[f"Settlement {sid} also matches {competing_txns}"],
                    reason="No unique deterministic candidate found.",
                    timestamp=now,
                ))
                continue

            delta_days = (candidates[0].date - t.date).days
            results.append(MatchResult(
                txn_id=t.txn_id, settlement_id=sid, status="MATCHED_LAYER1",
                match_method="exact_amount_merchant_date",
                evidence=[
                    "Exact amount match",
                    "Exact merchant match",
                    f"Settlement date +{delta_days} days",
                    "Unique candidate",
                ],
                reason="Unique exact amount+merchant+date-window match.",
                timestamp=now,
            ))
            used_settlements.add(sid)

        return results
