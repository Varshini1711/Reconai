"""
ReconAI - Layer 2: Candidate Generation.

Does NOT decide any match. Its only job: for each still-unresolved
transaction, find a small pool of *plausible* settlements using a wider
amount tolerance and date window than Layer 1's strict exact match, and
the reverse (settlement -> plausible transactions), for later layers to
use for fuzzy matching, grouping, and agent investigation.

Kept deliberately dumb and cheap: no string similarity, no arithmetic
beyond a tolerance check. That belongs to Layer 3 (fuzzy) and Layer 4
(grouped/fee/refund arithmetic).
"""

from reconciliation.models import Transaction, Settlement

AMOUNT_TOLERANCE_RATIO = 0.15     # allow up to 15% amount difference (covers fees/refunds/partials)
DATE_WINDOW_DAYS = 10              # wider than Layer 1's 3-day window


def _amount_within_tolerance(a: float, b: float) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= max(a, b) * AMOUNT_TOLERANCE_RATIO


def _date_within_window(t: Transaction, s: Settlement, window_days: int = DATE_WINDOW_DAYS) -> bool:
    delta = (s.date - t.date).days
    return -1 <= delta <= window_days   # allow -1 day for same-day timezone edge cases


def generate_candidates(transactions: list, settlements: list, merchant_filter: bool = True) -> dict:
    """
    Returns {txn_id: [Settlement, ...]} - plausible settlement candidates
    per transaction, widened beyond Layer 1's strict criteria.

    merchant_filter=True restricts to settlements sharing SOME textual
    overlap in merchant name (cheap prefilter, not similarity scoring -
    that's Layer 3's job) to keep pools small; set False to search
    everything (used sparingly, e.g. for orphan investigation in Layer 5).
    """
    candidates = {}
    for t in transactions:
        pool = []
        for s in settlements:
            if not _amount_within_tolerance(t.amount, s.amount):
                continue
            if not _date_within_window(t, s):
                continue
            if merchant_filter:
                t_tokens = set(t.merchant_name.lower().split())
                s_tokens = set(s.merchant_name.lower().split())
                if not (t_tokens & s_tokens):
                    continue
            pool.append(s)
        candidates[t.txn_id] = pool
    return candidates


def generate_settlement_candidates(settlements: list, transactions: list, merchant_filter: bool = True) -> dict:
    """Reverse direction: {settlement_id: [Transaction, ...]} - used by
    Layer 4 for many-to-one / one-to-many grouping search."""
    candidates = {}
    for s in settlements:
        pool = []
        for t in transactions:
            if not _amount_within_tolerance(t.amount, s.amount):
                # NOTE: for grouping this per-item check is loose on purpose;
                # Layer 4 does its own sum-based tolerance check on subsets.
                if t.amount > s.amount * (1 + AMOUNT_TOLERANCE_RATIO):
                    continue
            if not _date_within_window(t, s):
                continue
            if merchant_filter:
                t_tokens = set(t.merchant_name.lower().split())
                s_tokens = set(s.merchant_name.lower().split())
                if not (t_tokens & s_tokens):
                    continue
            pool.append(t)
        candidates[s.settlement_id] = pool
    return candidates
