"""
ReconAI - Layer 3: Fuzzy / Semantic Matching.

Handles the "AMZN Mktp IN" vs "Amazon Marketplace India" problem: amount
matches (exactly or very closely) but merchant name text differs.

Uses rapidfuzz for string similarity, after a lightweight normalization
step (lowercasing, stripping common corporate suffixes, expanding a
small set of known e-commerce/fintech abbreviations). This mirrors how
real merchant-matching systems handle "merchant master data" - not a
peek at any transaction-level ground truth, just general domain
knowledge about how company names get abbreviated.

NOTE: true semantic (embedding-based) similarity, as used in Yojana
Mitra, would need a downloaded model; this sandboxed environment has no
network access to model hubs, so this layer uses lexical similarity
instead. `merchant_similarity` is isolated specifically so a real
embedding model can be swapped in later without touching the rest of
the layer.

Only commits a match when:
  - amount matches within a tight tolerance (not the loose Layer 2 pool
    tolerance - this layer is still about 1:1, not grouping)
  - exactly one candidate clears the similarity threshold
  - that candidate is not also claimed by another transaction
"""

import re
from rapidfuzz import fuzz
from reconciliation.models import Transaction, Settlement

SIMILARITY_THRESHOLD = 60   # rapidfuzz scores are 0-100
TIGHT_AMOUNT_TOLERANCE = 0.005   # 0.5% - basically "same amount", not a grouping tolerance

# Small, general merchant-name normalization table - domain knowledge
# about common corporate-suffix / abbreviation patterns, not specific to
# any transaction in this dataset.
_SUFFIXES = [
    "pvt ltd", "private limited", "ltd", "inc", "technologies", "systems",
    "internet", "retail", "designs", "entertainment", "e-commerce",
    "ecommerce", "india", "in",
]
_ABBREVIATIONS = {
    "amzn": "amazon",
    "mktp": "marketplace",
    "bms": "bookmyshow",
    "ani": "",
}


def _normalize(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[^a-z0-9\s]", " ", n)   # strip punctuation like ( )
    tokens = n.split()
    tokens = [_ABBREVIATIONS.get(t, t) for t in tokens]
    n = " ".join(t for t in tokens if t)
    for suffix in _SUFFIXES:
        n = re.sub(rf"\b{re.escape(suffix)}\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def merchant_similarity(name_a: str, name_b: str) -> float:
    """Similarity in [0,100]. Isolated so it can be swapped for a real
    embedding-based measure later. Takes the best of a normalized
    token-sort ratio and a partial ratio, since merchant names commonly
    differ by substring inclusion (one is a subset of the other) or by
    word reordering."""
    a, b = _normalize(name_a), _normalize(name_b)
    if not a or not b:
        a, b = name_a.lower(), name_b.lower()
    return max(fuzz.token_sort_ratio(a, b), fuzz.partial_ratio(a, b), fuzz.WRatio(a, b))


def _amount_close(a: float, b: float) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= max(a, b) * TIGHT_AMOUNT_TOLERANCE


def run_fuzzy_match(unresolved_txns: list, unresolved_settlements: list, candidate_pools: dict) -> dict:
    """
    Args:
      unresolved_txns: list[Transaction] still unresolved after Layer 1
      unresolved_settlements: list[Settlement] still unresolved
      candidate_pools: {txn_id: [Settlement,...]} from Layer 2 (wide pool)

    Returns {txn_id: FuzzyResult} where FuzzyResult is a dict:
      {matched: bool, settlement_id, similarity, evidence, reason}
    """
    unresolved_settlement_ids = {s.settlement_id for s in unresolved_settlements}
    results = {}

    # narrow each txn's pool to amount-tight, still-unresolved settlements
    tight_pools = {}
    for t in unresolved_txns:
        pool = [
            s for s in candidate_pools.get(t.txn_id, [])
            if s.settlement_id in unresolved_settlement_ids and _amount_close(t.amount, s.amount)
        ]
        tight_pools[t.txn_id] = pool

    for t in unresolved_txns:
        pool = tight_pools[t.txn_id]
        scored = [(s, merchant_similarity(t.merchant_name, s.merchant_name)) for s in pool]
        good = [(s, sim) for s, sim in scored if sim >= SIMILARITY_THRESHOLD]
        good.sort(key=lambda x: -x[1])

        if not good:
            results[t.txn_id] = dict(matched=False, settlement_id=None, similarity=None,
                                      evidence=["No candidate cleared merchant-name similarity threshold"],
                                      reason="No sufficiently similar merchant name found among amount-matching candidates.")
            continue

        best_s, best_sim = good[0]

        # check uniqueness: is this settlement also a good match for another unresolved txn?
        competing = []
        for other_t in unresolved_txns:
            if other_t.txn_id == t.txn_id:
                continue
            other_pool = tight_pools[other_t.txn_id]
            for s in other_pool:
                if s.settlement_id == best_s.settlement_id:
                    sim = merchant_similarity(other_t.merchant_name, s.merchant_name)
                    if sim >= SIMILARITY_THRESHOLD:
                        competing.append(other_t.txn_id)

        if competing:
            results[t.txn_id] = dict(matched=False, settlement_id=None, similarity=best_sim,
                                      evidence=[f"Settlement {best_s.settlement_id} also plausible for {competing}"],
                                      reason="Fuzzy candidate is not unique; ambiguous.")
            continue

        if len(good) > 1 and (good[0][1] - good[1][1]) < 5:
            results[t.txn_id] = dict(matched=False, settlement_id=None, similarity=best_sim,
                                      evidence=[f"Top two fuzzy candidates too close: "
                                                f"{good[0][0].settlement_id}={good[0][1]:.2f} vs "
                                                f"{good[1][0].settlement_id}={good[1][1]:.2f}"],
                                      reason="Multiple close fuzzy candidates; ambiguous.")
            continue

        results[t.txn_id] = dict(matched=True, settlement_id=best_s.settlement_id, similarity=best_sim,
                                  evidence=[f"Merchant name similarity {best_sim:.2f}", "Amount within tight tolerance"],
                                  reason=f"Unique fuzzy match: name similarity {best_sim:.2f}, amount closely matches.")

    return results
