"""
ReconAI - Unit tests for Layer 1 deterministic matching.

Uses small hand-crafted fixtures, not the big synthetic dataset, so each
test isolates exactly one behavior.
"""

import sys
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
from reconciliation.models import Transaction, Settlement
from reconciliation.layer1_exact_match import Layer1ExactMatcher


def d(s):
    return datetime.strptime(s, "%Y-%m-%d")


def result_for(results, txn_id):
    return next(r for r in results if r.txn_id == txn_id)


# 1. Exact 1-to-1 match
def test_exact_1to1_match():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 1000.0, d("2026-09-02"), "Zomato")]
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "MATCHED_LAYER1"
    assert r.settlement_id == "S1"


# 2. Different merchant name -> not matched by Layer 1
def test_different_merchant_not_matched():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 1000.0, d("2026-09-02"), "Swiggy")]
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "UNMATCHED_LAYER1"
    assert r.settlement_id is None


# 3. Settlement outside date window -> not matched
def test_outside_date_window_not_matched():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 1000.0, d("2026-09-10"), "Zomato")]  # 9 days later, window is 3
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "UNMATCHED_LAYER1"


# 4. Multiple equally valid candidates -> not matched
def test_multiple_candidates_not_matched():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [
        Settlement("S1", 1000.0, d("2026-09-02"), "Zomato"),
        Settlement("S2", 1000.0, d("2026-09-03"), "Zomato"),
    ]
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "UNMATCHED_LAYER1"
    assert r.settlement_id is None


# 5. Exact amount + merchant + valid date + unique candidate -> matched
def test_unique_candidate_matched():
    txns = [
        Transaction("T1", 1000.0, d("2026-09-01"), "Zomato"),
        Transaction("T2", 2000.0, d("2026-09-01"), "Swiggy"),
    ]
    sets = [
        Settlement("S1", 1000.0, d("2026-09-02"), "Zomato"),
        Settlement("S2", 2000.0, d("2026-09-02"), "Swiggy"),
    ]
    results = Layer1ExactMatcher(txns, sets).run()
    assert result_for(results, "T1").settlement_id == "S1"
    assert result_for(results, "T2").settlement_id == "S2"


# 6. Missing settlement -> unmatched
def test_missing_settlement_unmatched():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = []
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "UNMATCHED_LAYER1"
    assert r.settlement_id is None


# 7. Duplicate transaction situation -> must not double-match
def test_duplicate_transactions_do_not_double_match():
    txns = [
        Transaction("T1", 1000.0, d("2026-09-01"), "Zomato"),
        Transaction("T2", 1000.0, d("2026-09-01"), "Zomato"),  # duplicate of T1
    ]
    sets = [Settlement("S1", 1000.0, d("2026-09-02"), "Zomato")]  # only ONE settlement exists
    results = Layer1ExactMatcher(txns, sets).run()
    matched = [r for r in results if r.status == "MATCHED_LAYER1"]
    # Neither should be force-matched since both are equally plausible
    # for the single settlement - Layer 1 must not guess which is real.
    assert len(matched) == 0
    assert result_for(results, "T1").status == "UNMATCHED_LAYER1"
    assert result_for(results, "T2").status == "UNMATCHED_LAYER1"


# 8. Basic invalid/malformed data handling
def test_empty_input_lists_handled_gracefully():
    results = Layer1ExactMatcher([], []).run()
    assert results == []


def test_zero_amount_and_same_day_handled():
    # Edge case: zero amount, same-day settlement (0-day window)
    txns = [Transaction("T1", 0.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 0.0, d("2026-09-01"), "Zomato")]
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "MATCHED_LAYER1"


def test_negative_date_delta_not_matched():
    # Settlement BEFORE transaction date should never match (money can't
    # be paid out before the transaction occurred).
    txns = [Transaction("T1", 1000.0, d("2026-09-05"), "Zomato")]
    sets = [Settlement("S1", 1000.0, d("2026-09-01"), "Zomato")]
    results = Layer1ExactMatcher(txns, sets).run()
    r = result_for(results, "T1")
    assert r.status == "UNMATCHED_LAYER1"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
