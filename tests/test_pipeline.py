"""
ReconAI - Tests for Layers 2-5 and the integrated pipeline.
Small hand-crafted fixtures, isolating each layer's behavior.
"""

import sys
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
from reconciliation.models import Transaction, Settlement
from reconciliation.layer2_candidates import generate_candidates
from reconciliation.layer3_fuzzy_match import run_fuzzy_match, merchant_similarity
from reconciliation.layer4_grouping import (
    try_many_to_one, try_one_to_many, try_refund_adjusted,
    try_fee_adjusted_single, try_partial_settlement,
)
from reconciliation.layer5_agent import investigate, _investigate_rule_based
from reconciliation.confidence import score_layer1, score_fuzzy, score_grouping
from reconciliation.decision import decide


def d(s):
    return datetime.strptime(s, "%Y-%m-%d")


# ---- Layer 2 ----

def test_layer2_finds_wider_candidates_than_exact():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 1000.0, d("2026-09-08"), "Zomato")]  # outside Layer1's 3-day window
    cands = generate_candidates(txns, sets)
    assert len(cands["T1"]) == 1


def test_layer2_respects_amount_tolerance():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Zomato")]
    sets = [Settlement("S1", 5000.0, d("2026-09-02"), "Zomato")]  # way outside tolerance
    cands = generate_candidates(txns, sets)
    assert len(cands["T1"]) == 0


# ---- Layer 3 ----

def test_fuzzy_match_handles_known_abbreviation():
    sim = merchant_similarity("Flipkart", "Flipkart Internet Pvt Ltd")
    assert sim >= 60


def test_fuzzy_match_resolves_unique_close_candidate():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Myntra")]
    sets = [Settlement("S1", 1000.0, d("2026-09-02"), "Myntra Designs")]
    pools = generate_candidates(txns, sets)
    result = run_fuzzy_match(txns, sets, pools)
    assert result["T1"]["matched"] is True
    assert result["T1"]["settlement_id"] == "S1"


def test_fuzzy_match_refuses_ambiguous_candidates():
    txns = [Transaction("T1", 1000.0, d("2026-09-01"), "Cafe A")]
    sets = [
        Settlement("S1", 1000.0, d("2026-09-02"), "Cafe A"),
        Settlement("S2", 1000.0, d("2026-09-02"), "Cafe A Branch 2"),
    ]
    pools = generate_candidates(txns, sets)
    result = run_fuzzy_match(txns, sets, pools)
    # both candidates are highly similar and close - should not force a guess
    assert result["T1"]["matched"] is False


# ---- Layer 4 ----

def test_many_to_one_with_fee():
    txns = [
        Transaction("T1", 1000.0, d("2026-09-01"), "Swiggy"),
        Transaction("T2", 2000.0, d("2026-09-01"), "Swiggy"),
    ]
    sets = [Settlement("S1", 2940.0, d("2026-09-03"), "Swiggy")]  # 3000 - 60 fee
    fees = [dict(fee_id="F1", related_txn_ids="T1;T2", amount=60.0, type="platform_fee")]
    hits = try_many_to_one(txns, sets, fees)
    assert len(hits) == 1
    assert set(hits[0]["txn_ids"]) == {"T1", "T2"}
    assert hits[0]["fee_id"] == "F1"


def test_one_to_many_split_payout():
    txns = [Transaction("T1", 10000.0, d("2026-09-01"), "Flipkart")]
    sets = [
        Settlement("S1", 6000.0, d("2026-09-03"), "Flipkart"),
        Settlement("S2", 4000.0, d("2026-09-05"), "Flipkart"),
    ]
    hits = try_one_to_many(txns, sets)
    assert len(hits) == 1
    assert set(hits[0]["settlement_ids"]) == {"S1", "S2"}


def test_refund_adjusted_match():
    txns = [Transaction("T1", 2000.0, d("2026-09-01"), "Myntra")]
    sets = [Settlement("S1", 1500.0, d("2026-09-05"), "Myntra")]
    refunds = [dict(refund_id="R1", original_txn_id="T1", amount=500.0, date="2026-09-04")]
    hits = try_refund_adjusted(txns, sets, refunds)
    assert len(hits) == 1
    assert hits[0]["settlement_id"] == "S1"
    assert hits[0]["refund_id"] == "R1"


def test_fee_adjusted_single_match():
    txns = [Transaction("T1", 4000.0, d("2026-09-01"), "Nykaa")]
    sets = [Settlement("S1", 3920.0, d("2026-09-03"), "Nykaa")]
    fees = [dict(fee_id="F1", related_txn_ids="T1", amount=80.0, type="platform_fee")]
    hits = try_fee_adjusted_single(txns, sets, fees)
    assert len(hits) == 1
    assert hits[0]["fee_id"] == "F1"


def test_partial_settlement_requires_settlement_after_transaction():
    txns = [Transaction("T1", 8000.0, d("2026-09-05"), "Ola")]
    sets = [Settlement("S1", 5000.0, d("2026-09-01"), "Ola")]  # BEFORE the transaction - invalid
    hits = try_partial_settlement(txns, sets)
    assert hits == []


def test_partial_settlement_detects_valid_partial():
    txns = [Transaction("T1", 8000.0, d("2026-09-01"), "Ola")]
    sets = [Settlement("S1", 5000.0, d("2026-09-03"), "Ola")]
    hits = try_partial_settlement(txns, sets)
    assert len(hits) == 1
    assert hits[0]["settlement_id"] == "S1"


# ---- Layer 5 rule-based logic (tested directly, since the public
# investigate() entry point now requires a real LLM key by design - see
# test_agent_public_entry_point_requires_llm_key below for that contract) ----

def test_rule_based_marks_no_candidate_as_unresolved():
    txn = Transaction("T1", 9999.0, d("2026-09-01"), "Uber")
    result = _investigate_rule_based(txn, [], [txn], [], [])
    assert result["decision"] == "UNRESOLVED"


def test_rule_based_flags_close_ambiguous_candidates_for_review():
    txn = Transaction("T1", 1000.0, d("2026-09-01"), "Cafe X")
    sets = [
        Settlement("S1", 1000.0, d("2026-09-02"), "Cafe X"),
        Settlement("S2", 1005.0, d("2026-09-02"), "Cafe X Branch"),
    ]
    result = _investigate_rule_based(txn, sets, [txn], [], [])
    assert result["decision"] in ("NEEDS_REVIEW",)


def test_agent_public_entry_point_requires_llm_key(monkeypatch):
    """Without any LLM API key configured, the public investigate() must
    NOT silently fall back to rule-based reasoning - this is an
    agentic-AI project, so a missing key is a genuine failure state
    (NEEDS_REVIEW + an explicit 'unavailable' audit message), not a
    working substitute."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("RECONAI_ALLOW_RULE_BASED_FALLBACK", raising=False)

    txn = Transaction("T1", 9999.0, d("2026-09-01"), "Uber")
    result = investigate(txn, [], [txn], [], [])
    assert result["decision"] == "NEEDS_REVIEW"
    assert "unavailable" in result["reasoning"].lower()


# ---- Confidence / Decision ----

def test_decision_thresholds():
    assert decide(90) == "AUTO_MATCHED"
    assert decide(50) == "NEEDS_REVIEW"
    assert decide(10) == "UNRESOLVED"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
