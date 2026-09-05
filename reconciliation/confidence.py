"""
ReconAI - Confidence scoring.

Turns evidence strings/flags gathered by layers 1-5 into a 0-100 score.
Weights are explicit and configurable here (not invented by an LLM).
This module does not decide AUTO/REVIEW/UNRESOLVED - that's decision.py;
this only produces the number those thresholds are compared against.
"""

WEIGHTS = dict(
    exact_amount=25,
    exact_merchant=20,
    unique_candidate=20,
    date_within_layer1_window=15,
    date_within_wider_window=7,
    fuzzy_high=15,       # similarity >= 0.8
    fuzzy_medium=8,       # similarity >= 0.55
    grouping_validated=25,
    fee_explained=10,
    refund_explained=10,
    partial_detected=-10,      # partial settlements are real but incomplete - not full confidence
    ambiguity_penalty=-35,
    no_candidate_penalty=-40,
)


def score_layer1(match_result) -> float:
    """Layer 1 matches are exact+unique by construction - always max confidence."""
    return 100.0 if match_result.status == "MATCHED_LAYER1" else 0.0


def score_fuzzy(fuzzy_result: dict) -> float:
    """NOTE: `similarity` is on rapidfuzz's 0-100 scale, not 0-1."""
    if not fuzzy_result["matched"]:
        return 0.0
    sim = fuzzy_result["similarity"] or 0
    # Exact amount + unique candidate is already strong; similarity on
    # top of that (especially >=80) makes this comparable to Layer 1.
    base = 80 if sim >= 80 else 65 if sim >= 60 else 45
    score = base + WEIGHTS["date_within_wider_window"]
    return min(score, 100.0)


def score_grouping(evidence: list, has_fee: bool, has_refund: bool) -> float:
    # An exact sum-match (within $1 tolerance) across a validated group is
    # strong, near-deterministic evidence - comparable to Layer 1's exact
    # match, just requiring one extra arithmetic step (fee/refund netting).
    score = 80 + WEIGHTS["date_within_wider_window"]
    if has_fee:
        score += WEIGHTS["fee_explained"]
    if has_refund:
        score += WEIGHTS["refund_explained"]
    return min(score, 100.0)


def score_partial(ratio: float) -> float:
    # partial settlements get a moderate score reflecting genuine partial
    # evidence, capped well below AUTO_MATCHED territory
    base = 40 + ratio * 20
    return min(base, 65.0)


def score_agent(decision: str, evidence: list) -> float:
    if decision == "UNRESOLVED":
        return max(5.0, 20.0 + WEIGHTS["no_candidate_penalty"])
    if decision == "AUTO_MATCHED":
        return 80.0
    # NEEDS_REVIEW - moderate, reflects "plausible but not safe" evidence
    return 45.0
