"""
ReconAI - Decision thresholds.

Configurable, explicit thresholds turning a confidence score into one of
the three required categories. NEVER force AUTO_MATCHED just because a
record needs some output - UNRESOLVED and NEEDS_REVIEW are first-class,
correct outcomes.
"""

AUTO_MATCHED_THRESHOLD = 75.0
NEEDS_REVIEW_THRESHOLD = 20.0   # below this -> UNRESOLVED


def decide(confidence: float) -> str:
    if confidence >= AUTO_MATCHED_THRESHOLD:
        return "AUTO_MATCHED"
    if confidence >= NEEDS_REVIEW_THRESHOLD:
        return "NEEDS_REVIEW"
    return "UNRESOLVED"
