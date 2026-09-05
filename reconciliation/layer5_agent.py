"""
ReconAI - Layer 5: Agentic Investigation.

Runs on whatever is STILL unresolved after Layers 1-4: ambiguous
same-amount collisions, orphan settlements/refunds, missing settlements,
unusually long date-shift cases, and duplicate-record leftovers.

This is an agentic-AI hackathon project, so Layer 5 ALWAYS uses a real
LLM agent by default. It does NOT silently substitute rule-based
heuristics when a key is missing - see `investigate()` below for the
exact contract.

Provider selection:

  GEMINI_API_KEY (or GOOGLE_API_KEY) set -> real Gemini agent
  (reconciliation/llm_agent_gemini.py + llm_tools.py). Gemini is the only
  supported LLM provider for this project.

  No key configured -> NEEDS_REVIEW with an explicit "AI investigation
  unavailable" audit message. This is a genuine failure state, not a
  working fallback.

Whichever provider is used, the model investigates via ten controlled
tools (search_candidate_records, inspect_transaction, inspect_settlement,
inspect_refund, inspect_fee, calculate_amount_difference,
calculate_group_total, check_date_difference, compare_merchant_names,
verify_candidate_relationship), proposes a decision, is forced through an
explicit self-critique turn, then records a final decision. It can never
invent an ID/amount/date - every fact it uses must come back from a tool
call - and it can never do arithmetic itself.

On ANY failure of a real LLM call - missing package, network error, bad
response, tool-budget exhausted, whatever - the result is the same
NEEDS_REVIEW/"AI investigation unavailable" fallback, per the reliability
requirement that the AI layer must never crash reconciliation.

The ORIGINAL deterministic tool-sequence logic (below, in
`_investigate_rule_based`) still exists in this file for reference and is
still directly unit-tested, but the public `investigate()` entry point
does NOT call it by default. Set RECONAI_ALLOW_RULE_BASED_FALLBACK=1 if
you deliberately want that old behavior back for offline debugging.
"""

import os

# Auto-load a .env file if present, so GEMINI_API_KEY survives across
# terminal sessions instead of needing $env:GEMINI_API_KEY="..." re-run
# every time you open a new terminal/VS Code window. Never fatal if the
# python-dotenv package isn't installed or no .env file exists - falls
# through silently, same as before.
#
# IMPORTANT: uses an explicit absolute path (project root's .env), not
# load_dotenv()'s default CWD-relative search. A server process (e.g.
# uvicorn's --reload subprocess) may not have the same working directory
# as the terminal you launched it from, which could silently fail to
# find .env even though it's sitting right there in the project folder.
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PROJECT_ROOT_FOR_ENV, ".env"))
except ImportError:
    pass
from reconciliation.layer2_candidates import generate_candidates, generate_settlement_candidates
from reconciliation.layer3_fuzzy_match import merchant_similarity


# ---- Tools -----------------------------------------------------------

def exact_match(txn, settlements):
    """Tool: re-check for an exact amount+merchant+date match (cheap,
    catches nothing new by this point but keeps the tool sequence
    complete/inspectable per the spec)."""
    for s in settlements:
        if abs(txn.amount - s.amount) < 0.001 and txn.merchant_name.strip().lower() == s.merchant_name.strip().lower():
            return s
    return None


def find_candidates(txn, settlements):
    """Tool: wide candidate search (reuses Layer 2)."""
    return generate_candidates([txn], settlements, merchant_filter=False)[txn.txn_id]


def compare_amounts(txn, settlement):
    diff = abs(txn.amount - settlement.amount)
    return dict(diff=diff, ratio=diff / txn.amount if txn.amount else None)


def compare_dates(txn, settlement):
    return (settlement.date - txn.date).days


def fuzzy_match(txn, settlement):
    return merchant_similarity(txn.merchant_name, settlement.merchant_name)


def find_grouped_matches(txn, settlements, other_txns, fees):
    """Tool: last-chance grouping search with a slightly wider tolerance
    than Layer 4 used, for stragglers only."""
    from reconciliation.layer4_grouping import try_many_to_one, try_fee_adjusted_single
    single = try_fee_adjusted_single([txn], settlements, fees)
    if single:
        return single[0]
    group = try_many_to_one(other_txns + [txn], settlements, fees)
    for g in group:
        if txn.txn_id in g["txn_ids"]:
            return g
    return None


def check_refunds(txn, refunds):
    for r in refunds:
        if r["original_txn_id"] == txn.txn_id:
            return r
    return None


def check_fees(txn, fees):
    for f in fees:
        if txn.txn_id in f["related_txn_ids"].split(";"):
            return f
    return None


def retrieve_transaction_details(txn):
    return dict(txn_id=txn.txn_id, amount=txn.amount, date=str(txn.date.date()), merchant=txn.merchant_name)


# ---- Explanation (rule-based; swap point for a real LLM call) --------

def explain_decision(evidence: list, decision: str) -> str:
    """
    Builds a plain-English explanation from gathered evidence.
    SWAP POINT: replace this function's body with a real LLM call
    (e.g. passing `evidence` as context and asking for a one-paragraph
    explanation) if/when a live model API is available. Must remain
    grounded strictly in `evidence` - never invent facts not present here.
    """
    if not evidence:
        return f"No supporting evidence found; decision: {decision}."
    return f"Decision: {decision}. Evidence considered: " + "; ".join(evidence) + "."


# ---- Controller (rule-based fallback) -----------------------------------

def _investigate_rule_based(txn, all_settlements, all_txns, refunds, fees):
    """
    Runs the full deterministic tool sequence for one unresolved
    transaction and returns a dict:
    {decision, evidence, audit_steps, settlement_id or None, reasoning}
    """
    audit_steps = []
    evidence = []

    audit_steps.append("Tool: exact_match - re-checking for a missed exact match")
    exact = exact_match(txn, all_settlements)
    if exact:
        # Before trusting this, make sure no OTHER settlement is a close
        # enough competitor - an "exact" match found in isolation can
        # still be genuinely ambiguous if something else is nearly as
        # plausible (e.g. same amount, very similar name).
        competitors = [
            s for s in all_settlements
            if s.settlement_id != exact.settlement_id
            and abs(txn.amount - s.amount) <= max(txn.amount, s.amount) * 0.02
            and fuzzy_match(txn, s) >= 55
        ]
        if not competitors:
            evidence.append(f"Exact match found on re-check: {exact.settlement_id}, no close competitors")
            audit_steps.append(f"Result: found {exact.settlement_id}, verified unique")
            return dict(decision="AUTO_MATCHED", evidence=evidence, audit_steps=audit_steps,
                         settlement_id=exact.settlement_id,
                         reasoning=explain_decision(evidence, "AUTO_MATCHED"))
        evidence.append(f"Exact match {exact.settlement_id} found, but {[c.settlement_id for c in competitors]} "
                         f"are close enough competitors to be unsafe")
        audit_steps.append(f"Result: exact match not safely unique - competitors: {[c.settlement_id for c in competitors]}")

    audit_steps.append("Tool: find_candidates - wide search ignoring merchant-name prefilter")
    candidates = find_candidates(txn, all_settlements)
    audit_steps.append(f"Result: {len(candidates)} candidate(s) found: {[c.settlement_id for c in candidates]}")

    if len(candidates) == 0:
        audit_steps.append("Tool: check_refunds")
        refund = check_refunds(txn, refunds)
        audit_steps.append(f"Result: {'found ' + refund['refund_id'] if refund else 'none'}")

        audit_steps.append("Tool: check_fees")
        fee = check_fees(txn, fees)
        audit_steps.append(f"Result: {'found ' + fee['fee_id'] if fee else 'none'}")

        evidence.append("No plausible settlement candidate found by wide search")
        if refund and not fee:
            evidence.append(f"Refund {refund['refund_id']} exists but no matching settlement was found for the net amount")
        decision = "UNRESOLVED"
        audit_steps.append(f"Tool: record_decision -> {decision}")
        return dict(decision=decision, evidence=evidence, audit_steps=audit_steps, settlement_id=None,
                     reasoning=explain_decision(evidence, decision))

    audit_steps.append("Tool: compare_amounts / compare_dates / fuzzy_match on each candidate")
    scored = []
    for s in candidates:
        amt = compare_amounts(txn, s)
        days = compare_dates(txn, s)
        sim = fuzzy_match(txn, s)
        scored.append((s, amt, days, sim))
        evidence.append(f"Candidate {s.settlement_id}: amount diff {amt['diff']:.2f}, "
                         f"date +{days}d, name similarity {sim:.2f}")

    # try grouping as a last resort for this straggler
    audit_steps.append("Tool: find_grouped_matches - last-chance grouping search")
    other_txns = [t for t in all_txns if t.txn_id != txn.txn_id]
    group = find_grouped_matches(txn, all_settlements, other_txns, fees)
    if group:
        audit_steps.append(f"Result: grouping found - {group}")
        evidence.append(f"Grouped match found: {group}")
        return dict(decision="AUTO_MATCHED", evidence=evidence, audit_steps=audit_steps,
                     settlement_id=group.get("settlement_id"),
                     reasoning=explain_decision(evidence, "AUTO_MATCHED"))
    audit_steps.append("Result: no grouping explains the gap")

    # rank candidates: prefer close amount + high similarity
    scored.sort(key=lambda x: (x[1]["diff"], -x[3]))
    best = scored[0]
    if len(scored) > 1:
        second = scored[1]
        close_call = (best[1]["diff"] - second[1]["diff"] < 50) and (best[3] - second[3] < 0.15)
    else:
        close_call = False

    if close_call:
        evidence.append(f"Top two candidates ({best[0].settlement_id}, {second[0].settlement_id}) "
                         f"too close on amount and name similarity to decide safely")
        decision = "NEEDS_REVIEW"
    elif best[3] < 0.4 and best[1]["diff"] > best[0].amount * 0.02 if best[0].amount else True:
        evidence.append(f"Best candidate {best[0].settlement_id} has low name similarity "
                         f"({best[3]:.2f}) and a non-trivial amount gap; not confident enough to auto-match")
        decision = "NEEDS_REVIEW"
    else:
        evidence.append(f"Best candidate {best[0].settlement_id} is plausible but not unique/clean "
                         f"enough for earlier layers; escalating for human judgment")
        decision = "NEEDS_REVIEW"

    audit_steps.append(f"Tool: record_decision -> {decision}")
    return dict(decision=decision, evidence=evidence, audit_steps=audit_steps,
                settlement_id=best[0].settlement_id if decision == "AUTO_MATCHED" else None,
                reasoning=explain_decision(evidence, decision))


# ---- Public entry point (always uses the real LLM agent) ----------------

def has_llm_key() -> bool:
    """Cheap check other entry points (run_pipeline.py, evaluate.py) can
    use to print a loud upfront warning instead of letting a missing key
    silently show up as scattered per-case 'AI investigation unavailable'
    lines that are easy to miss in a long run."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def investigate(txn, all_settlements, all_txns, refunds, fees):
    """
    Public Layer 5 entry point used by pipeline.py.

    This is an agentic-AI hackathon project, so Layer 5 ALWAYS uses a real
    LLM agent by default - it does not silently substitute rule-based
    heuristics for missing credentials, since that would misrepresent
    what's actually doing the reasoning.

    Provider selection:
      GEMINI_API_KEY (or GOOGLE_API_KEY) set -> real Gemini agent
      (reconciliation/llm_agent_gemini.py). Gemini is the only supported
      LLM provider for this project.

      No key configured -> NEEDS_REVIEW with an "AI investigation
      unavailable" audit message. This is a genuine failure state, not
      a working fallback - Layer 5 does no real investigation in this
      case.

    On ANY failure of a real LLM call - missing package, network error,
    bad response, tool-budget exhausted, whatever - falls back to the same
    NEEDS_REVIEW/"AI investigation unavailable" result, per the reliability
    requirement that the AI layer must never crash reconciliation.

    The original deterministic tool-sequence logic still exists as
    `_investigate_rule_based` in this file (kept for reference/offline
    debugging, and still directly unit-tested), but it is NOT used by this
    entry point unless you call it yourself or set
    RECONAI_ALLOW_RULE_BASED_FALLBACK=1.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    allow_rule_based = os.environ.get("RECONAI_ALLOW_RULE_BASED_FALLBACK") == "1"

    if not gemini_key:
        if allow_rule_based:
            return _investigate_rule_based(txn, all_settlements, all_txns, refunds, fees)
        return dict(
            decision="NEEDS_REVIEW",
            evidence=["No LLM API key configured"],
            audit_steps=["AI investigation unavailable (no API key configured). Escalated for human review."],
            settlement_id=None,
            reasoning="AI investigation unavailable. Escalated for human review.",
        )

    provider = "gemini"
    try:
        from reconciliation.llm_agent_gemini import investigate_with_llm
        return investigate_with_llm(txn, all_settlements, all_txns, refunds, fees)
    except Exception as e:
        return dict(
            decision="NEEDS_REVIEW",
            evidence=[f"LLM investigation failed ({provider}): {e}"],
            audit_steps=[f"AI investigation unavailable ({provider}: {e}). Escalated for human review."],
            settlement_id=None,
            reasoning="AI investigation unavailable. Escalated for human review.",
        )
