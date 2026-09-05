"""
ReconAI - Layer 5 real LLM agent (Google Gemini), with tool use and a
mandatory self-critique turn.

REQUIRES, in your own environment (not this sandbox):
    pip install google-genai
    export GEMINI_API_KEY=...      (get one free, no credit card, at
                                     https://aistudio.google.com/apikey)

Uses the current, non-deprecated `google-genai` unified SDK (NOT the old
`google-generativeai` package, which Google has deprecated). Verified
against google-genai 2.21.0's actual installed type signatures while
building this, not just documentation.

This module is only ever called from layer5_agent.investigate(), which
wraps every call to this file in a try/except and falls back to
NEEDS_REVIEW with an "AI investigation unavailable" audit message if
ANYTHING here fails - missing package, missing key, network error, rate
limit, malformed response, whatever. Reconciliation must never crash
because the AI layer had a bad day.

Flow:
  1. Agent investigates using tools.
  2. It self-critiques against a fixed checklist internally in the same context.
  3. It calls `record_decision` once with the final decision and critique.

Grounding: the system prompt forbids inventing any ID/amount/date not
returned by a tool, and forbids doing arithmetic itself - every number
the agent uses must come from calculate_amount_difference,
calculate_group_total, check_date_difference, or
verify_candidate_relationship. Automatic function calling is explicitly
DISABLED so every tool call is executed by our own code against our own
data - the model never gets to run anything on its own.

Determinism note: temperature is set to 0 (supported by this SDK, unlike
the provider SDK version used by the project) and a simple
on-disk cache (keyed by exact case inputs + model name) means repeated
evaluation runs reuse the same result instead of re-calling the API.
"""

import os
import json
import hashlib
import time

from google import genai
from google.genai import types

from reconciliation.llm_tools import TOOL_SCHEMAS, ToolExecutor

DEFAULT_MODEL = os.environ.get("RECONAI_LLM_MODEL", "gemini-3.5-flash-lite")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "synthetic", ".llm_agent_cache_gemini.json")

MAX_TOOL_ITERATIONS = 10
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = [2, 5, 10]


def _generate_with_retry(client, **kwargs):
    """Wraps client.models.generate_content with a short retry-with-backoff
    on transient rate-limit (429 RESOURCE_EXHAUSTED) errors - these are
    usually per-minute bursts that clear within seconds, not the harder
    per-day cap, so a brief wait-and-retry recovers most of them instead
    of failing the whole case outright."""
    last_error = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            msg = str(e)
            is_transient = (
                "RESOURCE_EXHAUSTED" in msg
                or "429" in msg
                or "UNAVAILABLE" in msg
                or "503" in msg
            )
            if not is_transient or attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            last_error = e
            delay = RATE_LIMIT_BACKOFF_SECONDS[min(attempt, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)]
            time.sleep(delay)
    raise last_error

SYSTEM_PROMPT = """You are ReconAI's financial reconciliation investigator.

You are given ONE unresolved transaction. Your job is to determine, using
ONLY the tools provided, whether it should be:
  - AUTO_MATCHED to a specific settlement (or group of settlements/transactions),
  - NEEDS_REVIEW (plausible but not safe to auto-match), or
  - UNRESOLVED (no plausible explanation found).

STRICT RULES:
1. You must NEVER state a transaction ID, settlement ID, amount, date, or
   fee/refund value that did not come from a tool result. If you don't know
   something, call a tool to find out - never guess or infer.
2. You must NEVER do arithmetic yourself. Always use
   calculate_amount_difference, calculate_group_total, check_date_difference,
   or verify_candidate_relationship.
3. AUTO_MATCHED requires verify_candidate_relationship to report
   balances_within_tolerance=true AND no other equally plausible candidate
   left unexplained. If you are not sure, choose NEEDS_REVIEW - never force
   a match to produce an answer.
4. A transaction can be settled by MORE THAN ONE settlement (a split
   payout). If a single settlement candidate does not balance, use
   search_candidate_records to gather same-merchant candidates, then use
   calculate_group_total on plausible subsets to check whether a GROUP of
   settlements sums to the transaction amount, and confirm with
   verify_candidate_relationship passing multiple settlement_ids before
   concluding no match exists.
5. Distinguish NEEDS_REVIEW from UNRESOLVED carefully: choose UNRESOLVED
   when, after checking single candidates AND grouped candidates, nothing
   balances even loosely and no candidate is genuinely plausible - do not
   default to NEEDS_REVIEW just because search_candidate_records returned
   some coincidentally nearby settlement that doesn't actually explain the
   transaction. Reserve NEEDS_REVIEW for cases where a plausible candidate
   genuinely exists but you cannot be fully certain (e.g. two equally
   strong candidates, or a partial settlement with a sensible remaining
   balance).
6. When you believe you have enough evidence, call propose_decision with
   your provisional decision and reasoning. You will then be asked to
   self-critique before your decision is final.
"""

SELF_CRITIQUE_PROMPT = """Before calling record_decision, silently self-check the proposed conclusion against this checklist:
- Could another candidate be equally valid?
- Does the amount arithmetic actually work according to the verification tools?
- Did you check whether a group of settlements could explain this transaction?
- Does the date relationship make sense?
- Could this be a duplicate transaction?
- Could this be a partial settlement?
- Could a refund or fee explain a remaining difference?
- If nothing balances, should this be UNRESOLVED rather than NEEDS_REVIEW?
- Is there truly enough evidence to AUTO_MATCH, or should this be NEEDS_REVIEW?

Do not invent facts. Include a concise summary of this self-check in the self_critique field of record_decision.
"""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["AUTO_MATCHED", "NEEDS_REVIEW", "UNRESOLVED"]},
        "matched_transaction_ids": {"type": "array", "items": {"type": "string"}},
        "matched_settlement_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["decision", "reasoning"],
}

RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["AUTO_MATCHED", "NEEDS_REVIEW", "UNRESOLVED"]},
        "matched_transaction_ids": {"type": "array", "items": {"type": "string"}},
        "matched_settlement_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "self_critique": {"type": "string"},
    },
    "required": ["decision", "reasoning", "self_critique"],
}


def _investigation_tools():
    """Converts the shared, provider-agnostic TOOL_SCHEMAS (written in
    provider-neutral {name, description, input_schema} shape) into Gemini
    FunctionDeclaration objects."""
    return [
        types.FunctionDeclaration(
            name=s["name"], description=s["description"], parameters_json_schema=s["input_schema"],
        )
        for s in TOOL_SCHEMAS
    ]


def _cache_key(txn, settlements, refunds, fees):
    # Include a hash of the current prompts so that changing the system
    # prompt or self-critique checklist automatically invalidates old
    # cached answers, instead of silently replaying stale reasoning from
    # before the prompt changed.
    prompt_fingerprint = hashlib.sha256((SYSTEM_PROMPT + SELF_CRITIQUE_PROMPT).encode()).hexdigest()[:16]
    payload = dict(
        txn=dict(id=txn.txn_id, amount=txn.amount, date=str(txn.date.date()), merchant=txn.merchant_name),
        settlement_ids=sorted(s.settlement_id for s in settlements),
        refund_ids=sorted(r["refund_id"] for r in refunds),
        fee_ids=sorted(f["fee_id"] for f in fees),
        model=DEFAULT_MODEL,
        prompt_fingerprint=prompt_fingerprint,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass   # cache is a nice-to-have, never fatal


def _run_tool_loop(client, contents, terminal_decl, terminal_name, executor):
    """Runs the generate_content/function-call loop until the model calls
    `terminal_name`, or MAX_TOOL_ITERATIONS is hit. Returns
    (terminal_args_dict_or_None, updated_contents)."""
    tool = types.Tool(function_declarations=_investigation_tools() + [terminal_decl])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        response = _generate_with_retry(client, model=DEFAULT_MODEL, contents=contents, config=config)
        fn_calls = response.function_calls or []
        if not fn_calls:
            return None, contents

        model_content = response.candidates[0].content
        contents = contents + [model_content]

        terminal_result = None
        response_parts = []
        for fc in fn_calls:
            if fc.name == terminal_name:
                terminal_result = dict(fc.args or {})
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"status": "recorded"}))
            else:
                result = executor.run(fc.name, dict(fc.args or {}))
                response_parts.append(types.Part.from_function_response(name=fc.name, response=result))

        contents = contents + [types.Content(role="user", parts=response_parts)]
        if terminal_result is not None:
            return terminal_result, contents

    return None, contents


def investigate_with_llm(txn, all_settlements, all_txns, refunds, fees) -> dict:
    """
    Real Gemini-backed investigation with tool use + self-critique.
    Raises on any failure - the caller (layer5_agent.investigate) is
    responsible for catching and falling back to NEEDS_REVIEW.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    cache = _load_cache()
    key = _cache_key(txn, all_settlements, refunds, fees)
    if key in cache:
        return cache[key]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    )
    executor = ToolExecutor(all_txns, all_settlements, refunds, fees)

    record_decl = types.FunctionDeclaration(
        name="record_decision",
        description="Record the FINAL decision after investigating and self-checking the evidence. This ends the investigation.",
        parameters_json_schema=RECORD_SCHEMA,
    )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=(
        f"Investigate transaction {txn.txn_id}. Use inspect_transaction first, then "
        f"use search_candidate_records and other investigation tools only when useful. "
        f"You have a limited tool budget. Do not repeatedly inspect the same evidence. "
        f"When you have enough evidence, stop investigating, perform the self-critique "
        f"checklist internally, and call record_decision exactly once with the FINAL "
        f"decision, concise evidence-based reasoning, and self_critique. "
        f"If evidence is insufficient, choose NEEDS_REVIEW or UNRESOLVED rather than forcing a match.\n\n"
        f"{SELF_CRITIQUE_PROMPT}"
    ))])]

    final_decision, contents = _run_tool_loop(client, contents, record_decl, "record_decision", executor)
    if final_decision is None:
        raise RuntimeError("Agent did not record a final decision within the tool-call budget")

    audit_steps = [f"Tool: {c['tool']}({c['input']}) -> {c['result']}" for c in executor.tool_call_log]
    audit_steps.append(f"Self-critique: {str(final_decision.get('self_critique', ''))[:300]}")
    audit_steps.append(f"Final: {final_decision.get('decision')}")

    settlement_ids = list(final_decision.get("matched_settlement_ids") or [])
    result = dict(
        decision=final_decision.get("decision", "NEEDS_REVIEW"),
        evidence=[f"{c['tool']}: {c['result']}" for c in executor.tool_call_log],
        audit_steps=audit_steps,
        settlement_id=settlement_ids[0] if settlement_ids else None,
        settlement_ids=settlement_ids,
        matched_transaction_ids=list(final_decision.get("matched_transaction_ids") or []),
        reasoning=final_decision.get("reasoning", ""),
        self_critique=final_decision.get("self_critique", ""),
        llm_used=True,
        llm_provider="gemini",
    )

    cache[key] = result
    _save_cache(cache)
    return result
