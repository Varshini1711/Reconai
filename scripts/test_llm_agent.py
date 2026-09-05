"""
ReconAI - Quick smoke test for the real LLM agent (Layer 5).

Run this FIRST, before scripts/run_pipeline.py, to confirm your
GEMINI_API_KEY and the Gemini package actually work end-to-end,
against a single simple case - not the whole unresolved pool.

Usage:
    export GEMINI_API_KEY=...
    python scripts/test_llm_agent.py
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
from reconciliation.models import Transaction, Settlement
from reconciliation.layer5_agent import investigate


def main():
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("No LLM API key is set. Set one first, e.g.:")
        print('  export GEMINI_API_KEY="..."         (macOS/Linux)')
        print('  $env:GEMINI_API_KEY="..."            (PowerShell)')
        return
    print("Using provider: gemini")

    # A simple, clearly-resolvable case: one transaction, one settlement,
    # amounts match exactly minus a small fee that requires a tool call
    # to discover - good for confirming the agent actually calls tools
    # rather than guessing.
    txn = Transaction("TXN_TEST_1", 1000.0, datetime(2026, 9, 1), "Test Cafe")
    settlement = Settlement("SET_TEST_1", 980.0, datetime(2026, 9, 3), "Test Cafe")
    fees = [dict(fee_id="FEE_TEST_1", related_txn_ids="TXN_TEST_1", amount=20.0, type="platform_fee")]

    print("Investigating TXN_TEST_1 (expect the agent to find the fee and consider AUTO_MATCHED)...")
    result = investigate(txn, [settlement], [txn], [], fees)

    print("\n--- Result ---")
    print(f"Decision: {result['decision']}")
    print(f"Reasoning: {result.get('reasoning', '')}")
    if "self_critique" in result:
        print(f"Self-critique: {result['self_critique']}")
    print(f"\nUsed real LLM: {result.get('llm_used', False)}")
    if not result.get("llm_used"):
        print("NOTE: This did NOT use the real LLM - check the 'evidence'/'audit_steps' "
              "below for why it fell back.")
    print("\n--- Audit trail ---")
    for step in result.get("audit_steps", []):
        print(f"  {step}")


if __name__ == "__main__":
    main()
