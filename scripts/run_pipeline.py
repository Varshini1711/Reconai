import sys
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from reconciliation.pipeline import run_pipeline, write_pipeline_results
from reconciliation.layer5_agent import has_llm_key
from ask_ledger.ask_ledger import answer

def main():
    if not has_llm_key():
        print("=" * 70)
        print("WARNING: GEMINI_API_KEY is not set in this environment.")
        print("Layer 5 will report NEEDS_REVIEW for every unresolved case")
        print("instead of running real AI investigation. If you meant to")
        print("test the real agent, set it first:")
        print('  $env:GEMINI_API_KEY="your-key-here"   (PowerShell)')
        print("or create a .env file (see .env.example) so it persists")
        print("across terminal sessions.")
        print("=" * 70)
        print()

    records, transactions, settlements, refunds, fees = run_pipeline()
    path = write_pipeline_results(records)

    auto = [r for r in records if r.decision == "AUTO_MATCHED"]
    review = [r for r in records if r.decision == "NEEDS_REVIEW"]
    unresolved = [r for r in records if r.decision == "UNRESOLVED"]

    print("=== ReconAI Full Pipeline Report ===")
    print(f"Total records      : {len(records)}")
    print(f"AUTO_MATCHED       : {len(auto)}")
    print(f"NEEDS_REVIEW       : {len(review)}")
    print(f"UNRESOLVED         : {len(unresolved)}")
    print(f"Results written to : {path}")

    print("\n--- Sample AUTO_MATCHED (by method) ---")
    seen_methods = set()
    for r in auto:
        if r.match_method not in seen_methods:
            print(f"  [{r.match_method}] {r.transaction_ids} -> {r.settlement_ids} | {r.reasoning[:100]}")
            seen_methods.add(r.match_method)

    print("\n--- Sample NEEDS_REVIEW ---")
    for r in review[:5]:
        print(f"  {r.transaction_ids or r.settlement_ids}: {r.reasoning[:120]}")

    print("\n--- Sample UNRESOLVED ---")
    for r in unresolved[:5]:
        print(f"  {r.transaction_ids or r.settlement_ids}: {r.reasoning[:120]}")

    print("\n--- Ask the Ledger demo ---")
    demo_questions = [
        "Give me a summary",
        "Show me all unresolved records",
        "Which merchants have the most exceptions?",
    ]
    for q in demo_questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(q, records, transactions, settlements)}")

if __name__ == "__main__":
    main()
