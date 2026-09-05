"""
ReconAI - Diagnose one specific case's Layer 5 audit trail.

Reads the most recent pipeline_results.csv (from scripts/run_pipeline.py
or evaluation/evaluate.py) and prints everything recorded for the given
transaction ID or case_ref - full audit trail, reasoning, self-critique.

Usage:
    python scripts/diagnose_case.py TXN1087
    python scripts/diagnose_case.py REC0066
"""

import sys
import os
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "pipeline_results.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_case.py <TXN_ID or case_ref>")
        return

    target = sys.argv[1]

    if not os.path.exists(RESULTS_PATH):
        print(f"No pipeline_results.csv found at {RESULTS_PATH}.")
        print("Run scripts/run_pipeline.py or evaluation/evaluate.py first.")
        return

    with open(RESULTS_PATH) as f:
        rows = list(csv.DictReader(f))

    matches = [
        r for r in rows
        if target == r["case_ref"]
        or target in r["transaction_ids"].split(";")
        or target in r["settlement_ids"].split(";")
    ]

    if not matches:
        print(f"No record found matching '{target}'.")
        return

    for r in matches:
        print("=" * 70)
        print(f"case_ref:          {r['case_ref']}")
        print(f"transaction_ids:   {r['transaction_ids']}")
        print(f"settlement_ids:    {r['settlement_ids']}")
        print(f"refund_ids:        {r['refund_ids']}")
        print(f"fee_ids:           {r['fee_ids']}")
        print(f"match_method:      {r['match_method']}")
        print(f"decision:          {r['decision']}")
        print(f"confidence:        {r['confidence']}")
        print()
        print("reasoning:")
        print(f"  {r['reasoning']}")
        print()
        print("audit_trail:")
        for step in r["audit_trail"].split(" | "):
            print(f"  - {step}")
        print()
        print("evidence:")
        for e in r["evidence"].split("; "):
            print(f"  - {e}")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
