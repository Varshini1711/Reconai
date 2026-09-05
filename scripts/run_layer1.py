"""
ReconAI - Run Layer 1 against the synthetic dataset.

Loads visible data -> runs Layer 1 -> writes results CSV -> prints report.
Also runs an OPTIONAL sanity check against DEV ground truth only (never
TEST ground truth), purely for our own confidence - not used to tune
Layer 1's logic itself.
"""

import sys
import csv
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from reconciliation.loader import load_transactions, load_settlements
from reconciliation.layer1_exact_match import Layer1ExactMatcher

RESULTS_PATH = PROJECT_ROOT + "/data/synthetic/layer1_results.csv"
DEV_GT_PATH = PROJECT_ROOT + "/data/ground_truth/ground_truth_dev.csv"


def write_results(results):
    with open(RESULTS_PATH, "w", newline="") as f:
        fieldnames = ["txn_id", "settlement_id", "status", "match_method", "evidence", "reason", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def dev_sanity_check(results):
    """
    Sanity check ONLY - compares Layer 1 output to DEV ground truth to
    build confidence the layer behaves as designed. This is NOT used to
    tune Layer 1's matching logic, and TEST ground truth is never read
    here or anywhere near reconciliation code.
    """
    result_by_txn = {r.txn_id: r for r in results}
    with open(DEV_GT_PATH) as f:
        dev_cases = list(csv.DictReader(f))

    print("\n--- DEV ground truth sanity check (not used to tune Layer 1) ---")
    by_type_total = {}
    by_type_matched = {}
    for case in dev_cases:
        txn_ids = case["transaction_ids"].split(";") if case["transaction_ids"] else []
        for tid in txn_ids:
            if tid not in result_by_txn:
                continue
            ct = case["case_type"]
            by_type_total[ct] = by_type_total.get(ct, 0) + 1
            if result_by_txn[tid].status == "MATCHED_LAYER1":
                by_type_matched[ct] = by_type_matched.get(ct, 0) + 1

    for ct, total in sorted(by_type_total.items()):
        matched = by_type_matched.get(ct, 0)
        print(f"  {ct:28s} matched={matched}/{total}")


def main():
    transactions = load_transactions()
    settlements = load_settlements()

    matcher = Layer1ExactMatcher(transactions, settlements)
    results = matcher.run()
    write_results(results)

    matched = [r for r in results if r.status == "MATCHED_LAYER1"]
    unmatched = [r for r in results if r.status == "UNMATCHED_LAYER1"]

    print("=== Layer 1 Report ===")
    print(f"Total transactions : {len(transactions)}")
    print(f"Total settlements  : {len(settlements)}")
    print(f"Matched (Layer 1)  : {len(matched)}")
    print(f"Unmatched (passed forward) : {len(unmatched)}")
    print(f"Match rate         : {len(matched)/len(transactions)*100:.1f}%")

    print("\n--- Example successful matches ---")
    for r in matched[:5]:
        print(f"  {r.txn_id} -> {r.settlement_id}  |  {r.reason}")

    print("\n--- Example correctly passed-forward (unmatched) ---")
    for r in unmatched[:5]:
        print(f"  {r.txn_id} -> UNMATCHED  |  {r.reason}  |  {r.evidence}")

    dev_sanity_check(results)


if __name__ == "__main__":
    main()
