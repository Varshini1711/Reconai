"""
ReconAI - Evaluation.

The ONLY module in the whole project allowed to import data/ground_truth.
Nothing under reconciliation/ imports this module or ground_truth - the
import direction only goes one way: evaluation -> reconciliation (to run
the pipeline), never reconciliation -> ground_truth.

Computes, per the design doc:
  - precision on auto-matches (of AUTO_MATCHED, how many were correct)
  - coverage (% resolved without human review)
  - escalation correctness (of ground-truth NEEDS_REVIEW cases, how many
    did we also flag as review rather than force-matching or giving up)
  - false match rate (AUTO_MATCHED that were actually wrong - the
    scariest number)
  - unresolved accuracy (of ground-truth UNRESOLVED cases, how many did
    we also correctly mark unresolved)

Reports overall AND broken down by case_type, since that's what actually
tells us which parts of the system work and which don't.
"""

import csv
import sys
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from reconciliation.pipeline import run_pipeline, write_pipeline_results
from reconciliation.layer5_agent import has_llm_key

GT_DEV_PATH = PROJECT_ROOT + "/data/ground_truth/ground_truth_dev.csv"
GT_TEST_PATH = PROJECT_ROOT + "/data/ground_truth/ground_truth_test.csv"


def load_ground_truth(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _record_for_case(records_by_txn, records_by_settlement, records_by_refund, case):
    """Finds the pipeline record covering this ground-truth case's
    primary transaction, settlement, or (for orphan-refund cases) refund."""
    txn_ids = case["transaction_ids"].split(";") if case["transaction_ids"] else []
    if txn_ids:
        return records_by_txn.get(txn_ids[0])
    set_ids = case["settlement_ids"].split(";") if case["settlement_ids"] else []
    if set_ids:
        return records_by_settlement.get(set_ids[0])
    ref_ids = case["refund_ids"].split(";") if case["refund_ids"] else []
    if ref_ids:
        return records_by_refund.get(ref_ids[0])
    return None


def _predicted_groups_correct(record, case):
    """For AUTO_MATCHED grouped cases, checks the predicted txn/settlement
    sets match ground truth's sets (order-independent)."""
    gt_txns = set(case["transaction_ids"].split(";")) if case["transaction_ids"] else set()
    gt_sets = set(case["settlement_ids"].split(";")) if case["settlement_ids"] else set()
    pred_txns = set(record.transaction_ids)
    pred_sets = set(record.settlement_ids)
    return gt_txns == pred_txns and gt_sets == pred_sets


def evaluate(records, ground_truth_cases, label="DEV"):
    records_by_txn = {}
    records_by_settlement = {}
    records_by_refund = {}
    for r in records:
        for tid in r.transaction_ids:
            records_by_txn[tid] = r
        for sid in r.settlement_ids:
            records_by_settlement[sid] = r
        for rid in r.refund_ids:
            records_by_refund[rid] = r

    total = 0
    auto_correct = 0
    auto_incorrect = 0
    auto_total = 0
    review_expected_and_flagged = 0
    review_expected_total = 0
    unresolved_expected_and_flagged = 0
    unresolved_expected_total = 0
    missing_prediction = 0

    by_type = {}

    for case in ground_truth_cases:
        total += 1
        ct = case["case_type"]
        by_type.setdefault(ct, dict(total=0, correct_decision=0))
        by_type[ct]["total"] += 1

        record = _record_for_case(records_by_txn, records_by_settlement, records_by_refund, case)
        expected = case["expected_decision"]

        if record is None:
            missing_prediction += 1
            continue

        predicted = record.decision

        if expected == "AUTO_MATCHED":
            if predicted == "AUTO_MATCHED":
                auto_total += 1
                if _predicted_groups_correct(record, case):
                    auto_correct += 1
                    by_type[ct]["correct_decision"] += 1
                else:
                    auto_incorrect += 1
            # if predicted something else, it's a coverage miss (counted implicitly)
        elif expected == "NEEDS_REVIEW":
            review_expected_total += 1
            if predicted == "NEEDS_REVIEW":
                review_expected_and_flagged += 1
                by_type[ct]["correct_decision"] += 1
            elif predicted == "AUTO_MATCHED":
                pass  # forced a match on a genuinely ambiguous/partial case - a real error, not counted as correct
        elif expected == "UNRESOLVED":
            unresolved_expected_total += 1
            if predicted == "UNRESOLVED":
                unresolved_expected_and_flagged += 1
                by_type[ct]["correct_decision"] += 1

        # also count any AUTO_MATCHED prediction where ground truth expected
        # something else, as a false match - the scariest metric
        if predicted == "AUTO_MATCHED" and expected != "AUTO_MATCHED":
            auto_total += 1
            auto_incorrect += 1

    precision = (auto_correct / auto_total * 100) if auto_total else None
    coverage = (auto_correct / total * 100) if total else None
    escalation_correctness = (review_expected_and_flagged / review_expected_total * 100) if review_expected_total else None
    unresolved_accuracy = (unresolved_expected_and_flagged / unresolved_expected_total * 100) if unresolved_expected_total else None
    false_match_rate = (auto_incorrect / auto_total * 100) if auto_total else None

    print(f"\n=== Evaluation ({label} set, {total} cases) ===")
    print(f"Missing predictions (no record found)       : {missing_prediction}")
    print(f"Auto-match precision                          : {precision:.1f}%  ({auto_correct}/{auto_total})" if precision is not None else "Auto-match precision: n/a")
    print(f"Coverage (correct auto-matches / all cases)   : {coverage:.1f}%" if coverage is not None else "Coverage: n/a")
    print(f"Escalation correctness (NEEDS_REVIEW cases)    : {escalation_correctness:.1f}%  ({review_expected_and_flagged}/{review_expected_total})" if escalation_correctness is not None else "Escalation correctness: n/a")
    print(f"Unresolved accuracy (UNRESOLVED cases)         : {unresolved_accuracy:.1f}%  ({unresolved_expected_and_flagged}/{unresolved_expected_total})" if unresolved_accuracy is not None else "Unresolved accuracy: n/a")
    print(f"False match rate (auto-matched but wrong)      : {false_match_rate:.1f}%  ({auto_incorrect}/{auto_total})" if false_match_rate is not None else "False match rate: n/a")

    print("\nPer case-type breakdown (correct decision / total):")
    for ct, v in sorted(by_type.items()):
        print(f"  {ct:28s} {v['correct_decision']}/{v['total']}")

    return dict(precision=precision, coverage=coverage, escalation_correctness=escalation_correctness,
                unresolved_accuracy=unresolved_accuracy, false_match_rate=false_match_rate, by_type=by_type)


def run_dev_evaluation():
    if not has_llm_key():
        print("=" * 70)
        print("WARNING: GEMINI_API_KEY is not set. Layer 5 metrics below will")
        print("reflect 'AI investigation unavailable' NEEDS_REVIEW fallbacks,")
        print("not real agent decisions. Set your key before running this if")
        print("you want real evaluation numbers.")
        print("=" * 70)
        print()
    records, *_ = run_pipeline()
    write_pipeline_results(records)
    gt = load_ground_truth(GT_DEV_PATH)
    return evaluate(records, gt, label="DEV")


def run_test_evaluation(confirm: bool = False):
    """
    Guarded: requires explicit confirm=True so this is never run
    accidentally while tuning. Intended to be run ONCE, at the very end.
    """
    if not confirm:
        raise RuntimeError(
            "Refusing to run TEST evaluation without confirm=True. "
            "This should only be run once, deliberately, after Layer 1-5 "
            "and thresholds are finalized using DEV only."
        )
    records, *_ = run_pipeline()
    write_pipeline_results(records)
    gt = load_ground_truth(GT_TEST_PATH)
    return evaluate(records, gt, label="TEST")


if __name__ == "__main__":
    run_dev_evaluation()
