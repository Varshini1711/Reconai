"""
ReconAI - Pipeline orchestrator.

Connects Layer 1 (untouched, reused) through Layer 5, applying
confidence scoring and final decisions, and building a per-record audit
trail as it goes.

Flow:
  Layer 1 (exact 1:1)
    -> unresolved txns/settlements passed to Layer 2
  Layer 2 (candidate generation - no decisions)
    -> candidate pools passed to Layer 3 and Layer 4
  Layer 3 (fuzzy 1:1 on amount-tight candidates)
    -> still-unresolved passed to Layer 4
  Layer 4 (grouped: many-to-one, one-to-many, fee/refund-adjusted, partial)
    -> still-unresolved passed to Layer 5
  Layer 5 (agentic investigation, tool-based)
    -> everything gets a final decision here if not resolved earlier

Every record, regardless of which layer resolved it, ends up with a
confidence score, a decision, and a full audit trail of what was tried.

Never imports data/ground_truth - this file and everything it imports
stays fully blind to the answer key.
"""

from reconciliation.loader import load_transactions, load_settlements
from reconciliation.layer1_exact_match import Layer1ExactMatcher
from reconciliation.layer2_candidates import generate_candidates, generate_settlement_candidates
from reconciliation.layer3_fuzzy_match import run_fuzzy_match
from reconciliation.layer4_grouping import (
    try_many_to_one, try_one_to_many, try_refund_adjusted,
    try_fee_adjusted_single, try_partial_settlement,
)
from reconciliation.layer5_agent import investigate
from reconciliation.pipeline_types import ReconciliationRecord
from reconciliation import confidence as conf
from reconciliation import decision as dec

import csv
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA_DIR = PROJECT_ROOT + "/data/synthetic"
_rec_counter = 0


def _next_case_ref():
    global _rec_counter
    _rec_counter += 1
    return f"REC{_rec_counter:04d}"


def load_refunds(data_dir: str = None):
    data_dir = data_dir or DATA_DIR
    with open(f"{data_dir}/refunds.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["amount"] = float(r["amount"])
    return rows


def load_fees(data_dir: str = None):
    data_dir = data_dir or DATA_DIR
    with open(f"{data_dir}/fees.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["amount"] = float(r["amount"])
    return rows


def run_pipeline(data_dir: str = None):
    """
    data_dir: optional override pointing at a different folder containing
    transactions.csv/settlements.csv/refunds.csv/fees.csv (same schema as
    data/synthetic/). Used by the web UI's upload feature. Defaults to
    the built-in demo dataset when not given - existing callers
    (scripts/run_pipeline.py, evaluation/evaluate.py, tests) are
    unaffected.
    """
    global _rec_counter
    _rec_counter = 0

    txn_path = f"{data_dir}/transactions.csv" if data_dir else None
    set_path = f"{data_dir}/settlements.csv" if data_dir else None
    transactions = load_transactions(txn_path)
    settlements = load_settlements(set_path)
    refunds = load_refunds(data_dir)
    fees = load_fees(data_dir)

    txn_by_id = {t.txn_id: t for t in transactions}
    set_by_id = {s.settlement_id: s for s in settlements}

    records = {}          # txn_id -> ReconciliationRecord (for txn-anchored records)
    settlement_records = {}  # settlement_id -> ReconciliationRecord (for orphan settlements)
    resolved_txn_ids = set()
    resolved_settlement_ids = set()

    # ---------------- Layer 1 ----------------
    l1_results = Layer1ExactMatcher(transactions, settlements).run()
    for r in l1_results:
        rec = ReconciliationRecord(
            case_ref=_next_case_ref(),
            transaction_ids=[r.txn_id],
            settlement_ids=[r.settlement_id] if r.settlement_id else [],
            match_method="layer1_exact",
            evidence=list(r.evidence),
        )
        rec.add_audit(f"Layer 1: {r.reason}")
        records[r.txn_id] = rec
        if r.status == "MATCHED_LAYER1":
            rec.confidence = conf.score_layer1(r)
            rec.decision = dec.decide(rec.confidence)
            rec.reasoning = "Resolved by Layer 1: unique exact amount+merchant+date match."
            rec.resolved = True
            resolved_txn_ids.add(r.txn_id)
            resolved_settlement_ids.add(r.settlement_id)

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    # ---------------- Layer 2 ----------------
    candidate_pools = generate_candidates(unresolved_txns, unresolved_settlements, merchant_filter=True)
    # (settlement-side pools generated on demand inside Layer 4 functions via merchant grouping)

    for t in unresolved_txns:
        records[t.txn_id].add_audit(
            f"Layer 2: {len(candidate_pools.get(t.txn_id, []))} candidate settlement(s) found "
            f"(amount tolerance + {10}-day window)"
        )

    # ---------------- Layer 3 ----------------
    fuzzy_results = run_fuzzy_match(unresolved_txns, unresolved_settlements, candidate_pools)
    for txn_id, fr in fuzzy_results.items():
        rec = records[txn_id]
        rec.add_audit(f"Layer 3: {fr['reason']}")
        if fr["matched"]:
            rec.settlement_ids = [fr["settlement_id"]]
            rec.match_method = "layer3_fuzzy"
            rec.evidence.extend(fr["evidence"])
            rec.confidence = conf.score_fuzzy(fr)
            rec.decision = dec.decide(rec.confidence)
            rec.reasoning = fr["reason"]
            rec.resolved = True
            resolved_txn_ids.add(txn_id)
            resolved_settlement_ids.add(fr["settlement_id"])

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    # ---------------- Layer 4 ----------------
    # Order matters: try the more specific/cheaper explanations first.
    refund_hits = try_refund_adjusted(unresolved_txns, unresolved_settlements, refunds)
    for hit in refund_hits:
        if hit["txn_id"] in resolved_txn_ids or hit["settlement_id"] in resolved_settlement_ids:
            continue
        rec = records[hit["txn_id"]]
        rec.settlement_ids = [hit["settlement_id"]]
        rec.refund_ids = [hit["refund_id"]]
        rec.match_method = "layer4_refund_adjusted"
        rec.evidence.extend(hit["evidence"])
        rec.confidence = conf.score_grouping(hit["evidence"], has_fee=False, has_refund=True)
        rec.decision = dec.decide(rec.confidence)
        rec.reasoning = hit["evidence"][0]
        rec.resolved = True
        rec.add_audit(f"Layer 4 (refund-adjusted): {hit['evidence'][0]}")
        resolved_txn_ids.add(hit["txn_id"])
        resolved_settlement_ids.add(hit["settlement_id"])

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    fee_hits = try_fee_adjusted_single(unresolved_txns, unresolved_settlements, fees)
    for hit in fee_hits:
        if hit["txn_id"] in resolved_txn_ids or hit["settlement_id"] in resolved_settlement_ids:
            continue
        rec = records[hit["txn_id"]]
        rec.settlement_ids = [hit["settlement_id"]]
        rec.fee_ids = [hit["fee_id"]]
        rec.match_method = "layer4_fee_adjusted"
        rec.evidence.extend(hit["evidence"])
        rec.confidence = conf.score_grouping(hit["evidence"], has_fee=True, has_refund=False)
        rec.decision = dec.decide(rec.confidence)
        rec.reasoning = hit["evidence"][0]
        rec.resolved = True
        rec.add_audit(f"Layer 4 (fee-adjusted): {hit['evidence'][0]}")
        resolved_txn_ids.add(hit["txn_id"])
        resolved_settlement_ids.add(hit["settlement_id"])

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    many_to_one_hits = try_many_to_one(unresolved_txns, unresolved_settlements, fees)
    for hit in many_to_one_hits:
        if hit["settlement_id"] in resolved_settlement_ids:
            continue
        if any(tid in resolved_txn_ids for tid in hit["txn_ids"]):
            continue
        case_ref = _next_case_ref()
        rec = ReconciliationRecord(
            case_ref=case_ref, transaction_ids=hit["txn_ids"], settlement_ids=[hit["settlement_id"]],
            fee_ids=[hit["fee_id"]] if hit["fee_id"] else [],
            match_method="layer4_many_to_one", evidence=list(hit["evidence"]),
        )
        rec.confidence = conf.score_grouping(hit["evidence"], has_fee=bool(hit["fee_id"]), has_refund=False)
        rec.decision = dec.decide(rec.confidence)
        rec.reasoning = hit["evidence"][0]
        rec.resolved = True
        rec.add_audit(f"Layer 4 (many-to-one): {hit['evidence'][0]}")
        for tid in hit["txn_ids"]:
            records[tid] = rec   # all grouped txns point at the same record
            resolved_txn_ids.add(tid)
        resolved_settlement_ids.add(hit["settlement_id"])

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    one_to_many_hits = try_one_to_many(unresolved_txns, unresolved_settlements)
    for hit in one_to_many_hits:
        if hit["txn_id"] in resolved_txn_ids:
            continue
        if any(sid in resolved_settlement_ids for sid in hit["settlement_ids"]):
            continue
        rec = records[hit["txn_id"]]
        rec.settlement_ids = hit["settlement_ids"]
        rec.match_method = "layer4_one_to_many"
        rec.evidence.extend(hit["evidence"])
        rec.confidence = conf.score_grouping(hit["evidence"], has_fee=False, has_refund=False)
        rec.decision = dec.decide(rec.confidence)
        rec.reasoning = hit["evidence"][0]
        rec.resolved = True
        rec.add_audit(f"Layer 4 (one-to-many): {hit['evidence'][0]}")
        resolved_txn_ids.add(hit["txn_id"])
        resolved_settlement_ids.update(hit["settlement_ids"])

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    partial_hits = try_partial_settlement(unresolved_txns, unresolved_settlements)
    for hit in partial_hits:
        if hit["txn_id"] in resolved_txn_ids or hit["settlement_id"] in resolved_settlement_ids:
            continue
        rec = records[hit["txn_id"]]
        rec.settlement_ids = [hit["settlement_id"]]
        rec.match_method = "layer4_partial_settlement"
        rec.evidence.extend(hit["evidence"])
        rec.confidence = conf.score_partial(hit["ratio"])
        rec.decision = dec.decide(rec.confidence)
        rec.reasoning = hit["evidence"][0]
        rec.resolved = True
        rec.add_audit(f"Layer 4 (partial settlement): {hit['evidence'][0]}")
        resolved_txn_ids.add(hit["txn_id"])
        resolved_settlement_ids.add(hit["settlement_id"])
        # NOTE: intentionally do NOT block the settlement from other txns'
        # candidate pools here beyond this - partials are inherently
        # uncertain, but we treat the settlement as spoken-for to avoid
        # double-booking it against a second transaction.

    unresolved_txns = [t for t in transactions if t.txn_id not in resolved_txn_ids]
    unresolved_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]

    # ---------------- Layer 5 ----------------
    # IMPORTANT: pass only STILL-UNRESOLVED settlements/transactions, not
    # the full lists - otherwise the agent can "rediscover" a settlement
    # another record already legitimately claimed earlier in the
    # pipeline, wrongly treating it as a fresh unique match.
    for t in unresolved_txns:
        result = investigate(t, unresolved_settlements, unresolved_txns, refunds, fees)
        rec = records[t.txn_id]
        rec.match_method = "layer5_agent"
        rec.evidence.extend(result["evidence"])
        for step in result["audit_steps"]:
            rec.add_audit(f"Layer 5: {step}")
        rec.confidence = conf.score_agent(result["decision"], result["evidence"])

        # Safety gate:
        # Gemini is allowed to investigate and propose a match,
        # but it cannot independently override the reconciliation
        # pipeline's safety rules.
        agent_decision = result["decision"]

        if agent_decision == "AUTO_MATCHED" and result.get("settlement_id"):
            settlement_id = result["settlement_id"]

            # Make sure the settlement is actually still available.
            settlement = next(
                (s for s in unresolved_settlements
                 if s.settlement_id == settlement_id),
                None
            )

            if settlement is None:
                agent_decision = "NEEDS_REVIEW"
                result["evidence"].append(
                    "Safety gate: proposed settlement is no longer available."
                )

            else:
                # Check merchant similarity independently.
                from reconciliation.layer3_fuzzy_match import merchant_similarity

                similarity = merchant_similarity(
                    t.merchant_name,
                    settlement.merchant_name
                )

                # A financial match should not rely only on matching money.
                if similarity < 55:
                    agent_decision = "NEEDS_REVIEW"
                    result["evidence"].append(
                        f"Safety gate: merchant similarity too low ({similarity:.2f})."
                    )

                # Settlement should not occur before the transaction.
                elif settlement.date < t.date:
                    agent_decision = "NEEDS_REVIEW"
                    result["evidence"].append(
                        "Safety gate: settlement date is before transaction date."
                    )

                else:
                    result["evidence"].append(
                        f"Safety gate passed: merchant similarity {similarity:.2f}, "
                        "settlement date is valid."
                    )

        rec.decision = agent_decision
        rec.reasoning = result["reasoning"]

        if agent_decision != result["decision"]:
            rec.add_audit(
                f"Layer 5 safety gate changed decision: "
                f"{result['decision']} -> {agent_decision}"
            )
        rec.resolved = True
        if result["settlement_id"]:
            rec.settlement_ids = [result["settlement_id"]]
            resolved_settlement_ids.add(result["settlement_id"])
        resolved_txn_ids.add(t.txn_id)

    # ---------------- Orphan settlements (no transaction at all) ----------------
    remaining_settlements = [s for s in settlements if s.settlement_id not in resolved_settlement_ids]
    for s in remaining_settlements:
        case_ref = _next_case_ref()
        rec = ReconciliationRecord(
            case_ref=case_ref, transaction_ids=[], settlement_ids=[s.settlement_id],
            match_method="orphan_settlement", evidence=["No transaction found for this settlement in any layer"],
        )
        rec.confidence = 5.0
        rec.decision = "UNRESOLVED"
        rec.reasoning = "No transaction could be linked to this settlement after all five layers."
        rec.resolved = True
        rec.add_audit("Orphan check: no transaction found across any layer")
        settlement_records[s.settlement_id] = rec

    # ---------------- Orphan refunds (reference a transaction that doesn't exist) ----------------
    orphan_refund_records = {}
    for r in refunds:
        if r["original_txn_id"] not in txn_by_id:
            case_ref = _next_case_ref()
            rec = ReconciliationRecord(
                case_ref=case_ref, transaction_ids=[], settlement_ids=[], refund_ids=[r["refund_id"]],
                match_method="orphan_refund",
                evidence=[f"References transaction {r['original_txn_id']} which does not exist in the dataset"],
            )
            rec.confidence = 5.0
            rec.decision = "UNRESOLVED"
            rec.reasoning = f"Refund {r['refund_id']} references a non-existent transaction; data error, genuinely unresolved."
            rec.resolved = True
            rec.add_audit("Orphan check: referenced transaction not found in dataset")
            orphan_refund_records[r["refund_id"]] = rec

    all_records = list({id(r): r for r in list(records.values()) + list(settlement_records.values())
                         + list(orphan_refund_records.values())}.values())
    return all_records, transactions, settlements, refunds, fees


def write_pipeline_results(records, path=PROJECT_ROOT + "/data/synthetic/pipeline_results.csv"):
    fieldnames = ["case_ref", "transaction_ids", "settlement_ids", "refund_ids", "fee_ids",
                  "match_method", "evidence", "confidence", "decision", "reasoning", "audit_trail"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())
    return path


if __name__ == "__main__":
    records, *_ = run_pipeline()
    path = write_pipeline_results(records)
    print(f"Pipeline produced {len(records)} records -> {path}")
