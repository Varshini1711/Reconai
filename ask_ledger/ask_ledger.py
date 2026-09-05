"""
ReconAI - Ask the Ledger.

Conversational-style Q&A over the reconciliation results. Answers are
grounded strictly in what the pipeline actually produced (records +
audit trails) - never invented.

HONESTY NOTE: this sandbox has no network access to an LLM API, so
question understanding here is pattern/keyword based, not a true
open-ended conversational model. It correctly answers the specific
question shapes the spec calls out ("why wasn't X matched", "show
unresolved", "which merchants have the most exceptions"). A real
deployment would swap the `answer()` function's routing for an LLM call
that still only reads from `records` - never from ground truth, and
never inventing figures - but the underlying grounding contract (answer
only from stored results) stays the same either way.
"""

import re
from collections import Counter


def _find_record_by_txn_or_settlement(records, ref_id):
    for r in records:
        if ref_id in r.transaction_ids or ref_id in r.settlement_ids:
            return r
    return None


def answer(question: str, records: list, transactions: list, settlements: list) -> str:
    q = question.strip()
    q_lower = q.lower()

    # "why wasn't/was TXNxxxx matched?"
    id_match = re.search(r'\b(TXN\d+|SET\d+)\b', q, re.IGNORECASE)
    if id_match and ("why" in q_lower or "wasn't" in q_lower or "was not" in q_lower or "explain" in q_lower):
        ref_id = id_match.group(1).upper()
        record = _find_record_by_txn_or_settlement(records, ref_id)
        if not record:
            return f"I don't have a record for {ref_id}. It may not exist in the current dataset."
        return (f"{ref_id} -> decision: {record.decision} (confidence {record.confidence:.0f}). "
                f"{record.reasoning} Audit trail: {' | '.join(record.audit_trail)}")

    # "show me all unresolved transactions"
    if "unresolved" in q_lower and ("show" in q_lower or "list" in q_lower or "all" in q_lower):
        unresolved = [r for r in records if r.decision == "UNRESOLVED"]
        if not unresolved:
            return "No unresolved records in the current results."
        lines = [f"  {';'.join(r.transaction_ids) or ';'.join(r.settlement_ids)}: {r.reasoning}" for r in unresolved[:20]]
        more = f"\n(+{len(unresolved)-20} more)" if len(unresolved) > 20 else ""
        return f"{len(unresolved)} unresolved record(s):\n" + "\n".join(lines) + more

    # "show me all cases needing review"
    if ("review" in q_lower or "needs review" in q_lower) and ("show" in q_lower or "list" in q_lower or "all" in q_lower):
        review = [r for r in records if r.decision == "NEEDS_REVIEW"]
        if not review:
            return "No records currently flagged for human review."
        lines = [f"  {';'.join(r.transaction_ids) or ';'.join(r.settlement_ids)}: {r.reasoning}" for r in review[:20]]
        return f"{len(review)} record(s) needing review:\n" + "\n".join(lines)

    # "which merchants have the most exceptions?"
    if "merchant" in q_lower and ("most" in q_lower or "highest" in q_lower or "exception" in q_lower):
        txn_by_id = {t.txn_id: t for t in transactions}
        set_by_id = {s.settlement_id: s for s in settlements}
        merchant_counts = Counter()
        for r in records:
            if r.decision == "AUTO_MATCHED":
                continue
            names = set()
            for tid in r.transaction_ids:
                if tid in txn_by_id:
                    names.add(txn_by_id[tid].merchant_name)
            for sid in r.settlement_ids:
                if sid in set_by_id:
                    names.add(set_by_id[sid].merchant_name)
            for n in names:
                merchant_counts[n] += 1
        if not merchant_counts:
            return "No exceptions found - every record auto-matched."
        top = merchant_counts.most_common(5)
        lines = [f"  {name}: {count} exception(s)" for name, count in top]
        return "Merchants with the most reconciliation exceptions:\n" + "\n".join(lines)

    # summary / dashboard-style question
    if "summary" in q_lower or "how many" in q_lower or "overview" in q_lower:
        total = len(records)
        auto = sum(1 for r in records if r.decision == "AUTO_MATCHED")
        review = sum(1 for r in records if r.decision == "NEEDS_REVIEW")
        unresolved = sum(1 for r in records if r.decision == "UNRESOLVED")
        return (f"{total} total records: {auto} auto-matched, {review} need review, "
                f"{unresolved} unresolved.")

    return ("I can answer questions like: \"why wasn't TXN1234 matched?\", "
            "\"show me all unresolved records\", \"which merchants have the most exceptions?\", "
            "or \"give me a summary\". I only answer from the stored reconciliation results, "
            "never from data I haven't actually processed.")
