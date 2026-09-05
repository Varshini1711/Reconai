"""
ReconAI - Layer 5 LLM tool definitions and executor.

Ten controlled tools the LLM agent may call. Every tool reads ONLY from
the visible dataset slice handed to Layer 5 (unresolved transactions/
settlements plus refunds/fees) - never data/ground_truth. Tools return
structured, factual data; the agent is never allowed to state a number
that didn't come from a tool result - that's enforced by the system
prompt in llm_agent.py, and helped structurally here by making every
number-producing tool (amount diffs, sums, date deltas, relationship
verification) a pure calculator the model must call rather than do
arithmetic itself.
"""

from datetime import datetime
from reconciliation.layer3_fuzzy_match import merchant_similarity


TOOL_SCHEMAS = [
    {
        "name": "search_candidate_records",
        "description": "Search for settlements plausibly related to a given amount and/or merchant name. "
                        "Returns a list of settlement summaries (id, amount, date, merchant). Use this first "
                        "to find candidates before inspecting them in detail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to search near (optional)"},
                "merchant": {"type": "string", "description": "Merchant name to search near (optional)"},
                "amount_tolerance_pct": {"type": "number", "description": "Percent tolerance around amount, default 20"},
            },
        },
    },
    {
        "name": "inspect_transaction",
        "description": "Get the full details of a specific transaction by ID: amount, date, merchant name, customer reference.",
        "input_schema": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}},
            "required": ["txn_id"],
        },
    },
    {
        "name": "inspect_settlement",
        "description": "Get the full details of a specific settlement by ID: amount, date, merchant name.",
        "input_schema": {
            "type": "object",
            "properties": {"settlement_id": {"type": "string"}},
            "required": ["settlement_id"],
        },
    },
    {
        "name": "inspect_refund",
        "description": "Look up any refund associated with a given transaction ID. Returns refund details if one exists, or null.",
        "input_schema": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}},
            "required": ["txn_id"],
        },
    },
    {
        "name": "inspect_fee",
        "description": "Look up any fee associated with a given transaction ID, including batched fees covering "
                        "multiple transactions. Returns fee details if any exist.",
        "input_schema": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}},
            "required": ["txn_id"],
        },
    },
    {
        "name": "calculate_amount_difference",
        "description": "Compute the numeric difference between two amounts. Always use this instead of doing arithmetic yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"amount_a": {"type": "number"}, "amount_b": {"type": "number"}},
            "required": ["amount_a", "amount_b"],
        },
    },
    {
        "name": "calculate_group_total",
        "description": "Sum a list of amounts. Always use this instead of adding numbers yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"amounts": {"type": "array", "items": {"type": "number"}}},
            "required": ["amounts"],
        },
    },
    {
        "name": "check_date_difference",
        "description": "Compute the number of days between two dates (YYYY-MM-DD). Positive means date_b is after date_a.",
        "input_schema": {
            "type": "object",
            "properties": {"date_a": {"type": "string"}, "date_b": {"type": "string"}},
            "required": ["date_a", "date_b"],
        },
    },
    {
        "name": "compare_merchant_names",
        "description": "Compute a 0-100 text similarity score between two merchant name strings.",
        "input_schema": {
            "type": "object",
            "properties": {"name_a": {"type": "string"}, "name_b": {"type": "string"}},
            "required": ["name_a", "name_b"],
        },
    },
    {
        "name": "verify_candidate_relationship",
        "description": "Given one or more transaction IDs and one or more settlement IDs, checks whether the "
                        "transactions' total (after automatically netting any fees/refunds found for those exact "
                        "transactions) equals the settlements' total, within a small tolerance. Returns whether it "
                        "balances and the exact numbers used - always use this before deciding AUTO_MATCHED, "
                        "never compute this arithmetic yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "txn_ids": {"type": "array", "items": {"type": "string"}},
                "settlement_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["txn_ids", "settlement_ids"],
        },
    },
]


class ToolExecutor:
    """Binds the ten tools to one investigation's visible data slice, and
    logs every call for the audit trail."""

    def __init__(self, transactions, settlements, refunds, fees):
        self.txn_by_id = {t.txn_id: t for t in transactions}
        self.set_by_id = {s.settlement_id: s for s in settlements}
        self.settlements = settlements
        self.refunds = refunds
        self.fees = fees
        self.tool_call_log = []

    def run(self, name: str, tool_input: dict) -> dict:
        try:
            handler = getattr(self, f"_{name}", None)
            if handler is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                result = handler(**tool_input)
        except Exception as e:
            result = {"error": str(e)}
        self.tool_call_log.append(dict(tool=name, input=tool_input, result=result))
        return result

    # ---- individual tools ----

    def _search_candidate_records(self, amount=None, merchant=None, amount_tolerance_pct=20):
        out = []
        for s in self.settlements:
            if amount is not None:
                tol = amount * amount_tolerance_pct / 100
                if abs(s.amount - amount) > tol:
                    continue
            if merchant is not None:
                sim = merchant_similarity(merchant, s.merchant_name)
                if sim < 40:
                    continue
            out.append(dict(settlement_id=s.settlement_id, amount=s.amount,
                             date=str(s.date.date()), merchant_name=s.merchant_name))
        return {"candidates": out[:15]}   # capped to keep prompts small

    def _inspect_transaction(self, txn_id):
        t = self.txn_by_id.get(txn_id)
        if not t:
            return {"error": f"No transaction {txn_id} found in the visible dataset"}
        return dict(txn_id=t.txn_id, amount=t.amount, date=str(t.date.date()),
                    merchant_name=t.merchant_name, customer_ref=t.customer_ref)

    def _inspect_settlement(self, settlement_id):
        s = self.set_by_id.get(settlement_id)
        if not s:
            return {"error": f"No settlement {settlement_id} found in the visible dataset"}
        return dict(settlement_id=s.settlement_id, amount=s.amount, date=str(s.date.date()),
                    merchant_name=s.merchant_name)

    def _inspect_refund(self, txn_id):
        for r in self.refunds:
            if r["original_txn_id"] == txn_id:
                return dict(refund_id=r["refund_id"], original_txn_id=r["original_txn_id"],
                            amount=r["amount"], date=r["date"])
        return {"refund": None}

    def _inspect_fee(self, txn_id):
        matches = []
        for f in self.fees:
            if txn_id in f["related_txn_ids"].split(";"):
                matches.append(dict(fee_id=f["fee_id"], related_txn_ids=f["related_txn_ids"],
                                     amount=f["amount"], type=f["type"]))
        return {"fees": matches}

    def _calculate_amount_difference(self, amount_a, amount_b):
        return {"difference": round(amount_a - amount_b, 2), "absolute_difference": round(abs(amount_a - amount_b), 2)}

    def _calculate_group_total(self, amounts):
        return {"total": round(sum(amounts), 2)}

    def _check_date_difference(self, date_a, date_b):
        da = datetime.strptime(date_a, "%Y-%m-%d")
        db = datetime.strptime(date_b, "%Y-%m-%d")
        return {"days_between": (db - da).days}

    def _compare_merchant_names(self, name_a, name_b):
        return {"similarity_0_to_100": round(merchant_similarity(name_a, name_b), 1)}

    def _verify_candidate_relationship(self, txn_ids, settlement_ids):
        txns = [self.txn_by_id.get(tid) for tid in txn_ids]
        missing_txns = [tid for tid, t in zip(txn_ids, txns) if t is None]
        sets = [self.set_by_id.get(sid) for sid in settlement_ids]
        missing_sets = [sid for sid, s in zip(settlement_ids, sets) if s is None]
        if missing_txns or missing_sets:
            return {"error": f"Unknown IDs - transactions: {missing_txns}, settlements: {missing_sets}"}

        txn_total = sum(t.amount for t in txns)

        fees_applied = []
        seen_fee_ids = set()
        for tid in txn_ids:
            for f in self.fees:
                if tid in f["related_txn_ids"].split(";") and f["fee_id"] not in seen_fee_ids:
                    fees_applied.append(f)
                    seen_fee_ids.add(f["fee_id"])

        refunds_applied = []
        for tid in txn_ids:
            for r in self.refunds:
                if r["original_txn_id"] == tid:
                    refunds_applied.append(r)

        net_expected = txn_total - sum(f["amount"] for f in fees_applied) - sum(r["amount"] for r in refunds_applied)
        settlement_total = sum(s.amount for s in sets)
        diff = round(net_expected - settlement_total, 2)
        return dict(
            transaction_total=round(txn_total, 2),
            fees_applied=[dict(fee_id=f["fee_id"], amount=f["amount"]) for f in fees_applied],
            refunds_applied=[dict(refund_id=r["refund_id"], amount=r["amount"]) for r in refunds_applied],
            net_expected_settlement=round(net_expected, 2),
            actual_settlement_total=round(settlement_total, 2),
            difference=diff,
            balances_within_tolerance=abs(diff) <= 1.0,
        )
