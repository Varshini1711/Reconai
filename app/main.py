"""
ReconAI - FastAPI backend for the web dashboard.

This is a NEW top-level module (not inside reconciliation/), so it is
allowed to import evaluation/evaluate.py to DISPLAY metrics computed
against ground truth after the fact - exactly the same pattern
evaluate.py itself already uses. It never feeds ground truth into any
reconciliation decision; it only reads pre-computed evaluation numbers
for the dashboard.

Endpoints:
  GET  /api/summary              - dashboard totals + eval metrics (demo dataset only)
  GET  /api/records               - list of reconciliation records, optional ?status= filter
  GET  /api/records/{case_ref}   - full detail for one record: raw txn/settlement/
                                    refund/fee data, evidence, audit trail, AI reasoning
  POST /api/ask                   - Ask the Ledger
  POST /api/run                   - (re)run the full 5-layer pipeline
  POST /api/upload                - upload custom transactions/settlements/refunds/fees CSVs
  GET  /                          - serves the dashboard UI (static/index.html)

HONESTY NOTE on "live progress": /api/run is a single blocking HTTP call
that runs all five layers synchronously and returns final results. The
UI shows a "Layer 1 -> 5" progress list for clarity while waiting, but
this is NOT true per-layer streaming (that would need websockets/SSE,
out of scope here) - it's a labeled loading state, not a live feed. Say
so if asked, don't oversell it as real-time.
"""

import os
import sys
import csv
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reconciliation.pipeline import run_pipeline, write_pipeline_results
from reconciliation.layer5_agent import has_llm_key
from ask_ledger.ask_ledger import answer as ask_ledger_answer

DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
UPLOAD_DATA_DIR = os.path.join(PROJECT_ROOT, "app", "uploads")
DEMO_RESULTS_PATH = os.path.join(DEMO_DATA_DIR, "pipeline_results.csv")
UPLOAD_RESULTS_PATH = os.path.join(UPLOAD_DATA_DIR, "pipeline_results.csv")

app = FastAPI(title="ReconAI")

# ---------------------------------------------------------------------
# In-memory state - the currently loaded run (demo or uploaded dataset)
# ---------------------------------------------------------------------
STATE = dict(
    dataset="demo",         # "demo" or "uploaded"
    records=[],              # list of dicts (parsed pipeline_results.csv rows)
    transactions={},         # txn_id -> dict
    settlements={},          # settlement_id -> dict
    refunds={},               # refund_id -> dict
    fees={},                   # fee_id -> dict
    raw_txn_objs=[],          # Transaction objects (for ask_ledger)
    raw_set_objs=[],          # Settlement objects (for ask_ledger)
    raw_records_objs=[],     # ReconciliationRecord objects (for ask_ledger)
)


def _split(val):
    return [v for v in val.split(";") if v] if val else []


def _load_state_from_disk(data_dir: str, results_path: str, dataset_label: str):
    """Loads raw CSVs + a previously-written pipeline_results.csv into
    STATE, WITHOUT re-running the pipeline. Used at startup and after a
    fresh /api/run completes."""
    from reconciliation.loader import load_transactions, load_settlements
    from reconciliation.pipeline import load_refunds, load_fees

    txns = load_transactions(f"{data_dir}/transactions.csv")
    sets = load_settlements(f"{data_dir}/settlements.csv")
    refunds = load_refunds(data_dir)
    fees = load_fees(data_dir)

    STATE["transactions"] = {t.txn_id: dict(txn_id=t.txn_id, amount=t.amount,
                                             date=str(t.date.date()), merchant_name=t.merchant_name,
                                             customer_ref=t.customer_ref) for t in txns}
    STATE["settlements"] = {s.settlement_id: dict(settlement_id=s.settlement_id, amount=s.amount,
                                                   date=str(s.date.date()), merchant_name=s.merchant_name) for s in sets}
    STATE["refunds"] = {r["refund_id"]: r for r in refunds}
    STATE["fees"] = {f["fee_id"]: f for f in fees}
    STATE["raw_txn_objs"] = txns
    STATE["raw_set_objs"] = sets
    STATE["dataset"] = dataset_label

    if os.path.exists(results_path):
        with open(results_path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["transaction_ids"] = _split(r["transaction_ids"])
            r["settlement_ids"] = _split(r["settlement_ids"])
            r["refund_ids"] = _split(r["refund_ids"])
            r["fee_ids"] = _split(r["fee_ids"])
            r["audit_trail"] = r["audit_trail"].split(" | ") if r["audit_trail"] else []
            r["evidence"] = r["evidence"].split("; ") if r["evidence"] else []
            try:
                r["confidence"] = float(r["confidence"]) if r["confidence"] not in ("", None) else None
            except ValueError:
                pass
        STATE["records"] = rows
    else:
        STATE["records"] = []


@app.on_event("startup")
def _startup():
    os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
    try:
        _load_state_from_disk(DEMO_DATA_DIR, DEMO_RESULTS_PATH, "demo")
    except Exception as e:
        print(f"Startup load failed (this is fine on first run before any pipeline run exists): {e}")


# ---------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------

@app.get("/api/summary")
def get_summary():
    records = STATE["records"]
    total = len(records)
    auto = sum(1 for r in records if r["decision"] == "AUTO_MATCHED")
    review = sum(1 for r in records if r["decision"] == "NEEDS_REVIEW")
    unresolved = sum(1 for r in records if r["decision"] == "UNRESOLVED")

    eval_metrics = None
    if STATE["dataset"] == "demo" and total > 0:
        try:
            from evaluation.evaluate import load_ground_truth, evaluate as run_eval, GT_DEV_PATH
            from reconciliation.pipeline_types import ReconciliationRecord

            rec_objs = []
            for r in records:
                rec = ReconciliationRecord(
                    case_ref=r["case_ref"], transaction_ids=r["transaction_ids"],
                    settlement_ids=r["settlement_ids"], refund_ids=r["refund_ids"], fee_ids=r["fee_ids"],
                    match_method=r["match_method"], decision=r["decision"],
                )
                rec_objs.append(rec)
            gt = load_ground_truth(GT_DEV_PATH)
            metrics = run_eval(rec_objs, gt, label="DEV")
            eval_metrics = dict(
                precision=metrics["precision"], coverage=metrics["coverage"],
                escalation_correctness=metrics["escalation_correctness"],
                unresolved_accuracy=metrics["unresolved_accuracy"],
                false_match_rate=metrics["false_match_rate"],
            )
        except Exception as e:
            eval_metrics = dict(error=f"Evaluation unavailable: {e}")

    return dict(
        dataset=STATE["dataset"],
        total_records=total,
        auto_matched=auto,
        needs_review=review,
        unresolved=unresolved,
        match_rate=round(auto / total * 100, 1) if total else 0,
        llm_key_configured=has_llm_key(),
        evaluation=eval_metrics,
    )


# ---------------------------------------------------------------------
# Records list + detail
# ---------------------------------------------------------------------

@app.get("/api/records")
def get_records(status: str = "ALL"):
    records = STATE["records"]
    if status != "ALL":
        records = [r for r in records if r["decision"] == status]
    return [
        dict(
            case_ref=r["case_ref"],
            transaction_ids=r["transaction_ids"],
            settlement_ids=r["settlement_ids"],
            decision=r["decision"],
            confidence=r["confidence"],
            match_method=r["match_method"],
        )
        for r in records
    ]


@app.get("/api/records/{case_ref}")
def get_record_detail(case_ref: str):
    record = next((r for r in STATE["records"] if r["case_ref"] == case_ref), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"No record {case_ref}")

    transactions = [STATE["transactions"].get(tid) for tid in record["transaction_ids"]]
    settlements = [STATE["settlements"].get(sid) for sid in record["settlement_ids"]]
    refunds = [STATE["refunds"].get(rid) for rid in record["refund_ids"]]
    fees = [STATE["fees"].get(fid) for fid in record["fee_ids"]]

    is_ai = record["match_method"] == "layer5_agent"

    return dict(
        case_ref=record["case_ref"],
        decision=record["decision"],
        confidence=record["confidence"],
        match_method=record["match_method"],
        reasoning=record["reasoning"],
        transactions=[t for t in transactions if t],
        settlements=[s for s in settlements if s],
        refunds=[r for r in refunds if r],
        fees=[f for f in fees if f],
        evidence=record["evidence"],
        audit_trail=record["audit_trail"],
        is_ai_investigated=is_ai,
    )


# ---------------------------------------------------------------------
# Ask the Ledger
# ---------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask(req: AskRequest):
    from reconciliation.pipeline_types import ReconciliationRecord
    rec_objs = []
    for r in STATE["records"]:
        rec = ReconciliationRecord(
            case_ref=r["case_ref"], transaction_ids=r["transaction_ids"],
            settlement_ids=r["settlement_ids"], refund_ids=r["refund_ids"], fee_ids=r["fee_ids"],
            match_method=r["match_method"], decision=r["decision"],
            confidence=r["confidence"] or 0, reasoning=r["reasoning"], audit_trail=r["audit_trail"],
        )
        rec_objs.append(rec)

    ans = ask_ledger_answer(req.question, rec_objs, STATE["raw_txn_objs"], STATE["raw_set_objs"])
    return dict(answer=ans)


# ---------------------------------------------------------------------
# Run pipeline (demo or uploaded dataset)
# ---------------------------------------------------------------------

@app.post("/api/run")
def run_reconciliation(use_uploaded: bool = False):
    if use_uploaded:
        required = ["transactions.csv", "settlements.csv", "refunds.csv", "fees.csv"]
        missing = [f for f in required if not os.path.exists(os.path.join(UPLOAD_DATA_DIR, f))]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing uploaded files: {missing}. Upload all four first.")
        data_dir = UPLOAD_DATA_DIR
        results_path = UPLOAD_RESULTS_PATH
        dataset_label = "uploaded"
    else:
        data_dir = DEMO_DATA_DIR
        results_path = DEMO_RESULTS_PATH
        dataset_label = "demo"

    records, transactions, settlements, refunds, fees = run_pipeline(data_dir=data_dir)
    write_pipeline_results(records, results_path)
    _load_state_from_disk(data_dir, results_path, dataset_label)

    auto = sum(1 for r in records if r.decision == "AUTO_MATCHED")
    review = sum(1 for r in records if r.decision == "NEEDS_REVIEW")
    unresolved = sum(1 for r in records if r.decision == "UNRESOLVED")

    return dict(
        dataset=dataset_label,
        total_records=len(records),
        auto_matched=auto,
        needs_review=review,
        unresolved=unresolved,
        llm_key_configured=has_llm_key(),
        layers_run=[
            "Layer 1 - Exact matching",
            "Layer 2 - Candidate generation",
            "Layer 3 - Fuzzy matching",
            "Layer 4 - Grouped reconciliation",
            "Layer 5 - AI investigation" if has_llm_key() else "Layer 5 - AI investigation (no key - NEEDS_REVIEW fallback)",
        ],
    )


# ---------------------------------------------------------------------
# Upload custom dataset
# ---------------------------------------------------------------------

REQUIRED_UPLOAD_COLUMNS = dict(
    transactions=["txn_id", "amount", "date", "merchant_name"],
    settlements=["settlement_id", "amount", "date", "merchant_name"],
    refunds=["refund_id", "original_txn_id", "amount", "date"],
    fees=["fee_id", "related_txn_ids", "amount", "type"],
)


@app.post("/api/upload")
async def upload_dataset(
    transactions: UploadFile = File(...),
    settlements: UploadFile = File(...),
    refunds: UploadFile = File(...),
    fees: UploadFile = File(...),
):
    os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
    files = dict(transactions=transactions, settlements=settlements, refunds=refunds, fees=fees)
    errors = []

    for name, upload in files.items():
        content = (await upload.read()).decode("utf-8", errors="replace")
        header = content.splitlines()[0].split(",") if content.splitlines() else []
        missing_cols = [c for c in REQUIRED_UPLOAD_COLUMNS[name] if c not in header]
        if missing_cols:
            errors.append(f"{name}.csv is missing required column(s): {missing_cols}")
            continue
        dest = os.path.join(UPLOAD_DATA_DIR, f"{name}.csv")
        with open(dest, "w", encoding="utf-8") as out:
            out.write(content)

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    return dict(status="uploaded", message="All 4 files uploaded. Click 'Run on Uploaded Data' to reconcile.")


# ---------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
