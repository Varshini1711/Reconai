"""
ReconAI - Data loader.

Reads ONLY the visible data files (transactions.csv, settlements.csv).
This module must NEVER import anything from data/ground_truth/ - that
folder is reserved for evaluation code, imported nowhere near matching
logic. Enforced here by simple omission: there is no path in this file
that points at ground_truth.
"""

import csv
from datetime import datetime
from reconciliation.models import Transaction, Settlement
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA_DIR = PROJECT_ROOT + "/data/synthetic"


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def load_transactions(path: str = None) -> list[Transaction]:
    path = path or f"{DATA_DIR}/transactions.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        Transaction(
            txn_id=r["txn_id"],
            amount=float(r["amount"]),
            date=_parse_date(r["date"]),
            merchant_name=r["merchant_name"],
            customer_ref=r.get("customer_ref", ""),
        )
        for r in rows
    ]


def load_settlements(path: str = None) -> list[Settlement]:
    path = path or f"{DATA_DIR}/settlements.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        Settlement(
            settlement_id=r["settlement_id"],
            amount=float(r["amount"]),
            date=_parse_date(r["date"]),
            merchant_name=r["merchant_name"],
        )
        for r in rows
    ]
