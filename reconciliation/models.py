"""
ReconAI - Data models.

Plain dataclasses shared across all reconciliation layers. Keeping these
separate from loading/matching logic so later layers (fuzzy, grouped,
agent) can import the same types without duplicating field definitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Transaction:
    txn_id: str
    amount: float
    date: datetime
    merchant_name: str
    customer_ref: str = ""


@dataclass
class Settlement:
    settlement_id: str
    amount: float
    date: datetime
    merchant_name: str


@dataclass
class MatchResult:
    """
    Structured result for one transaction, produced by a reconciliation
    layer. Later layers append to / build on this rather than replacing
    it, so the full evidence trail survives to the audit trail feature.
    """
    txn_id: str
    settlement_id: Optional[str]      # None if unmatched at this layer
    status: str                        # e.g. "MATCHED_LAYER1", "UNMATCHED_LAYER1"
    match_method: str                  # e.g. "exact_amount_merchant_date"
    evidence: list = field(default_factory=list)   # concise evidence strings
    reason: str = ""                   # human-readable explanation
    timestamp: str = ""                # when this decision was made

    def to_dict(self):
        return dict(
            txn_id=self.txn_id,
            settlement_id=self.settlement_id or "",
            status=self.status,
            match_method=self.match_method,
            evidence="; ".join(self.evidence),
            reason=self.reason,
            timestamp=self.timestamp,
        )
