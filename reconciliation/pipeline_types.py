"""
ReconAI - Shared pipeline record type.

Layer 1 (existing, untouched) speaks in `MatchResult` (one transaction ->
one settlement or none). Layers 2-5 need a richer unit that can represent
a GROUP (many-to-one, one-to-many), carry fee/refund involvement, and
accumulate an audit trail across layers. `ReconciliationRecord` is that
richer unit. The pipeline converts Layer 1's MatchResult objects into
ReconciliationRecords at the start, so Layer 1's own code never needs to
change.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReconciliationRecord:
    case_ref: str                          # internal id for this record, e.g. "REC001"
    transaction_ids: list                  # one or more txn ids
    settlement_ids: list                   # zero or more settlement ids
    refund_ids: list = field(default_factory=list)
    fee_ids: list = field(default_factory=list)
    match_method: str = ""                 # which layer/method resolved (or attempted) this
    evidence: list = field(default_factory=list)     # concise evidence strings
    audit_trail: list = field(default_factory=list)  # ordered list of step strings, across all layers
    confidence: Optional[float] = None     # 0-100, set by confidence.py
    decision: str = ""                     # AUTO_MATCHED / NEEDS_REVIEW / UNRESOLVED, set by decision.py
    reasoning: str = ""                    # human-readable explanation (Layer 5 / decision)
    resolved: bool = False                 # True once a layer has produced a final grouping for this txn/settlement

    def add_audit(self, step: str):
        self.audit_trail.append(step)

    def to_dict(self):
        return dict(
            case_ref=self.case_ref,
            transaction_ids=";".join(self.transaction_ids),
            settlement_ids=";".join(self.settlement_ids),
            refund_ids=";".join(self.refund_ids),
            fee_ids=";".join(self.fee_ids),
            match_method=self.match_method,
            evidence="; ".join(self.evidence),
            confidence=self.confidence,
            decision=self.decision,
            reasoning=self.reasoning,
            audit_trail=" | ".join(self.audit_trail),
        )
