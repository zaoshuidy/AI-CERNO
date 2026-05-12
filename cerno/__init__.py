"""cerno public surface.

Only stable, intended-for-callers names live here. Internal helpers stay
in their respective modules.
"""

from __future__ import annotations

from cerno.fact_checker import FactChecker, check_claim
from cerno.observability import (
    AuditStep,
    CostBreakdown,
    TraceRecorder,
    record_step,
)
from cerno.types import (
    AtomicClaim,
    ClaimProfile,
    ClaimType,
    ConflictReport,
    Disagreement,
    EvidenceJudgement,
    EvidenceScore,
    EvidenceSpan,
    LLMProvider,
    ModelVote,
    Relation,
    RetrievalPlan,
    RiskLevel,
    RiskLevelSource,
    SourceTier,
    Verdict,
    VerificationRequest,
    VerificationResult,
)

__all__ = [
    # types
    "AtomicClaim",
    "ClaimProfile",
    "ClaimType",
    "ConflictReport",
    "Disagreement",
    "EvidenceJudgement",
    "EvidenceScore",
    "EvidenceSpan",
    "LLMProvider",
    "ModelVote",
    "Relation",
    "RetrievalPlan",
    "RiskLevel",
    "RiskLevelSource",
    "SourceTier",
    "Verdict",
    "VerificationRequest",
    "VerificationResult",
    # observability
    "AuditStep",
    "CostBreakdown",
    "TraceRecorder",
    "record_step",
    # orchestrator (Stage 7)
    "FactChecker",
    "check_claim",
]

__version__ = "0.1.0"
