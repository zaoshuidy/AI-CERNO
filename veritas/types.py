"""Core dataclasses, literals, and enums for veritas-core.

Single source of truth: design v0.5 §7. Field invariants enforced via
__post_init__ where cheap; cross-object invariants (e.g.
EvidenceJudgement.quote ⊂ EvidenceSpan.quote) are checked in their owning
module, not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from veritas.observability import AuditStep, CostBreakdown

# ---------------------------------------------------------------------------
# Literals / enums
# ---------------------------------------------------------------------------

ClaimType = Literal[
    "entity_fact",
    "temporal_fact",
    "quantitative_fact",
    "quotation",
    "domain_fact",
    "other",
]

RiskLevel = Literal["low", "medium", "high", "critical"]
RiskLevelSource = Literal["user_hint", "llm_inferred", "max_of_both"]

SourceTier = Literal["T0", "T1", "T2", "T3", "BLOCKED"]

Relation = Literal["supports", "refutes", "neutral", "insufficient"]

Verdict = Literal[
    "likely_correct",
    "likely_error",
    "needs_review",
    "unverifiable",
    "conflicting_sources",
]

ConfidenceLevel = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Request / config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationRequest:
    """P0 surface: 4 fields only. source_policy / max_cost / max_latency_ms → P1."""

    claim: str
    context: str | None = None
    domain_hint: str | None = None       # legal / academic / medical / policy / general / etc.
    risk_hint: str | None = None         # low / medium / high / critical


@dataclass(frozen=True)
class LLMProvider:
    """All callers configure their own LLM endpoints. veritas never holds keys."""

    name: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0
    max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Claim understanding
# ---------------------------------------------------------------------------

@dataclass
class ClaimProfile:
    is_checkable: bool
    claim_type: ClaimType
    domain: str
    risk_level: RiskLevel
    risk_level_source: RiskLevelSource
    required_evidence: list[str]
    strict_mode: bool
    reason: str
    allow_single_t1_source: bool = True
    allow_single_t0_source: bool = True


@dataclass
class AtomicClaim:
    id: str
    text: str                            # 规范化后的可检索文本
    original_span: str                   # 原 claim 中对应的原始片段（非空）
    parent_claim: str
    check_priority: int
    required_evidence_type: list[str]

    def __post_init__(self) -> None:
        if not self.original_span:
            raise ValueError(
                f"AtomicClaim[{self.id}] requires non-empty original_span; "
                "preserve the original phrasing from the parent claim."
            )


@dataclass
class RetrievalPlan:
    queries: list[str]
    source_targets: list[str]            # P0: ["tavily", "wikipedia_zh"]
    min_independent_sources: int
    allow_discovery_sources: bool
    allow_single_source_medium: bool
    require_official_source: bool
    require_freshness_check: bool


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSpan:
    id: str
    title: str
    url: str
    quote: str                           # 锚词窗口 ±200 字 (~400 chars total)
    source_name: str
    source_tier: SourceTier
    retrieved_at: str
    content_hash: str                    # SHA-256(normalize(raw_content))[:16]
    raw_score: float | None = None
    metadata: dict = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class EvidenceScore:
    """P0: 3 mandatory + 2 conditional. independence_score deferred to P1."""

    source_tier_score: float
    lexical_overlap_score: float
    semantic_support_score: float
    freshness_score: float | None = None
    injection_risk_penalty: float = 0.0
    final_score: float = 0.0


@dataclass
class ModelVote:
    """One model's verdict on (atomic_claim, evidence). On failure: error is set,
    relation and confidence are None.

    `quote` carries the model's claimed support quote (verification.py will
    substring-check it against EvidenceSpan.quote). `raw_response` is reserved
    for the model's raw output / audit trail and must NOT be reused for quote.
    """

    model_name: str
    relation: Relation | None
    confidence: ConfidenceLevel | None
    reason: str = ""
    error: str | None = None
    quote: str | None = None
    raw_response: str | None = None


@dataclass
class EvidenceJudgement:
    atomic_claim_id: str
    evidence_id: str
    relation: Relation
    confidence: ConfidenceLevel
    quote: str                           # MUST be a substring of the source EvidenceSpan.quote
    reason: str
    model_votes: list[ModelVote] = field(default_factory=list)


@dataclass
class Disagreement:
    """Recorded when models do not agree on a single (atomic_claim, evidence)."""

    atomic_claim_id: str
    evidence_id: str
    model_votes: list[ModelVote]
    summary: str = ""


@dataclass
class ConflictReport:
    """Recorded when supporting and refuting evidence coexist for one atomic_claim."""

    atomic_claim_id: str
    supporting_evidence_ids: list[str]
    refuting_evidence_ids: list[str]
    summary: str = ""


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    # 必选展示字段
    verdict: Verdict
    confidence: float
    claim: str
    sources: list[EvidenceSpan]
    reasoning: str

    # 审计可选字段
    model_votes: list[ModelVote] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    conflicts: list[ConflictReport] = field(default_factory=list)
    consensus_method: str = "strictest"
    atomic_claims: list[AtomicClaim] = field(default_factory=list)
    supporting_sources: list[EvidenceSpan] = field(default_factory=list)
    refuting_sources: list[EvidenceSpan] = field(default_factory=list)
    audit_trace: list[AuditStep] = field(default_factory=list)
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    def to_compact_dict(self) -> dict[str, Any]:
        """Compact representation for UI / front-end. Full audit lives in the full dataclass."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "claim": self.claim,
            "sources": [asdict(s) for s in self.sources],
            "reasoning": self.reasoning,
        }
