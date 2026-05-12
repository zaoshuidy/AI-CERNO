"""Unit tests for cerno.types — dataclass invariants, defaults, serialization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pytest

from cerno import (
    AtomicClaim,
    ClaimProfile,
    ConflictReport,
    Disagreement,
    EvidenceJudgement,
    EvidenceScore,
    EvidenceSpan,
    LLMProvider,
    ModelVote,
    RetrievalPlan,
    VerificationRequest,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# VerificationRequest
# ---------------------------------------------------------------------------

def test_verification_request_has_only_p0_fields() -> None:
    req = VerificationRequest(claim="嫦娥四号于2019年1月3日着陆月球背面。")
    assert req.claim == "嫦娥四号于2019年1月3日着陆月球背面。"
    assert req.context is None
    assert req.domain_hint is None
    assert req.risk_hint is None


def test_verification_request_is_frozen() -> None:
    req = VerificationRequest(claim="...")
    with pytest.raises(Exception):
        req.claim = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------

def test_llm_provider_is_frozen_and_has_defaults() -> None:
    p = LLMProvider(
        name="deepseek",
        api_key="sk-xxx",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    assert p.timeout == 30.0
    assert p.max_tokens == 4096
    with pytest.raises(Exception):
        p.api_key = "leaked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ClaimProfile
# ---------------------------------------------------------------------------

def test_claim_profile_has_dual_single_source_flags() -> None:
    profile = ClaimProfile(
        is_checkable=True,
        claim_type="temporal_fact",
        domain="science",
        risk_level="medium",
        risk_level_source="llm_inferred",
        required_evidence=["date", "official"],
        strict_mode=False,
        reason="...",
    )
    assert profile.allow_single_t1_source is True
    assert profile.allow_single_t0_source is True
    # Both flags must be independently controllable
    profile.allow_single_t1_source = False
    profile.allow_single_t0_source = True
    assert profile.allow_single_t1_source is False
    assert profile.allow_single_t0_source is True


def test_claim_profile_has_no_legacy_authoritative_field() -> None:
    profile = ClaimProfile(
        is_checkable=True,
        claim_type="entity_fact",
        domain="general",
        risk_level="low",
        risk_level_source="user_hint",
        required_evidence=[],
        strict_mode=False,
        reason="",
    )
    # legacy field name must not exist anywhere on the dataclass
    assert not hasattr(profile, "allow_single_authoritative_source")


# ---------------------------------------------------------------------------
# AtomicClaim
# ---------------------------------------------------------------------------

def test_atomic_claim_requires_non_empty_original_span() -> None:
    with pytest.raises(ValueError, match="original_span"):
        AtomicClaim(
            id="c1",
            text="嫦娥四号于2019年1月3日着陆",
            original_span="",
            parent_claim="嫦娥四号于2019年1月3日着陆月球背面。",
            check_priority=1,
            required_evidence_type=["date"],
        )


def test_atomic_claim_text_may_equal_original_span() -> None:
    c = AtomicClaim(
        id="c1",
        text="嫦娥四号于2019年1月3日着陆",
        original_span="嫦娥四号于2019年1月3日着陆",
        parent_claim="嫦娥四号于2019年1月3日着陆月球背面。",
        check_priority=1,
        required_evidence_type=["date"],
    )
    assert c.text == c.original_span


# ---------------------------------------------------------------------------
# EvidenceSpan / EvidenceScore
# ---------------------------------------------------------------------------

def test_evidence_span_carries_audit_fields() -> None:
    span = EvidenceSpan(
        id="e1",
        title="嫦娥四号 - 维基百科",
        url="https://zh.wikipedia.org/wiki/...",
        quote="嫦娥四号于2019年1月3日成功着陆在月球背面的冯·卡门撞击坑...",
        source_name="wikipedia_zh",
        source_tier="T1",
        retrieved_at="2026-05-12T00:00:00Z",
        content_hash="0123456789abcdef",
    )
    # url / quote / retrieved_at / content_hash are mandatory (no defaults)
    assert span.url and span.quote and span.retrieved_at and span.content_hash
    assert span.raw_score is None
    assert span.metadata == {}
    assert span.risk_flags == []


def test_evidence_score_defaults() -> None:
    s = EvidenceScore(
        source_tier_score=0.9,
        lexical_overlap_score=0.8,
        semantic_support_score=0.85,
    )
    assert s.freshness_score is None
    assert s.injection_risk_penalty == 0.0
    assert s.final_score == 0.0


# ---------------------------------------------------------------------------
# Judgement & disagreement
# ---------------------------------------------------------------------------

def test_model_vote_supports_failure_record() -> None:
    ok = ModelVote(model_name="deepseek", relation="supports", confidence="high", reason="ok")
    failed = ModelVote(model_name="mimo", relation=None, confidence=None, error="timeout")
    assert ok.error is None
    assert failed.relation is None
    assert failed.confidence is None
    assert failed.error == "timeout"


def test_evidence_judgement_default_model_votes_empty() -> None:
    j = EvidenceJudgement(
        atomic_claim_id="c1",
        evidence_id="e1",
        relation="supports",
        confidence="high",
        quote="嫦娥四号于2019年1月3日成功着陆",
        reason="日期与实体均直接匹配",
    )
    assert j.model_votes == []


def test_disagreement_dataclass_shape() -> None:
    d = Disagreement(
        atomic_claim_id="c1",
        evidence_id="e1",
        model_votes=[
            ModelVote("deepseek", "supports", "high"),
            ModelVote("mimo", "refutes", "medium"),
        ],
        summary="supports vs refutes",
    )
    assert is_dataclass(d)
    assert len(d.model_votes) == 2


def test_conflict_report_dataclass_shape() -> None:
    c = ConflictReport(
        atomic_claim_id="c1",
        supporting_evidence_ids=["e1"],
        refuting_evidence_ids=["e2"],
        summary="two contradictory T1 sources",
    )
    assert is_dataclass(c)


# ---------------------------------------------------------------------------
# RetrievalPlan
# ---------------------------------------------------------------------------

def test_retrieval_plan_carries_p0_source_targets() -> None:
    plan = RetrievalPlan(
        queries=["嫦娥四号 月球背面"],
        source_targets=["tavily", "wikipedia_zh"],
        min_independent_sources=2,
        allow_discovery_sources=True,
        allow_single_source_medium=False,
        require_official_source=False,
        require_freshness_check=False,
    )
    assert plan.source_targets == ["tavily", "wikipedia_zh"]


# ---------------------------------------------------------------------------
# VerificationResult & to_compact_dict
# ---------------------------------------------------------------------------

def test_verification_result_defaults() -> None:
    span = EvidenceSpan(
        id="e1", title="t", url="https://x", quote="q",
        source_name="wikipedia_zh", source_tier="T1",
        retrieved_at="2026-05-12T00:00:00Z", content_hash="abc",
    )
    result = VerificationResult(
        verdict="likely_correct",
        confidence=0.82,
        claim="...",
        sources=[span],
        reasoning="两源一致",
    )
    assert result.consensus_method == "strictest"
    assert result.model_votes == []
    assert result.disagreements == []
    assert result.conflicts == []
    assert result.audit_trace == []
    # CostBreakdown initialized via default_factory
    assert result.cost.llm_calls == 0
    assert result.cost.retrieval_calls == 0


def test_verification_result_compact_dict_only_exposes_5_fields() -> None:
    span = EvidenceSpan(
        id="e1", title="t", url="https://x", quote="q",
        source_name="wikipedia_zh", source_tier="T1",
        retrieved_at="2026-05-12T00:00:00Z", content_hash="abc",
    )
    result = VerificationResult(
        verdict="likely_correct",
        confidence=0.82,
        claim="hello",
        sources=[span],
        reasoning="reason",
    )
    compact = result.to_compact_dict()
    assert set(compact.keys()) == {"verdict", "confidence", "claim", "sources", "reasoning"}
    assert compact["verdict"] == "likely_correct"
    assert compact["confidence"] == 0.82
    assert compact["claim"] == "hello"
    assert isinstance(compact["sources"], list) and len(compact["sources"]) == 1
    assert compact["sources"][0]["id"] == "e1"
    assert compact["reasoning"] == "reason"


def test_verification_result_serializable_via_asdict() -> None:
    span = EvidenceSpan(
        id="e1", title="t", url="https://x", quote="q",
        source_name="wikipedia_zh", source_tier="T1",
        retrieved_at="2026-05-12T00:00:00Z", content_hash="abc",
    )
    result = VerificationResult(
        verdict="needs_review",
        confidence=0.5,
        claim="...",
        sources=[span],
        reasoning="...",
    )
    d = asdict(result)
    # The compact subset must be present in the full dict
    for key in ("verdict", "confidence", "claim", "sources", "reasoning"):
        assert key in d
    # Plus audit fields
    for key in ("model_votes", "disagreements", "conflicts", "audit_trace", "cost"):
        assert key in d
