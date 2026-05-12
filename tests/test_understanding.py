"""Unit tests for cerno.understanding — claim profile, decompose, retrieval plan.

No real LLMs, no real network. All adapters are local FakeAdapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cerno.consensus import LLMResponse, MultiModelConsensus
from cerno.types import (
    AtomicClaim,
    ClaimProfile,
    VerificationRequest,
)
from cerno.understanding import (
    DECOMPOSE_SCHEMA,
    MAX_ATOMIC_CLAIMS,
    P0_SOURCE_TARGETS,
    PROFILE_SCHEMA,
    VALID_CLAIM_TYPES,
    build_claim_profile,
    build_retrieval_plan,
    decompose_claim,
)

# ---------------------------------------------------------------------------
# Test stubs — never reach the real network / real LLMs
# ---------------------------------------------------------------------------

@dataclass
class FakeAdapter:
    """Duck-typed adapter that feeds canned LLMResponse objects in order.
    Does NOT validate schema — the test author controls the parsed payload.
    """

    name: str
    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls.append((system, user, schema))
        if not self.responses:
            raise AssertionError(f"FakeAdapter[{self.name}] out of canned responses")
        return self.responses.pop(0)


def _profile_response(
    name: str,
    *,
    is_checkable: bool = True,
    claim_type: str = "entity_fact",
    domain: str = "general",
    risk_level: str = "low",
    required_evidence: list[str] | None = None,
    strict_mode: bool = False,
    reason: str = "test",
) -> LLMResponse:
    return LLMResponse(
        model_name=name,
        parsed={
            "is_checkable": is_checkable,
            "claim_type": claim_type,
            "domain": domain,
            "risk_level": risk_level,
            "required_evidence": list(required_evidence or []),
            "strict_mode": strict_mode,
            "reason": reason,
        },
        raw_text="{}",
    )


def _decompose_response(name: str, items: list[dict[str, Any]]) -> LLMResponse:
    return LLMResponse(
        model_name=name,
        parsed={"atomic_claims": items},
        raw_text="{}",
    )


def _fail_response(name: str, error: str = "json_parse_failed") -> LLMResponse:
    return LLMResponse(model_name=name, parsed=None, raw_text="garbage", error=error)


def _consensus(*responses: LLMResponse) -> MultiModelConsensus:
    adapter = FakeAdapter("fake", list(responses))
    return MultiModelConsensus([adapter])


def _stub_profile(
    *,
    risk_level: str = "low",
    claim_type: str = "entity_fact",
    strict_mode: bool = False,
    required_evidence: list[str] | None = None,
    domain: str = "general",
) -> ClaimProfile:
    return ClaimProfile(
        is_checkable=True,
        claim_type=claim_type,  # type: ignore[arg-type]
        domain=domain,
        risk_level=risk_level,  # type: ignore[arg-type]
        risk_level_source="llm_inferred",
        required_evidence=list(required_evidence or []),
        strict_mode=strict_mode,
        reason="stub",
    )


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_profile_schema_locks_claim_type_enum() -> None:
    enum = PROFILE_SCHEMA["properties"]["claim_type"]["enum"]
    assert set(enum) == set(VALID_CLAIM_TYPES)


def test_profile_schema_locks_risk_level_enum() -> None:
    enum = PROFILE_SCHEMA["properties"]["risk_level"]["enum"]
    assert set(enum) == {"low", "medium", "high", "critical"}


def test_decompose_schema_enforces_non_empty_original_span() -> None:
    item_schema = DECOMPOSE_SCHEMA["properties"]["atomic_claims"]["items"]
    assert item_schema["properties"]["original_span"]["minLength"] == 1


# ---------------------------------------------------------------------------
# build_claim_profile — LLM happy path
# ---------------------------------------------------------------------------

def test_build_claim_profile_uses_llm_when_valid() -> None:
    request = VerificationRequest(claim="水的沸点是 100 摄氏度", domain_hint="science")
    consensus = _consensus(
        _profile_response(
            "fake",
            claim_type="quantitative_fact",
            domain="science",
            risk_level="low",
            required_evidence=["scientific_paper"],
            reason="basic_fact",
        )
    )
    profile = build_claim_profile(request, consensus)
    assert profile.is_checkable is True
    assert profile.claim_type == "quantitative_fact"
    assert profile.domain == "science"
    assert profile.risk_level == "low"
    assert profile.required_evidence == ["scientific_paper"]
    assert profile.strict_mode is False
    assert profile.reason == "basic_fact"


def test_build_claim_profile_normalizes_invalid_claim_type_to_other() -> None:
    """Defense in depth: even if a future schema relaxes, business code clamps
    claim_type to the canonical set."""
    request = VerificationRequest(claim="x")
    bad_resp = LLMResponse(
        model_name="fake",
        parsed={
            "is_checkable": True,
            "claim_type": "invalid_kind",
            "domain": "general",
            "risk_level": "low",
            "required_evidence": [],
            "strict_mode": False,
            "reason": "test",
        },
        raw_text="{}",
    )
    consensus = _consensus(bad_resp)
    profile = build_claim_profile(request, consensus)
    assert profile.claim_type == "other"


def test_build_claim_profile_normalizes_invalid_risk_level_to_low() -> None:
    request = VerificationRequest(claim="x")
    bad_resp = LLMResponse(
        model_name="fake",
        parsed={
            "is_checkable": True,
            "claim_type": "entity_fact",
            "domain": "general",
            "risk_level": "extreme",
            "required_evidence": [],
            "strict_mode": False,
            "reason": "test",
        },
        raw_text="{}",
    )
    consensus = _consensus(bad_resp)
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "low"


def test_build_claim_profile_forces_strict_mode_when_risk_is_high() -> None:
    request = VerificationRequest(claim="x")
    consensus = _consensus(
        _profile_response("fake", risk_level="high", strict_mode=False)
    )
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "high"
    assert profile.strict_mode is True


def test_build_claim_profile_forces_strict_mode_when_risk_is_critical() -> None:
    request = VerificationRequest(claim="x")
    consensus = _consensus(
        _profile_response("fake", risk_level="critical", strict_mode=False)
    )
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "critical"
    assert profile.strict_mode is True


# ---------------------------------------------------------------------------
# build_claim_profile — risk reconciliation (max-of-both)
# ---------------------------------------------------------------------------

def test_risk_hint_higher_than_llm_takes_hint_with_max_of_both() -> None:
    request = VerificationRequest(claim="x", risk_hint="high")
    consensus = _consensus(_profile_response("fake", risk_level="low"))
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "high"
    assert profile.risk_level_source == "max_of_both"
    # strict_mode coerced True because reconciled risk is "high"
    assert profile.strict_mode is True


def test_llm_risk_higher_than_hint_takes_llm_with_max_of_both() -> None:
    request = VerificationRequest(claim="x", risk_hint="low")
    consensus = _consensus(_profile_response("fake", risk_level="critical"))
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "critical"
    assert profile.risk_level_source == "max_of_both"


def test_risk_hint_equal_to_llm_keeps_llm_inferred() -> None:
    request = VerificationRequest(claim="x", risk_hint="medium")
    consensus = _consensus(_profile_response("fake", risk_level="medium"))
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level == "medium"
    assert profile.risk_level_source == "llm_inferred"


def test_no_risk_hint_keeps_llm_inferred() -> None:
    request = VerificationRequest(claim="x", risk_hint=None)
    consensus = _consensus(_profile_response("fake", risk_level="medium"))
    profile = build_claim_profile(request, consensus)
    assert profile.risk_level_source == "llm_inferred"


def test_invalid_risk_hint_is_ignored() -> None:
    request = VerificationRequest(claim="x", risk_hint="ULTRA")
    consensus = _consensus(_profile_response("fake", risk_level="medium"))
    profile = build_claim_profile(request, consensus)
    # Invalid hint is dropped → source stays llm_inferred
    assert profile.risk_level == "medium"
    assert profile.risk_level_source == "llm_inferred"


# ---------------------------------------------------------------------------
# build_claim_profile — rule fallback
# ---------------------------------------------------------------------------

def test_build_claim_profile_falls_back_to_rule_when_llm_fails() -> None:
    request = VerificationRequest(claim="爱因斯坦于 1879 年出生")
    consensus = _consensus(_fail_response("fake"))
    profile = build_claim_profile(request, consensus)
    assert profile.reason == "rule_fallback"
    # 1879 年 → temporal_fact
    assert profile.claim_type == "temporal_fact"


def test_rule_fallback_marks_opinion_claims_uncheckable() -> None:
    request = VerificationRequest(claim="我认为这个项目会成功")
    consensus = _consensus(_fail_response("fake"))
    profile = build_claim_profile(request, consensus)
    assert profile.reason == "rule_fallback"
    assert profile.is_checkable is False


def test_rule_fallback_honors_risk_hint() -> None:
    request = VerificationRequest(claim="x", risk_hint="high")
    consensus = _consensus(_fail_response("fake"))
    profile = build_claim_profile(request, consensus)
    # rule-inferred risk = "low", hint = "high" → max-of-both gives high
    assert profile.risk_level == "high"
    assert profile.risk_level_source == "max_of_both"
    assert profile.strict_mode is True


# ---------------------------------------------------------------------------
# decompose_claim — LLM happy path
# ---------------------------------------------------------------------------

def test_decompose_claim_uses_llm_when_valid() -> None:
    request = VerificationRequest(claim="爱因斯坦出生于德国,获得过诺贝尔奖")
    profile = _stub_profile()
    consensus = _consensus(
        _decompose_response(
            "fake",
            [
                {
                    "text": "爱因斯坦出生于德国",
                    "original_span": "爱因斯坦出生于德国",
                    "check_priority": 1,
                    "required_evidence_type": ["biographical"],
                },
                {
                    "text": "爱因斯坦获得过诺贝尔奖",
                    "original_span": "获得过诺贝尔奖",
                    "check_priority": 2,
                    "required_evidence_type": ["award"],
                },
            ],
        )
    )
    claims = decompose_claim(request, profile, consensus)
    assert len(claims) == 2
    assert claims[0].id == "c1"
    assert claims[0].text == "爱因斯坦出生于德国"
    assert claims[0].original_span == "爱因斯坦出生于德国"
    assert claims[0].parent_claim == request.claim
    assert claims[0].required_evidence_type == ["biographical"]
    assert claims[1].id == "c2"
    assert claims[1].check_priority == 2


def test_decompose_claim_caps_at_max_atomic_claims() -> None:
    request = VerificationRequest(claim="a; b; c; d; e")
    profile = _stub_profile()
    items = [
        {
            "text": f"part{i}",
            "original_span": f"span{i}",
            "check_priority": i,
            "required_evidence_type": [],
        }
        for i in range(1, 6)  # 5 items, MAX_ATOMIC_CLAIMS = 3
    ]
    consensus = _consensus(_decompose_response("fake", items))
    claims = decompose_claim(request, profile, consensus)
    assert len(claims) == MAX_ATOMIC_CLAIMS == 3
    assert [c.id for c in claims] == ["c1", "c2", "c3"]


def test_decompose_claim_drops_items_with_empty_original_span() -> None:
    request = VerificationRequest(claim="x")
    profile = _stub_profile()
    items = [
        {
            "text": "good",
            "original_span": "good span",
            "check_priority": 1,
            "required_evidence_type": [],
        },
        {
            "text": "bad",
            "original_span": "   ",  # whitespace-only must be dropped
            "check_priority": 2,
            "required_evidence_type": [],
        },
    ]
    consensus = _consensus(_decompose_response("fake", items))
    claims = decompose_claim(request, profile, consensus)
    assert len(claims) == 1
    assert claims[0].text == "good"


def test_decompose_claim_falls_back_to_rule_when_llm_fails() -> None:
    request = VerificationRequest(claim="第一句。第二句;第三句；第四句")
    profile = _stub_profile()
    consensus = _consensus(_fail_response("fake"))
    claims = decompose_claim(request, profile, consensus)
    assert 1 <= len(claims) <= MAX_ATOMIC_CLAIMS
    # rule fallback keeps original spans non-empty
    assert all(c.original_span for c in claims)
    assert all(c.parent_claim == request.claim for c in claims)


def test_decompose_claim_falls_back_when_all_items_empty_original_span() -> None:
    """If every LLM item has an empty/whitespace original_span, we must fall back
    to the rule decomposer rather than emit zero claims."""
    request = VerificationRequest(claim="独立可验证语句")
    profile = _stub_profile()
    consensus = _consensus(
        _decompose_response(
            "fake",
            [
                {
                    "text": "ignored",
                    "original_span": "",  # bypass schema (FakeAdapter doesn't enforce)
                    "check_priority": 1,
                    "required_evidence_type": [],
                }
            ],
        )
    )
    claims = decompose_claim(request, profile, consensus)
    assert len(claims) >= 1
    assert all(c.original_span for c in claims)


def test_atomic_claim_rejects_empty_original_span_directly() -> None:
    """Type-level invariant: AtomicClaim.__post_init__ raises on empty span."""
    with pytest.raises(ValueError):
        AtomicClaim(
            id="c1",
            text="x",
            original_span="",
            parent_claim="parent",
            check_priority=1,
            required_evidence_type=[],
        )


# ---------------------------------------------------------------------------
# build_retrieval_plan
# ---------------------------------------------------------------------------

def test_retrieval_plan_source_targets_locked_to_p0_pair() -> None:
    profile = _stub_profile()
    claims = [
        AtomicClaim(
            id="c1",
            text="q1",
            original_span="q1",
            parent_claim="x",
            check_priority=1,
            required_evidence_type=[],
        )
    ]
    plan = build_retrieval_plan(profile, claims)
    assert plan.source_targets == ["tavily", "wikipedia_zh"]
    assert P0_SOURCE_TARGETS == ("tavily", "wikipedia_zh")


def test_retrieval_plan_has_no_legacy_allow_single_authoritative_source() -> None:
    """The earlier design draft mentioned `allow_single_authoritative_source` —
    that name must never reappear on the RetrievalPlan type."""
    profile = _stub_profile()
    plan = build_retrieval_plan(profile, [])
    field_names = set(plan.__dataclass_fields__.keys())
    assert "allow_single_authoritative_source" not in field_names


def test_retrieval_plan_queries_match_atomic_claims() -> None:
    profile = _stub_profile()
    claims = [
        AtomicClaim(
            id="c1",
            text="alpha",
            original_span="alpha",
            parent_claim="x",
            check_priority=1,
            required_evidence_type=[],
        ),
        AtomicClaim(
            id="c2",
            text="beta",
            original_span="beta",
            parent_claim="x",
            check_priority=2,
            required_evidence_type=[],
        ),
    ]
    plan = build_retrieval_plan(profile, claims)
    assert plan.queries == ["alpha", "beta"]


def test_retrieval_plan_defaults() -> None:
    profile = _stub_profile()
    plan = build_retrieval_plan(profile, [])
    assert plan.min_independent_sources == 2
    assert plan.allow_single_source_medium is True
    assert plan.allow_discovery_sources is True  # strict_mode=False
    assert plan.require_official_source is False  # risk_level=low
    assert plan.require_freshness_check is False  # claim_type=entity_fact


def test_retrieval_plan_strict_mode_disables_discovery_sources() -> None:
    profile = _stub_profile(strict_mode=True, risk_level="high")
    plan = build_retrieval_plan(profile, [])
    assert plan.allow_discovery_sources is False


def test_retrieval_plan_high_risk_requires_official_source() -> None:
    profile = _stub_profile(risk_level="high", strict_mode=True)
    plan = build_retrieval_plan(profile, [])
    assert plan.require_official_source is True


def test_retrieval_plan_critical_risk_requires_official_source() -> None:
    profile = _stub_profile(risk_level="critical", strict_mode=True)
    plan = build_retrieval_plan(profile, [])
    assert plan.require_official_source is True


def test_retrieval_plan_low_risk_does_not_require_official_source() -> None:
    profile = _stub_profile(risk_level="low")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_official_source is False


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def test_build_claim_profile_accumulates_cost() -> None:
    from cerno.observability import CostBreakdown

    request = VerificationRequest(claim="test")
    consensus = _consensus(
        LLMResponse(
            model_name="fake",
            parsed={
                "is_checkable": True,
                "claim_type": "entity_fact",
                "domain": "general",
                "risk_level": "low",
                "required_evidence": [],
                "strict_mode": False,
                "reason": "ok",
            },
            raw_text="{}",
            input_tokens=10,
            output_tokens=5,
        )
    )
    cost = CostBreakdown()
    profile = build_claim_profile(request, consensus, cost=cost)
    assert profile.is_checkable is True
    assert cost.llm_calls == 1
    assert cost.input_tokens == 10
    assert cost.output_tokens == 5


def test_decompose_claim_accumulates_cost() -> None:
    from cerno.observability import CostBreakdown

    request = VerificationRequest(claim="test")
    profile = _stub_profile()
    consensus = _consensus(
        LLMResponse(
            model_name="fake",
            parsed={
                "atomic_claims": [
                    {
                        "text": "t1",
                        "original_span": "t1",
                        "check_priority": 1,
                        "required_evidence_type": [],
                    }
                ]
            },
            raw_text="{}",
            input_tokens=20,
            output_tokens=8,
        )
    )
    cost = CostBreakdown()
    claims = decompose_claim(request, profile, consensus, cost=cost)
    assert len(claims) == 1
    assert cost.llm_calls == 1
    assert cost.input_tokens == 20
    assert cost.output_tokens == 8


def test_build_claim_profile_without_cost_does_not_crash() -> None:
    request = VerificationRequest(claim="test")
    consensus = _consensus(
        _profile_response("fake", claim_type="entity_fact", risk_level="low")
    )
    profile = build_claim_profile(request, consensus)
    assert profile.is_checkable is True


def test_retrieval_plan_temporal_fact_requires_freshness_check() -> None:
    profile = _stub_profile(claim_type="temporal_fact")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_freshness_keyword_in_evidence_triggers_check() -> None:
    profile = _stub_profile(required_evidence=["fresh_source"])
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_policy_domain_requires_freshness_check() -> None:
    """design §11.5: domain=policy → freshness check, even for entity_fact."""
    profile = _stub_profile(domain="policy", claim_type="entity_fact")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_legal_domain_requires_freshness_check() -> None:
    profile = _stub_profile(domain="legal")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_law_domain_requires_freshness_check() -> None:
    """Design draft uses ``law``; we accept it as well as ``legal``."""
    profile = _stub_profile(domain="law")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_finance_domain_requires_freshness_check() -> None:
    profile = _stub_profile(domain="finance")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_news_domain_requires_freshness_check() -> None:
    profile = _stub_profile(domain="news")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is True


def test_retrieval_plan_general_no_freshness_keyword_no_check() -> None:
    """domain=general, claim_type=entity_fact, no fresh keyword → no check."""
    profile = _stub_profile(domain="general", claim_type="entity_fact")
    plan = build_retrieval_plan(profile, [])
    assert plan.require_freshness_check is False


# ---------------------------------------------------------------------------
# No-network, no-real-LLM guard
# ---------------------------------------------------------------------------

def test_no_real_network_or_llm_calls_in_this_module() -> None:
    """Smoke-test the test surface: FakeAdapter is the only adapter type used.
    Combined with the explicit import list at the top (no OpenAICompatibleAdapter),
    this guarantees no real LLM endpoint is hit during this test module.
    """
    adapter = FakeAdapter("guard", [_profile_response("guard")])
    consensus = MultiModelConsensus([adapter])
    assert type(adapter).__name__ == "FakeAdapter"
    request = VerificationRequest(claim="x")
    profile = build_claim_profile(request, consensus)
    # Exactly one call was made to the fake adapter.
    assert len(adapter.calls) == 1
    assert profile is not None
