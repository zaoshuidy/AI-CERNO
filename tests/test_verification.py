"""Unit tests for cerno.verification.

No real LLMs, no real network. Tests inject a stubbed MultiModelConsensus
subclass that returns a pre-canned ConsensusResult so we can assert every
branch of ``verify_evidence`` independently of the consensus pipeline.
"""

from __future__ import annotations

from typing import Any

import pytest

from cerno.consensus import ConsensusResult, MultiModelConsensus
from cerno.types import (
    AtomicClaim,
    EvidenceSpan,
    ModelVote,
)
from cerno.verification import (
    CITATION_MAX_CHARS,
    LEXICAL_OVERLAP_FLOOR,
    VERIFY_RESPONSE_SCHEMA,
    VERIFY_SYSTEM_PROMPT,
    anchor_citation,
    verify_evidence,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class _NoopAdapter:
    """Satisfies the _AdapterLike protocol so MultiModelConsensus.__init__ is
    happy. ``chat_json`` raises if the stub's ``judge`` ever delegates back to
    the real path — which it never should once we override ``judge``.
    """

    name = "noop-adapter"

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> Any:
        raise AssertionError(
            "_NoopAdapter.chat_json should never run when judge() is stubbed"
        )


class _StubConsensus(MultiModelConsensus):
    """Returns a pre-canned ConsensusResult on every ``judge`` call and
    records the kwargs for assertion."""

    def __init__(self, result: ConsensusResult) -> None:
        super().__init__([_NoopAdapter()])
        self._result = result
        self.judge_calls: list[dict[str, Any]] = []

    def judge(  # type: ignore[override]
        self,
        *,
        step: str,
        claim: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        high_risk: bool = False,
    ) -> ConsensusResult:
        self.judge_calls.append({
            "step": step,
            "claim": claim,
            "system": system,
            "user": user,
            "schema": schema,
            "high_risk": high_risk,
        })
        return self._result


class _AssertNoCallConsensus(MultiModelConsensus):
    """Drop-in MultiModelConsensus that asserts ``judge`` is never invoked.
    Used for the hard-rule short-circuit paths."""

    def __init__(self) -> None:
        super().__init__([_NoopAdapter()])

    def judge(self, **kwargs: Any) -> ConsensusResult:  # type: ignore[override]
        raise AssertionError(
            f"consensus.judge should not be called; got kwargs={kwargs}"
        )


def _make_evidence(**overrides: Any) -> EvidenceSpan:
    defaults: dict[str, Any] = {
        "id": "ev1",
        "title": "title",
        "url": "https://example.com/article",
        "quote": "爱因斯坦因光电效应获得1921年诺贝尔物理学奖。",
        "source_name": "tavily",
        "source_tier": "T1",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "content_hash": "deadbeefcafe1234",
        "raw_score": 0.9,
        "metadata": {},
        "risk_flags": [],
    }
    defaults.update(overrides)
    return EvidenceSpan(**defaults)


def _make_claim(text: str = "爱因斯坦获得诺贝尔物理学奖", **overrides: Any) -> AtomicClaim:
    defaults: dict[str, Any] = {
        "id": "c1",
        "text": text,
        "original_span": text,
        "parent_claim": text,
        "check_priority": 1,
        "required_evidence_type": [],
    }
    defaults.update(overrides)
    return AtomicClaim(**defaults)


def _make_consensus_result(
    *,
    relation: str = "supports",
    confidence: str = "high",
    quote: str = "",
    reason: str = "dual_supports",
    model_votes: list[ModelVote] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ConsensusResult:
    return ConsensusResult(
        relation=relation,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        quote=quote,
        reason=reason,
        model_votes=list(model_votes or []),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# 1. EvidenceSpan contract — every missing field short-circuits to insufficient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("url", "evidence_missing_url"),
        ("quote", "evidence_missing_quote"),
        ("retrieved_at", "evidence_missing_retrieved_at"),
        ("content_hash", "evidence_missing_content_hash"),
    ],
)
def test_verify_evidence_contract_violations_skip_consensus(
    field: str, expected_reason: str
) -> None:
    claim = _make_claim()
    evidence = _make_evidence(**{field: ""})
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == expected_reason
    assert judgement.model_votes == []
    # Substring invariant: quote must be a substring of evidence.quote.
    assert judgement.quote in evidence.quote


# ---------------------------------------------------------------------------
# 2. Lexical overlap floor
# ---------------------------------------------------------------------------

def test_verify_evidence_low_lexical_overlap_returns_neutral_low() -> None:
    claim = _make_claim(text="苹果公司发布了新的智能手机")
    # Evidence quote has zero overlap with the claim's anchors.
    evidence = _make_evidence(quote="天气很好今天阳光明媚适合出门散步")
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "neutral"
    assert judgement.confidence == "low"
    assert judgement.reason.startswith("lexical_overlap_too_low")
    assert judgement.model_votes == []


def test_verify_evidence_lexical_overlap_floor_constant_is_documented() -> None:
    # If this assertion fails, update the threshold in the design doc too.
    assert LEXICAL_OVERLAP_FLOOR == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 3. Claim-type hard rules
# ---------------------------------------------------------------------------

def test_verify_evidence_quantitative_missing_unit_short_circuits() -> None:
    claim = _make_claim(text="中国人口约14亿人")
    # Evidence quote has the number but no qualifying unit token.
    evidence = _make_evidence(quote="中国人口大约14")
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "quantitative_missing_number_or_unit"


def test_verify_evidence_quantitative_missing_number_short_circuits() -> None:
    claim = _make_claim(text="中国人口约14亿人")
    # Evidence quote has a unit-like token but no digit.
    evidence = _make_evidence(quote="中国人口很多人")
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "quantitative_missing_number_or_unit"


def test_verify_evidence_temporal_missing_year_short_circuits() -> None:
    claim = _make_claim(text="中华人民共和国于1949年成立")
    evidence = _make_evidence(quote="中华人民共和国成立是历史大事件")
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "temporal_missing_date_or_year"


def test_verify_evidence_quotation_missing_phrase_short_circuits() -> None:
    claim = _make_claim(text="马克思说过人是社会关系的总和")
    evidence = _make_evidence(quote="马克思深入研究了人类社会关系的本质问题")
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "quotation_missing_in_evidence"


# ---------------------------------------------------------------------------
# 4. Core-entity rule
# ---------------------------------------------------------------------------

def test_verify_evidence_core_entity_missing_short_circuits() -> None:
    # "Anthropic" is the longest anchor; evidence quote talks about OpenAI.
    claim = _make_claim(text="Anthropic是一家人工智能公司")
    evidence = _make_evidence(
        quote="OpenAI是一家人工智能公司,总部位于旧金山。",
    )
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason.startswith("core_entity_missing:")


# ---------------------------------------------------------------------------
# 5. Mutually exclusive term rule
# ---------------------------------------------------------------------------

def test_verify_evidence_lunar_near_side_claim_far_side_evidence_refutes() -> None:
    claim = _make_claim(
        text="\u5ae6\u5a25\u56db\u53f7\u7740\u9646\u5728"
        "\u6708\u7403\u6b63\u9762"
    )
    evidence = _make_evidence(
        quote="\u5ae6\u5a25\u56db\u53f7\u63a2\u6d4b\u5668"
        "\u7740\u9646\u5728\u6708\u7403\u80cc\u9762\u3002",
    )
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "refutes"
    assert judgement.confidence == "low"
    assert judgement.reason.startswith("mutually_exclusive_terms:")
    assert judgement.model_votes == []


def test_verify_evidence_lunar_near_side_claim_yuebei_evidence_refutes() -> None:
    claim = _make_claim(
        text="\u5ae6\u5a25\u56db\u53f7\u6210\u4e3a\u4eba\u7c7b"
        "\u9996\u6b21\u5728\u6708\u7403\u6b63\u9762"
        "\u5b9e\u73b0\u8f6f\u7740\u9646\u7684\u63a2\u6d4b\u5668"
    )
    evidence = _make_evidence(
        quote="\u5ae6\u5a25\u56db\u53f7\u5b9e\u73b0\u4eba\u7c7b"
        "\u63a2\u6d4b\u5668\u9996\u6b21\u6708\u80cc"
        "\u8f6f\u7740\u9646\u3002",
    )
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "refutes"
    assert judgement.confidence == "low"
    assert judgement.reason.startswith("mutually_exclusive_terms:")
    assert judgement.model_votes == []


def test_verify_evidence_competing_mission_sample_return_refutes() -> None:
    claim = _make_claim(
        text="\u5ae6\u5a25\u56db\u53f7\u5f00\u542f\u4e86\u4eba\u7c7b"
        "\u7b2c\u4e00\u6b21\u4ece\u6708\u7403\u80cc\u9762"
        "\u91c7\u96c6\u6708\u58e4\u5e76\u5e26\u56de\u5730\u7403"
        "\u7684\u5de5\u7a0b\u5b9e\u8df5"
    )
    evidence = _make_evidence(
        quote="\u5ae6\u5a25\u516d\u53f7\u5b8c\u6210\u4eba\u7c7b"
        "\u9996\u6b21\u6708\u7403\u80cc\u9762\u91c7\u6837\u8fd4\u56de\u3002",
    )
    consensus = _AssertNoCallConsensus()

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "refutes"
    assert judgement.confidence == "low"
    assert judgement.reason.startswith("competing_mission_attribution:")
    assert judgement.model_votes == []


# ---------------------------------------------------------------------------
# 6. Consensus dispatch — happy paths
# ---------------------------------------------------------------------------

def _passing_pair() -> tuple[AtomicClaim, EvidenceSpan]:
    """A claim + evidence that passes every hard rule so consensus is called."""
    claim = _make_claim(text="爱因斯坦获得诺贝尔物理学奖")
    evidence = _make_evidence(
        quote="爱因斯坦因光电效应获得1921年诺贝尔物理学奖。",
    )
    return claim, evidence


def test_verify_evidence_consensus_supports_passes_through() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports",
            confidence="high",
            quote="爱因斯坦因光电效应获得1921年诺贝尔物理学奖",
            reason="dual_supports",
            model_votes=[
                ModelVote(model_name="m1", relation="supports", confidence="high"),
                ModelVote(model_name="m2", relation="supports", confidence="high"),
            ],
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "supports"
    assert judgement.confidence == "high"
    assert judgement.reason == "dual_supports"
    assert judgement.quote == "爱因斯坦因光电效应获得1921年诺贝尔物理学奖"
    assert len(judgement.model_votes) == 2
    assert len(consensus.judge_calls) == 1
    assert consensus.judge_calls[0]["high_risk"] is False
    assert consensus.judge_calls[0]["schema"] is VERIFY_RESPONSE_SCHEMA


def test_verify_evidence_consensus_refutes_passes_through() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="refutes",
            confidence="medium",
            quote="爱因斯坦因光电效应获得",
            reason="refutes_with_weaker_partner",
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "refutes"
    assert judgement.confidence == "medium"
    assert judgement.reason == "refutes_with_weaker_partner"


def test_verify_evidence_aligned_negative_claim_refute_normalizes_to_supports() -> None:
    """A negative claim supported by negative evidence is still supports.

    Some LLMs incorrectly label "the regulation is not legally effective" as
    refutes because the quote negates the regulation. Relation is about whether
    evidence supports the claim text, not whether the evidence negates an entity.
    """
    claim = _make_claim(text="《中华人民共和国印章管理办法》不具有法律效力")
    evidence = _make_evidence(
        quote=(
            "公安部工作人员表示《中华人民共和国印章管理办法》"
            "均未出台或施行，不具有法律效力。"
        ),
    )
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="refutes",
            confidence="high",
            quote="不具有法律效力",
            reason="model_misread_negative_claim",
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "supports"
    assert judgement.confidence == "high"
    assert judgement.reason == (
        "negative_claim_relation_normalized:model_misread_negative_claim"
    )


def test_verify_evidence_true_refute_of_negative_claim_stays_refutes() -> None:
    claim = _make_claim(text="《中华人民共和国印章管理办法》不具有法律效力")
    evidence = _make_evidence(
        quote="官方说明《中华人民共和国印章管理办法》已经出台并具有法律效力。",
    )
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="refutes",
            confidence="high",
            quote="具有法律效力",
            reason="evidence_says_legally_effective",
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "refutes"
    assert judgement.confidence == "high"
    assert judgement.reason == "evidence_says_legally_effective"


def test_verify_evidence_consensus_empty_quote_falls_back_to_evidence_quote() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="insufficient", confidence="low",
            quote="", reason="evidence_not_relevant",
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.quote == evidence.quote
    assert judgement.reason == "evidence_not_relevant"


# ---------------------------------------------------------------------------
# 6. Consensus dispatch — conflict / abstain
# ---------------------------------------------------------------------------

def test_verify_evidence_conflict_downgrades_to_insufficient_low() -> None:
    claim, evidence = _passing_pair()
    votes = [
        ModelVote(model_name="m1", relation="supports", confidence="high"),
        ModelVote(model_name="m2", relation="refutes", confidence="high"),
    ]
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="conflict",
            confidence="abstain",
            quote="爱因斯坦",  # would-be substring, but conflict trumps it
            reason="supports_vs_refutes",
            model_votes=votes,
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "model_conflict"
    assert judgement.quote == evidence.quote
    assert judgement.model_votes == votes


def test_verify_evidence_abstain_without_conflict_also_downgrades() -> None:
    claim, evidence = _passing_pair()
    votes = [
        ModelVote(model_name="m1", relation=None, confidence=None,
                  error="transport_error"),
    ]
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="insufficient",
            confidence="abstain",
            quote="",
            reason="all_models_failed",
            model_votes=votes,
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "model_conflict"
    assert judgement.model_votes == votes


# ---------------------------------------------------------------------------
# 7. Substring rule — fabricated quote → downgrade
# ---------------------------------------------------------------------------

def test_verify_evidence_model_quote_not_in_evidence_downgrades() -> None:
    claim, evidence = _passing_pair()
    votes = [
        ModelVote(model_name="m1", relation="supports", confidence="high"),
        ModelVote(model_name="m2", relation="supports", confidence="high"),
    ]
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports",
            confidence="high",
            quote="爱因斯坦发明了相对论",  # not a substring of evidence.quote
            reason="dual_supports",
            model_votes=votes,
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "insufficient"
    assert judgement.confidence == "low"
    assert judgement.reason == "model_quote_not_in_evidence"
    assert judgement.quote == evidence.quote  # falls back to anchor window
    assert judgement.model_votes == votes


def test_verify_evidence_model_quote_substring_is_preserved() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports",
            confidence="high",
            quote="爱因斯坦因光电效应",  # exact substring
            reason="dual_supports",
        )
    )

    judgement = verify_evidence(claim, evidence, consensus)

    assert judgement.relation == "supports"
    assert judgement.quote == "爱因斯坦因光电效应"
    assert judgement.quote in evidence.quote


# ---------------------------------------------------------------------------
# 8. UNTRUSTED_EVIDENCE boundary in the prompt
# ---------------------------------------------------------------------------

def test_verify_evidence_system_prompt_marks_evidence_untrusted() -> None:
    assert "<UNTRUSTED_EVIDENCE>" in VERIFY_SYSTEM_PROMPT
    assert "</UNTRUSTED_EVIDENCE>" in VERIFY_SYSTEM_PROMPT
    # The prompt must explicitly forbid memory-based truth judgements.
    assert "记忆" in VERIFY_SYSTEM_PROMPT or "memory" in VERIFY_SYSTEM_PROMPT.lower()


def test_verify_evidence_user_message_wraps_evidence_in_untrusted_tags() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports", confidence="high",
            quote="爱因斯坦", reason="ok",
        )
    )

    verify_evidence(claim, evidence, consensus)

    assert len(consensus.judge_calls) == 1
    user_msg = consensus.judge_calls[0]["user"]
    assert "<UNTRUSTED_EVIDENCE" in user_msg
    assert "</UNTRUSTED_EVIDENCE>" in user_msg
    assert evidence.quote in user_msg
    assert claim.text in user_msg


def test_verify_evidence_passes_correct_step_and_claim_to_judge() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports", confidence="high",
            quote="", reason="ok",
        )
    )

    verify_evidence(claim, evidence, consensus)

    call = consensus.judge_calls[0]
    assert call["step"] == "verify_evidence"
    assert call["claim"] == claim.text
    assert call["system"] is VERIFY_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 9. anchor_citation
# ---------------------------------------------------------------------------

def test_anchor_citation_returns_input_when_quote_is_clean() -> None:
    evidence = _make_evidence(quote="干净的引用文本")

    result = anchor_citation(evidence)

    # No transformation needed → same object identity.
    assert result is evidence


def test_anchor_citation_collapses_internal_whitespace() -> None:
    messy = "  文本  含有\t多个 \n  空白  "
    evidence = _make_evidence(quote=messy)

    result = anchor_citation(evidence)

    assert result is not evidence
    assert result.quote == "文本 含有 多个 空白"
    # All other fields preserved.
    assert result.id == evidence.id
    assert result.url == evidence.url
    assert result.content_hash == evidence.content_hash


def test_anchor_citation_caps_at_max_chars() -> None:
    long_quote = "字" * (CITATION_MAX_CHARS + 50)
    evidence = _make_evidence(quote=long_quote)

    result = anchor_citation(evidence)

    assert len(result.quote) == CITATION_MAX_CHARS


def test_anchor_citation_rejects_empty_quote() -> None:
    evidence = _make_evidence(quote="")

    with pytest.raises(ValueError):
        anchor_citation(evidence)


# ---------------------------------------------------------------------------
# 10. End-to-end sanity: judgement.quote always substrings evidence.quote
# ---------------------------------------------------------------------------

def test_verify_evidence_judgement_quote_always_substrings_evidence_quote() -> None:
    """Substring invariant — every code path obeys it, even hard-fail paths."""
    claim, evidence = _passing_pair()

    scenarios = [
        # Happy-path supports with substring quote.
        _StubConsensus(
            _make_consensus_result(
                relation="supports", confidence="high",
                quote="爱因斯坦", reason="ok",
            )
        ),
        # Conflict → falls back to evidence.quote.
        _StubConsensus(
            _make_consensus_result(
                relation="conflict", confidence="abstain",
                quote="anything", reason="x",
            )
        ),
        # Fabricated quote → falls back to evidence.quote.
        _StubConsensus(
            _make_consensus_result(
                relation="supports", confidence="high",
                quote="不在原文里的伪造引用", reason="x",
            )
        ),
    ]
    for consensus in scenarios:
        judgement = verify_evidence(claim, evidence, consensus)
        assert judgement.quote in evidence.quote, (
            f"judgement.quote {judgement.quote!r} not in evidence.quote"
        )


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def test_verify_evidence_accumulates_cost() -> None:
    from cerno.observability import CostBreakdown

    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports", confidence="high",
            quote="爱因斯坦", reason="ok",
            input_tokens=15,
            output_tokens=7,
        )
    )
    cost = CostBreakdown()
    judgement = verify_evidence(claim, evidence, consensus, cost=cost)
    assert judgement.relation == "supports"
    assert cost.llm_calls == 1
    assert cost.input_tokens == 15
    assert cost.output_tokens == 7


def test_verify_evidence_without_cost_does_not_crash() -> None:
    claim, evidence = _passing_pair()
    consensus = _StubConsensus(
        _make_consensus_result(
            relation="supports", confidence="high",
            quote="爱因斯坦", reason="ok",
        )
    )
    judgement = verify_evidence(claim, evidence, consensus)
    assert judgement.relation == "supports"


