"""Unit tests for veritas.consensus — adapter contract, retry, StrictestStrategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from veritas.consensus import (
    JSON_RETRY_PROMPT_TEMPLATE,
    TEMPERATURE,
    ConsensusResult,
    LLMResponse,
    MultiModelConsensus,
    OpenAICompatibleAdapter,
    StrictestStrategy,
    _parse_and_validate,
    _response_to_vote,
)
from veritas.types import LLMProvider, ModelVote

JUDGEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["relation", "confidence", "quote", "reason"],
    "properties": {
        "relation": {"enum": ["supports", "refutes", "neutral", "insufficient"]},
        "confidence": {"enum": ["high", "medium", "low"]},
        "quote": {"type": "string"},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Fake adapter (matches _AdapterLike duck type)
# ---------------------------------------------------------------------------

@dataclass
class FakeAdapter:
    """Test stub: feeds canned LLMResponse objects in order."""

    name: str
    responses: list[LLMResponse] = field(default_factory=list)
    raise_on_call: Exception | None = None
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> LLMResponse:
        self.calls.append((system, user, schema))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if not self.responses:
            raise AssertionError(
                f"FakeAdapter[{self.name}] ran out of canned responses"
            )
        return self.responses.pop(0)


def _vote(relation: str | None, confidence: str | None, quote: str = "q") -> dict[str, Any]:
    return {"relation": relation, "confidence": confidence, "quote": quote, "reason": "r"}


def _ok_response(name: str, relation: str, confidence: str, quote: str = "q") -> LLMResponse:
    return LLMResponse(
        model_name=name,
        parsed=_vote(relation, confidence, quote),
        raw_text="{}",
    )


def _fail_response(name: str, error: str = "json_parse_failed") -> LLMResponse:
    return LLMResponse(
        model_name=name,
        parsed=None,
        raw_text="garbage",
        error=error,
    )


# ---------------------------------------------------------------------------
# Hard constants
# ---------------------------------------------------------------------------

def test_temperature_constant_is_hardcoded_to_0() -> None:
    assert TEMPERATURE == 0.0


def test_json_retry_prompt_template_mentions_schema() -> None:
    rendered = JSON_RETRY_PROMPT_TEMPLATE.format(schema_json='{"x": 1}')
    assert "JSON Schema" in rendered
    assert '{"x": 1}' in rendered


def test_llm_provider_has_no_temperature_field() -> None:
    p = LLMProvider(
        name="x", api_key="x", base_url="https://x", model="x",
    )
    assert not hasattr(p, "temperature")


# ---------------------------------------------------------------------------
# _parse_and_validate
# ---------------------------------------------------------------------------

def test_parse_and_validate_accepts_valid_json() -> None:
    raw = '{"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"}'
    parsed, err = _parse_and_validate(raw, JUDGEMENT_SCHEMA)
    assert err is None
    assert parsed == {
        "relation": "supports", "confidence": "high", "quote": "q", "reason": "r",
    }


def test_parse_and_validate_rejects_bad_json() -> None:
    parsed, err = _parse_and_validate("not json", JUDGEMENT_SCHEMA)
    assert parsed is None
    assert err is not None and err.startswith("json_decode_error")


def test_parse_and_validate_rejects_schema_violation() -> None:
    raw = '{"relation": "INVALID", "confidence": "high", "quote": "q", "reason": "r"}'
    parsed, err = _parse_and_validate(raw, JUDGEMENT_SCHEMA)
    assert parsed is None
    assert err is not None and err.startswith("schema_violation")


def test_parse_and_validate_rejects_top_level_list() -> None:
    parsed, err = _parse_and_validate("[1,2,3]", JUDGEMENT_SCHEMA)
    assert parsed is None
    assert err == "top_level_not_object"


# ---------------------------------------------------------------------------
# OpenAICompatibleAdapter: retry behaviour via subclassing
# ---------------------------------------------------------------------------

class _StubbedOpenAIAdapter(OpenAICompatibleAdapter):
    """Override _raw_chat to feed canned (text, in_tok, out_tok) tuples."""

    def __init__(self, provider: LLMProvider, tuples: list[tuple[str, int, int]]):
        super().__init__(provider)
        self._tuples = tuples
        self.raw_calls: list[list[dict[str, str]]] = []

    def _raw_chat(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        self.raw_calls.append(list(messages))
        if not self._tuples:
            raise AssertionError("no canned raw_chat tuple available")
        return self._tuples.pop(0)


@pytest.fixture()
def provider() -> LLMProvider:
    return LLMProvider(
        name="deepseek",
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )


def test_adapter_returns_parsed_on_first_try(provider: LLMProvider) -> None:
    good = '{"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"}'
    adapter = _StubbedOpenAIAdapter(provider, [(good, 100, 30)])
    resp = adapter.chat_json("sys", "user", JUDGEMENT_SCHEMA)
    assert resp.error is None
    assert resp.parsed is not None
    assert resp.parsed["relation"] == "supports"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 30
    assert resp.retries == 0
    assert len(adapter.raw_calls) == 1


def test_adapter_retries_once_with_explicit_prompt_on_bad_json(
    provider: LLMProvider,
) -> None:
    good = '{"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"}'
    adapter = _StubbedOpenAIAdapter(provider, [
        ("not json at all", 50, 10),
        (good, 80, 20),
    ])
    resp = adapter.chat_json("sys", "user", JUDGEMENT_SCHEMA)
    assert resp.error is None
    assert resp.parsed is not None
    assert resp.retries == 1
    assert resp.input_tokens == 50 + 80
    assert resp.output_tokens == 10 + 20
    # Retry must include the explicit retry prompt in the user message.
    assert len(adapter.raw_calls) == 2
    retry_user = adapter.raw_calls[1][1]["content"]
    assert "JSON Schema" in retry_user


def test_adapter_returns_error_after_two_bad_responses(provider: LLMProvider) -> None:
    adapter = _StubbedOpenAIAdapter(provider, [
        ("garbage 1", 10, 5),
        ("garbage 2", 20, 8),
    ])
    resp = adapter.chat_json("sys", "user", JUDGEMENT_SCHEMA)
    assert resp.parsed is None
    assert resp.error is not None and "json_parse_failed" in resp.error
    assert resp.retries == 1


def test_adapter_handles_transport_error(provider: LLMProvider) -> None:
    class ExplodingAdapter(OpenAICompatibleAdapter):
        def _raw_chat(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
            raise RuntimeError("network down")

    adapter = ExplodingAdapter(provider)
    resp = adapter.chat_json("sys", "user", JUDGEMENT_SCHEMA)
    assert resp.parsed is None
    assert resp.error is not None and "transport_error" in resp.error


# ---------------------------------------------------------------------------
# MultiModelConsensus: invoke_all + judge
# ---------------------------------------------------------------------------

def test_multimodel_consensus_requires_at_least_one_adapter() -> None:
    with pytest.raises(ValueError):
        MultiModelConsensus([])


def test_invoke_all_returns_one_response_per_adapter() -> None:
    a = FakeAdapter("deepseek", [_ok_response("deepseek", "supports", "high")])
    b = FakeAdapter("mimo", [_ok_response("mimo", "supports", "high")])
    consensus = MultiModelConsensus([a, b])
    results = consensus.invoke_all("sys", "user", JUDGEMENT_SCHEMA)
    assert set(results.keys()) == {"deepseek", "mimo"}
    assert results["deepseek"].parsed is not None
    assert results["mimo"].parsed is not None


def test_invoke_all_records_executor_failure() -> None:
    a = FakeAdapter("deepseek", raise_on_call=RuntimeError("boom"))
    b = FakeAdapter("mimo", [_ok_response("mimo", "supports", "high")])
    consensus = MultiModelConsensus([a, b])
    results = consensus.invoke_all("sys", "user", JUDGEMENT_SCHEMA)
    assert results["deepseek"].error is not None
    assert "executor_error" in results["deepseek"].error
    assert results["mimo"].parsed is not None


# ---------------------------------------------------------------------------
# _response_to_vote
# ---------------------------------------------------------------------------

def test_response_to_vote_success_path() -> None:
    resp = _ok_response("deepseek", "supports", "high", quote="hello")
    vote = _response_to_vote(resp)
    assert vote.model_name == "deepseek"
    assert vote.relation == "supports"
    assert vote.confidence == "high"
    assert vote.error is None


def test_response_to_vote_carries_failure() -> None:
    resp = _fail_response("deepseek", error="json_parse_failed: x")
    vote = _response_to_vote(resp)
    assert vote.relation is None
    assert vote.confidence is None
    assert vote.error is not None and "json_parse_failed" in vote.error


def test_response_to_vote_rejects_invalid_relation() -> None:
    resp = LLMResponse(
        model_name="x",
        parsed={"relation": "weird", "confidence": "high", "quote": "q", "reason": ""},
        raw_text="{}",
    )
    vote = _response_to_vote(resp)
    assert vote.relation is None
    assert vote.error is not None and "invalid_relation" in vote.error


def test_response_to_vote_rejects_invalid_confidence() -> None:
    resp = LLMResponse(
        model_name="x",
        parsed={"relation": "supports", "confidence": "very_high", "quote": "q", "reason": ""},
        raw_text="{}",
    )
    vote = _response_to_vote(resp)
    assert vote.relation is None
    assert vote.error is not None and "invalid_confidence" in vote.error


def test_response_to_vote_success_sets_quote_field_not_raw_response() -> None:
    """On success, the model's quote lives on vote.quote; raw_response stays unused
    (reserved for the model's raw audit trail, not the quote payload).
    """
    resp = _ok_response("deepseek", "supports", "high", quote="hello world")
    vote = _response_to_vote(resp)
    assert vote.quote == "hello world"
    assert vote.raw_response is None  # not overwritten by the quote


def test_response_to_vote_failure_preserves_raw_response_and_leaves_quote_none() -> None:
    """On parse / schema failure, raw_response carries the model's raw text and
    quote stays None (we don't have a validated quote).
    """
    resp = _fail_response("deepseek", error="json_parse_failed: x")  # raw_text="garbage"
    vote = _response_to_vote(resp)
    assert vote.quote is None
    assert vote.raw_response == "garbage"


# ---------------------------------------------------------------------------
# StrictestStrategy: the 7 mandatory cases from v3 §3.3
# ---------------------------------------------------------------------------

def _v(name: str, relation: str | None, confidence: str | None, quote: str = "q") -> ModelVote:
    return ModelVote(
        model_name=name,
        relation=relation,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        quote=quote,
    )


def test_strictest_supports_supports_to_supports_high() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "supports", "high"), _v("mimo", "supports", "high")]
    )
    assert out.relation == "supports"
    assert out.confidence == "high"
    assert out.disagreement is None


def test_strictest_supports_neutral_to_supports_medium() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "supports", "high"), _v("mimo", "neutral", "low")]
    )
    assert out.relation == "supports"
    assert out.confidence == "medium"


def test_strictest_supports_insufficient_to_supports_medium() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "supports", "high"), _v("mimo", "insufficient", "low")]
    )
    assert out.relation == "supports"
    assert out.confidence == "medium"


def test_strictest_supports_refutes_to_conflict_abstain() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "supports", "high"), _v("mimo", "refutes", "high")]
    )
    assert out.relation == "conflict"
    assert out.confidence == "abstain"
    assert out.disagreement is not None
    assert out.disagreement.summary == "supports_vs_refutes"


def test_strictest_refutes_neutral_to_refutes_high() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "refutes", "high"), _v("mimo", "neutral", "low")]
    )
    assert out.relation == "refutes"
    assert out.confidence == "high"


def test_strictest_neutral_neutral_to_neutral_low() -> None:
    out = StrictestStrategy.merge(
        [_v("deepseek", "neutral", "low"), _v("mimo", "neutral", "low")]
    )
    assert out.relation == "neutral"
    assert out.confidence == "low"


def test_strictest_one_model_fail_under_high_risk_cap_below_high() -> None:
    failed = ModelVote("deepseek", None, None, error="json_parse_failed")
    ok = _v("mimo", "supports", "high")
    out = StrictestStrategy.merge([failed, ok], high_risk=True)
    # The cap rule: any model fail under high_risk → never confidence "high".
    # Single valid model after failure → cap at low.
    assert out.confidence in {"medium", "low", "abstain"}
    assert out.confidence != "high"
    assert out.disagreement is not None


# ---------------------------------------------------------------------------
# Extra coverage: insufficient + insufficient, single-model, all-fail
# ---------------------------------------------------------------------------

def test_strictest_insufficient_insufficient_to_insufficient_low() -> None:
    out = StrictestStrategy.merge(
        [_v("a", "insufficient", "low"), _v("b", "insufficient", "low")]
    )
    assert out.relation == "insufficient"
    assert out.confidence == "low"


def test_strictest_neutral_insufficient_to_neutral_low() -> None:
    out = StrictestStrategy.merge(
        [_v("a", "neutral", "low"), _v("b", "insufficient", "low")]
    )
    assert out.relation == "neutral"
    assert out.confidence == "low"


def test_strictest_refutes_supports_symmetric_with_supports_refutes() -> None:
    out = StrictestStrategy.merge(
        [_v("a", "refutes", "high"), _v("b", "supports", "high")]
    )
    assert out.relation == "conflict"
    assert out.confidence == "abstain"


def test_strictest_all_models_failed_to_insufficient_abstain() -> None:
    out = StrictestStrategy.merge([
        ModelVote("a", None, None, error="err"),
        ModelVote("b", None, None, error="err"),
    ])
    assert out.relation == "insufficient"
    assert out.confidence == "abstain"


def test_strictest_empty_votes_to_insufficient_abstain() -> None:
    out = StrictestStrategy.merge([])
    assert out.relation == "insufficient"
    assert out.confidence == "abstain"


def test_strictest_single_model_configured_caps_at_low() -> None:
    out = StrictestStrategy.merge(
        [_v("solo", "supports", "high")],
        single_model=True,
    )
    assert out.relation == "supports"
    assert out.confidence == "low"


# ---------------------------------------------------------------------------
# End-to-end judge() via FakeAdapter
# ---------------------------------------------------------------------------

def test_judge_dual_supports_to_supports_high() -> None:
    consensus = MultiModelConsensus([
        FakeAdapter("deepseek", [_ok_response("deepseek", "supports", "high")]),
        FakeAdapter("mimo", [_ok_response("mimo", "supports", "high")]),
    ])
    result = consensus.judge(
        step="verify", claim="c",
        system="sys", user="user", schema=JUDGEMENT_SCHEMA,
    )
    assert result.relation == "supports"
    assert result.confidence == "high"
    assert isinstance(result, ConsensusResult)


def test_judge_records_failure_in_model_votes() -> None:
    consensus = MultiModelConsensus([
        FakeAdapter("deepseek", [_fail_response("deepseek", "json_parse_failed: x")]),
        FakeAdapter("mimo", [_ok_response("mimo", "supports", "high")]),
    ])
    result = consensus.judge(
        step="verify", claim="c",
        system="sys", user="user", schema=JUDGEMENT_SCHEMA,
        high_risk=False,
    )
    # one fail + one valid → caps low + disagreement
    assert result.confidence in {"low", "medium"}
    assert result.disagreement is not None
    failed_votes = [v for v in result.model_votes if v.error is not None]
    assert len(failed_votes) == 1
    assert failed_votes[0].model_name == "deepseek"


def test_judge_high_risk_with_failure_never_high() -> None:
    consensus = MultiModelConsensus([
        FakeAdapter("deepseek", [_fail_response("deepseek", "json_parse_failed")]),
        FakeAdapter("mimo", [_ok_response("mimo", "supports", "high")]),
    ])
    result = consensus.judge(
        step="verify", claim="c",
        system="sys", user="user", schema=JUDGEMENT_SCHEMA,
        high_risk=True,
    )
    assert result.confidence != "high"


# ---------------------------------------------------------------------------
# Usage tracking through judge()
# ---------------------------------------------------------------------------

def test_judge_accumulates_input_output_tokens() -> None:
    """ConsensusResult must carry the summed usage from all adapters."""
    consensus = MultiModelConsensus([
        FakeAdapter("deepseek", [
            LLMResponse(
                model_name="deepseek",
                parsed={"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"},
                raw_text="{}",
                input_tokens=10,
                output_tokens=5,
            )
        ]),
        FakeAdapter("mimo", [
            LLMResponse(
                model_name="mimo",
                parsed={"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"},
                raw_text="{}",
                input_tokens=8,
                output_tokens=4,
            )
        ]),
    ])
    result = consensus.judge(
        step="verify", claim="c",
        system="sys", user="user", schema=JUDGEMENT_SCHEMA,
    )
    assert result.input_tokens == 18
    assert result.output_tokens == 9


def test_judge_includes_failure_tokens_in_usage() -> None:
    """Even adapters that fail parsing contribute their token usage."""
    consensus = MultiModelConsensus([
        FakeAdapter("deepseek", [
            LLMResponse(
                model_name="deepseek",
                parsed=None,
                raw_text="garbage",
                error="json_parse_failed",
                input_tokens=12,
                output_tokens=6,
            )
        ]),
        FakeAdapter("mimo", [
            LLMResponse(
                model_name="mimo",
                parsed={"relation": "supports", "confidence": "high", "quote": "q", "reason": "r"},
                raw_text="{}",
                input_tokens=7,
                output_tokens=3,
            )
        ]),
    ])
    result = consensus.judge(
        step="verify", claim="c",
        system="sys", user="user", schema=JUDGEMENT_SCHEMA,
    )
    assert result.input_tokens == 19
    assert result.output_tokens == 9
