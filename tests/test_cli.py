"""Unit tests for cerno.cli — no real LLMs, no real network.

All tests monkeypatch ``FactChecker`` and environment variables so the CLI
entry point runs offline.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from typing import Any

import pytest

from cerno.cli import main
from cerno.observability import AuditStep, CostBreakdown
from cerno.types import (
    EvidenceSpan,
    ModelVote,
    VerificationRequest,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Fake checker
# ---------------------------------------------------------------------------

class _FakeChecker:
    """Records calls and returns canned results."""

    def __init__(
        self,
        single_result: VerificationResult | None = None,
        batch_results: list[VerificationResult] | None = None,
    ) -> None:
        self._single_result = single_result
        self._batch_results = batch_results
        self.verify_calls: list[VerificationRequest] = []
        self.verify_batch_calls: list[list[VerificationRequest]] = []

    def verify(self, request: VerificationRequest) -> VerificationResult:
        self.verify_calls.append(request)
        if self._single_result is None:
            raise AssertionError("_FakeChecker.verify called without single_result")
        return self._single_result

    def verify_batch(
        self, requests: list[VerificationRequest]
    ) -> list[VerificationResult]:
        self.verify_batch_calls.append(list(requests))
        if self._batch_results is None:
            raise AssertionError("_FakeChecker.verify_batch called without batch_results")
        return list(self._batch_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(span_id: str = "ev1") -> EvidenceSpan:
    return EvidenceSpan(
        id=span_id,
        title=f"Article {span_id}",
        url=f"https://example.com/{span_id}",
        quote=f"Quote body for {span_id}.",
        source_name="tavily",
        source_tier="T1",
        retrieved_at="2026-05-12T00:00:00+00:00",
        content_hash="hash-" + span_id,
        raw_score=0.85,
        metadata={},
        risk_flags=[],
    )


def _make_result(
    claim: str,
    verdict: str = "likely_correct",
    confidence: float = 0.85,
    reasoning: str = "two T1 sources support",
    sources: list[EvidenceSpan] | None = None,
    model_votes: list[ModelVote] | None = None,
    audit_trace: list[AuditStep] | None = None,
    cost: CostBreakdown | None = None,
) -> VerificationResult:
    return VerificationResult(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        claim=claim,
        sources=sources if sources is not None else [_make_evidence("ev1")],
        reasoning=reasoning,
        model_votes=model_votes or [],
        audit_trace=audit_trace or [],
        cost=cost or CostBreakdown(),
    )


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> tuple[int, dict[str, Any]]:
    """Run ``main(argv)``, capture stdout, return (exit_code, parsed_json)."""
    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = main(argv)
    out = buf.getvalue()
    return code, json.loads(out)


# ---------------------------------------------------------------------------
# 1. Single claim — happy path JSON shape
# ---------------------------------------------------------------------------

def test_verify_single_json_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen-key")

    vote = ModelVote(
        model_name="qwen-turbo",
        relation="supports",
        confidence="high",
        reason="matches",
        quote="Quote body for ev1.",
    )
    result = _make_result(
        claim="c1",
        verdict="likely_correct",
        confidence=0.92,
        reasoning="strong",
        model_votes=[vote],
        audit_trace=[AuditStep("step", "in", "out")],
        cost=CostBreakdown(
            input_tokens=100, output_tokens=20,
            llm_calls=1, retrieval_calls=2, cache_hits=0,
        ),
    )
    fake = _FakeChecker(single_result=result)
    monkeypatch.setattr("cerno.cli.FactChecker", lambda **_: fake)

    code, out = _run_cli(monkeypatch, ["verify", "--claim", "c1", "--provider", "qwen"])

    assert code == 0
    assert out["claim"] == "c1"
    assert out["verdict"] == "likely_correct"
    assert out["confidence"] == 0.92
    assert out["reasoning"] == "strong"
    assert len(out["sources"]) == 1
    assert out["sources"][0]["id"] == "ev1"
    assert "content_hash" not in out["sources"][0]
    assert len(out["model_votes"]) == 1
    assert out["model_votes"][0]["model_name"] == "qwen-turbo"
    assert out["model_votes"][0]["quote"] == "Quote body for ev1."
    assert out["audit_steps_count"] == 1
    assert out["cost"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "llm_calls": 1,
        "retrieval_calls": 2,
        "cache_hits": 0,
    }
    assert out["warnings"] == []
    assert "error" not in out

    assert len(fake.verify_calls) == 1
    assert fake.verify_calls[0].claim == "c1"


# ---------------------------------------------------------------------------
# 2. Batch claim — happy path order preserved
# ---------------------------------------------------------------------------

def test_verify_batch_order_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen-key")

    r1 = _make_result(claim="A", verdict="likely_correct", confidence=0.9)
    r2 = _make_result(claim="B", verdict="needs_review", confidence=0.5)
    r3 = _make_result(claim="C", verdict="likely_error", confidence=0.8)
    fake = _FakeChecker(batch_results=[r1, r2, r3])
    monkeypatch.setattr("cerno.cli.FactChecker", lambda **_: fake)

    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps(["A", "B", "C"]), encoding="utf-8")

    code, out = _run_cli(
        monkeypatch,
        ["verify-batch", str(claims_file), "--provider", "qwen"],
    )

    assert code == 0
    results = out["results"]
    assert len(results) == 3
    assert [r["claim"] for r in results] == ["A", "B", "C"]
    assert [r["verdict"] for r in results] == [
        "likely_correct", "needs_review", "likely_error",
    ]

    assert len(fake.verify_batch_calls) == 1
    assert [r.claim for r in fake.verify_batch_calls[0]] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# 3. Missing API key — JSON error, no traceback
# ---------------------------------------------------------------------------

def test_verify_missing_key_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    code, out = _run_cli(monkeypatch, ["verify", "--claim", "x", "--provider", "qwen"])

    assert code == 1
    assert "error" in out
    assert "QWEN_API_KEY" in out["error"]
    assert out["verdict"] == "unverifiable"
    assert out["confidence"] == 0.0


# ---------------------------------------------------------------------------
# 4. Multiple providers
# ---------------------------------------------------------------------------

def test_verify_multiple_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")

    result = _make_result(claim="multi")
    fake = _FakeChecker(single_result=result)
    monkeypatch.setattr("cerno.cli.FactChecker", lambda **_: fake)

    code, out = _run_cli(
        monkeypatch,
        ["verify", "--claim", "multi", "--provider", "qwen", "deepseek"],
    )

    assert code == 0
    assert out["claim"] == "multi"


# ---------------------------------------------------------------------------
# 5. Tavily key forwarded from env
# ---------------------------------------------------------------------------

def test_verify_forwards_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")

    captured: dict[str, Any] = {}

    def _capture_fact_checker(**kwargs: Any) -> _FakeChecker:
        captured.update(kwargs)
        return _FakeChecker(single_result=_make_result(claim="t"))

    monkeypatch.setattr("cerno.cli.FactChecker", _capture_fact_checker)

    code, _ = _run_cli(monkeypatch, ["verify", "--claim", "t", "--provider", "qwen"])

    assert code == 0
    assert captured.get("tavily_api_key") == "fake-tavily"


# ---------------------------------------------------------------------------
# 6. Provider timeout set to 120s
# ---------------------------------------------------------------------------

def test_verify_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen")

    captured_providers: list[Any] = []

    def _capture_fact_checker(**kwargs: Any) -> _FakeChecker:
        captured_providers.extend(kwargs.get("llm_providers", []))
        return _FakeChecker(single_result=_make_result(claim="t"))

    monkeypatch.setattr("cerno.cli.FactChecker", _capture_fact_checker)

    code, _ = _run_cli(monkeypatch, ["verify", "--claim", "t", "--provider", "qwen"])

    assert code == 0
    assert len(captured_providers) == 1
    assert captured_providers[0].timeout == 120.0


# ---------------------------------------------------------------------------
# 7. Invalid batch input
# ---------------------------------------------------------------------------

def test_verify_batch_invalid_json_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen")

    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    code, out = _run_cli(
        monkeypatch,
        ["verify-batch", str(claims_file), "--provider", "qwen"],
    )

    assert code == 1
    assert "error" in out
    assert "list of strings" in out["error"]


# ---------------------------------------------------------------------------
# 8. Batch with non-string items
# ---------------------------------------------------------------------------

def test_verify_batch_non_string_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-qwen")

    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps(["ok", 123]), encoding="utf-8")

    code, out = _run_cli(
        monkeypatch,
        ["verify-batch", str(claims_file), "--provider", "qwen"],
    )

    assert code == 1
    assert "error" in out
    assert "list of strings" in out["error"]


# ---------------------------------------------------------------------------
# build_provider_from_env
# ---------------------------------------------------------------------------

def test_build_provider_from_env_qwen_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-key")
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)

    from cerno.cli import build_provider_from_env

    p = build_provider_from_env("qwen")
    assert p.name == "qwen"
    assert p.api_key == "fake-key"
    assert p.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert p.model == "qwen-turbo"
    assert p.timeout == 120.0


def test_build_provider_from_env_qwen_custom_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "custom-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://custom.example.com/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen-max")

    from cerno.cli import build_provider_from_env

    p = build_provider_from_env("qwen")
    assert p.api_key == "custom-key"
    assert p.base_url == "https://custom.example.com/v1"
    assert p.model == "qwen-max"


def test_build_provider_from_env_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    from cerno.cli import build_provider_from_env

    with pytest.raises(ValueError) as excinfo:
        build_provider_from_env("qwen")
    assert "QWEN_API_KEY" in str(excinfo.value)


def test_build_provider_from_env_generic_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYMODEL_API_KEY", "sk-xxx")
    monkeypatch.setenv("MYMODEL_BASE_URL", "https://api.mymodel.com/v1")
    monkeypatch.setenv("MYMODEL_MODEL", "mymodel-v1")

    from cerno.cli import build_provider_from_env

    p = build_provider_from_env("mymodel")
    assert p.name == "mymodel"
    assert p.api_key == "sk-xxx"
    assert p.base_url == "https://api.mymodel.com/v1"
    assert p.model == "mymodel-v1"
    assert p.timeout == 120.0


def test_build_provider_from_env_generic_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MYMODEL_API_KEY", raising=False)
    monkeypatch.delenv("MYMODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MYMODEL_MODEL", raising=False)

    from cerno.cli import build_provider_from_env

    with pytest.raises(ValueError) as excinfo:
        build_provider_from_env("mymodel")
    msg = str(excinfo.value)
    assert "mymodel" in msg
    assert "MYMODEL_API_KEY" in msg
    assert "MYMODEL_BASE_URL" in msg
    assert "MYMODEL_MODEL" in msg


def test_build_provider_from_env_all_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "q")
    monkeypatch.setenv("XIAOMI_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")

    from cerno.cli import build_provider_from_env

    pq = build_provider_from_env("qwen")
    px = build_provider_from_env("xiaomi")
    pd = build_provider_from_env("deepseek")

    assert pq.name == "qwen"
    assert px.name == "xiaomi"
    assert pd.name == "deepseek"
    assert pq.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert px.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
    assert pd.base_url == "https://api.deepseek.com/v1"


def test_build_provider_from_env_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "fake-key")
    monkeypatch.delenv("QWEN_MODEL", raising=False)

    from cerno.cli import build_provider_from_env

    p = build_provider_from_env("qwen", model_override="qwen3.5-122b-a10b")
    assert p.model == "qwen3.5-122b-a10b"
