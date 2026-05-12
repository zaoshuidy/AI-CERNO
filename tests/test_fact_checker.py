"""Mocked unit tests for cerno.fact_checker.

No real LLMs, no real network. Each test monkey-patches the upstream stage
functions imported INTO ``cerno.fact_checker`` (build_claim_profile,
decompose_claim, build_retrieval_plan, verify_evidence, resolve_claim,
apply_failure_matrix) and injects deterministic ``tavily_search`` /
``wiki_search`` callables. The consensus argument is a stub that asserts it
is never actually invoked — every code path the orchestrator exercises is
routed through the stubbed stage functions, so consensus.judge must stay
untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from cerno.consensus import MultiModelConsensus
from cerno.fact_checker import FactChecker, check_claim
from cerno.types import (
    AtomicClaim,
    ClaimProfile,
    EvidenceJudgement,
    EvidenceSpan,
    LLMProvider,
    RetrievalPlan,
    VerificationRequest,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class _NoopAdapter:
    """Satisfies the _AdapterLike protocol so MultiModelConsensus.__init__
    accepts it. ``chat_json`` must never run; if it does, the orchestrator
    is calling consensus without being properly stubbed."""

    name = "noop-adapter"

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> Any:
        raise AssertionError(
            "_NoopAdapter.chat_json should never run when stage functions "
            "are stubbed"
        )


class _AssertNoCallConsensus(MultiModelConsensus):
    """Drop-in MultiModelConsensus that asserts ``judge`` is never invoked."""

    def __init__(self) -> None:
        super().__init__([_NoopAdapter()])

    def judge(self, **kwargs: Any) -> Any:  # type: ignore[override]
        raise AssertionError(
            f"consensus.judge should not be called from check_claim when "
            f"stages are stubbed; got kwargs={kwargs}"
        )


def _make_request(
    claim: str = "爱因斯坦获得诺贝尔物理学奖", **overrides: Any
) -> VerificationRequest:
    return VerificationRequest(claim=claim, **overrides)


def _make_profile(
    *,
    is_checkable: bool = True,
    claim_type: str = "entity_fact",
    risk_level: str = "medium",
    reason: str = "ok",
) -> ClaimProfile:
    return ClaimProfile(
        is_checkable=is_checkable,
        claim_type=claim_type,  # type: ignore[arg-type]
        domain="general",
        risk_level=risk_level,  # type: ignore[arg-type]
        risk_level_source="llm_inferred",
        required_evidence=[],
        strict_mode=False,
        reason=reason,
    )


def _make_atomic(
    id: str = "c1", text: str = "爱因斯坦获得诺贝尔物理学奖"
) -> AtomicClaim:
    return AtomicClaim(
        id=id,
        text=text,
        original_span=text,
        parent_claim=text,
        check_priority=1,
        required_evidence_type=[],
    )


def _make_plan(queries: tuple[str, ...] = ("q1",)) -> RetrievalPlan:
    return RetrievalPlan(
        queries=list(queries),
        source_targets=["tavily", "wikipedia_zh"],
        min_independent_sources=2,
        allow_discovery_sources=False,
        allow_single_source_medium=True,
        require_official_source=False,
        require_freshness_check=False,
    )


def _make_evidence(
    *,
    id: str = "ev1",
    source_name: str = "tavily",
    source_tier: str = "T1",
    quote: str = "爱因斯坦因光电效应获得1921年诺贝尔物理学奖。",
) -> EvidenceSpan:
    return EvidenceSpan(
        id=id,
        title=f"title-{id}",
        url=f"https://example.com/{id}",
        quote=quote,
        source_name=source_name,
        source_tier=source_tier,  # type: ignore[arg-type]
        retrieved_at="2026-01-01T00:00:00+00:00",
        content_hash="deadbeefcafe1234",
        raw_score=0.9,
        metadata={},
        risk_flags=[],
    )


def _make_judgement(
    *,
    atomic_claim_id: str = "c1",
    evidence_id: str = "ev1",
    relation: str = "supports",
    confidence: str = "high",
) -> EvidenceJudgement:
    return EvidenceJudgement(
        atomic_claim_id=atomic_claim_id,
        evidence_id=evidence_id,
        relation=relation,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        quote="爱因斯坦",
        reason="ok",
        model_votes=[],
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: ClaimProfile,
    atomic_claims: list[AtomicClaim],
    plan: RetrievalPlan,
    judgement_fn: Callable[..., EvidenceJudgement] | None = None,
    resolver_result: VerificationResult | None = None,
    failure_matrix_result: VerificationResult | None = None,
) -> None:
    """Patch the four upstream stage functions imported into fact_checker.

    Only the patches the caller cares about are actually wired; missing ones
    are guarded with ``pytest.fail`` so an unexpected call is loud."""
    monkeypatch.setattr(
        "cerno.fact_checker.build_claim_profile",
        lambda req, cons, **kw: profile,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.decompose_claim",
        lambda req, prof, cons, **kw: atomic_claims,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.build_retrieval_plan",
        lambda prof, acs: plan,
    )
    if judgement_fn is None:
        judgement_fn = lambda ac, ev, cons, **kw: _make_judgement(  # noqa: E731
            atomic_claim_id=ac.id, evidence_id=ev.id,
        )
    monkeypatch.setattr(
        "cerno.fact_checker.verify_evidence",
        judgement_fn,
    )
    if resolver_result is not None:
        monkeypatch.setattr(
            "cerno.fact_checker.resolve_claim",
            lambda acs, judgements, evidence, prof: resolver_result,
        )
    if failure_matrix_result is not None:
        monkeypatch.setattr(
            "cerno.fact_checker.apply_failure_matrix",
            lambda ctx: failure_matrix_result,
        )


# ---------------------------------------------------------------------------
# 1. is_checkable=False short-circuits the whole pipeline
# ---------------------------------------------------------------------------

def test_unchecked_profile_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile says claim is not factually checkable → inline unverifiable
    result, no retrieval, no verification, no resolution."""
    profile = _make_profile(is_checkable=False, reason="claim_is_opinion")

    monkeypatch.setattr(
        "cerno.fact_checker.build_claim_profile",
        lambda req, cons, **kw: profile,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.decompose_claim",
        lambda *a, **kw: pytest.fail("decompose_claim must not be called"),
    )
    monkeypatch.setattr(
        "cerno.fact_checker.build_retrieval_plan",
        lambda *a, **kw: pytest.fail("build_retrieval_plan must not be called"),
    )
    monkeypatch.setattr(
        "cerno.fact_checker.verify_evidence",
        lambda *a, **kw: pytest.fail("verify_evidence must not be called"),
    )
    monkeypatch.setattr(
        "cerno.fact_checker.resolve_claim",
        lambda *a, **kw: pytest.fail("resolve_claim must not be called"),
    )

    result = check_claim(_make_request(), _AssertNoCallConsensus())

    assert result.verdict == "unverifiable"
    assert result.confidence == 0.0
    assert result.reasoning == "claim_is_opinion"
    assert result.sources == []
    # Audit + cost are still attached.
    assert any(
        s.name == "build_claim_profile" for s in result.audit_trace
    )
    assert result.cost.retrieval_calls == 0
    assert result.cost.llm_calls == 0


def test_unchecked_profile_falls_back_to_default_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty profile.reason → orchestrator fills in a default reasoning."""
    profile = _make_profile(is_checkable=False, reason="")
    monkeypatch.setattr(
        "cerno.fact_checker.build_claim_profile",
        lambda req, cons, **kw: profile,
    )

    result = check_claim(_make_request(), _AssertNoCallConsensus())

    assert result.verdict == "unverifiable"
    assert result.reasoning == "claim is not factually checkable"


# ---------------------------------------------------------------------------
# 2. Happy path — every stage runs, audit + cost populate
# ---------------------------------------------------------------------------

def test_happy_path_runs_all_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _make_profile()
    atomic = _make_atomic()
    plan = _make_plan()
    tavily_span = _make_evidence(id="ev_t", source_name="tavily")
    wiki_span = _make_evidence(
        id="ev_w", source_name="wikipedia_zh", source_tier="T1"
    )

    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.9,
        claim=_make_request().claim,
        sources=[tavily_span, wiki_span],
        reasoning="dual_supports",
    )

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[atomic],
        plan=plan,
        resolver_result=expected,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.apply_failure_matrix",
        lambda *a, **kw: pytest.fail("failure matrix not expected on happy path"),
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=lambda q: [tavily_span],
        wiki_search=lambda q: [wiki_span],
    )

    assert result.verdict == "likely_correct"
    assert result.confidence == 0.9
    assert result.cost.retrieval_calls == 2  # one tavily + one wiki
    names = [s.name for s in result.audit_trace]
    for required in (
        "build_claim_profile",
        "decompose_claim",
        "build_retrieval_plan",
        "search_tavily",
        "search_wikipedia_zh",
        "verify_evidence",
        "resolve_claim",
    ):
        assert required in names, f"missing audit step {required!r}: {names}"


def test_audit_trace_respects_pipeline_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile → decompose → plan → retrieval → verify → resolve order."""
    profile = _make_profile()
    plan = _make_plan()
    tavily_span = _make_evidence(id="ev_t")
    wiki_span = _make_evidence(id="ev_w", source_name="wikipedia_zh")
    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.7,
        claim="x",
        sources=[tavily_span, wiki_span],
        reasoning="dual",
    )

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
        resolver_result=expected,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=lambda q: [tavily_span],
        wiki_search=lambda q: [wiki_span],
    )

    seq = [s.name for s in result.audit_trace]
    pi = seq.index("build_claim_profile")
    di = seq.index("decompose_claim")
    pli = seq.index("build_retrieval_plan")
    ri = seq.index("search_tavily")
    vi = seq.index("verify_evidence")
    re = seq.index("resolve_claim")
    assert pi < di < pli < ri < vi < re


# ---------------------------------------------------------------------------
# 3. Zero evidence → apply_failure_matrix routing
# ---------------------------------------------------------------------------

def test_zero_evidence_routes_to_failure_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both channels return empty → orchestrator calls apply_failure_matrix,
    not resolve_claim."""
    profile = _make_profile()
    plan = _make_plan()
    matrix_result = VerificationResult(
        verdict="unverifiable",
        confidence=0.0,
        claim=_make_request().claim,
        sources=[],
        reasoning="no_evidence_found",
    )

    captured: list[dict[str, Any]] = []

    def fake_failure_matrix(ctx: dict[str, Any]) -> VerificationResult:
        captured.append(ctx)
        return matrix_result

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.apply_failure_matrix", fake_failure_matrix,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.resolve_claim",
        lambda *a, **kw: pytest.fail("resolve_claim must not run on empty evidence"),
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=lambda q: [],
        wiki_search=lambda q: [],
    )

    assert result.verdict == "unverifiable"
    assert result.confidence == 0.0
    assert captured, "apply_failure_matrix should have been called once"
    # Channels reported OK but produced nothing → both flags stay True.
    assert captured[0]["tavily_available"] is True
    assert captured[0]["wiki_available"] is True
    # Audit + cost still attached.
    assert any(s.name == "search_tavily" for s in result.audit_trace)
    assert any(s.name == "search_wikipedia_zh" for s in result.audit_trace)


def test_both_channels_fail_marks_both_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors on both stubs flip the availability flags AND route
    through the failure matrix."""
    profile = _make_profile()
    plan = _make_plan()
    matrix_result = VerificationResult(
        verdict="unverifiable",
        confidence=0.10,
        claim="x",
        sources=[],
        reasoning="retrieval_failed_all",
    )

    captured: list[dict[str, Any]] = []

    def fake_failure_matrix(ctx: dict[str, Any]) -> VerificationResult:
        captured.append(ctx)
        return matrix_result

    def boom(q: str) -> list[EvidenceSpan]:
        raise ConnectionError("simulated down")

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.apply_failure_matrix", fake_failure_matrix,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=boom,
        wiki_search=boom,
    )

    assert result.verdict == "unverifiable"
    assert result.confidence == 0.10
    assert captured[0]["tavily_available"] is False
    assert captured[0]["wiki_available"] is False
    # No successful retrieval → cost.retrieval_calls stays at 0.
    assert result.cost.retrieval_calls == 0
    names = [s.name for s in result.audit_trace]
    assert "search_tavily.error" in names
    assert "search_wikipedia_zh.error" in names


def test_tavily_error_falls_back_to_wiki_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily transport error doesn't crash the pipeline; wiki evidence
    flows through to resolve_claim."""
    profile = _make_profile()
    plan = _make_plan()
    wiki_span = _make_evidence(
        id="ev_w", source_name="wikipedia_zh", source_tier="T1"
    )
    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.7,
        claim="x",
        sources=[wiki_span],
        reasoning="wiki_only_supports",
    )

    def boom(q: str) -> list[EvidenceSpan]:
        raise ConnectionError("simulated tavily down")

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
        resolver_result=expected,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=boom,
        wiki_search=lambda q: [wiki_span],
    )

    assert result.verdict == "likely_correct"
    # Only the wiki call succeeded.
    assert result.cost.retrieval_calls == 1
    names = [s.name for s in result.audit_trace]
    assert "search_tavily.error" in names
    assert "search_wikipedia_zh" in names


# ---------------------------------------------------------------------------
# 4. Tavily opt-in: no api_key and no stub → skip silently
# ---------------------------------------------------------------------------

def test_tavily_skipped_when_no_key_and_no_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tavily_api_key + no tavily_search → tavily.skip recorded, wiki
    continues normally."""
    profile = _make_profile()
    plan = _make_plan()
    wiki_span = _make_evidence(id="ev_w", source_name="wikipedia_zh")
    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.7,
        claim="x",
        sources=[wiki_span],
        reasoning="wiki_only",
    )

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
        resolver_result=expected,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        # NO tavily_api_key, NO tavily_search stub.
        wiki_search=lambda q: [wiki_span],
    )

    assert result.verdict == "likely_correct"
    names = [s.name for s in result.audit_trace]
    assert "search_tavily.skip" in names
    # Tavily was never invoked.
    assert "search_tavily" not in names
    assert "search_tavily.error" not in names


# ---------------------------------------------------------------------------
# 5. Dedupe by evidence id
# ---------------------------------------------------------------------------

def test_evidence_deduped_by_id_across_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same id from both Tavily and Wiki → verify_evidence called once per
    (atomic_claim, evidence) pair, not twice."""
    profile = _make_profile()
    plan = _make_plan()
    dup_span = _make_evidence(id="ev_dup", source_name="tavily")

    judge_calls: list[tuple[str, str]] = []

    def fake_verify(
        ac: AtomicClaim, ev: EvidenceSpan, cons: MultiModelConsensus, **kw: Any
    ) -> EvidenceJudgement:
        judge_calls.append((ac.id, ev.id))
        return _make_judgement(atomic_claim_id=ac.id, evidence_id=ev.id)

    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.7,
        claim="x",
        sources=[dup_span],
        reasoning="ok",
    )

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic(id="c1")],
        plan=plan,
        judgement_fn=fake_verify,
        resolver_result=expected,
    )

    check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=lambda q: [dup_span],
        wiki_search=lambda q: [dup_span],
    )

    # One atomic claim × one deduplicated evidence span → one judgement call.
    assert judge_calls == [("c1", "ev_dup")]


# ---------------------------------------------------------------------------
# 6. Plan with multiple atomic claims and queries fans out correctly
# ---------------------------------------------------------------------------

def test_multiple_atomic_claims_and_queries_fan_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two atomic claims × two queries × two channels (4 retrievals, possibly
    yielding more spans) should produce one judgement per (atomic, evidence)
    pair, not per (atomic, query, channel) tuple."""
    profile = _make_profile()
    atomic_a = _make_atomic(id="c1", text="claim_a")
    atomic_b = _make_atomic(id="c2", text="claim_b")
    plan = _make_plan(queries=("q1", "q2"))

    # Each query returns a different span. Two queries × two channels = 4
    # potentially-different spans; we use 4 unique ids.
    spans: dict[str, list[EvidenceSpan]] = {
        ("tavily", "q1"): [_make_evidence(id="t_q1")],
        ("tavily", "q2"): [_make_evidence(id="t_q2")],
        ("wiki", "q1"): [_make_evidence(id="w_q1", source_name="wikipedia_zh")],
        ("wiki", "q2"): [_make_evidence(id="w_q2", source_name="wikipedia_zh")],
    }

    def tavily_stub(q: str) -> list[EvidenceSpan]:
        return spans[("tavily", q)]

    def wiki_stub(q: str) -> list[EvidenceSpan]:
        return spans[("wiki", q)]

    judge_calls: list[tuple[str, str]] = []

    def fake_verify(
        ac: AtomicClaim, ev: EvidenceSpan, cons: MultiModelConsensus, **kw: Any
    ) -> EvidenceJudgement:
        judge_calls.append((ac.id, ev.id))
        return _make_judgement(atomic_claim_id=ac.id, evidence_id=ev.id)

    expected = VerificationResult(
        verdict="likely_correct",
        confidence=0.8,
        claim="x",
        sources=list(spans[("tavily", "q1")]),
        reasoning="ok",
    )

    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[atomic_a, atomic_b],
        plan=plan,
        judgement_fn=fake_verify,
        resolver_result=expected,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=tavily_stub,
        wiki_search=wiki_stub,
    )

    # 4 unique evidence ids × 2 atomic claims = 8 judgement calls.
    assert len(judge_calls) == 8
    # Each atomic claim sees every unique evidence id exactly once.
    by_ac: dict[str, set[str]] = {"c1": set(), "c2": set()}
    for ac_id, ev_id in judge_calls:
        by_ac[ac_id].add(ev_id)
    assert by_ac["c1"] == {"t_q1", "t_q2", "w_q1", "w_q2"}
    assert by_ac["c2"] == {"t_q1", "t_q2", "w_q1", "w_q2"}
    # cost counts every successful retrieval call (2 queries × 2 channels).
    assert result.cost.retrieval_calls == 4


# ---------------------------------------------------------------------------
# 7. consensus is forwarded to stage functions, not consumed by orchestrator
# ---------------------------------------------------------------------------

def test_consensus_object_is_forwarded_to_stage_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must hand the same consensus instance to every stage
    function — it never invokes consensus.judge directly."""
    consensus = _AssertNoCallConsensus()
    seen_in_profile: list[Any] = []
    seen_in_decompose: list[Any] = []
    seen_in_verify: list[Any] = []

    profile = _make_profile()
    plan = _make_plan()
    ev = _make_evidence(id="ev1")
    expected = VerificationResult(
        verdict="likely_correct", confidence=0.7,
        claim="x", sources=[ev], reasoning="ok",
    )

    def fake_profile(req: Any, cons: Any, **kw: Any) -> ClaimProfile:
        seen_in_profile.append(cons)
        return profile

    def fake_decompose(
        req: Any, prof: Any, cons: Any, **kw: Any
    ) -> list[AtomicClaim]:
        seen_in_decompose.append(cons)
        return [_make_atomic()]

    def fake_verify(
        ac: Any, evidence: Any, cons: Any, **kw: Any
    ) -> EvidenceJudgement:
        seen_in_verify.append(cons)
        return _make_judgement(atomic_claim_id=ac.id, evidence_id=evidence.id)

    monkeypatch.setattr(
        "cerno.fact_checker.build_claim_profile", fake_profile,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.decompose_claim", fake_decompose,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.build_retrieval_plan",
        lambda prof, acs: plan,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.verify_evidence", fake_verify,
    )
    monkeypatch.setattr(
        "cerno.fact_checker.resolve_claim",
        lambda acs, judgements, evidence, prof: expected,
    )

    check_claim(
        _make_request(),
        consensus,
        tavily_search=lambda q: [ev],
        wiki_search=lambda q: [],
    )

    assert seen_in_profile == [consensus]
    assert seen_in_decompose == [consensus]
    assert seen_in_verify == [consensus]


# ---------------------------------------------------------------------------
# 8. Tavily live path uses real search_tavily when api_key given (no stub)
# ---------------------------------------------------------------------------

def test_tavily_api_key_without_stub_invokes_real_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tavily_api_key is supplied but no tavily_search stub, the
    orchestrator must call into ``cerno.retrieval.search_tavily``. We
    patch ``search_tavily`` (imported into fact_checker) to confirm dispatch
    without actually hitting the network."""
    profile = _make_profile()
    plan = _make_plan()
    span = _make_evidence(id="ev_real", source_name="tavily")

    captured_args: list[tuple[str, str, str | None]] = []

    def fake_search_tavily(
        query: str, api_key: str, cache_dir: str | None = None
    ) -> list[EvidenceSpan]:
        captured_args.append((query, api_key, cache_dir))
        return [span]

    monkeypatch.setattr(
        "cerno.fact_checker.search_tavily", fake_search_tavily,
    )
    expected = VerificationResult(
        verdict="likely_correct", confidence=0.7,
        claim="x", sources=[span], reasoning="ok",
    )
    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
        resolver_result=expected,
    )

    check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_api_key="sk-test",
        cache_dir="/tmp/veritas-test-cache",
        # NO tavily_search stub — must fall through to fake_search_tavily.
        wiki_search=lambda q: [],
    )

    assert captured_args == [("q1", "sk-test", "/tmp/veritas-test-cache")]


# ---------------------------------------------------------------------------
# 9. Audit + cost survive across all return paths
# ---------------------------------------------------------------------------

def test_audit_and_cost_attached_on_failure_matrix_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _make_profile()
    plan = _make_plan()
    matrix_result = VerificationResult(
        verdict="unverifiable", confidence=0.0,
        claim="x", sources=[], reasoning="no_evidence",
    )
    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[_make_atomic()],
        plan=plan,
        failure_matrix_result=matrix_result,
    )

    result = check_claim(
        _make_request(),
        _AssertNoCallConsensus(),
        tavily_search=lambda q: [],
        wiki_search=lambda q: [],
    )

    # apply_failure_matrix returns a result with default empty audit/cost;
    # _attach_audit must splice them in.
    assert result.audit_trace, "audit_trace must be attached on matrix path"
    # The fake matrix returned a fresh CostBreakdown; orchestrator replaces
    # it with the per-request CostBreakdown which counts the two retrieval
    # calls that returned empty.
    assert result.cost.retrieval_calls == 2


# ---------------------------------------------------------------------------
# 10. FactChecker construction (Stage 7 收口)
# ---------------------------------------------------------------------------

def _make_provider(name: str = "p1") -> LLMProvider:
    return LLMProvider(
        name=name,
        api_key=f"sk-fake-{name}",
        base_url="https://example.com/v1",
        model="gpt-fake",
    )


def test_factchecker_requires_providers_or_consensus() -> None:
    # Neither path supplied → constructor must refuse, not silently build
    # a half-initialised checker.
    with pytest.raises(ValueError, match="llm_providers"):
        FactChecker()


def test_factchecker_consensus_kwarg_takes_precedence() -> None:
    cons = _AssertNoCallConsensus()
    # llm_providers is ignored when an explicit consensus is injected — this
    # is the documented test-injection path.
    checker = FactChecker(
        consensus=cons,
        llm_providers=[_make_provider("p1"), _make_provider("p2")],
    )
    assert checker.consensus is cons


def test_factchecker_llm_providers_builds_consensus() -> None:
    providers = [_make_provider("p1"), _make_provider("p2")]
    checker = FactChecker(llm_providers=providers)
    assert isinstance(checker.consensus, MultiModelConsensus)
    # One OpenAICompatibleAdapter is built per provider.
    assert len(checker.consensus.adapters) == 2


def test_factchecker_max_workers_must_be_positive() -> None:
    # max_workers=0 is nonsense; reject it loudly at construction time.
    with pytest.raises(ValueError):
        FactChecker(consensus=_AssertNoCallConsensus(), max_workers=0)


# ---------------------------------------------------------------------------
# 11. FactChecker.verify — string wrapping
# ---------------------------------------------------------------------------

def test_factchecker_verify_wraps_string_into_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``checker.verify("claim", context="ctx")`` must build a
    ``VerificationRequest`` internally and pass it on to ``check_claim``."""
    captured: dict[str, Any] = {}

    def fake_check_claim(
        req: VerificationRequest, cons: Any, **kwargs: Any
    ) -> VerificationResult:
        captured["request"] = req
        captured["kwargs"] = kwargs
        return VerificationResult(
            verdict="supported", confidence=0.9,
            claim=req.claim, sources=[], reasoning="ok",
        )

    monkeypatch.setattr("cerno.fact_checker.check_claim", fake_check_claim)

    checker = FactChecker(
        consensus=_AssertNoCallConsensus(),
        cache_dir="/tmp/x",
    )
    result = checker.verify("某个声明", context="ctx-string")

    assert isinstance(captured["request"], VerificationRequest)
    assert captured["request"].claim == "某个声明"
    assert captured["request"].context == "ctx-string"
    # cache_dir/tavily_search/wiki_search must be forwarded as kwargs.
    assert captured["kwargs"]["cache_dir"] == "/tmp/x"
    assert result.claim == "某个声明"


# ---------------------------------------------------------------------------
# 12. FactChecker.verify_batch — 5 required acceptance tests
# ---------------------------------------------------------------------------

def test_factchecker_verify_batch_empty_returns_empty() -> None:
    checker = FactChecker(consensus=_AssertNoCallConsensus())
    assert checker.verify_batch([]) == []


def test_factchecker_verify_batch_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output order must match input order, even when completion order is
    reversed by deliberate sleep delays."""
    import time as _time

    def fake_check_claim(
        req: VerificationRequest, cons: Any, **kwargs: Any
    ) -> VerificationResult:
        # c0 sleeps longest so it finishes LAST under the thread pool — yet
        # results[0] must still hold c0's result.
        idx = int(req.claim[-1])
        _time.sleep((2 - idx) * 0.02)
        return VerificationResult(
            verdict="supported", confidence=0.9,
            claim=req.claim, sources=[], reasoning="ok",
        )

    monkeypatch.setattr("cerno.fact_checker.check_claim", fake_check_claim)

    requests = [_make_request(claim=f"c{i}") for i in range(3)]
    checker = FactChecker(consensus=_AssertNoCallConsensus(), max_workers=4)
    results = checker.verify_batch(requests)

    assert [r.claim for r in results] == ["c0", "c1", "c2"]


def test_factchecker_verify_batch_isolates_single_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single failing claim becomes an ``unverifiable`` result with a
    ``verify_batch.error`` audit step, but does NOT poison the batch."""

    def fake_check_claim(
        req: VerificationRequest, cons: Any, **kwargs: Any
    ) -> VerificationResult:
        if req.claim == "boom":
            raise RuntimeError("simulated_failure")
        return VerificationResult(
            verdict="supported", confidence=0.9,
            claim=req.claim, sources=[], reasoning="ok",
        )

    monkeypatch.setattr("cerno.fact_checker.check_claim", fake_check_claim)

    requests = [
        _make_request(claim="ok-1"),
        _make_request(claim="boom"),
        _make_request(claim="ok-2"),
    ]
    checker = FactChecker(consensus=_AssertNoCallConsensus())
    results = checker.verify_batch(requests)

    assert results[0].verdict == "supported"
    assert results[0].claim == "ok-1"

    assert results[1].verdict == "unverifiable"
    assert results[1].confidence == 0.0
    assert results[1].claim == "boom"
    assert "RuntimeError" in results[1].reasoning
    assert "simulated_failure" in results[1].reasoning
    # Per-claim audit must record the batch error — required for traceability.
    assert any(
        step.name == "verify_batch.error" for step in results[1].audit_trace
    )

    assert results[2].verdict == "supported"
    assert results[2].claim == "ok-2"


def test_factchecker_verify_batch_uses_max_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ThreadPoolExecutor must be constructed with the configured
    ``max_workers``, not a hard-coded value."""
    from concurrent.futures import ThreadPoolExecutor as RealPool

    captured: dict[str, int] = {}

    class RecordingPool:
        def __init__(self, *, max_workers: int) -> None:
            captured["max_workers"] = max_workers
            self._pool = RealPool(max_workers=max_workers)

        def __enter__(self) -> Any:
            return self._pool.__enter__()

        def __exit__(self, *args: Any) -> Any:
            return self._pool.__exit__(*args)

    monkeypatch.setattr(
        "cerno.fact_checker.ThreadPoolExecutor", RecordingPool
    )
    monkeypatch.setattr(
        "cerno.fact_checker.check_claim",
        lambda req, cons, **k: VerificationResult(
            verdict="supported", confidence=0.5,
            claim=req.claim, sources=[], reasoning="ok",
        ),
    )

    checker = FactChecker(
        consensus=_AssertNoCallConsensus(), max_workers=7,
    )
    checker.verify_batch([_make_request(claim="x")])

    assert captured["max_workers"] == 7


def _wire_cache_dir_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    """Patch the upstream stages so ``check_claim`` flows end-to-end, and
    record the ``cache_dir`` each retrieval helper is invoked with."""
    profile = _make_profile()
    plan = _make_plan(queries=("q1",))
    atomic = _make_atomic()
    judgement = _make_judgement()
    resolver_result = VerificationResult(
        verdict="supported", confidence=0.8,
        claim=atomic.text, sources=[], reasoning="ok",
    )
    _patch_pipeline(
        monkeypatch,
        profile=profile,
        atomic_claims=[atomic],
        plan=plan,
        judgement_fn=lambda ac, ev, cons, **kw: judgement,
        resolver_result=resolver_result,
    )

    def fake_tavily(
        query: str, api_key: str, *, cache_dir: Any = None,
    ) -> list[EvidenceSpan]:
        captured["tavily_cache_dir"] = cache_dir
        return [_make_evidence(id="t1")]

    def fake_wiki(
        query: str, *, cache_dir: Any = None,
    ) -> list[EvidenceSpan]:
        captured["wiki_cache_dir"] = cache_dir
        return [
            _make_evidence(
                id="w1", source_name="wikipedia_zh", source_tier="T0",
            )
        ]

    monkeypatch.setattr("cerno.fact_checker.search_tavily", fake_tavily)
    monkeypatch.setattr(
        "cerno.fact_checker.search_wikipedia_zh", fake_wiki
    )


def test_factchecker_cache_dir_forwarded_to_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _wire_cache_dir_pipeline(monkeypatch, captured)

    checker = FactChecker(
        consensus=_AssertNoCallConsensus(),
        tavily_api_key="dummy-key",
        cache_dir="/tmp/veritas-cache",
    )
    checker.verify("某个声明")

    # Both retrieval helpers must see the configured cache_dir.
    assert captured["tavily_cache_dir"] == "/tmp/veritas-cache"
    assert captured["wiki_cache_dir"] == "/tmp/veritas-cache"


def test_factchecker_cache_dir_none_forwarded_to_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _wire_cache_dir_pipeline(monkeypatch, captured)

    checker = FactChecker(
        consensus=_AssertNoCallConsensus(),
        tavily_api_key="dummy-key",
        # cache_dir omitted → defaults to None
    )
    checker.verify("某个声明")

    # No cache_dir set → retrieval helpers must be called with cache_dir=None
    # (i.e. caching disabled), NOT some accidental empty-string fallback.
    assert captured["tavily_cache_dir"] is None
    assert captured["wiki_cache_dir"] is None
