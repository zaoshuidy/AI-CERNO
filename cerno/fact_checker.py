"""Stage 7: end-to-end orchestrator.

Public surface:

- ``FactChecker`` — the production entry point. Constructed once with
  ``llm_providers`` / ``tavily_api_key`` / ``max_workers`` / ``cache_dir``;
  exposes ``verify(request)`` for a single claim and ``verify_batch(requests)``
  for parallel fan-out via ``ThreadPoolExecutor``.
- ``check_claim(request, consensus, ...)`` — the functional helper that
  ``FactChecker.verify`` delegates to. Kept public so tests and advanced
  callers can drive the pipeline with a pre-built ``MultiModelConsensus``.

Pipeline:

1. Stage 3 (understanding) — ``build_claim_profile`` → ``decompose_claim`` →
   ``build_retrieval_plan``. If the profile reports ``is_checkable=False``,
   short-circuit to ``verdict="unverifiable"`` with the profile's reason as
   ``reasoning`` (no retrieval, no verification).
2. Stage 4 (retrieval) — for each query in ``plan.queries``, fan out to
   Tavily and Wikipedia (or injected stubs). Transport errors mark the
   channel as unavailable but do not crash the pipeline. Spans accumulate
   and are deduplicated by id.
3. If retrieval ends with zero evidence, route to ``apply_failure_matrix``
   with the observed availability flags.
4. Stage 5 (verification) — for each ``(atomic_claim, evidence)`` pair,
   call ``verify_evidence``. Hard-rule short-circuits inside
   ``verify_evidence`` keep the LLM call count modest.
5. Stage 6 (resolution) — ``resolve_claim`` aggregates the per-pair
   judgements into a ``VerificationResult``. The orchestrator attaches
   the ``TraceRecorder.steps`` and ``CostBreakdown`` to the returned
   result via ``dataclasses.replace`` (both fields default to empty on
   the resolver side).

Boundary constraints (per Stage 7 release directive):

- Never imports / depends on the OpenClawFix adapter.
- Never imports / depends on backend business code.
- Live Tavily / Wikipedia calls stay behind injectable callables so
  unit tests run offline and live tests are gated by ``LIVE_TEST=1``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from cerno.consensus import MultiModelConsensus, OpenAICompatibleAdapter
from cerno.observability import CostBreakdown, TraceRecorder
from cerno.resolver import apply_failure_matrix, resolve_claim
from cerno.retrieval import search_tavily, search_wikipedia_zh
from cerno.types import (
    EvidenceJudgement,
    EvidenceSpan,
    LLMProvider,
    VerificationRequest,
    VerificationResult,
)
from cerno.understanding import (
    build_claim_profile,
    build_retrieval_plan,
    decompose_claim,
)
from cerno.verification import verify_evidence

#: Injectable search callable. Production callers don't supply this; tests
#: substitute a deterministic stub so they never touch the network. The
#: orchestrator only ever calls it with a single positional ``query`` arg.
SearchCallable = Callable[..., list[EvidenceSpan]]


def check_claim(
    request: VerificationRequest,
    consensus: MultiModelConsensus,
    *,
    tavily_api_key: str | None = None,
    cache_dir: str | None = None,
    tavily_search: SearchCallable | None = None,
    wiki_search: SearchCallable | None = None,
) -> VerificationResult:
    """Run the full 6-stage verification pipeline on a single claim.

    Tavily is silently disabled when neither ``tavily_api_key`` nor
    ``tavily_search`` is supplied — the pipeline falls back to Wikipedia
    only and the resulting result reflects the missing channel through
    ``apply_failure_matrix`` if no evidence is collected.
    """
    recorder = TraceRecorder()
    cost = CostBreakdown()

    # --- Stage 3: understanding ---------------------------------------
    profile = build_claim_profile(request, consensus, cost=cost)
    recorder.record(
        "build_claim_profile",
        request.claim,
        (
            f"is_checkable={profile.is_checkable} "
            f"type={profile.claim_type} "
            f"risk={profile.risk_level} "
            f"source={profile.risk_level_source}"
        ),
    )

    # An opinion or non-claim never benefits from retrieval/verification.
    # Surface profile.reason directly so callers see *why* it's unverifiable.
    if not profile.is_checkable:
        result = VerificationResult(
            verdict="unverifiable",
            confidence=0.0,
            claim=request.claim,
            sources=[],
            reasoning=profile.reason or "claim is not factually checkable",
        )
        return _attach_audit(result, recorder, cost)

    atomic_claims = decompose_claim(request, profile, consensus, cost=cost)
    recorder.record(
        "decompose_claim",
        request.claim,
        f"{len(atomic_claims)} atomic_claims",
    )

    plan = build_retrieval_plan(profile, atomic_claims)
    recorder.record(
        "build_retrieval_plan",
        f"{len(atomic_claims)} atomic_claims",
        f"queries={len(plan.queries)} targets={list(plan.source_targets)}",
    )

    # --- Stage 4: retrieval -------------------------------------------
    tavily_available = True
    wiki_available = True
    evidence: list[EvidenceSpan] = []
    seen_ids: set[str] = set()

    # Tavily is opt-in: caller must either pass an api_key or inject a stub.
    # Missing both is a normal degraded mode, not an error.
    tavily_enabled = "tavily" in plan.source_targets and (
        tavily_search is not None or bool(tavily_api_key)
    )
    if "tavily" in plan.source_targets and not tavily_enabled:
        tavily_available = False
        recorder.record(
            "search_tavily.skip",
            "",
            "no_api_key_and_no_stub",
        )

    wiki_enabled = "wikipedia_zh" in plan.source_targets

    for query in plan.queries:
        if tavily_enabled:
            ok, spans = _run_tavily(
                query, tavily_search, tavily_api_key, cache_dir,
                recorder, cost,
            )
            if not ok:
                tavily_available = False
            _merge(spans, evidence, seen_ids)
        if wiki_enabled:
            ok, spans = _run_wiki(
                query, wiki_search, cache_dir, recorder, cost,
            )
            if not ok:
                wiki_available = False
            _merge(spans, evidence, seen_ids)

    # No evidence at all → upstream-degradation routing. Even if both
    # channels reported ``ok``, the failure matrix's "no flags" branch
    # returns ``unverifiable`` conf=0.0 which is the correct semantics
    # for "we tried, found nothing useful".
    if not evidence:
        result = apply_failure_matrix({
            "tavily_available": tavily_available,
            "wiki_available": wiki_available,
            "claim": request.claim,
            "sources": [],
        })
        return _attach_audit(result, recorder, cost)

    # --- Stage 5: verification ----------------------------------------
    judgements: list[EvidenceJudgement] = []
    for ac in atomic_claims:
        for ev in evidence:
            judgement = verify_evidence(ac, ev, consensus, cost=cost)
            judgements.append(judgement)
            recorder.record(
                "verify_evidence",
                f"{ac.id}|{ev.id}",
                f"{judgement.relation}/{judgement.confidence}",
            )

    # --- Stage 6: resolution ------------------------------------------
    result = resolve_claim(atomic_claims, judgements, evidence, profile)
    recorder.record(
        "resolve_claim",
        f"{len(atomic_claims)}ac×{len(evidence)}ev",
        f"{result.verdict}/{result.confidence:.2f}",
    )

    return _attach_audit(result, recorder, cost)


# ---------------------------------------------------------------------------
# Internal: retrieval helpers
# ---------------------------------------------------------------------------

def _run_tavily(
    query: str,
    stub: SearchCallable | None,
    api_key: str | None,
    cache_dir: str | None,
    recorder: TraceRecorder,
    cost: CostBreakdown,
) -> tuple[bool, list[EvidenceSpan]]:
    """One Tavily call. Returns ``(ok, spans)`` — ``ok=False`` on transport
    failure so the caller can flip ``tavily_available``."""
    try:
        if stub is not None:
            spans = stub(query)
        else:
            # caller (check_claim) guarantees api_key is truthy here.
            assert api_key, "_run_tavily reached without stub and without api_key"
            spans = search_tavily(query, api_key, cache_dir=cache_dir)
    except Exception as exc:
        recorder.record(
            "search_tavily.error",
            query,
            type(exc).__name__,
        )
        return False, []
    cost.add_retrieval()
    recorder.record(
        "search_tavily",
        query,
        f"{len(spans)} spans",
    )
    return True, spans


def _run_wiki(
    query: str,
    stub: SearchCallable | None,
    cache_dir: str | None,
    recorder: TraceRecorder,
    cost: CostBreakdown,
) -> tuple[bool, list[EvidenceSpan]]:
    """One Wikipedia call. Returns ``(ok, spans)`` — ``ok=False`` on
    transport failure so the caller can flip ``wiki_available``."""
    try:
        if stub is not None:
            spans = stub(query)
        else:
            spans = search_wikipedia_zh(query, cache_dir=cache_dir)
    except Exception as exc:
        recorder.record(
            "search_wikipedia_zh.error",
            query,
            type(exc).__name__,
        )
        return False, []
    cost.add_retrieval()
    recorder.record(
        "search_wikipedia_zh",
        query,
        f"{len(spans)} spans",
    )
    return True, spans


def _merge(
    new_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    seen_ids: set[str],
) -> None:
    """In-place dedupe-by-id merge so the verification stage never wastes
    cycles on duplicate spans."""
    for span in new_spans:
        if span.id not in seen_ids:
            seen_ids.add(span.id)
            evidence.append(span)


def _attach_audit(
    result: VerificationResult,
    recorder: TraceRecorder,
    cost: CostBreakdown,
) -> VerificationResult:
    """Splice ``TraceRecorder.steps`` and ``CostBreakdown`` onto a result
    from ``resolve_claim`` / ``apply_failure_matrix`` / the inline
    not-checkable branch. All three leave these fields at default values.
    """
    return replace(result, audit_trace=list(recorder.steps), cost=cost)


# ---------------------------------------------------------------------------
# Public class: FactChecker
# ---------------------------------------------------------------------------

class FactChecker:
    """Production entry point for the Stage 7 verification pipeline.

    Constructed once with the runtime config (LLM providers, Tavily key,
    worker count, cache directory). ``verify`` runs a single claim through
    ``check_claim``; ``verify_batch`` fans claims out across a thread pool
    while preserving input order and isolating per-claim failures.

    Two construction modes are supported:

    - **Production:** pass ``llm_providers`` (a non-empty list of
      ``LLMProvider``). One ``OpenAICompatibleAdapter`` is built per
      provider, then wrapped in a single ``MultiModelConsensus``.
    - **Test injection:** pass a pre-built ``consensus`` (typically a
      mocked ``MultiModelConsensus`` subclass). When both are passed,
      ``consensus`` takes precedence and ``llm_providers`` is ignored —
      this keeps unit tests offline without needing real adapters.

    At least one of the two must be supplied; otherwise a ``ValueError``
    is raised. ``tavily_search`` / ``wiki_search`` exist solely so tests
    can substitute deterministic stubs — production code leaves them at
    ``None`` and lets ``check_claim`` route through the real retrieval
    layer.
    """

    def __init__(
        self,
        *,
        tavily_api_key: str | None = None,
        llm_providers: list[LLMProvider] | None = None,
        max_workers: int = 4,
        cache_dir: str | None = None,
        tavily_search: Callable[..., list[EvidenceSpan]] | None = None,
        wiki_search: Callable[..., list[EvidenceSpan]] | None = None,
        consensus: MultiModelConsensus | None = None,
    ) -> None:
        if consensus is not None:
            # Test-injection path takes precedence so unit tests stay offline.
            self.consensus: MultiModelConsensus = consensus
        elif llm_providers:
            adapters = [OpenAICompatibleAdapter(p) for p in llm_providers]
            self.consensus = MultiModelConsensus(adapters)
        else:
            raise ValueError(
                "FactChecker requires either llm_providers (production) "
                "or consensus (test injection)"
            )
        if max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1, got {max_workers}"
            )
        self.tavily_api_key = tavily_api_key
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self._tavily_search = tavily_search
        self._wiki_search = wiki_search

    def verify(
        self,
        request: VerificationRequest | str,
        context: str | None = None,
    ) -> VerificationResult:
        """Verify a single claim. Accepts a string for the convenient
        ``checker.verify("some claim")`` style or a fully-formed
        ``VerificationRequest`` for callers that need to set ``domain_hint``
        or ``risk_hint``."""
        if isinstance(request, str):
            request = VerificationRequest(claim=request, context=context)
        return check_claim(
            request,
            self.consensus,
            tavily_api_key=self.tavily_api_key,
            cache_dir=self.cache_dir,
            tavily_search=self._tavily_search,
            wiki_search=self._wiki_search,
        )

    def verify_batch(
        self, requests: list[VerificationRequest]
    ) -> list[VerificationResult]:
        """Verify many claims in parallel.

        Output order matches input order regardless of completion order;
        per-claim failures are caught and converted to ``unverifiable``
        results so one bad claim never poisons the batch. Each result
        carries its own ``audit_trace`` and ``cost`` (``check_claim``
        constructs fresh ones per call, so no cross-claim leakage)."""
        if not requests:
            return []
        # Pre-allocate by index so as_completed (which yields out of order)
        # writes results back to their original slot.
        results: list[VerificationResult | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._verify_safe, req): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        # All slots are filled; the cast is safe because every future
        # either returned a VerificationResult or raised (in which case
        # _verify_safe already converted it to one).
        return results  # type: ignore[return-value]

    def _verify_safe(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """Wrap ``verify`` so a single claim's failure is isolated.

        Any exception becomes an ``unverifiable`` result with a fresh
        audit step (``verify_batch.error``) and a fresh ``CostBreakdown``.
        Keeps the rest of the batch from crashing."""
        try:
            return self.verify(request)
        except Exception as exc:  # noqa: BLE001
            recorder = TraceRecorder()
            cost = CostBreakdown()
            recorder.record(
                "verify_batch.error",
                request.claim,
                f"{type(exc).__name__}: {exc}",
            )
            return VerificationResult(
                verdict="unverifiable",
                confidence=0.0,
                claim=request.claim,
                sources=[],
                reasoning=f"batch_error: {type(exc).__name__}: {exc}",
                audit_trace=list(recorder.steps),
                cost=cost,
            )
