"""Stage 6: verdict resolution.

Three public functions:

- ``resolve_claim``: aggregate per-evidence judgements into a final
  ``VerificationResult`` (the happy path; LLM judgements are already in hand).
- ``decide_verdict``: strict-priority verdict picker; surfaced so callers that
  only need (verdict, confidence) can skip result construction.
- ``apply_failure_matrix``: build a ``VerificationResult`` for an upstream
  degradation (retrieval down, model down, etc.) where ``resolve_claim`` has
  no judgements to work with.

``resolve_claim`` runs in this fixed order:

1. Split judgements into supporting / refuting / other (neutral|insufficient).
2. Detect signals from *judged* evidence + judgements:
   - injection risk: any judged ``EvidenceSpan.risk_flags`` non-empty
   - model disagreement: at least one judgement has supports + refutes votes
   - model failure: any ``ModelVote.error`` non-None
   - tier distribution: only-T3, has-T0-or-T1
   Unjudged evidence (retrieved but not consumed by verification) is ignored
   on purpose — a stray T0 source can't paper over absent authority for the
   claims that were actually judged.
3. Pick verdict via ``decide_verdict`` (strict priority — never averaged).
4. Apply confidence caps via ``_cap_confidence`` (take MIN of every match).
5. Build a ``VerificationResult`` with full audit fields preserved.

Verdict priority (highest wins):

1. supporting + refuting both present  →  ``conflicting_sources``, conf=0.30
2. refuting only                       →  ``likely_error``, conf=base_refute (capped)
3. no supports and no refutes          →  ``unverifiable``, conf=0.20
4. supports + (injection|disagreement) →  ``needs_review``, conf=capped
5. high/critical risk + single source  →  ``needs_review``, conf=capped
   (one source never enough to auto-confirm a high-stakes claim, any tier)
6. supports, no flags                  →  verdict by (capped_conf, risk_level):
   - high/critical risk: ≥0.75 likely_correct, ≥0.50 needs_review, else unverifiable
   - low/medium  risk:  ≥0.70 likely_correct, ≥0.50 needs_review, else unverifiable

Confidence caps (applied as MIN; never averaged or summed):

- single T0 supporting source         → 0.75
- single T1 supporting source         → 0.70
- single T2 supporting source         → 0.60
- any model failure                   → 0.60
- supports + neutral/insufficient     → 0.70
- injection risk                      → 0.40
- only-T3 sources                     → 0.45
- high/critical risk no T0/T1 source  → 0.40

``apply_failure_matrix`` covers 11 upstream-failure cases at the ``context``
dict level. The pipeline (Stage 7) calls it when it cannot even invoke
``resolve_claim`` — e.g. Tavily and Wikipedia both 503. It mirrors the same
verdict priorities so callers can reason uniformly.
"""

from __future__ import annotations

from typing import Any

from veritas.types import (
    AtomicClaim,
    ClaimProfile,
    ConflictReport,
    Disagreement,
    EvidenceJudgement,
    EvidenceSpan,
    ModelVote,
    Verdict,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Map ConfidenceLevel to base float used as the starting confidence before caps.
_CONF_FLOOR: dict[str, float] = {"high": 0.90, "medium": 0.70, "low": 0.50}

#: Fixed confidence for ``conflicting_sources``. Deliberately not averaged from
#: supports/refutes — see module docstring.
CONFLICT_CONFIDENCE: float = 0.30

#: Fixed confidence for ``unverifiable`` when no supports / refutes exist.
UNVERIFIABLE_CONFIDENCE: float = 0.20

#: Single-source confidence caps by tier.
SINGLE_T0_CAP: float = 0.75
SINGLE_T1_CAP: float = 0.70
SINGLE_T2_CAP: float = 0.60

#: Other named caps.
MODEL_FAILURE_CAP: float = 0.60
SUPPORTS_PLUS_OTHER_CAP: float = 0.70
INJECTION_RISK_CAP: float = 0.40
ONLY_T3_CAP: float = 0.45
HIGH_RISK_NO_AUTHORITY_CAP: float = 0.40

#: Verdict-threshold tiers (capped confidence → verdict).
_HIGH_RISK_CORRECT_THRESHOLD: float = 0.75
_LOW_RISK_CORRECT_THRESHOLD: float = 0.70
_REVIEW_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Public: resolve_claim
# ---------------------------------------------------------------------------

def resolve_claim(
    atomic_claims: list[AtomicClaim],
    judgements: list[EvidenceJudgement],
    evidence: list[EvidenceSpan],
    profile: ClaimProfile,
) -> VerificationResult:  # type: ignore[name-defined]  # see import below
    """Aggregate (atomic_claims, judgements, evidence, profile) → VerificationResult.

    Hard order: split → signal detection → verdict pick → cap → assemble.
    """
    parent_claim = atomic_claims[0].parent_claim if atomic_claims else ""
    evidence_by_id = {e.id: e for e in evidence}

    supporting = [j for j in judgements if j.relation == "supports"]
    refuting = [j for j in judgements if j.relation == "refutes"]
    other = [j for j in judgements if j.relation in {"neutral", "insufficient"}]

    # Signals are computed only over evidence actually referenced by a
    # judgement. Unjudged spans (e.g. retrieval returned more than verification
    # consumed) must not affect verdict signals — otherwise a stray T0 source
    # could hide the lack of authority for the claims that were judged.
    judged_ids = {j.evidence_id for j in judgements}
    judged_evidence = [e for e in evidence if e.id in judged_ids]

    has_injection = any(e.risk_flags for e in judged_evidence)
    has_disagreement = _has_model_disagreement(judgements)
    has_model_failure = _has_model_failure(judgements)
    has_direct_conflict = _has_direct_conflict(judgements)
    only_t3 = bool(judged_evidence) and all(
        e.source_tier in {"T3", "BLOCKED"} for e in judged_evidence
    )
    has_t0_or_t1 = any(e.source_tier in {"T0", "T1"} for e in judged_evidence)

    verdict, confidence = decide_verdict(
        supporting=supporting,
        refuting=refuting,
        other=other,
        evidence_by_id=evidence_by_id,
        profile=profile,
        has_injection_risk=has_injection,
        has_model_disagreement=has_disagreement,
        has_model_failure=has_model_failure,
        has_direct_conflict=has_direct_conflict,
        only_t3=only_t3,
        has_t0_or_t1=has_t0_or_t1,
    )

    reasoning = _build_reasoning(
        verdict=verdict,
        confidence=confidence,
        supporting=supporting,
        refuting=refuting,
        other=other,
        has_injection=has_injection,
        has_disagreement=has_disagreement,
        has_model_failure=has_model_failure,
        only_t3=only_t3,
    )

    sources, supporting_sources, refuting_sources = _collect_sources(
        supporting, refuting, other, evidence_by_id,
    )
    disagreements = _collect_disagreements(judgements)
    conflicts = _collect_conflicts(judgements)
    model_votes = [v for j in judgements for v in j.model_votes]

    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        claim=parent_claim,
        sources=sources,
        reasoning=reasoning,
        model_votes=model_votes,
        disagreements=disagreements,
        conflicts=conflicts,
        consensus_method="strictest",
        atomic_claims=list(atomic_claims),
        supporting_sources=supporting_sources,
        refuting_sources=refuting_sources,
    )


# ---------------------------------------------------------------------------
# Public: decide_verdict
# ---------------------------------------------------------------------------

def decide_verdict(
    *,
    supporting: list[EvidenceJudgement],
    refuting: list[EvidenceJudgement],
    other: list[EvidenceJudgement],
    evidence_by_id: dict[str, EvidenceSpan],
    profile: ClaimProfile,
    has_injection_risk: bool,
    has_model_disagreement: bool,
    has_model_failure: bool,
    only_t3: bool,
    has_t0_or_t1: bool,
    has_direct_conflict: bool = True,
) -> tuple[Verdict, float]:
    """Strict-priority verdict picker. Never averages opposing evidence."""
    # Priority 1: conflict — supporting and refuting both exist for the same
    # atomic claim. Cross-atomic support/refute in a decomposed parent means one
    # sub-claim is false, so the parent should route to likely_error instead.
    if supporting and refuting and has_direct_conflict:
        return "conflicting_sources", CONFLICT_CONFIDENCE

    # Priority 2: refutes — a high-confidence refute is "likely_error".
    if refuting:
        base = max(_CONF_FLOOR[j.confidence] for j in refuting)
        capped = base
        if has_model_failure:
            capped = min(capped, MODEL_FAILURE_CAP)
        if only_t3:
            capped = min(capped, ONLY_T3_CAP)
        if has_injection_risk:
            capped = min(capped, INJECTION_RISK_CAP)
        return "likely_error", capped

    # Priority 3: no supports and no refutes — nothing to anchor a verdict.
    if not supporting:
        return "unverifiable", UNVERIFIABLE_CONFIDENCE

    # From here on: supports present, no refutes.
    base = max(_CONF_FLOOR[j.confidence] for j in supporting)
    capped = _cap_confidence(
        base=base,
        supporting=supporting,
        other=other,
        evidence_by_id=evidence_by_id,
        profile=profile,
        has_model_failure=has_model_failure,
        has_injection_risk=has_injection_risk,
        only_t3=only_t3,
        has_t0_or_t1=has_t0_or_t1,
    )

    # Priority 4: supports + (injection risk OR model disagreement) → review.
    if has_injection_risk or has_model_disagreement:
        return "needs_review", capped

    # Priority 5: high/critical risk with only one supporting source never
    # auto-confirms — one source is not enough authority to clear a high-stakes
    # claim, regardless of tier. (Single T0 caps at 0.75, which equals the
    # high-risk correct threshold, so without this guard the boundary case
    # leaks through to likely_correct.)
    if profile.risk_level in {"high", "critical"} and len(supporting) == 1:
        return "needs_review", capped

    # Priority 6: verdict by (capped confidence, risk level).
    correct_threshold = (
        _HIGH_RISK_CORRECT_THRESHOLD
        if profile.risk_level in {"high", "critical"}
        else _LOW_RISK_CORRECT_THRESHOLD
    )
    if capped >= correct_threshold:
        return "likely_correct", capped
    if capped >= _REVIEW_THRESHOLD:
        return "needs_review", capped
    return "unverifiable", capped


# ---------------------------------------------------------------------------
# Public: apply_failure_matrix
# ---------------------------------------------------------------------------

def apply_failure_matrix(context: dict[str, Any]) -> VerificationResult:  # type: ignore[name-defined]
    """Build a VerificationResult for an upstream-degradation context.

    Recognized context flags (all optional, default True for *_available):

    - ``tavily_available`` / ``wiki_available`` — retrieval health
    - ``deepseek_available`` / ``mimo_available`` — model health
    - ``models_disagree`` — two models gave incompatible relations
    - ``has_conflict`` — supports + refutes coexist
    - ``has_injection_risk`` — an evidence carries an injection risk flag
    - ``only_t3_sources`` — every retrieved source is T3 / BLOCKED
    - ``high_risk_single_source`` — high-risk claim with only one source

    Payload fields:

    - ``claim`` (str): parent claim text. Defaults to ``""``.
    - ``sources`` (list[EvidenceSpan]): partial sources to surface. Defaults
      to ``[]``.

    Verdict priority mirrors ``decide_verdict``:

    1. both retrievals down  → ``unverifiable``, conf=0.10
    2. both models down      → ``unverifiable``, conf=0.10
    3. has_conflict          → ``conflicting_sources``, conf=0.30
    4. any other flag        → ``needs_review`` with MIN of matching caps
    5. no flags at all       → ``unverifiable``, conf=0.0 (mis-call defense)
    """
    claim = str(context.get("claim", ""))
    sources_in = context.get("sources", []) or []
    sources = list(sources_in)

    tavily_ok = bool(context.get("tavily_available", True))
    wiki_ok = bool(context.get("wiki_available", True))
    deepseek_ok = bool(context.get("deepseek_available", True))
    mimo_ok = bool(context.get("mimo_available", True))
    models_disagree = bool(context.get("models_disagree", False))
    has_conflict = bool(context.get("has_conflict", False))
    has_injection = bool(context.get("has_injection_risk", False))
    only_t3 = bool(context.get("only_t3_sources", False))
    high_risk_single = bool(context.get("high_risk_single_source", False))

    # Priority 1: both retrievals down — no evidence at all.
    if not tavily_ok and not wiki_ok:
        return _failure_result(
            verdict="unverifiable",
            confidence=0.10,
            claim=claim,
            sources=sources,
            reasoning="retrieval_failed_all: tavily_down, wiki_down",
        )

    # Priority 2: both models down — no judgement possible.
    if not deepseek_ok and not mimo_ok:
        return _failure_result(
            verdict="unverifiable",
            confidence=0.10,
            claim=claim,
            sources=sources,
            reasoning="model_failed_all: deepseek_down, mimo_down",
        )

    # Priority 3: conflict — opposing evidence both present.
    if has_conflict:
        return _failure_result(
            verdict="conflicting_sources",
            confidence=CONFLICT_CONFIDENCE,
            claim=claim,
            sources=sources,
            reasoning="source_conflict: supporting and refuting evidence both present",
        )

    # Priority 4: needs_review with MIN of matching caps.
    flags: list[str] = []
    caps: list[float] = []

    if not tavily_ok:
        flags.append("tavily_down")
        caps.append(0.50)
    if not wiki_ok:
        flags.append("wiki_down")
        caps.append(0.50)
    if not deepseek_ok:
        flags.append("deepseek_down")
        caps.append(MODEL_FAILURE_CAP)
    if not mimo_ok:
        flags.append("mimo_down")
        caps.append(MODEL_FAILURE_CAP)
    if models_disagree:
        flags.append("model_disagreement")
        caps.append(INJECTION_RISK_CAP)
    if has_injection:
        flags.append("injection_risk")
        caps.append(INJECTION_RISK_CAP)
    if only_t3:
        flags.append("only_t3_sources")
        caps.append(ONLY_T3_CAP)
    if high_risk_single:
        flags.append("high_risk_insufficient_authority")
        caps.append(HIGH_RISK_NO_AUTHORITY_CAP)

    if not flags:
        return _failure_result(
            verdict="unverifiable",
            confidence=0.0,
            claim=claim,
            sources=sources,
            reasoning="apply_failure_matrix called without any degradation flags",
        )

    confidence = min(caps)
    return _failure_result(
        verdict="needs_review",
        confidence=confidence,
        claim=claim,
        sources=sources,
        reasoning="degradation: " + ", ".join(flags),
    )


# ---------------------------------------------------------------------------
# Internal: signal detection
# ---------------------------------------------------------------------------

def _has_model_disagreement(judgements: list[EvidenceJudgement]) -> bool:
    """True iff some judgement's ModelVotes carry both supports and refutes."""
    for j in judgements:
        relations = {v.relation for v in j.model_votes if v.relation is not None}
        if "supports" in relations and "refutes" in relations:
            return True
    return False


def _has_model_failure(judgements: list[EvidenceJudgement]) -> bool:
    """True iff any ModelVote has a non-None error."""
    return any(v.error is not None for j in judgements for v in j.model_votes)


def _has_direct_conflict(judgements: list[EvidenceJudgement]) -> bool:
    by_claim: dict[str, set[str]] = {}
    for judgement in judgements:
        by_claim.setdefault(judgement.atomic_claim_id, set()).add(judgement.relation)
    return any(
        "supports" in relations and "refutes" in relations
        for relations in by_claim.values()
    )


# ---------------------------------------------------------------------------
# Internal: confidence cap stacking
# ---------------------------------------------------------------------------

def _cap_confidence(
    *,
    base: float,
    supporting: list[EvidenceJudgement],
    other: list[EvidenceJudgement],
    evidence_by_id: dict[str, EvidenceSpan],
    profile: ClaimProfile,
    has_model_failure: bool,
    has_injection_risk: bool,
    only_t3: bool,
    has_t0_or_t1: bool,
) -> float:
    """Return MIN(base, every applicable cap). Never averages, never sums."""
    capped = base

    # Single-supporting-source caps depend on tier.
    if len(supporting) == 1:
        single = supporting[0]
        span = evidence_by_id.get(single.evidence_id)
        if span is not None:
            if span.source_tier == "T0":
                capped = min(capped, SINGLE_T0_CAP)
            elif span.source_tier == "T1":
                capped = min(capped, SINGLE_T1_CAP)
            elif span.source_tier == "T2":
                capped = min(capped, SINGLE_T2_CAP)

    if has_model_failure:
        capped = min(capped, MODEL_FAILURE_CAP)
    if supporting and other:
        capped = min(capped, SUPPORTS_PLUS_OTHER_CAP)
    if has_injection_risk:
        capped = min(capped, INJECTION_RISK_CAP)
    if only_t3:
        capped = min(capped, ONLY_T3_CAP)
    if profile.risk_level in {"high", "critical"} and not has_t0_or_t1:
        capped = min(capped, HIGH_RISK_NO_AUTHORITY_CAP)

    return capped


# ---------------------------------------------------------------------------
# Internal: result assembly
# ---------------------------------------------------------------------------

def _collect_sources(
    supporting: list[EvidenceJudgement],
    refuting: list[EvidenceJudgement],
    other: list[EvidenceJudgement],
    evidence_by_id: dict[str, EvidenceSpan],
) -> tuple[list[EvidenceSpan], list[EvidenceSpan], list[EvidenceSpan]]:
    """Order: supporting → refuting → other. De-duplicates by id."""
    seen: set[str] = set()
    all_sources: list[EvidenceSpan] = []
    supporting_sources: list[EvidenceSpan] = []
    refuting_sources: list[EvidenceSpan] = []

    for j in supporting:
        span = evidence_by_id.get(j.evidence_id)
        if span is None:
            continue
        supporting_sources.append(span)
        if span.id not in seen:
            all_sources.append(span)
            seen.add(span.id)

    for j in refuting:
        span = evidence_by_id.get(j.evidence_id)
        if span is None:
            continue
        refuting_sources.append(span)
        if span.id not in seen:
            all_sources.append(span)
            seen.add(span.id)

    for j in other:
        span = evidence_by_id.get(j.evidence_id)
        if span is None:
            continue
        if span.id not in seen:
            all_sources.append(span)
            seen.add(span.id)

    return all_sources, supporting_sources, refuting_sources


def _collect_disagreements(
    judgements: list[EvidenceJudgement],
) -> list[Disagreement]:
    """One Disagreement per judgement whose ModelVotes mix supports/refutes."""
    out: list[Disagreement] = []
    for j in judgements:
        relations = {v.relation for v in j.model_votes if v.relation is not None}
        if "supports" in relations and "refutes" in relations:
            out.append(Disagreement(
                atomic_claim_id=j.atomic_claim_id,
                evidence_id=j.evidence_id,
                model_votes=list(j.model_votes),
                summary="supports vs refutes",
            ))
    return out


def _collect_conflicts(
    judgements: list[EvidenceJudgement],
) -> list[ConflictReport]:
    """One ConflictReport per atomic_claim whose evidence mixes supports/refutes."""
    by_claim: dict[str, list[EvidenceJudgement]] = {}
    for j in judgements:
        by_claim.setdefault(j.atomic_claim_id, []).append(j)

    out: list[ConflictReport] = []
    for claim_id, claim_judgements in by_claim.items():
        supports_ids = [j.evidence_id for j in claim_judgements if j.relation == "supports"]
        refutes_ids = [j.evidence_id for j in claim_judgements if j.relation == "refutes"]
        if supports_ids and refutes_ids:
            out.append(ConflictReport(
                atomic_claim_id=claim_id,
                supporting_evidence_ids=supports_ids,
                refuting_evidence_ids=refutes_ids,
                summary=(
                    f"{len(supports_ids)} supporting + "
                    f"{len(refutes_ids)} refuting evidence"
                ),
            ))
    return out


def _build_reasoning(
    *,
    verdict: Verdict,
    confidence: float,
    supporting: list[EvidenceJudgement],
    refuting: list[EvidenceJudgement],
    other: list[EvidenceJudgement],
    has_injection: bool,
    has_disagreement: bool,
    has_model_failure: bool,
    only_t3: bool,
) -> str:
    parts = [
        f"verdict={verdict} confidence={confidence:.2f}",
        f"supports={len(supporting)} refutes={len(refuting)} other={len(other)}",
    ]
    flags: list[str] = []
    if has_injection:
        flags.append("injection_risk")
    if has_disagreement:
        flags.append("model_disagreement")
    if has_model_failure:
        flags.append("model_failure")
    if only_t3:
        flags.append("only_t3_sources")
    if flags:
        parts.append("flags=" + ",".join(flags))
    return "; ".join(parts)


def _failure_result(
    *,
    verdict: Verdict,
    confidence: float,
    claim: str,
    sources: list[EvidenceSpan],
    reasoning: str,
) -> VerificationResult:  # type: ignore[name-defined]
    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        claim=claim,
        sources=sources,
        reasoning=reasoning,
    )


# Local import to keep VerificationResult at the bottom (avoids circular import
# risk if types ever pulls anything from resolver). VerificationResult itself
# has no runtime dependency on this module.
from veritas.types import VerificationResult  # noqa: E402

# Silence "unused import" warnings for re-export consumers; ModelVote is part
# of the documented input shape and AtomicClaim is on the public boundary.
_ = (AtomicClaim, ModelVote)
