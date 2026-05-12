"""Unit tests for veritas.resolver.

No LLM, no network. Every test constructs ``EvidenceJudgement`` /
``EvidenceSpan`` directly because Stage 6 consumes already-built
judgements — it never calls consensus itself.

Coverage map (each numbered section matches a user-spec requirement):

1. ``decide_verdict`` strict-priority order:
   - P1 supports + refutes → conflicting_sources (fixed 0.30)
   - P2 refutes only → likely_error (capped)
   - P3 no supports / no refutes → unverifiable (fixed 0.20)
   - P4 supports + (injection | model disagreement) → needs_review
   - P5 high/critical risk + single supporting source → needs_review
     (one source never enough to auto-confirm a high-stakes claim)
   - P6 supports only (multi-source or low/medium risk) →
     verdict by (capped, risk_level)

2. Confidence caps (8 named caps, all stack as MIN):
   - SINGLE_T0_CAP, SINGLE_T1_CAP, SINGLE_T2_CAP
   - MODEL_FAILURE_CAP
   - SUPPORTS_PLUS_OTHER_CAP
   - INJECTION_RISK_CAP
   - ONLY_T3_CAP
   - HIGH_RISK_NO_AUTHORITY_CAP

3. ``resolve_claim`` integration:
   - source ordering / de-duplication
   - Disagreement + ConflictReport assembly
   - model_votes flattening
   - consensus_method == "strictest"
   - to_compact_dict() invariant survives

4. ``apply_failure_matrix`` priority + 11 documented cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from veritas.resolver import (
    CONFLICT_CONFIDENCE,
    HIGH_RISK_NO_AUTHORITY_CAP,
    INJECTION_RISK_CAP,
    MODEL_FAILURE_CAP,
    ONLY_T3_CAP,
    SINGLE_T0_CAP,
    SINGLE_T1_CAP,
    SINGLE_T2_CAP,
    SUPPORTS_PLUS_OTHER_CAP,
    UNVERIFIABLE_CONFIDENCE,
    apply_failure_matrix,
    decide_verdict,
    resolve_claim,
)
from veritas.types import (
    AtomicClaim,
    ClaimProfile,
    ConfidenceLevel,
    ConflictReport,
    Disagreement,
    EvidenceJudgement,
    EvidenceSpan,
    ModelVote,
    Relation,
    SourceTier,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Factories — keep tests terse and intent-first
# ---------------------------------------------------------------------------


def _make_evidence(
    eid: str = "ev1",
    tier: SourceTier = "T1",
    risk_flags: list[str] | None = None,
    quote: str = "evidence quote",
) -> EvidenceSpan:
    return EvidenceSpan(
        id=eid,
        title=f"title-{eid}",
        url=f"https://example.com/{eid}",
        quote=quote,
        source_name="wikipedia_zh" if tier == "T1" else "tavily",
        source_tier=tier,
        retrieved_at="2026-01-01T00:00:00+00:00",
        content_hash=f"hash{eid}",
        risk_flags=list(risk_flags or []),
    )


def _make_judgement(
    eid: str = "ev1",
    relation: Relation = "supports",
    confidence: ConfidenceLevel = "high",
    claim_id: str = "c1",
    model_votes: list[ModelVote] | None = None,
    reason: str = "ok",
) -> EvidenceJudgement:
    return EvidenceJudgement(
        atomic_claim_id=claim_id,
        evidence_id=eid,
        relation=relation,
        confidence=confidence,
        quote="qq",
        reason=reason,
        model_votes=list(model_votes or []),
    )


def _make_atomic_claim(cid: str = "c1", text: str = "claim text") -> AtomicClaim:
    return AtomicClaim(
        id=cid,
        text=text,
        original_span=text,
        parent_claim=text,
        check_priority=1,
        required_evidence_type=[],
    )


def _make_profile(
    risk_level: str = "medium",
    *,
    allow_single_t1: bool = True,
    allow_single_t0: bool = True,
) -> ClaimProfile:
    return ClaimProfile(
        is_checkable=True,
        claim_type="entity_fact",
        domain="general",
        risk_level=risk_level,  # type: ignore[arg-type]
        risk_level_source="user_hint",
        required_evidence=[],
        strict_mode=False,
        reason="test",
        allow_single_t1_source=allow_single_t1,
        allow_single_t0_source=allow_single_t0,
    )


def _common_decide_kwargs(
    *,
    profile: ClaimProfile | None = None,
    evidence_by_id: dict[str, EvidenceSpan] | None = None,
    has_injection: bool = False,
    has_disagreement: bool = False,
    has_model_failure: bool = False,
    only_t3: bool = False,
    has_t0_or_t1: bool = True,
) -> dict[str, Any]:
    return {
        "evidence_by_id": evidence_by_id or {},
        "profile": profile or _make_profile(),
        "has_injection_risk": has_injection,
        "has_model_disagreement": has_disagreement,
        "has_model_failure": has_model_failure,
        "only_t3": only_t3,
        "has_t0_or_t1": has_t0_or_t1,
    }


# ---------------------------------------------------------------------------
# 1. decide_verdict — strict priority
# ---------------------------------------------------------------------------


def test_decide_verdict_priority1_supports_plus_refutes_is_conflict() -> None:
    """P1: supports and refutes both present → conflicting_sources, fixed 0.30."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    refuting = [_make_judgement(eid="e2", relation="refutes", confidence="high")]

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=refuting, other=[],
        **_common_decide_kwargs(
            evidence_by_id={
                "e1": _make_evidence("e1", "T0"),
                "e2": _make_evidence("e2", "T0"),
            },
        ),
    )

    assert verdict == "conflicting_sources"
    # Fixed — must never be averaged from the two sides' base confidences.
    assert conf == CONFLICT_CONFIDENCE == 0.30


def test_decide_verdict_conflict_never_averages_opposing_confidences() -> None:
    """Even high+high supports vs high+high refutes stays at fixed 0.30."""
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    refuting = [
        _make_judgement(eid="e3", relation="refutes", confidence="high"),
        _make_judgement(eid="e4", relation="refutes", confidence="high"),
    ]

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=refuting, other=[],
        **_common_decide_kwargs(),
    )

    assert verdict == "conflicting_sources"
    assert conf == 0.30


def test_decide_verdict_priority2_refutes_only_is_likely_error() -> None:
    """P2: refutes only → likely_error with base from max refute confidence."""
    refuting = [_make_judgement(eid="e1", relation="refutes", confidence="high")]

    verdict, conf = decide_verdict(
        supporting=[], refuting=refuting, other=[],
        **_common_decide_kwargs(),
    )

    assert verdict == "likely_error"
    # high → 0.90 base, no caps applied → 0.90.
    assert conf == pytest.approx(0.90)


def test_decide_verdict_priority2_refutes_with_injection_caps_at_040() -> None:
    """Refute base 0.90 but injection cap → final 0.40."""
    refuting = [_make_judgement(eid="e1", relation="refutes", confidence="high")]

    verdict, conf = decide_verdict(
        supporting=[], refuting=refuting, other=[],
        **_common_decide_kwargs(has_injection=True),
    )

    assert verdict == "likely_error"
    assert conf == pytest.approx(INJECTION_RISK_CAP) == 0.40


def test_decide_verdict_priority2_refutes_with_model_failure_caps_at_060() -> None:
    refuting = [_make_judgement(eid="e1", relation="refutes", confidence="high")]

    verdict, conf = decide_verdict(
        supporting=[], refuting=refuting, other=[],
        **_common_decide_kwargs(has_model_failure=True),
    )

    assert verdict == "likely_error"
    assert conf == pytest.approx(MODEL_FAILURE_CAP)


def test_decide_verdict_priority3_no_supports_no_refutes_is_unverifiable() -> None:
    """P3: nothing to anchor → unverifiable, fixed 0.20."""
    other = [_make_judgement(eid="e1", relation="neutral", confidence="low")]

    verdict, conf = decide_verdict(
        supporting=[], refuting=[], other=other,
        **_common_decide_kwargs(),
    )

    assert verdict == "unverifiable"
    assert conf == UNVERIFIABLE_CONFIDENCE == 0.20


def test_decide_verdict_priority3_empty_everything_is_unverifiable() -> None:
    verdict, conf = decide_verdict(
        supporting=[], refuting=[], other=[],
        **_common_decide_kwargs(has_t0_or_t1=False),
    )

    assert verdict == "unverifiable"
    assert conf == 0.20


def test_decide_verdict_priority4_supports_plus_injection_is_needs_review() -> None:
    """P4: supports + injection_risk → needs_review (not likely_correct)."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0", risk_flags=["prompt_injection_pattern"])}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence, has_injection=True,
        ),
    )

    assert verdict == "needs_review"
    # Injection cap dominates: 0.40.
    assert conf == pytest.approx(INJECTION_RISK_CAP)


def test_decide_verdict_priority4_supports_plus_model_disagreement_is_needs_review() -> None:
    """P4: supports + model disagreement → needs_review."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    verdict, _conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id={"e1": _make_evidence("e1", "T0")},
            has_disagreement=True,
        ),
    )

    assert verdict == "needs_review"


def test_decide_verdict_priority5_low_risk_high_capped_is_likely_correct() -> None:
    """P5: low-risk profile, capped ≥0.70 → likely_correct."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    # base 0.90 capped by SINGLE_T0_CAP 0.75 → 0.75 ≥ 0.70 → likely_correct.
    assert verdict == "likely_correct"
    assert conf == pytest.approx(SINGLE_T0_CAP)


def test_decide_verdict_priority5_high_risk_075_threshold() -> None:
    """High-risk + single source → needs_review via P5 guard, regardless of tier.

    Pre-patch this test passed because T1 cap 0.70 fell short of the 0.75
    high-risk auto-confirm threshold. Post-patch the P5 guard short-circuits
    earlier: high/critical + len(supporting) == 1 returns needs_review without
    consulting the threshold at all. The assertion shape is unchanged.
    """
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T1")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            profile=_make_profile(risk_level="high"),
        ),
    )

    # base 0.90 capped by SINGLE_T1_CAP 0.70 → < 0.75 high-risk threshold,
    # ≥ 0.50 review threshold → needs_review.
    assert verdict == "needs_review"
    assert conf == pytest.approx(SINGLE_T1_CAP)


def test_decide_verdict_priority5_medium_capped_is_needs_review() -> None:
    """Capped ≥0.50 but <0.70 → needs_review."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="medium")]
    evidence = {"e1": _make_evidence("e1", "T2")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    # base 0.70 capped by SINGLE_T2_CAP 0.60 → between 0.50 and 0.70 → needs_review.
    assert verdict == "needs_review"
    assert conf == pytest.approx(SINGLE_T2_CAP)


def test_decide_verdict_priority5_very_low_capped_is_unverifiable() -> None:
    """Capped <0.50 → unverifiable."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="low")]
    evidence = {"e1": _make_evidence("e1", "T3")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence, only_t3=True, has_t0_or_t1=False,
        ),
    )

    # base 0.50 capped by ONLY_T3_CAP 0.45 → <0.50 → unverifiable.
    assert verdict == "unverifiable"
    assert conf == pytest.approx(ONLY_T3_CAP)


def test_decide_verdict_high_risk_single_t0_supports_high_is_needs_review() -> None:
    """P5 guard: high risk + single T0 + supports/high → needs_review (NOT likely_correct).

    Boundary case the closing patch fixes. SINGLE_T0_CAP == 0.75 ==
    _HIGH_RISK_CORRECT_THRESHOLD, so without the explicit single-source guard
    the P6 threshold check (capped >= correct_threshold) leaks this case
    through as likely_correct, in violation of the P0 rule that high-risk
    claims need ≥2 independent sources.
    """
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            profile=_make_profile(risk_level="high"),
        ),
    )

    assert verdict == "needs_review"
    assert conf == pytest.approx(SINGLE_T0_CAP)


def test_decide_verdict_critical_risk_single_t0_supports_high_is_needs_review() -> None:
    """P5 guard applies to critical risk identically."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0")}

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            profile=_make_profile(risk_level="critical"),
        ),
    )

    assert verdict == "needs_review"
    assert conf == pytest.approx(SINGLE_T0_CAP)


def test_decide_verdict_high_risk_two_supports_with_t0_allows_likely_correct() -> None:
    """Regression guard: high risk + ≥2 independent supports + has T0/T1 → P6 path.

    The single-source guard must not over-trigger. With two supports the
    single-source cap doesn't apply, so base 0.90 survives, ≥0.75 high-risk
    threshold → likely_correct.
    """
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T0"),
        "e2": _make_evidence("e2", "T1"),
    }

    verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            profile=_make_profile(risk_level="high"),
        ),
    )

    assert verdict == "likely_correct"
    assert conf == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# 2. Confidence caps — each cap, isolated
# ---------------------------------------------------------------------------


def test_cap_single_t0_supporting_is_075() -> None:
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0")}

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    assert conf == pytest.approx(SINGLE_T0_CAP) == 0.75


def test_cap_single_t1_supporting_is_070() -> None:
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T1")}

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    assert conf == pytest.approx(SINGLE_T1_CAP) == 0.70


def test_cap_single_t2_supporting_is_060() -> None:
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T2")}

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    assert conf == pytest.approx(SINGLE_T2_CAP) == 0.60


def test_cap_two_supports_no_single_source_cap() -> None:
    """Single-source cap is gated by len(supporting) == 1. Two supports skip it."""
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T2"),
        "e2": _make_evidence("e2", "T2"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    # 2× T2 evidence → no single-source cap; base 0.90 untouched.
    assert conf == pytest.approx(0.90)


def test_cap_model_failure_is_060() -> None:
    """Model-failure cap applies even with strong T0 source."""
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T0"),
        "e2": _make_evidence("e2", "T0"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence, has_model_failure=True,
        ),
    )

    assert conf == pytest.approx(MODEL_FAILURE_CAP) == 0.60


def test_cap_supports_plus_other_is_070() -> None:
    """Supports + neutral/insufficient → cap at 0.70."""
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    other = [_make_judgement(eid="e3", relation="neutral", confidence="low")]
    evidence = {
        "e1": _make_evidence("e1", "T0"),
        "e2": _make_evidence("e2", "T0"),
        "e3": _make_evidence("e3", "T0"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=other,
        **_common_decide_kwargs(evidence_by_id=evidence),
    )

    assert conf == pytest.approx(SUPPORTS_PLUS_OTHER_CAP) == 0.70


def test_cap_injection_risk_is_040() -> None:
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T0", risk_flags=["prompt_injection_pattern"]),
        "e2": _make_evidence("e2", "T0"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence, has_injection=True,
        ),
    )

    assert conf == pytest.approx(INJECTION_RISK_CAP) == 0.40


def test_cap_only_t3_sources_is_045() -> None:
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T3"),
        "e2": _make_evidence("e2", "T3"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence, only_t3=True, has_t0_or_t1=False,
        ),
    )

    assert conf == pytest.approx(ONLY_T3_CAP) == 0.45


def test_cap_high_risk_no_t0_or_t1_is_040() -> None:
    """High-risk claim without any T0/T1 source → cap at 0.40."""
    supporting = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]
    evidence = {
        "e1": _make_evidence("e1", "T2"),
        "e2": _make_evidence("e2", "T2"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            profile=_make_profile(risk_level="high"),
            has_t0_or_t1=False,
        ),
    )

    assert conf == pytest.approx(HIGH_RISK_NO_AUTHORITY_CAP) == 0.40


# ---------------------------------------------------------------------------
# 3. Cap stacking — multiple caps stack as MIN, never avg / sum
# ---------------------------------------------------------------------------


def test_cap_stacking_takes_min_not_average() -> None:
    """Model failure (0.60) + injection (0.40) → MIN = 0.40."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    evidence = {"e1": _make_evidence("e1", "T0", risk_flags=["x"])}

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=[],
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            has_injection=True,
            has_model_failure=True,
        ),
    )

    # Caps in play: SINGLE_T0 (0.75), MODEL_FAILURE (0.60), INJECTION (0.40).
    # MIN = 0.40 — NOT averaged to ~0.58.
    assert conf == pytest.approx(0.40)


def test_cap_stacking_with_three_caps_still_takes_min() -> None:
    """Single T2 + supports+other + only_t3 → MIN of {0.60, 0.70, 0.45} = 0.45."""
    supporting = [_make_judgement(eid="e1", relation="supports", confidence="high")]
    other = [_make_judgement(eid="e2", relation="neutral", confidence="low")]
    evidence = {
        "e1": _make_evidence("e1", "T2"),
        "e2": _make_evidence("e2", "T3"),
    }

    _verdict, conf = decide_verdict(
        supporting=supporting, refuting=[], other=other,
        **_common_decide_kwargs(
            evidence_by_id=evidence,
            only_t3=True, has_t0_or_t1=False,
        ),
    )

    # SINGLE_T2 0.60, SUPPORTS_PLUS_OTHER 0.70, ONLY_T3 0.45 → MIN = 0.45.
    assert conf == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# 4. resolve_claim — integration
# ---------------------------------------------------------------------------


def test_resolve_claim_happy_path_two_t0_supports_is_likely_correct() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    e2 = _make_evidence("e2", "T0")
    judgements = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]

    result = resolve_claim([claim], judgements, [e1, e2], _make_profile())

    assert result.verdict == "likely_correct"
    assert result.confidence == pytest.approx(0.90)
    assert result.claim == claim.parent_claim
    assert result.consensus_method == "strictest"


def test_resolve_claim_sources_ordered_supporting_refuting_other() -> None:
    claim = _make_atomic_claim()
    e_sup = _make_evidence("e_sup", "T0")
    e_ref = _make_evidence("e_ref", "T0")
    e_other = _make_evidence("e_other", "T1")
    judgements = [
        # Deliberately out of order in the input list:
        _make_judgement(eid="e_other", relation="neutral", confidence="low"),
        _make_judgement(eid="e_ref", relation="refutes", confidence="high"),
        _make_judgement(eid="e_sup", relation="supports", confidence="high"),
    ]
    evidence = [e_other, e_ref, e_sup]

    result = resolve_claim([claim], judgements, evidence, _make_profile())

    # Order: supporting → refuting → other.
    assert [s.id for s in result.sources] == ["e_sup", "e_ref", "e_other"]
    assert [s.id for s in result.supporting_sources] == ["e_sup"]
    assert [s.id for s in result.refuting_sources] == ["e_ref"]


def test_resolve_claim_dedupes_sources_by_id() -> None:
    """Same evidence judged twice (e.g. by re-call) appears once in sources."""
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    judgements = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        # Second judgement on the same evidence — must not double-emit.
        _make_judgement(eid="e1", relation="supports", confidence="medium"),
    ]

    result = resolve_claim([claim], judgements, [e1], _make_profile())

    assert len(result.sources) == 1
    assert result.sources[0].id == "e1"


def test_resolve_claim_emits_disagreements_when_modelvotes_mix() -> None:
    """A judgement whose ModelVotes carry both supports + refutes → Disagreement."""
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    votes = [
        ModelVote(model_name="m1", relation="supports", confidence="high"),
        ModelVote(model_name="m2", relation="refutes", confidence="high"),
    ]
    judgements = [
        _make_judgement(
            eid="e1", relation="supports", confidence="high",
            model_votes=votes,
        ),
    ]

    result = resolve_claim([claim], judgements, [e1], _make_profile())

    assert len(result.disagreements) == 1
    d = result.disagreements[0]
    assert isinstance(d, Disagreement)
    assert d.atomic_claim_id == "c1"
    assert d.evidence_id == "e1"
    assert d.model_votes == votes
    # Disagreement is a signal — verdict should be needs_review.
    assert result.verdict == "needs_review"


def test_resolve_claim_emits_conflicts_for_supports_plus_refutes_in_one_claim() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    e2 = _make_evidence("e2", "T0")
    judgements = [
        _make_judgement(eid="e1", relation="supports", confidence="high"),
        _make_judgement(eid="e2", relation="refutes", confidence="high"),
    ]

    result = resolve_claim([claim], judgements, [e1, e2], _make_profile())

    assert len(result.conflicts) == 1
    c = result.conflicts[0]
    assert isinstance(c, ConflictReport)
    assert c.atomic_claim_id == "c1"
    assert c.supporting_evidence_ids == ["e1"]
    assert c.refuting_evidence_ids == ["e2"]
    assert result.verdict == "conflicting_sources"


def test_resolve_claim_cross_atomic_refute_makes_parent_likely_error() -> None:
    c1 = _make_atomic_claim("c1", "landing date is correct")
    c2 = _make_atomic_claim("c2", "landing side is near side")
    e1 = _make_evidence("e1", "T0")
    e2 = _make_evidence("e2", "T0")
    judgements = [
        _make_judgement(
            eid="e1", claim_id="c1", relation="supports", confidence="high",
        ),
        _make_judgement(
            eid="e2", claim_id="c2", relation="refutes", confidence="high",
        ),
    ]

    result = resolve_claim([c1, c2], judgements, [e1, e2], _make_profile())

    assert result.verdict == "likely_error"
    assert result.conflicts == []
    assert result.supporting_sources == [e1]
    assert result.refuting_sources == [e2]


def test_resolve_claim_flattens_model_votes_from_all_judgements() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    e2 = _make_evidence("e2", "T0")
    v1 = ModelVote(model_name="m1", relation="supports", confidence="high")
    v2 = ModelVote(model_name="m2", relation="supports", confidence="high")
    v3 = ModelVote(model_name="m1", relation="supports", confidence="medium")
    judgements = [
        _make_judgement(eid="e1", relation="supports", confidence="high",
                        model_votes=[v1, v2]),
        _make_judgement(eid="e2", relation="supports", confidence="high",
                        model_votes=[v3]),
    ]

    result = resolve_claim([claim], judgements, [e1, e2], _make_profile())

    # Flattened in input order: [v1, v2, v3].
    assert result.model_votes == [v1, v2, v3]


def test_resolve_claim_consensus_method_is_strictest() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    judgements = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    result = resolve_claim([claim], judgements, [e1], _make_profile())

    assert result.consensus_method == "strictest"


def test_resolve_claim_attaches_atomic_claims_for_audit() -> None:
    claims = [_make_atomic_claim("c1"), _make_atomic_claim("c2", "another fact")]
    e1 = _make_evidence("e1", "T0")
    judgements = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    result = resolve_claim(claims, judgements, [e1], _make_profile())

    assert result.atomic_claims == claims


def test_resolve_claim_empty_atomic_claims_yields_empty_parent() -> None:
    """No atomic claims → parent_claim defaults to empty string, not crash."""
    e1 = _make_evidence("e1", "T0")
    judgements = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    result = resolve_claim([], judgements, [e1], _make_profile())

    assert result.claim == ""
    assert result.verdict == "likely_correct"


def test_resolve_claim_to_compact_dict_invariant_preserved() -> None:
    """The 5-display-field contract from test_types.py must survive resolver output."""
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    judgements = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    result = resolve_claim([claim], judgements, [e1], _make_profile())
    compact = result.to_compact_dict()

    assert set(compact.keys()) == {"verdict", "confidence", "claim", "sources", "reasoning"}


def test_resolve_claim_reasoning_mentions_flags() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0", risk_flags=["prompt_injection_pattern"])
    votes = [
        ModelVote(model_name="m1", relation="supports", confidence="high"),
        ModelVote(model_name="m2", relation="refutes", confidence="high"),
    ]
    judgements = [
        _make_judgement(
            eid="e1", relation="supports", confidence="high",
            model_votes=votes,
        ),
    ]

    result = resolve_claim([claim], judgements, [e1], _make_profile())

    assert "injection_risk" in result.reasoning
    assert "model_disagreement" in result.reasoning


def test_resolve_claim_model_failure_in_votes_caps_confidence() -> None:
    """A ModelVote with non-None error triggers MODEL_FAILURE_CAP."""
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    e2 = _make_evidence("e2", "T0")
    votes_with_error = [
        ModelVote(model_name="m1", relation="supports", confidence="high"),
        ModelVote(model_name="m2", relation=None, confidence=None,
                  error="transport_error"),
    ]
    judgements = [
        _make_judgement(eid="e1", relation="supports", confidence="high",
                        model_votes=votes_with_error),
        _make_judgement(eid="e2", relation="supports", confidence="high"),
    ]

    result = resolve_claim([claim], judgements, [e1, e2], _make_profile())

    # MODEL_FAILURE_CAP wins.
    assert result.confidence == pytest.approx(MODEL_FAILURE_CAP)


def test_resolve_claim_signal_detection_ignores_unjudged_evidence_injection() -> None:
    """MEDIUM-fix: unjudged evidence with risk_flags must NOT trigger has_injection.

    Retrieval may return more spans than verification judged. Without the
    judged-evidence filter, a stray unjudged span carrying a risk_flag would
    leak into the injection signal and force needs_review even when the
    judged sources are clean. Two T0 supports + one unjudged T0 injection
    span → likely_correct, NOT needs_review.
    """
    claim = _make_atomic_claim()
    judged_a = _make_evidence("ev_judged_a", "T0")
    judged_b = _make_evidence("ev_judged_b", "T0")
    unjudged_risky = _make_evidence(
        "ev_unjudged", "T0", risk_flags=["prompt_injection_pattern"],
    )
    judgements = [
        _make_judgement(eid="ev_judged_a", relation="supports", confidence="high"),
        _make_judgement(eid="ev_judged_b", relation="supports", confidence="high"),
    ]

    result = resolve_claim(
        [claim], judgements,
        [judged_a, judged_b, unjudged_risky],
        _make_profile(),
    )

    # Injection signal must come only from judged evidence — unjudged risky
    # span is invisible to the verdict.
    assert result.verdict == "likely_correct"
    assert "injection_risk" not in result.reasoning


def test_resolve_claim_signal_detection_ignores_unjudged_evidence_authority() -> None:
    """MEDIUM-fix: unjudged T0/T1 must NOT satisfy has_t0_or_t1 for the authority cap.

    High-risk claim whose judged evidence is all T2 should still trip
    HIGH_RISK_NO_AUTHORITY_CAP, even if an unjudged T0 is sitting in the
    evidence pool. The cap is about whether the *judged* sources include
    authoritative ones — unjudged spans don't count.
    """
    claim = _make_atomic_claim()
    judged_t2_a = _make_evidence("ev_t2_a", "T2")
    judged_t2_b = _make_evidence("ev_t2_b", "T2")
    unjudged_t0 = _make_evidence("ev_unjudged_t0", "T0")
    judgements = [
        _make_judgement(eid="ev_t2_a", relation="supports", confidence="high"),
        _make_judgement(eid="ev_t2_b", relation="supports", confidence="high"),
    ]

    result = resolve_claim(
        [claim], judgements,
        [judged_t2_a, judged_t2_b, unjudged_t0],
        _make_profile(risk_level="high"),
    )

    # No T0/T1 among judged evidence → HIGH_RISK_NO_AUTHORITY_CAP applies.
    # Two supports → no single-source cap. base 0.90 → capped at 0.40.
    assert result.confidence == pytest.approx(HIGH_RISK_NO_AUTHORITY_CAP)
    # capped 0.40 < 0.50 review threshold → unverifiable.
    assert result.verdict == "unverifiable"


# ---------------------------------------------------------------------------
# 5. apply_failure_matrix — 11 documented cases + priority
# ---------------------------------------------------------------------------


def test_failure_matrix_both_retrievals_down_is_unverifiable_010() -> None:
    result = apply_failure_matrix({
        "tavily_available": False,
        "wiki_available": False,
        "claim": "x",
    })

    assert result.verdict == "unverifiable"
    assert result.confidence == pytest.approx(0.10)
    assert "tavily_down" in result.reasoning
    assert "wiki_down" in result.reasoning


def test_failure_matrix_both_models_down_is_unverifiable_010() -> None:
    result = apply_failure_matrix({
        "deepseek_available": False,
        "mimo_available": False,
        "claim": "x",
    })

    assert result.verdict == "unverifiable"
    assert result.confidence == pytest.approx(0.10)
    assert "deepseek_down" in result.reasoning
    assert "mimo_down" in result.reasoning


def test_failure_matrix_has_conflict_is_conflicting_sources_030() -> None:
    result = apply_failure_matrix({"has_conflict": True, "claim": "x"})

    assert result.verdict == "conflicting_sources"
    assert result.confidence == pytest.approx(CONFLICT_CONFIDENCE)


def test_failure_matrix_tavily_down_only_is_needs_review_050() -> None:
    result = apply_failure_matrix({"tavily_available": False, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(0.50)
    assert "tavily_down" in result.reasoning


def test_failure_matrix_wiki_down_only_is_needs_review_050() -> None:
    result = apply_failure_matrix({"wiki_available": False, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(0.50)
    assert "wiki_down" in result.reasoning


def test_failure_matrix_deepseek_down_only_is_needs_review_060() -> None:
    result = apply_failure_matrix({"deepseek_available": False, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(MODEL_FAILURE_CAP)


def test_failure_matrix_mimo_down_only_is_needs_review_060() -> None:
    result = apply_failure_matrix({"mimo_available": False, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(MODEL_FAILURE_CAP)


def test_failure_matrix_models_disagree_is_needs_review_040() -> None:
    result = apply_failure_matrix({"models_disagree": True, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(INJECTION_RISK_CAP)


def test_failure_matrix_injection_risk_is_needs_review_040() -> None:
    result = apply_failure_matrix({"has_injection_risk": True, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(INJECTION_RISK_CAP)


def test_failure_matrix_only_t3_sources_is_needs_review_045() -> None:
    result = apply_failure_matrix({"only_t3_sources": True, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(ONLY_T3_CAP)


def test_failure_matrix_high_risk_single_source_is_needs_review_040() -> None:
    result = apply_failure_matrix({"high_risk_single_source": True, "claim": "x"})

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(HIGH_RISK_NO_AUTHORITY_CAP)


def test_failure_matrix_multiple_flags_takes_min_of_caps() -> None:
    """Tavily down (0.50) + injection (0.40) + only_t3 (0.45) → MIN = 0.40."""
    result = apply_failure_matrix({
        "tavily_available": False,
        "has_injection_risk": True,
        "only_t3_sources": True,
        "claim": "x",
    })

    assert result.verdict == "needs_review"
    assert result.confidence == pytest.approx(INJECTION_RISK_CAP)


def test_failure_matrix_no_flags_is_unverifiable_000() -> None:
    """Defense against mis-call: every *_available defaults True, no flags set."""
    result = apply_failure_matrix({"claim": "x"})

    assert result.verdict == "unverifiable"
    assert result.confidence == pytest.approx(0.0)
    assert "without any degradation flags" in result.reasoning


def test_failure_matrix_preserves_claim_text() -> None:
    result = apply_failure_matrix({
        "claim": "中国人口约14亿人",
        "tavily_available": False,
    })

    assert result.claim == "中国人口约14亿人"


def test_failure_matrix_preserves_partial_sources() -> None:
    """If retrieval partially succeeded, surface what we did get."""
    e1 = _make_evidence("e1", "T1")
    result = apply_failure_matrix({
        "claim": "x",
        "tavily_available": False,
        "sources": [e1],
    })

    assert result.sources == [e1]


def test_failure_matrix_priority_retrievals_dominate_conflict() -> None:
    """P1 retrievals-down beats P3 has_conflict in priority."""
    result = apply_failure_matrix({
        "tavily_available": False,
        "wiki_available": False,
        "has_conflict": True,
        "claim": "x",
    })

    # Both retrievals down → unverifiable wins, even with has_conflict set.
    assert result.verdict == "unverifiable"
    assert result.confidence == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 6. Return-type sanity — every code path returns a usable VerificationResult
# ---------------------------------------------------------------------------


def test_resolve_claim_returns_verification_result_instance() -> None:
    claim = _make_atomic_claim()
    e1 = _make_evidence("e1", "T0")
    judgements = [_make_judgement(eid="e1", relation="supports", confidence="high")]

    result = resolve_claim([claim], judgements, [e1], _make_profile())

    assert isinstance(result, VerificationResult)


def test_failure_matrix_returns_verification_result_instance() -> None:
    result = apply_failure_matrix({"tavily_available": False, "claim": "x"})

    assert isinstance(result, VerificationResult)
