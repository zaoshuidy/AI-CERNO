"""Unit tests for veritas.observability — CostBreakdown / AuditStep / TraceRecorder."""

from __future__ import annotations

from veritas import (
    AuditStep,
    CostBreakdown,
    TraceRecorder,
    record_step,
)

# ---------------------------------------------------------------------------
# CostBreakdown
# ---------------------------------------------------------------------------

def test_cost_breakdown_defaults_are_zero() -> None:
    c = CostBreakdown()
    assert c.input_tokens == 0
    assert c.output_tokens == 0
    assert c.llm_calls == 0
    assert c.retrieval_calls == 0
    assert c.cache_hits == 0


def test_cost_breakdown_add_llm_accumulates() -> None:
    c = CostBreakdown()
    c.add_llm(input_tokens=100, output_tokens=50)
    c.add_llm(input_tokens=200, output_tokens=80)
    assert c.input_tokens == 300
    assert c.output_tokens == 130
    assert c.llm_calls == 2


def test_cost_breakdown_add_retrieval_tracks_cache_hits() -> None:
    c = CostBreakdown()
    c.add_retrieval(cache_hit=False)
    c.add_retrieval(cache_hit=True)
    c.add_retrieval(cache_hit=True)
    assert c.retrieval_calls == 3
    assert c.cache_hits == 2


def test_cost_breakdown_independent_instances_do_not_share_state() -> None:
    a = CostBreakdown()
    b = CostBreakdown()
    a.add_llm(10, 5)
    assert b.input_tokens == 0
    assert b.output_tokens == 0
    assert b.llm_calls == 0


# ---------------------------------------------------------------------------
# AuditStep / record_step
# ---------------------------------------------------------------------------

def test_audit_step_auto_timestamps() -> None:
    step = record_step("understanding", "claim=hello", "claim_type=entity_fact")
    assert step.name == "understanding"
    assert step.input_summary == "claim=hello"
    assert step.output_summary == "claim_type=entity_fact"
    assert step.timestamp  # non-empty ISO string
    assert "T" in step.timestamp


def test_audit_step_explicit_timestamp_kept() -> None:
    step = AuditStep(
        name="custom",
        input_summary="in",
        output_summary="out",
        timestamp="2026-05-12T00:00:00+00:00",
    )
    assert step.timestamp == "2026-05-12T00:00:00+00:00"


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------

def test_trace_recorder_starts_empty() -> None:
    rec = TraceRecorder()
    assert rec.steps == []


def test_trace_recorder_record_appends_and_returns_step() -> None:
    rec = TraceRecorder()
    a = rec.record("step_a", "in_a", "out_a")
    b = rec.record("step_b", "in_b", "out_b")
    assert rec.steps == [a, b]
    assert a.name == "step_a"
    assert b.name == "step_b"


def test_trace_recorder_extend_supports_external_steps() -> None:
    rec = TraceRecorder()
    rec.record("first", "in", "out")
    external = [
        record_step("second", "in", "out"),
        record_step("third", "in", "out"),
    ]
    rec.extend(external)
    assert [s.name for s in rec.steps] == ["first", "second", "third"]


def test_trace_recorder_independent_instances_do_not_share_state() -> None:
    rec_a = TraceRecorder()
    rec_b = TraceRecorder()
    rec_a.record("only_in_a", "in", "out")
    assert rec_b.steps == []
