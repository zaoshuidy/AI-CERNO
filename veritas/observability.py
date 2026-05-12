"""Observability primitives: per-request cost and audit trace.

Both AuditTrace and CostBreakdown are per-VerificationResult — no global
accumulation, no shared mutable state across requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CostBreakdown:
    """Per-request cost & call counters.

    Money cost is not computed in P0; callers can aggregate tokens externally.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    cache_hits: int = 0

    def add_llm(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1

    def add_retrieval(self, *, cache_hit: bool = False) -> None:
        self.retrieval_calls += 1
        if cache_hit:
            self.cache_hits += 1


@dataclass
class AuditStep:
    """One step in the audit trace.

    `input_summary` and `output_summary` are intentionally short strings —
    callers must summarize, not dump raw payloads, to keep traces readable.
    """

    name: str
    input_summary: str
    output_summary: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def record_step(name: str, input_summary: str, output_summary: str) -> AuditStep:
    """Factory for an AuditStep with current UTC timestamp."""
    return AuditStep(
        name=name,
        input_summary=input_summary,
        output_summary=output_summary,
    )


@dataclass
class TraceRecorder:
    """Container for audit steps belonging to one VerificationResult."""

    steps: list[AuditStep] = field(default_factory=list)

    def record(self, name: str, input_summary: str, output_summary: str) -> AuditStep:
        step = record_step(name, input_summary, output_summary)
        self.steps.append(step)
        return step

    def extend(self, steps: list[AuditStep]) -> None:
        self.steps.extend(steps)
