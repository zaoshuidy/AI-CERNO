"""Multi-model consensus: OpenAI-compatible adapter + StrictestStrategy.

Hard contract:
- temperature = 0.1, hardcoded, never exposed to callers.
- JSON parse / schema fail → retry once with an explicit retry prompt → fail.
- Per-model failures are recorded as ModelVote(error=...), never silently dropped.
- StrictestStrategy implements the 14-row merge table from design v0.5 §9.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import jsonschema

from cerno.types import (
    Disagreement,
    LLMProvider,
    ModelVote,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hardcoded sampling temperature for all LLM calls. Not exposed to callers.
TEMPERATURE: float = 0.0

#: Retry prompt appended to the user message when the first response fails
#: JSON parsing or schema validation.
JSON_RETRY_PROMPT_TEMPLATE: str = (
    "上一次输出未通过 JSON Schema 校验。"
    "请严格按以下 schema 输出，不要附加解释文字：\n"
    "{schema_json}"
)

ConsensusRelation = Literal[
    "supports", "refutes", "neutral", "insufficient", "conflict"
]
ConsensusConfidence = Literal["high", "medium", "low", "abstain"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Outcome of one model call. `parsed` is non-None iff the JSON validated."""

    model_name: str
    parsed: dict[str, Any] | None
    raw_text: str
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0


@dataclass
class ConsensusResult:
    """Output of StrictestStrategy.merge — what verification.py will turn into
    an EvidenceJudgement (with the conflict / abstain case routed specially).
    """

    relation: ConsensusRelation
    confidence: ConsensusConfidence
    quote: str
    reason: str
    model_votes: list[ModelVote] = field(default_factory=list)
    disagreement: Disagreement | None = None
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Adapter interface (duck-typed; tests inject simple stubs)
# ---------------------------------------------------------------------------

class _AdapterLike(Protocol):
    """Duck-typed interface MultiModelConsensus expects."""

    @property
    def name(self) -> str: ...

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# OpenAICompatibleAdapter
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter:
    """Wraps openai.OpenAI client. One adapter per LLMProvider.

    Subclass and override `_raw_chat` for testing — or inject a fake adapter
    matching the _AdapterLike protocol directly into MultiModelConsensus.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self._client: Any = None  # lazy-constructed

    @property
    def name(self) -> str:
        return self.provider.name

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Local import keeps openai out of import time for callers that
            # inject their own adapter and never construct one of these.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.timeout,
            )
        return self._client

    def _raw_chat(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Send one chat completion. Returns (raw_text, input_tokens, output_tokens)."""
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.provider.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=self.provider.max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return text, in_tok, out_tok

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> LLMResponse:
        """Send a JSON-schema-constrained request; retry once on parse/schema fail."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw_text, in_tok, out_tok = self._raw_chat(messages)
        except Exception as exc:  # noqa: BLE001 — transport failure is opaque
            return LLMResponse(
                model_name=self.name,
                parsed=None,
                raw_text="",
                error=f"transport_error: {exc.__class__.__name__}: {exc}",
            )

        parsed, parse_err = _parse_and_validate(raw_text, schema)
        if parse_err is None and parsed is not None:
            return LLMResponse(
                model_name=self.name,
                parsed=parsed,
                raw_text=raw_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        # --- retry once with explicit retry prompt ---
        retry_user = user + "\n\n" + JSON_RETRY_PROMPT_TEMPLATE.format(
            schema_json=json.dumps(schema, ensure_ascii=False)
        )
        retry_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": retry_user},
        ]
        try:
            raw_text_2, in_tok_2, out_tok_2 = self._raw_chat(retry_messages)
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(
                model_name=self.name,
                parsed=None,
                raw_text=raw_text,
                error=f"transport_error_on_retry: {exc.__class__.__name__}: {exc}",
                input_tokens=in_tok,
                output_tokens=out_tok,
                retries=1,
            )

        parsed_2, parse_err_2 = _parse_and_validate(raw_text_2, schema)
        if parse_err_2 is None and parsed_2 is not None:
            return LLMResponse(
                model_name=self.name,
                parsed=parsed_2,
                raw_text=raw_text_2,
                input_tokens=in_tok + in_tok_2,
                output_tokens=out_tok + out_tok_2,
                retries=1,
            )
        return LLMResponse(
            model_name=self.name,
            parsed=None,
            raw_text=raw_text_2,
            error=f"json_parse_failed: {parse_err_2}",
            input_tokens=in_tok + in_tok_2,
            output_tokens=out_tok + out_tok_2,
            retries=1,
        )


def _parse_and_validate(
    raw_text: str, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON and validate against schema. Returns (parsed, error_message)."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "top_level_not_object"
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as exc:
        return None, f"schema_violation: {exc.message}"
    return parsed, None


# ---------------------------------------------------------------------------
# MultiModelConsensus
# ---------------------------------------------------------------------------

class MultiModelConsensus:
    """Drives N adapters in parallel (P0 default: DeepSeek + MiMo) and merges
    them through StrictestStrategy.
    """

    def __init__(self, adapters: list[_AdapterLike]) -> None:
        if not adapters:
            raise ValueError("MultiModelConsensus requires at least one adapter")
        self.adapters = list(adapters)

    @property
    def has_single_adapter(self) -> bool:
        """True when only one model is configured — confidence is capped accordingly."""
        return len(self.adapters) == 1

    def invoke_all(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, LLMResponse]:
        """Call every adapter in parallel and collect responses by adapter name."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(self.adapters)) as ex:
            futures = {
                ex.submit(adapter.chat_json, system, user, schema): adapter.name
                for adapter in self.adapters
            }
            results: dict[str, LLMResponse] = {}
            for fut, name in futures.items():
                try:
                    results[name] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[name] = LLMResponse(
                        model_name=name,
                        parsed=None,
                        raw_text="",
                        error=f"executor_error: {exc.__class__.__name__}: {exc}",
                    )
        return results

    def judge(
        self,
        *,
        step: str,
        claim: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        high_risk: bool = False,
    ) -> ConsensusResult:
        """Run all adapters, translate to ModelVote list, run StrictestStrategy."""
        del step, claim  # reserved for audit-side use; carried by callers' TraceRecorder
        responses = self.invoke_all(system, user, schema)
        votes = [_response_to_vote(resp) for resp in responses.values()]
        # If we have only one adapter configured, the StrictestStrategy applies
        # its single-model rule (confidence ≤ low).
        single_model = self.has_single_adapter
        result = StrictestStrategy.merge(
            votes, high_risk=high_risk, single_model=single_model
        )
        # Accumulate usage from all adapters, including failures.
        total_in = sum(r.input_tokens for r in responses.values())
        total_out = sum(r.output_tokens for r in responses.values())
        return ConsensusResult(
            relation=result.relation,
            confidence=result.confidence,
            quote=result.quote,
            reason=result.reason,
            model_votes=result.model_votes,
            disagreement=result.disagreement,
            input_tokens=total_in,
            output_tokens=total_out,
        )


def _response_to_vote(resp: LLMResponse) -> ModelVote:
    """Translate LLMResponse to ModelVote. Failures keep relation/confidence None."""
    if resp.error is not None or resp.parsed is None:
        return ModelVote(
            model_name=resp.model_name,
            relation=None,
            confidence=None,
            error=resp.error or "no_parsed_payload",
            raw_response=resp.raw_text or None,
        )
    parsed = resp.parsed
    relation = parsed.get("relation")
    confidence = parsed.get("confidence")
    reason = parsed.get("reason", "")
    quote = parsed.get("quote", "")
    if relation not in {"supports", "refutes", "neutral", "insufficient"}:
        return ModelVote(
            model_name=resp.model_name,
            relation=None,
            confidence=None,
            reason=reason,
            error=f"invalid_relation: {relation!r}",
            raw_response=resp.raw_text or None,
        )
    if confidence not in {"high", "medium", "low"}:
        return ModelVote(
            model_name=resp.model_name,
            relation=None,
            confidence=None,
            reason=reason,
            error=f"invalid_confidence: {confidence!r}",
            raw_response=resp.raw_text or None,
        )
    vote = ModelVote(
        model_name=resp.model_name,
        relation=relation,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        reason=reason,
        quote=quote,
    )
    return vote


# ---------------------------------------------------------------------------
# StrictestStrategy
# ---------------------------------------------------------------------------

class StrictestStrategy:
    """The 14-row merge table from design v0.5 §9, plus the failure rules."""

    @staticmethod
    def merge(
        votes: list[ModelVote],
        *,
        high_risk: bool = False,
        single_model: bool = False,
    ) -> ConsensusResult:
        if not votes:
            return ConsensusResult(
                relation="insufficient",
                confidence="abstain",
                quote="",
                reason="no_model_votes",
                model_votes=[],
            )

        # ---- Failure handling -------------------------------------------------
        failures = [v for v in votes if v.relation is None or v.confidence is None]
        valid = [v for v in votes if v.relation is not None and v.confidence is not None]

        # Pick a representative quote (first non-empty from valid models).
        def _quote_of(v: ModelVote) -> str:
            return v.quote or ""

        if not valid:
            # All models failed → insufficient / abstain
            return ConsensusResult(
                relation="insufficient",
                confidence="abstain",
                quote="",
                reason="all_models_failed",
                model_votes=votes,
            )

        if len(valid) == 1 and len(votes) > 1:
            # One model succeeded, the other(s) failed → cap to low, with disagreement
            vote = valid[0]
            disagreement = Disagreement(
                atomic_claim_id="",  # filled in by caller (verification.py)
                evidence_id="",
                model_votes=votes,
                summary=f"one model failed: {[f.model_name for f in failures]}",
            )
            confidence: ConsensusConfidence = (
                "low" if not high_risk else "low"
            )
            return ConsensusResult(
                relation=vote.relation,  # type: ignore[arg-type]
                confidence=confidence,
                quote=_quote_of(vote),
                reason="single_valid_model_after_failure",
                model_votes=votes,
                disagreement=disagreement,
            )

        # ---- Single-adapter configuration ------------------------------------
        if single_model and len(valid) == 1 and len(votes) == 1:
            vote = valid[0]
            return ConsensusResult(
                relation=vote.relation,  # type: ignore[arg-type]
                confidence="low",  # cap when only one model configured
                quote=_quote_of(vote),
                reason="single_model_configured_cap_low",
                model_votes=votes,
            )

        # ---- Two-or-more valid models: apply the 14-row table ----------------
        # P0 default is dual-model; for >2 we pick the strictest pair (first two
        # alphabetically by name to keep the merge deterministic — verification.py
        # is responsible for ordering adapters consistently).
        if len(valid) > 2:
            # We expect P0 callers to pass exactly 2 adapters; if not, fall back
            # to the most pessimistic relation.
            return _merge_n(valid, votes, high_risk=high_risk)

        return _merge_two(valid[0], valid[1], votes, high_risk=high_risk)


def _quote_of(v: ModelVote) -> str:
    return v.quote or ""


def _merge_two(
    a: ModelVote, b: ModelVote, all_votes: list[ModelVote], *, high_risk: bool
) -> ConsensusResult:
    rel_a, rel_b = a.relation, b.relation
    quote_a = _quote_of(a) or _quote_of(b)

    relations = {rel_a, rel_b}

    # Row 1: both supports → supports/high (with high-risk cap if any failure)
    if rel_a == "supports" and rel_b == "supports":
        return ConsensusResult(
            relation="supports",
            confidence="high",
            quote=quote_a,
            reason="dual_supports",
            model_votes=all_votes,
        )

    # Rows 6-7: supports vs refutes → conflict / abstain
    if "supports" in relations and "refutes" in relations:
        disagreement = Disagreement(
            atomic_claim_id="",
            evidence_id="",
            model_votes=all_votes,
            summary="supports_vs_refutes",
        )
        return ConsensusResult(
            relation="conflict",
            confidence="abstain",
            quote=quote_a,
            reason="supports_vs_refutes",
            model_votes=all_votes,
            disagreement=disagreement,
        )

    # Rows 8-11: refutes + (neutral|insufficient) → refutes/(high|medium)
    if "refutes" in relations:
        # Refutes wins. If high_risk and there was any failure → cap medium.
        conf: ConsensusConfidence = "high"
        if high_risk and any(v.error for v in all_votes):
            conf = "medium"
        return ConsensusResult(
            relation="refutes",
            confidence=conf,
            quote=quote_a,
            reason="refutes_with_weaker_partner",
            model_votes=all_votes,
        )

    # Rows 2-5: supports + (neutral|insufficient) → supports/medium
    if "supports" in relations and (
        "neutral" in relations or "insufficient" in relations
    ):
        return ConsensusResult(
            relation="supports",
            confidence="medium",
            quote=quote_a,
            reason="supports_with_weaker_partner",
            model_votes=all_votes,
        )

    # Rows 12-14: neutral / insufficient combinations
    if rel_a == "neutral" and rel_b == "neutral":
        return ConsensusResult(
            relation="neutral",
            confidence="low",
            quote=quote_a,
            reason="dual_neutral",
            model_votes=all_votes,
        )
    if rel_a == "insufficient" and rel_b == "insufficient":
        return ConsensusResult(
            relation="insufficient",
            confidence="low",
            quote=quote_a,
            reason="dual_insufficient",
            model_votes=all_votes,
        )
    # neutral + insufficient → neutral/low (the more permissive of the two)
    return ConsensusResult(
        relation="neutral",
        confidence="low",
        quote=quote_a,
        reason="neutral_insufficient_mix",
        model_votes=all_votes,
    )


def _merge_n(
    valid: list[ModelVote], all_votes: list[ModelVote], *, high_risk: bool
) -> ConsensusResult:
    """Fallback for >2 valid models — pick the most pessimistic outcome."""
    relations = {v.relation for v in valid}
    if "supports" in relations and "refutes" in relations:
        return ConsensusResult(
            relation="conflict",
            confidence="abstain",
            quote=_quote_of(valid[0]),
            reason="supports_vs_refutes_in_multi_model",
            model_votes=all_votes,
            disagreement=Disagreement(
                atomic_claim_id="",
                evidence_id="",
                model_votes=all_votes,
                summary="supports_vs_refutes_multi",
            ),
        )
    if "refutes" in relations:
        conf: ConsensusConfidence = "high"
        if high_risk and any(v.error for v in all_votes):
            conf = "medium"
        return ConsensusResult(
            relation="refutes",
            confidence=conf,
            quote=_quote_of(valid[0]),
            reason="refutes_in_multi_model",
            model_votes=all_votes,
        )
    if all(v.relation == "supports" for v in valid):
        return ConsensusResult(
            relation="supports",
            confidence="high",
            quote=_quote_of(valid[0]),
            reason="all_supports_multi_model",
            model_votes=all_votes,
        )
    if "supports" in relations:
        return ConsensusResult(
            relation="supports",
            confidence="medium",
            quote=_quote_of(valid[0]),
            reason="supports_with_weaker_partners_multi_model",
            model_votes=all_votes,
        )
    return ConsensusResult(
        relation="neutral",
        confidence="low",
        quote=_quote_of(valid[0]),
        reason="no_strong_relation_in_multi_model",
        model_votes=all_votes,
    )
