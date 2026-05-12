"""Stage 1 of the verification pipeline: claim understanding.

Three public functions, all LLM-first with rule-based fallback:

- ``build_claim_profile``: infer ClaimProfile (is_checkable / claim_type / domain /
  risk_level / required_evidence / strict_mode).
- ``decompose_claim``: split parent claim into ≤3 AtomicClaims (each carrying
  the original phrasing in ``original_span``).
- ``build_retrieval_plan``: deterministic derivation from profile + atomic_claims
  (no LLM call). Locks ``source_targets`` to the P0 pair
  ``["tavily", "wikipedia_zh"]``.

Risk reconciliation rule: when ``request.risk_hint`` differs from the LLM-inferred
risk, the stricter level wins and ``risk_level_source`` becomes
``"max_of_both"``.
"""

from __future__ import annotations

import re
from typing import Any

from veritas.consensus import MultiModelConsensus
from veritas.observability import CostBreakdown
from veritas.types import (
    AtomicClaim,
    ClaimProfile,
    ClaimType,
    RetrievalPlan,
    RiskLevel,
    RiskLevelSource,
    VerificationRequest,
)

# ---------------------------------------------------------------------------
# Constants — claim_type / risk / source policy
# ---------------------------------------------------------------------------

VALID_CLAIM_TYPES: tuple[ClaimType, ...] = (
    "entity_fact",
    "temporal_fact",
    "quantitative_fact",
    "quotation",
    "domain_fact",
    "other",
)

VALID_RISK_LEVELS: tuple[RiskLevel, ...] = ("low", "medium", "high", "critical")

_RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

#: P0: every RetrievalPlan uses this exact pair of source targets. Hardcoded.
P0_SOURCE_TARGETS: tuple[str, ...] = ("tavily", "wikipedia_zh")

#: Hard cap on the number of atomic claims a single parent claim can produce.
MAX_ATOMIC_CLAIMS: int = 3

#: Domains that always need a freshness check (design §11.5). ``law`` and
#: ``legal`` are both accepted because the design draft says ``law`` but the
#: in-repo prompts / domain_hint convention says ``legal``.
_FRESHNESS_DOMAINS: frozenset[str] = frozenset(
    {"news", "policy", "law", "legal", "finance"}
)


# ---------------------------------------------------------------------------
# Schemas (jsonschema-validated by the adapter inside consensus.py)
# ---------------------------------------------------------------------------

PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "is_checkable",
        "claim_type",
        "domain",
        "risk_level",
        "required_evidence",
        "strict_mode",
        "reason",
    ],
    "properties": {
        "is_checkable": {"type": "boolean"},
        "claim_type": {"enum": list(VALID_CLAIM_TYPES)},
        "domain": {"type": "string"},
        "risk_level": {"enum": list(VALID_RISK_LEVELS)},
        "required_evidence": {"type": "array", "items": {"type": "string"}},
        "strict_mode": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

DECOMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["atomic_claims"],
    "properties": {
        "atomic_claims": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "text",
                    "original_span",
                    "check_priority",
                    "required_evidence_type",
                ],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "original_span": {"type": "string", "minLength": 1},
                    "check_priority": {"type": "integer", "minimum": 1},
                    "required_evidence_type": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROFILE_SYSTEM_PROMPT = (
    "你是事实核查系统的“声明分析师”。给定一条声明,严格按 JSON Schema 输出:\n"
    "- is_checkable: 是否可被事实核查\n"
    "- claim_type: entity_fact / temporal_fact / quantitative_fact / quotation / "
    "domain_fact / other 之一\n"
    "- domain: 例如 science / legal / medical / policy / general\n"
    "- risk_level: low / medium / high / critical\n"
    "- required_evidence: 所需证据类型列表 (如 [\"date\",\"official\"])\n"
    "- strict_mode: 对高风险或需要权威源的场景置 true\n"
    "- reason: 简短判断依据\n"
    "\n"
    "只输出 JSON,不要附加解释文字。"
)

DECOMPOSE_SYSTEM_PROMPT = (
    "你是事实核查系统的“声明拆解器”。给定一条声明,最多拆成 3 条原子声明。\n"
    "每条原子声明必须:\n"
    "- 能被独立检索 / 验证\n"
    "- 保留原声明中对应的原始片段 (original_span,非空)\n"
    "- check_priority 为正整数,1 表示最关键\n"
    "\n"
    "只输出 JSON: { \"atomic_claims\": [...] }。不要附加解释文字。"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_claim_profile(
    request: VerificationRequest,
    consensus: MultiModelConsensus,
    *,
    cost: CostBreakdown | None = None,
) -> ClaimProfile:
    """LLM-first ClaimProfile inference with rule-based fallback.

    Returns a profile where ``risk_level`` reconciles ``request.risk_hint``
    against the LLM-inferred risk by taking the stricter of the two.
    """
    user_msg = _profile_user_message(request)
    parsed = _first_valid_parsed(
        consensus, PROFILE_SYSTEM_PROMPT, user_msg, PROFILE_SCHEMA, cost=cost
    )
    if parsed is None:
        return _rule_profile(request)
    return _profile_from_parsed(parsed, request)


def decompose_claim(
    request: VerificationRequest,
    profile: ClaimProfile,
    consensus: MultiModelConsensus,
    *,
    cost: CostBreakdown | None = None,
) -> list[AtomicClaim]:
    """LLM-first decomposition into ≤3 AtomicClaims with rule-based fallback."""
    user_msg = _decompose_user_message(request, profile)
    parsed = _first_valid_parsed(
        consensus, DECOMPOSE_SYSTEM_PROMPT, user_msg, DECOMPOSE_SCHEMA, cost=cost
    )
    if parsed is None:
        return _rule_decompose(request)
    return _atomic_claims_from_parsed(parsed, request)


def build_retrieval_plan(
    profile: ClaimProfile,
    atomic_claims: list[AtomicClaim],
) -> RetrievalPlan:
    """Deterministic plan derived from profile + atomic_claims. No LLM call."""
    queries = [c.text for c in atomic_claims]
    return RetrievalPlan(
        queries=queries,
        source_targets=list(P0_SOURCE_TARGETS),
        min_independent_sources=2,
        allow_discovery_sources=not profile.strict_mode,
        allow_single_source_medium=True,
        require_official_source=profile.risk_level in {"high", "critical"},
        require_freshness_check=_should_check_freshness(profile),
    )


# ---------------------------------------------------------------------------
# Internal: LLM call helpers
# ---------------------------------------------------------------------------

def _first_valid_parsed(
    consensus: MultiModelConsensus,
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    cost: CostBreakdown | None = None,
) -> dict[str, Any] | None:
    """Return the first model's parsed payload that survived schema validation.

    The adapter inside ``consensus`` already enforces JSON parsing + schema
    validation; this just picks the first survivor or returns None for fallback.
    """
    responses = consensus.invoke_all(system, user, schema)
    if cost is not None:
        total_in = sum(r.input_tokens for r in responses.values())
        total_out = sum(r.output_tokens for r in responses.values())
        cost.add_llm(total_in, total_out)
    for resp in responses.values():
        if resp.error is None and resp.parsed is not None:
            return resp.parsed
    return None


def _profile_user_message(request: VerificationRequest) -> str:
    context = request.context or ""
    domain_hint = request.domain_hint or ""
    return (
        f"声明: {request.claim}\n"
        f"上下文: {context}\n"
        f"领域提示: {domain_hint}\n"
    )


def _decompose_user_message(request: VerificationRequest, profile: ClaimProfile) -> str:
    return (
        f"父声明: {request.claim}\n"
        f"领域: {profile.domain}\n"
        f"声明类型: {profile.claim_type}\n"
        f"风险等级: {profile.risk_level}\n"
    )


# ---------------------------------------------------------------------------
# Internal: parse LLM responses into typed objects
# ---------------------------------------------------------------------------

def _profile_from_parsed(
    parsed: dict[str, Any], request: VerificationRequest
) -> ClaimProfile:
    claim_type = parsed["claim_type"]
    if claim_type not in VALID_CLAIM_TYPES:
        claim_type = "other"

    llm_risk = parsed["risk_level"]
    if llm_risk not in VALID_RISK_LEVELS:
        llm_risk = "low"

    final_risk, source = _reconcile_risk(llm_risk, request.risk_hint)

    # If the LLM said strict_mode=False but reconciled risk landed at high or
    # critical, force strict_mode True to keep risk + strict_mode consistent.
    parsed_strict = bool(parsed["strict_mode"])
    strict_mode = parsed_strict or final_risk in {"high", "critical"}

    return ClaimProfile(
        is_checkable=bool(parsed["is_checkable"]),
        claim_type=claim_type,
        domain=parsed.get("domain") or request.domain_hint or "general",
        risk_level=final_risk,
        risk_level_source=source,
        required_evidence=list(parsed.get("required_evidence", [])),
        strict_mode=strict_mode,
        reason=str(parsed.get("reason", "")),
    )


def _atomic_claims_from_parsed(
    parsed: dict[str, Any], request: VerificationRequest
) -> list[AtomicClaim]:
    items = list(parsed["atomic_claims"])[:MAX_ATOMIC_CLAIMS]
    out: list[AtomicClaim] = []
    for i, item in enumerate(items):
        span = str(item.get("original_span", "")).strip()
        if not span:
            # jsonschema enforces minLength=1, but defend against whitespace-only
            # spans and future schema drift.
            continue
        out.append(
            AtomicClaim(
                id=f"c{i + 1}",
                text=str(item.get("text", span)),
                original_span=span,
                parent_claim=request.claim,
                check_priority=int(item.get("check_priority", i + 1)),
                required_evidence_type=list(item.get("required_evidence_type", [])),
            )
        )
    if not out:
        return _rule_decompose(request)
    return out


# ---------------------------------------------------------------------------
# Internal: rule-based fallback
# ---------------------------------------------------------------------------

_TEMPORAL_TOKENS = re.compile(r"\d{1,4}\s*年|\d{1,2}\s*月|\d{1,2}\s*日|世纪|年代")
_QUANTITATIVE_TOKENS = re.compile(
    r"\d+(\.\d+)?\s*(%|百分|公里|米|千米|公斤|吨|小时|分钟|秒|岁|人|元|美元)"
)
_QUOTATION_MARKERS = re.compile(r"[\"”“][^\"”“]+[\"”“]|说过|表示|宣布")
_OPINION_MARKERS = ("我觉得", "我认为", "应该", "可能", "也许", "大概")

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "science": ("研究", "实验", "论文", "DNA", "基因", "粒子"),
    "medical": ("医院", "诊断", "症状", "药物", "疗法", "病例"),
    "legal": ("法律", "判决", "诉讼", "法院", "条款", "刑期"),
    "policy": ("政府", "政策", "法案", "总统", "国务院", "白宫"),
}

_SENTENCE_SPLIT = re.compile(r"[。;；]\s*")


def _rule_profile(request: VerificationRequest) -> ClaimProfile:
    text = request.claim
    is_checkable = not any(marker in text for marker in _OPINION_MARKERS)

    claim_type: ClaimType = "entity_fact"
    if _QUOTATION_MARKERS.search(text):
        claim_type = "quotation"
    elif _QUANTITATIVE_TOKENS.search(text):
        claim_type = "quantitative_fact"
    elif _TEMPORAL_TOKENS.search(text):
        claim_type = "temporal_fact"

    domain = request.domain_hint or "general"
    if domain == "general":
        for candidate, kws in _DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in kws):
                domain = candidate
                break

    rule_risk: RiskLevel = "low"
    final_risk, source = _reconcile_risk(rule_risk, request.risk_hint)

    return ClaimProfile(
        is_checkable=is_checkable,
        claim_type=claim_type,
        domain=domain,
        risk_level=final_risk,
        risk_level_source=source,
        required_evidence=[],
        strict_mode=final_risk in {"high", "critical"},
        reason="rule_fallback",
    )


def _rule_decompose(request: VerificationRequest) -> list[AtomicClaim]:
    text = request.claim
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    if not parts:
        parts = [text]
    parts = parts[:MAX_ATOMIC_CLAIMS]
    return [
        AtomicClaim(
            id=f"c{i + 1}",
            text=part,
            original_span=part,
            parent_claim=text,
            check_priority=i + 1,
            required_evidence_type=[],
        )
        for i, part in enumerate(parts)
    ]


# ---------------------------------------------------------------------------
# Internal: risk reconciliation
# ---------------------------------------------------------------------------

def _reconcile_risk(
    inferred: RiskLevel, hint: str | None
) -> tuple[RiskLevel, RiskLevelSource]:
    """Stricter-wins. ``risk_level_source`` becomes ``max_of_both`` on conflict."""
    if hint is None or hint not in VALID_RISK_LEVELS:
        return inferred, "llm_inferred"
    hint_lvl: RiskLevel = hint  # type: ignore[assignment]
    if hint_lvl == inferred:
        return inferred, "llm_inferred"
    final = inferred if _RISK_ORDER[inferred] >= _RISK_ORDER[hint_lvl] else hint_lvl
    return final, "max_of_both"


# ---------------------------------------------------------------------------
# Internal: retrieval-plan helpers
# ---------------------------------------------------------------------------

def _should_check_freshness(profile: ClaimProfile) -> bool:
    if profile.claim_type == "temporal_fact":
        return True
    if profile.domain in _FRESHNESS_DOMAINS:
        return True
    return any(
        keyword in evidence.lower()
        for evidence in profile.required_evidence
        for keyword in ("fresh", "date", "时效", "最新")
    )
