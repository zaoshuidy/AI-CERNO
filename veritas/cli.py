"""CLI entry point for veritas-core.

Usage::

    veritas verify --claim "爱因斯坦获得诺贝尔物理学奖" --provider qwen --json
    veritas verify-batch claims.json --provider qwen --json

All API keys are read from environment variables only — never from CLI args.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from veritas import FactChecker, VerificationRequest
from veritas.types import LLMProvider, VerificationResult

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

_BUILTIN_PROVIDERS: dict[str, dict[str, str]] = {
    "qwen": {
        "env_key": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "env_base_url": "QWEN_BASE_URL",
        "env_model": "QWEN_MODEL",
    },
    "xiaomi": {
        "env_key": "XIAOMI_API_KEY",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model": "mimo-v2.5",
        "env_base_url": "XIAOMI_BASE_URL",
        "env_model": "XIAOMI_MODEL",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_base_url": "DEEPSEEK_BASE_URL",
        "env_model": "DEEPSEEK_MODEL",
    },
}

# Single-claim CLI timeout (seconds) — generous for retrieval + LLM consensus.
_DEFAULT_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _env_var_name(prefix: str, suffix: str) -> str:
    return f"{prefix.upper()}_{suffix.upper()}"


def build_provider_from_env(name: str, model_override: str | None = None) -> LLMProvider:
    """Construct a single ``LLMProvider`` from environment variables.

    Built-in providers (``qwen``, ``xiaomi``, ``deepseek``) come with
    hard-coded defaults so callers only need the API key.  Any other
    provider name is treated as *generic* and must be fully configured
    via environment variables:

        {NAME}_API_KEY   (required)
        {NAME}_BASE_URL  (required, no default)
        {NAME}_MODEL     (required, no default)

    ``model_override`` takes precedence over both env and hard-coded defaults,
    allowing CLI callers to pin a specific model version without touching env.

    Raises ``ValueError`` if a required key or URL is missing.
    """
    cfg = _BUILTIN_PROVIDERS.get(name)
    if cfg is not None:
        api_key = (os.getenv(cfg["env_key"]) or "").strip()
        if not api_key:
            raise ValueError(f"missing provider API key: {cfg['env_key']}")
        base_url = (os.getenv(cfg["env_base_url"]) or cfg["base_url"]).strip()
        model = (model_override or os.getenv(cfg["env_model"]) or cfg["model"]).strip()
        return LLMProvider(
            name=name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=_DEFAULT_TIMEOUT,
        )

    # Generic provider — caller must supply all three pieces.
    prefix = name.upper()
    api_key = (os.getenv(_env_var_name(prefix, "API_KEY")) or "").strip()
    base_url = (os.getenv(_env_var_name(prefix, "BASE_URL")) or "").strip()
    model = (model_override or os.getenv(_env_var_name(prefix, "MODEL")) or "").strip()

    missing: list[str] = []
    if not api_key:
        missing.append(_env_var_name(prefix, "API_KEY"))
    if not base_url:
        missing.append(_env_var_name(prefix, "BASE_URL"))
    if not model:
        missing.append(_env_var_name(prefix, "MODEL"))
    if missing:
        raise ValueError(
            f"generic provider '{name}' missing: {', '.join(missing)}"
        )

    return LLMProvider(
        name=name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=_DEFAULT_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_providers(
    provider_names: list[str], model_override: str | None = None
) -> list[LLMProvider]:
    """Construct ``LLMProvider`` list from env vars.

    ``model_override`` is forwarded to each ``build_provider_from_env`` call
    so that ``--model`` CLI flags pin a specific model version across all
    selected providers.

    Raises ``ValueError`` with a actionable message when a required key is
    missing or no provider ends up configured.
    """
    providers: list[LLMProvider] = []
    missing: list[str] = []

    for name in provider_names:
        try:
            providers.append(build_provider_from_env(name, model_override))
        except ValueError as exc:
            msg = str(exc)
            if "missing" in msg.lower():
                missing.append(msg)
            else:
                raise

    if missing:
        raise ValueError(
            "provider setup failed:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    if not providers:
        raise ValueError(
            "no providers configured; set at least one of: "
            "QWEN_API_KEY, XIAOMI_API_KEY, DEEPSEEK_API_KEY "
            "or a generic {NAME}_API_KEY / {NAME}_BASE_URL / {NAME}_MODEL"
        )

    return providers


def _result_to_dict(result: VerificationResult) -> dict[str, Any]:
    """Stable JSON shape for CLI output."""
    return {
        "claim": result.claim,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "sources": [
            {
                "id": s.id,
                "title": s.title,
                "url": s.url,
                "quote": s.quote,
                "source_name": s.source_name,
                "source_tier": s.source_tier,
                "retrieved_at": s.retrieved_at,
            }
            for s in result.sources
        ],
        "model_votes": [
            {
                "model_name": v.model_name,
                "relation": v.relation,
                "confidence": v.confidence,
                "reason": v.reason,
                "error": v.error,
                "quote": v.quote,
            }
            for v in result.model_votes
        ],
        "audit_steps_count": len(result.audit_trace),
        "cost": {
            "input_tokens": result.cost.input_tokens,
            "output_tokens": result.cost.output_tokens,
            "llm_calls": result.cost.llm_calls,
            "retrieval_calls": result.cost.retrieval_calls,
            "cache_hits": result.cost.cache_hits,
        },
        "warnings": [],
    }


def _error_dict(message: str) -> dict[str, Any]:
    return {"error": message, "verdict": "unverifiable", "confidence": 0.0}


def _write_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    try:
        providers = _build_providers(args.provider, args.model)
    except ValueError as exc:
        _write_json(_error_dict(str(exc)))
        return 1

    checker = FactChecker(
        llm_providers=providers,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
    )
    result = checker.verify(VerificationRequest(claim=args.claim))
    _write_json(_result_to_dict(result))
    return 0


def cmd_verify_batch(args: argparse.Namespace) -> int:
    try:
        providers = _build_providers(args.provider, args.model)
    except ValueError as exc:
        _write_json(_error_dict(str(exc)))
        return 1

    with open(args.input_file, encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, list):
        _write_json(_error_dict("input JSON must be a list of strings"))
        return 1

    claims: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            _write_json(_error_dict("input JSON must be a list of strings"))
            return 1
        claims.append(item)

    checker = FactChecker(
        llm_providers=providers,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
    )
    requests = [VerificationRequest(claim=c) for c in claims]
    results = checker.verify_batch(requests)
    _write_json({"results": [_result_to_dict(r) for r in results]})
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veritas",
        description="Strict retrieval-augmented fact verification engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- verify ---
    verify_parser = sub.add_parser("verify", help="verify a single claim")
    verify_parser.add_argument(
        "--claim", required=True, help="claim text to verify"
    )
    verify_parser.add_argument(
        "--provider",
        nargs="+",
        default=["qwen"],
        help="LLM provider(s) to use (built-in: qwen, xiaomi, deepseek; or any generic name)",
    )
    verify_parser.add_argument(
        "--model",
        default=None,
        help="override the model version for all selected providers (e.g. qwen3.5-122b-a10b)",
    )
    verify_parser.add_argument(
        "--json", action="store_true", default=True, help="output JSON"
    )

    # --- verify-batch ---
    batch_parser = sub.add_parser("verify-batch", help="verify a batch of claims")
    batch_parser.add_argument(
        "input_file", help="JSON file containing a list of claim strings"
    )
    batch_parser.add_argument(
        "--provider",
        nargs="+",
        default=["qwen"],
        help="LLM provider(s) to use (built-in: qwen, xiaomi, deepseek; or any generic name)",
    )
    batch_parser.add_argument(
        "--model",
        default=None,
        help="override the model version for all selected providers (e.g. qwen3.5-122b-a10b)",
    )
    batch_parser.add_argument(
        "--json", action="store_true", default=True, help="output JSON"
    )

    args = parser.parse_args(argv)

    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "verify-batch":
        return cmd_verify_batch(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
