# Agent Usage Guide (Claude / Hermes)

This document describes how AI agents should call `veritas-core` to verify
factual claims. Agents should **not** perform their own web searches or
retrieve evidence directly — they must delegate to veritas.

## Principle

veritas-core is the single source of truth for fact verification. The agent's
role is:

1. Formulate a clear, checkable claim.
2. Call veritas via CLI.
3. Read the JSON result.
4. Present the verdict, confidence, and sources to the user.

The agent must **not**:
- Browse the web independently to "double-check" veritas results.
- Modify or filter the evidence returned by veritas.
- Override the verdict based on its own parametric knowledge.

## CLI Invocation

### Single claim

```bash
veritas verify --claim "CLAIM_TEXT" --provider qwen --json
```

### Batch claims

Create a JSON file:

```json
[
  "Claim one",
  "Claim two",
  "Claim three"
]
```

Then run:

```bash
veritas verify-batch claims.json --provider qwen --json
```

### Output schema

```json
{
  "claim": "CLAIM_TEXT",
  "verdict": "likely_correct | likely_error | needs_review | unverifiable | conflicting_sources",
  "confidence": 0.85,
  "reasoning": "human-readable summary",
  "sources": [
    {
      "id": "tav_1",
      "title": "Article title",
      "url": "https://example.com/article",
      "quote": "relevant excerpt",
      "source_name": "tavily | wikipedia_zh",
      "source_tier": "T0 | T1 | T2 | T3",
      "retrieved_at": "2026-05-12T13:00:00+00:00"
    }
  ],
  "model_votes": [
    {
      "model_name": "qwen",
      "relation": "supports | refutes | neutral | insufficient",
      "confidence": "high | medium | low",
      "reason": "...",
      "quote": "...",
      "error": null
    }
  ],
  "cost": {
    "input_tokens": 1200,
    "output_tokens": 300,
    "llm_calls": 3,
    "retrieval_calls": 4,
    "cache_hits": 0
  },
  "audit_steps_count": 12,
  "warnings": []
}
```

## Interpreting results for users

| Verdict | How to present |
|---|---|
| `likely_correct` | "Evidence supports this claim." Cite sources. |
| `likely_error` | "Evidence contradicts this claim." Cite sources and explain the contradiction. |
| `needs_review` | "Evidence is weak or ambiguous." Explain what is missing. |
| `unverifiable` | "This claim cannot be fact-checked with available evidence." |
| `conflicting_sources` | "Sources disagree." Present both sides. |

## Cost awareness

Every call consumes tokens and API quota. Report costs to the user when
relevant:

```
This verification used 3 LLM calls, 1,200 input tokens, and 300 output tokens.
```

For batch operations, aggregate costs across all claims.

## Error handling

If the CLI returns a non-zero exit code, parse the JSON error output:

```json
{
  "error": "missing provider API key: QWEN_API_KEY",
  "verdict": "unverifiable",
  "confidence": 0.0
}
```

Do not silently swallow errors. Inform the user and suggest fixing the
environment configuration.

## Security

- Never pass API keys as CLI arguments.
- Never log or echo the full JSON output if it contains sensitive evidence.
- Respect source URL privacy where applicable.
