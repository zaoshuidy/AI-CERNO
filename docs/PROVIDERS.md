# Provider Configuration Guide

## Recommended Default: Qwen

Qwen ( Alibaba Cloud / DashScope ) is currently the recommended default provider
for high-risk fact verification.

### Environment variables

```bash
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo
```

### Model selection

| Model | Notes |
|---|---|
| `qwen-turbo` | Fast, cost-effective. Good default. |
| `qwen-plus` | Higher capability, slower. |
| `qwen-max` | Best quality, highest latency. |

Do **not** use model names that have not been verified against the DashScope
API. In our testing `qwen3.5-122b-a10b` timed out repeatedly; stick to the
official model list.

### CLI usage

```bash
veritas verify --claim "..." --provider qwen --json
veritas verify --claim "..." --provider qwen --model qwen-plus --json
```

## DeepSeek

DeepSeek is supported as a secondary / consensus provider.

### Environment variables

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
```

### Quality notes

In live testing (2026-05-12) DeepSeek-V4-Flash showed **weaker recall on
historical-fact errors** compared to Qwen. Specifically, it failed to flag a
well-known Apollo 13 factual error and produced only 3 model votes with no
refutes, resulting in `verdict="unverifiable"` with very low confidence.

**Recommendation**: Use DeepSeek as a *candidate* or *consensus* model, not as
the sole verifier for high-stakes claims. When configured alongside Qwen, it
contributes additional signal without being the primary decision-maker.

## Xiaomi MiMo

### Environment variables

```bash
XIAOMI_API_KEY=sk-...
```

MiMo uses a hard-coded default endpoint and model; only the API key is required.

## Generic OpenAI-compatible provider

Any provider that speaks the OpenAI chat-completions API can be configured via
environment variables:

```bash
{NAME}_API_KEY=sk-...
{NAME}_BASE_URL=https://api.example.com/v1
{NAME}_MODEL=model-name
```

Then run:

```bash
veritas verify --claim "..." --provider {name} --json
```

## Tavily (web search)

Tavily provides web search results. Without it, the engine falls back to
Wikipedia only.

```bash
TAVILY_API_KEY=tvly-...
```

## Wikipedia (fallback)

Wikipedia search is always available and requires no API key. It is used as a
fallback when Tavily is disabled or returns no results.
