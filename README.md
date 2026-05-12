# veritas-core

Strict retrieval-augmented fact verification engine.

An independent, generic Python library for verifying factual claims against
external evidence. Not coupled to any business code.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quick start

```python
from veritas import FactChecker, VerificationRequest, LLMProvider

checker = FactChecker(
    llm_providers=[
        LLMProvider(
            name="qwen",
            api_key="your-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
        ),
    ],
    tavily_api_key="your-tavily-key",  # optional; without it Wikipedia is used
)

result = checker.verify(VerificationRequest(claim="爱因斯坦获得诺贝尔物理学奖"))
print(result.verdict)       # "likely_correct" | "likely_error" | ...
print(result.confidence)    # 0.0 - 1.0
print(result.reasoning)     # human-readable summary
print(result.cost.llm_calls)    # number of LLM calls made
print(result.cost.input_tokens) # total input tokens consumed
```

## CLI

```bash
# Single claim
veritas verify --claim "爱因斯坦获得诺贝尔物理学奖" --provider qwen --json

# Batch
veritas verify-batch claims.json --provider qwen --json

# Multiple providers for consensus
veritas verify --claim "..." --provider qwen deepseek --json

# Pin a specific model version
veritas verify --claim "..." --provider qwen --model qwen-plus --json
```

API keys are read from environment variables only — never from CLI arguments.

## Environment variables

Copy `.env.example` to `.env` and fill in the keys you need.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TAVILY_API_KEY` | No | — | Web search via Tavily |
| `QWEN_API_KEY` | No | — | Qwen LLM provider |
| `QWEN_BASE_URL` | No | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Qwen endpoint |
| `QWEN_MODEL` | No | `qwen-turbo` | Qwen model name |
| `XIAOMI_API_KEY` | No | — | Xiaomi MiMo provider |
| `DEEPSEEK_API_KEY` | No | — | DeepSeek provider |

At least one LLM provider key must be configured for the engine to start.

## Multi-provider consensus

Pass multiple providers to run consensus across models:

```python
from veritas import FactChecker
from veritas.cli import build_provider_from_env

checker = FactChecker(
    llm_providers=[
        build_provider_from_env("qwen"),
        build_provider_from_env("deepseek"),
    ],
)
```

Or via CLI:

```bash
veritas verify --claim "..." --provider qwen deepseek --json
```

## Verdicts

| Verdict | Meaning |
|---|---|
| `likely_correct` | Evidence strongly supports the claim |
| `likely_error` | Evidence contradicts the claim |
| `needs_review` | Evidence is weak or ambiguous |
| `unverifiable` | Not a checkable claim, or no evidence found |
| `conflicting_sources` | Supporting and refuting evidence both exist |

## Cost tracking

Every `VerificationResult` carries a `CostBreakdown`:

```python
result.cost.input_tokens    # total prompt tokens
result.cost.output_tokens   # total completion tokens
result.cost.llm_calls       # number of LLM API calls
result.cost.retrieval_calls # number of search/retrieval calls
result.cost.cache_hits      # retrieval cache hits
```

## Tests

Offline unit tests (no network):

```bash
pytest -q
```

Live smoke tests (hit real APIs, require keys):

```bash
LIVE_TEST=1 pytest -q -m live
```

## Provider quality notes

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for detailed provider recommendations.

- **Qwen** (`qwen-turbo` and up): Currently recommended as the default
  high-risk fact-checking model. Good at catching factual errors.
- **DeepSeek-V4-Flash**: Available as a candidate provider. In our tests it
  showed weaker recall on historical-fact errors; use as a secondary/
  consensus model rather than the sole verifier for high-stakes claims.

## Agent usage (Claude / Hermes)

See [docs/AGENT_USAGE.md](docs/AGENT_USAGE.md) for instructions on calling
veritas from AI agents.

## Disclaimer

This library performs **assistive** fact verification using external LLMs and
search APIs. It does **not** guarantee 100% accuracy. Output should be reviewed
by a human before being used for legal, medical, or other high-stakes
decisions.

## License

MIT — see [LICENSE](LICENSE).
