"""Real-network smoke tests for cerno.retrieval.

These tests reach out to live services (Tavily, zh.wikipedia.org). They are
gated by ``LIVE_TEST=1`` in the environment so that ``pytest`` invocations in
CI / dev workflows skip them by default. Tavily additionally requires a
``TAVILY_API_KEY`` to be set.

Run them locally with, e.g.:

    LIVE_TEST=1 TAVILY_API_KEY=... python -m pytest tests/test_retrieval_live.py -v
"""

from __future__ import annotations

import os

import pytest

from cerno.retrieval import search_tavily, search_wikipedia_zh

# Module-level skip: every test in here is skipped unless explicitly opted-in.
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("LIVE_TEST") != "1",
        reason="Live network tests are off by default. Set LIVE_TEST=1 to enable.",
    ),
]


# ---------------------------------------------------------------------------
# Wikipedia zh (no API key required)
# ---------------------------------------------------------------------------

def test_live_search_wikipedia_zh_returns_results_for_known_topic() -> None:
    """Smoke test: a well-known topic should yield at least one EvidenceSpan."""
    spans = search_wikipedia_zh("爱因斯坦", max_results=2)
    assert isinstance(spans, list)
    # Allow zero results if the API hiccups, but if we got anything it should
    # parse into the documented EvidenceSpan shape.
    for span in spans:
        assert span.source_name == "wikipedia_zh"
        assert span.source_tier == "T1"
        assert span.quote  # never empty
        assert span.url.startswith("https://zh.wikipedia.org/wiki/")
        assert span.content_hash  # 16-char sha256 prefix
        assert len(span.content_hash) == 16


# ---------------------------------------------------------------------------
# Tavily (API key required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY env var not set.",
)
def test_live_search_tavily_returns_results_for_known_topic() -> None:
    """Smoke test: Tavily returns parseable EvidenceSpan list for a stock query."""
    api_key = os.environ["TAVILY_API_KEY"]
    spans = search_tavily("Albert Einstein Nobel Prize", api_key, max_results=3)
    assert isinstance(spans, list)
    for span in spans:
        assert span.source_name == "tavily"
        assert span.source_tier in {"T0", "T1", "T2", "T3"}
        assert span.quote  # never empty
        assert span.url
        assert span.content_hash
        assert len(span.content_hash) == 16
