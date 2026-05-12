"""Stage 2 of the verification pipeline: retrieval + evidence screening.

P0 surfaces two real sources (Tavily, Wikipedia zh) plus the helpers that
turn raw search results into EvidenceSpan objects:

- ``search_tavily`` / ``search_wikipedia_zh``: real-network retrieval, with
  optional file-backed caching (TTL 24h, atomic write).
- ``classify_source_tier``: T0 / T1 / T2 / T3 / BLOCKED. Injection patterns
  force BLOCKED, ``.gov`` / ``.edu`` families give T0, ``wikipedia_zh`` gives
  T1, mainstream media gives T2, everything else lands at T3.
- ``detect_injection``: returns risk-flag strings (``prompt_injection_pattern``,
  ``content_farm``) used by both tier classification and the score penalty.
- ``extract_anchors`` + ``extract_quote``: anchor-window quote extraction
  (±``window`` chars around the first matching anchor).
- ``normalize_text`` + ``content_hash``: stable SHA-256(normalize)[:16] hash.
- ``score_evidence_initial``: EvidenceScore P0 fields (tier + lexical + penalty).
- ``load_cache`` / ``save_cache``: per-(source, query) JSON cache with TTL 24h.

Hard rules:
- ``EvidenceSpan`` is never produced with empty ``url``, ``quote``,
  ``retrieved_at`` or ``content_hash`` — enforced both on the live retrieval
  path and when re-hydrating from cache.
- ``cache_dir=None`` is an opt-out: no read, no write.
- Cache writes use ``tmp.replace(final)`` for atomic update.
- Cache only persists Tavily / Wiki retrieval results; LLM judgements never
  pass through this module.
- Tavily calls request ``search_depth="advanced"`` and
  ``include_raw_content=True`` so quote / hash / injection / tier all run
  against the full article body, not the short snippet.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

from veritas.types import EvidenceScore, EvidenceSpan, SourceTier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 24h. Cache entries older than this are treated as misses.
CACHE_TTL_SECONDS: int = 24 * 60 * 60

#: Cache key version. Bump to invalidate all on-disk caches.
CACHE_VERSION: str = "v1"

#: Default anchor window for ``extract_quote`` (chars on each side).
DEFAULT_QUOTE_WINDOW: int = 200

#: Wikipedia (zh) API endpoint.
WIKI_API_URL: str = "https://zh.wikipedia.org/w/api.php"

#: Wikipedia HTTP timeout (seconds).
WIKI_TIMEOUT: float = 15.0

#: User-Agent required by MediaWiki API etiquette.
WIKI_USER_AGENT: str = "veritas-core/0.1 (P0 fact verification)"

#: Domain suffixes that unconditionally produce T0.
_T0_DOMAIN_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".gov.cn",
    ".gov.uk",
    ".gov.au",
    ".edu",
    ".edu.cn",
    ".edu.au",
    ".ac.uk",
    ".ac.cn",
)

#: Recognized official-institution hosts that produce T0.
_T0_OFFICIAL_HOSTS: frozenset[str] = frozenset({
    "who.int",
    "un.org",
    "europa.eu",
    "imf.org",
    "worldbank.org",
    "oecd.org",
    "wto.org",
    "iaea.org",
})

#: Hosts that produce T1 directly (Wikipedia family).
_T1_WIKI_HOST_FRAGMENTS: tuple[str, ...] = ("wikipedia.org",)

#: Mainstream media + reputable publishers / professional sites → T2.
_T2_MAINSTREAM_HOSTS: frozenset[str] = frozenset({
    # Chinese mainstream / state media
    "xinhuanet.com",
    "people.com.cn",
    "peopledaily.com.cn",
    "cctv.com",
    "chinadaily.com.cn",
    "thepaper.cn",
    "caixin.com",
    "guancha.cn",
    "yicai.com",
    "21jingji.com",
    # International mainstream
    "nytimes.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "theguardian.com",
    "apnews.com",
    "ap.org",
    "npr.org",
    "washingtonpost.com",
    "scmp.com",
    "cnbc.com",
    # Publishers / academic
    "nature.com",
    "science.org",
    "sciencemag.org",
    "cell.com",
    "springer.com",
    "elsevier.com",
    "nejm.org",
    "thelancet.com",
    "ieee.org",
    "acm.org",
})

#: Prompt-injection regexes (case-insensitive search).
_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(the\s+)?(previous|above|prior|all)\s+(instructions?|prompts?)",
        r"disregard\s+(the\s+)?(previous|above|prior)",
        r"forget\s+(your|the)\s+(previous|prior)\s+(instructions?|prompts?)",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"new\s+instructions?\s*:",
        r"\[INST\]",
        r"</?s(?:ys|ystem)>",
        r"act\s+as\s+(a\s+)?(jailbreak|developer\s+mode|dan)",
        r"忽略(以上|之前|前面)(的)?(指令|提示|要求)",
        r"忘记(之前|以前|前面)(的)?(指令|提示)",
        r"请扮演",
        r"你现在(是|扮演)",
        r"新的(指令|任务|角色)",
    )
)

#: Promotional / SEO-spam tokens. ≥3 unique hits triggers ``content_farm``.
_CONTENT_FARM_TOKENS: tuple[str, ...] = (
    "立即购买",
    "马上抢购",
    "点击这里",
    "click here",
    "buy now",
    "limited offer",
    "special deal",
)

#: Per-tier base score for ``score_evidence_initial``.
_TIER_BASE_SCORE: dict[SourceTier, float] = {
    "T0": 1.0,
    "T1": 0.85,
    "T2": 0.70,
    "T3": 0.40,
    "BLOCKED": 0.0,
}

#: Stopwords excluded from anchor extraction.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "the", "and", "for", "with", "from", "that", "this", "have",
    "are", "was", "were", "but", "not", "you", "your", "his", "her",
    "its", "their", "they", "them", "what", "which", "who", "when",
    # Chinese (bigrams that show up in nearly every claim)
    "因为", "所以", "但是", "如果", "这是", "那是", "我们", "你们",
    "他们", "可以", "应该", "可能", "一个", "一种",
})


# ---------------------------------------------------------------------------
# Public: normalization + hashing
# ---------------------------------------------------------------------------

def normalize_text(raw_content: str) -> str:
    """Strip HTML, decode entities, collapse whitespace, lowercase English.

    Does NOT remove punctuation and does NOT convert traditional ↔ simplified.
    Returns the empty string when ``raw_content`` is falsy.
    """
    if not raw_content:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", raw_content)
    decoded = html.unescape(no_tags)
    collapsed = re.sub(r"\s+", " ", decoded).strip()
    return collapsed.lower()


def content_hash(raw_content: str) -> str:
    """SHA-256 of ``normalize_text(raw_content)``, truncated to 16 hex chars."""
    normalized = normalize_text(raw_content)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Public: anchors + quote
# ---------------------------------------------------------------------------

def extract_anchors(claim: str) -> list[str]:
    """Pull discriminative tokens from ``claim`` for use as quote anchors.

    English: alphanumeric tokens with length ≥ 3.
    Chinese: full CJK runs up to length 4; longer runs become 2-char sliding
    windows so a long claim still produces short, matchable anchors.
    Stopwords are dropped. Returns a list ordered by appearance, deduped.
    """
    if not claim:
        return []
    anchors: list[str] = []
    seen: set[str] = set()

    cjk_runs = re.findall(r"[一-鿿]+", claim)
    for run in cjk_runs:
        if len(run) <= 4:
            _push_unique(run, seen, anchors)
        else:
            for i in range(len(run) - 1):
                _push_unique(run[i:i + 2], seen, anchors)

    for tok in re.findall(r"[A-Za-z0-9]+", claim):
        if len(tok) >= 3:
            _push_unique(tok, seen, anchors)

    return anchors


def _push_unique(token: str, seen: set[str], out: list[str]) -> None:
    low = token.lower()
    if low in _STOPWORDS or low in seen:
        return
    seen.add(low)
    out.append(token)


def extract_quote(
    claim: str, raw_content: str, window: int = DEFAULT_QUOTE_WINDOW
) -> str:
    """Anchor-window method: first matching anchor's neighborhood, ±``window`` chars.

    Returns ``""`` when no anchor matches or content is empty. The returned
    quote is drawn from ``normalize_text(raw_content)`` (so it's safe to use
    as ``EvidenceSpan.quote`` directly).
    """
    text = normalize_text(raw_content)
    if not text:
        return ""
    anchors = extract_anchors(claim)
    if not anchors:
        return ""
    # Try longest anchors first — more discriminative.
    by_length = sorted(set(anchors), key=lambda s: -len(s))
    for anchor in by_length:
        idx = text.find(anchor.lower())
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(anchor) + window)
            return text[start:end]
    return ""


# ---------------------------------------------------------------------------
# Public: injection detection
# ---------------------------------------------------------------------------

def detect_injection(content: str) -> list[str]:
    """Return a list of risk-flag strings present in ``content``.

    P0 flags:
    - ``prompt_injection_pattern``: any of the prompt-injection regexes matched.
    - ``content_farm``: three or more distinct promotional tokens present.

    Empty list when content is clean.
    """
    if not content:
        return []
    flags: list[str] = []
    for pat in _PROMPT_INJECTION_PATTERNS:
        if pat.search(content):
            flags.append("prompt_injection_pattern")
            break
    lowered = content.lower()
    hit_tokens = {t for t in _CONTENT_FARM_TOKENS if t in lowered}
    if len(hit_tokens) >= 3:
        flags.append("content_farm")
    return flags


# ---------------------------------------------------------------------------
# Public: source tier classification
# ---------------------------------------------------------------------------

def classify_source_tier(url: str, content: str, source_name: str) -> SourceTier:
    """Classify a retrieved source into one of T0 / T1 / T2 / T3 / BLOCKED.

    Priority (first match wins):
    1. Injection or content-farm flags → BLOCKED.
    2. ``.gov`` / ``.edu`` family or recognized official host → T0.
    3. ``source_name == "wikipedia_zh"`` or any ``wikipedia.org`` host → T1.
    4. Mainstream media / reputable publisher → T2.
    5. Everything else → T3.
    """
    if detect_injection(content):
        return "BLOCKED"
    host = _extract_host(url)
    if any(host.endswith(suffix) for suffix in _T0_DOMAIN_SUFFIXES):
        return "T0"
    if host in _T0_OFFICIAL_HOSTS:
        return "T0"
    if source_name == "wikipedia_zh":
        return "T1"
    if any(frag in host for frag in _T1_WIKI_HOST_FRAGMENTS):
        return "T1"
    if host in _T2_MAINSTREAM_HOSTS:
        return "T2"
    if any(host.endswith("." + h) for h in _T2_MAINSTREAM_HOSTS):
        return "T2"
    return "T3"


def _extract_host(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


# ---------------------------------------------------------------------------
# Public: evidence score (initial)
# ---------------------------------------------------------------------------

def score_evidence_initial(claim: str, evidence: EvidenceSpan) -> EvidenceScore:
    """P0 initial score: tier baseline + lexical overlap, with injection penalty.

    ``semantic_support_score`` is left at 0.0 because P0 punts the LLM-based
    judgement to ``verification.py``. ``freshness_score`` is left at None for
    the same reason.
    """
    tier_score = _TIER_BASE_SCORE.get(evidence.source_tier, 0.0)
    anchors = extract_anchors(claim)
    if not anchors:
        lexical = 0.0
    else:
        quote_lower = evidence.quote.lower()
        hits = sum(1 for a in anchors if a.lower() in quote_lower)
        lexical = hits / len(anchors)

    penalty = 0.0
    if "prompt_injection_pattern" in evidence.risk_flags:
        penalty = 0.5
    if "content_farm" in evidence.risk_flags:
        penalty = max(penalty, 0.3)

    final = max(0.0, tier_score * 0.5 + lexical * 0.5 - penalty)

    return EvidenceScore(
        source_tier_score=tier_score,
        lexical_overlap_score=lexical,
        semantic_support_score=0.0,
        freshness_score=None,
        injection_risk_penalty=penalty,
        final_score=final,
    )


# ---------------------------------------------------------------------------
# Public: cache (file-backed, TTL 24h, atomic write)
# ---------------------------------------------------------------------------

def load_cache(
    cache_dir: str | None, source_name: str, query: str
) -> list[EvidenceSpan] | None:
    """Return cached EvidenceSpan list, or None on miss / expiry / disabled cache.

    Items that fail the EvidenceSpan contract (empty ``url``, ``quote``,
    ``retrieved_at`` or ``content_hash``) are silently dropped so that stale or
    corrupt cache files never replay a broken span back into the pipeline.
    """
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, source_name, query)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    ts = payload.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    if time.time() - ts > CACHE_TTL_SECONDS:
        return None
    items = payload.get("items", [])
    if not isinstance(items, list):
        return None
    spans: list[EvidenceSpan] = []
    for item in items:
        span = _span_from_cache_item(item)
        if span is not None:
            spans.append(span)
    return spans


def _span_from_cache_item(item: object) -> EvidenceSpan | None:
    """Reconstruct an EvidenceSpan from a cache payload item, or None on failure.

    Enforces the same non-empty invariants that live retrieval enforces:
    ``url``, ``quote``, ``retrieved_at`` and ``content_hash`` must all be
    non-empty. Anything else is treated as corrupt and dropped.
    """
    if not isinstance(item, dict):
        return None
    try:
        span = EvidenceSpan(**item)
    except (TypeError, ValueError):
        return None
    if not span.url or not span.quote or not span.retrieved_at or not span.content_hash:
        return None
    return span


def save_cache(
    cache_dir: str | None,
    source_name: str,
    query: str,
    items: list[EvidenceSpan],
) -> None:
    """Write the cache atomically: serialize to ``.tmp`` then ``replace``."""
    if cache_dir is None:
        return
    target = _cache_path(cache_dir, source_name, query)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "ts": time.time(),
        "query": query,
        "source_name": source_name,
        "items": [asdict(s) for s in items],
    }
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(target)


def _cache_key(query: str, source_name: str) -> str:
    raw = f"{query}|{source_name}|{CACHE_VERSION}".encode()
    return hashlib.sha256(raw).hexdigest()


def _cache_path(cache_dir: str, source_name: str, query: str) -> Path:
    return Path(cache_dir) / f"{_cache_key(query, source_name)}.json"


# ---------------------------------------------------------------------------
# Public: retrieval (real network, cache-aware)
# ---------------------------------------------------------------------------

def search_tavily(
    query: str,
    tavily_api_key: str,
    max_results: int = 5,
    cache_dir: str | None = None,
) -> list[EvidenceSpan]:
    """Search Tavily, return EvidenceSpan list. Honors cache when ``cache_dir`` set.

    Calls Tavily with ``search_depth="advanced"`` and ``include_raw_content=True``
    so we get full article text (``raw_content``) rather than the short snippet
    (``content``). Quote extraction, content hashing, injection detection and
    tier classification all run against the longest available body.
    """
    cached = load_cache(cache_dir, "tavily", query)
    if cached is not None:
        return cached
    # Local import keeps tavily-python out of import time when callers stub
    # `search_tavily` entirely (e.g., tests, alternative source plug-ins).
    from tavily import TavilyClient

    client = TavilyClient(api_key=tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=True,
    )
    raw_items = response.get("results", []) if isinstance(response, dict) else []
    spans = _build_tavily_spans(query, raw_items)
    save_cache(cache_dir, "tavily", query, spans)
    return spans


def _build_tavily_spans(query: str, items: list[dict]) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    timestamp = _now_iso()
    for i, item in enumerate(items):
        url = item.get("url", "") or ""
        if not url:
            # EvidenceSpan contract: url must be non-empty.
            continue
        title = item.get("title", "") or ""
        # Prefer the full article body over the short snippet so quote
        # extraction, injection detection and content_hash run against the
        # richest text available.
        content = item.get("raw_content") or item.get("content") or ""
        if not content:
            continue
        raw_score = item.get("score")
        risk_flags = detect_injection(content)
        tier = classify_source_tier(url, content, source_name="tavily")
        if tier == "BLOCKED":
            continue
        quote_text = extract_quote(query, content)
        if not quote_text:
            continue
        spans.append(
            EvidenceSpan(
                id=f"tav_{i + 1}",
                title=title,
                url=url,
                quote=quote_text,
                source_name="tavily",
                source_tier=tier,
                retrieved_at=timestamp,
                content_hash=content_hash(content),
                raw_score=raw_score if isinstance(raw_score, (int, float)) else None,
                metadata={},
                risk_flags=risk_flags,
            )
        )
    return spans


def search_wikipedia_zh(
    query: str,
    max_results: int = 3,
    cache_dir: str | None = None,
) -> list[EvidenceSpan]:
    """Search Chinese Wikipedia via MediaWiki API, return EvidenceSpan list."""
    cached = load_cache(cache_dir, "wikipedia_zh", query)
    if cached is not None:
        return cached

    titles = _wiki_search_titles(query, max_results)
    if not titles:
        save_cache(cache_dir, "wikipedia_zh", query, [])
        return []
    extracts = _wiki_fetch_extracts(titles)

    spans: list[EvidenceSpan] = []
    timestamp = _now_iso()
    for i, title in enumerate(titles):
        content = extracts.get(title, "") or ""
        url = f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"
        risk_flags = detect_injection(content)
        tier = classify_source_tier(url, content, source_name="wikipedia_zh")
        if tier == "BLOCKED":
            continue
        quote_text = extract_quote(query, content)
        if not quote_text:
            continue
        spans.append(
            EvidenceSpan(
                id=f"wiki_{i + 1}",
                title=title,
                url=url,
                quote=quote_text,
                source_name="wikipedia_zh",
                source_tier=tier,
                retrieved_at=timestamp,
                content_hash=content_hash(content),
                raw_score=None,
                metadata={},
                risk_flags=risk_flags,
            )
        )
    save_cache(cache_dir, "wikipedia_zh", query, spans)
    return spans


def _wiki_search_titles(query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "utf8": 1,
    }
    resp = requests.get(
        WIKI_API_URL,
        params=params,
        headers={"User-Agent": WIKI_USER_AGENT},
        timeout=WIKI_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("query", {}).get("search", [])
    return [h.get("title", "") for h in hits[:limit] if h.get("title")]


def _wiki_fetch_extracts(titles: list[str]) -> dict[str, str]:
    if not titles:
        return {}
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "titles": "|".join(titles),
        "format": "json",
        "utf8": 1,
        "redirects": 1,
    }
    resp = requests.get(
        WIKI_API_URL,
        params=params,
        headers={"User-Agent": WIKI_USER_AGENT},
        timeout=WIKI_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    out: dict[str, str] = {}
    for page in pages.values():
        title = page.get("title", "")
        extract = page.get("extract", "") or ""
        if title:
            out[title] = extract
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
