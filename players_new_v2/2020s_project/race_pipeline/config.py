"""Central configuration for the race-identification pipeline.

All tunable constants live here so the five stage scripts stay mechanical.
Intentionally NOT a config framework — just module-level constants, dicts, and
one function. Change values by editing this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"
PAGES_DIR = DATA_DIR / "pages"

PLAYERS_CSV      = DATA_DIR / "players.csv"
SEARCHES_JSONL   = DATA_DIR / "searches.jsonl"
FETCHES_CSV      = DATA_DIR / "fetches.csv"
CONTEXTS_JSONL   = DATA_DIR / "contexts.jsonl"
EXTRACTIONS_JSONL = DATA_DIR / "extractions.jsonl"
VERIFICATIONS_CSV = DATA_DIR / "verifications.csv"
LABELS_CSV       = DATA_DIR / "labels.csv"
EVIDENCE_CSV     = DATA_DIR / "evidence.csv"
COSTS_LOG        = PIPELINE_DIR / "costs.log"

for _d in (DATA_DIR, PAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys (loaded from the repo root .env if present)
# ---------------------------------------------------------------------------
_REPO_ROOT = PIPELINE_DIR.parents[2]  # cs329r/
for env_path in (_REPO_ROOT / ".env", PIPELINE_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

SERPER_API_KEY    = os.getenv("SERPER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Network / rate limiting
# ---------------------------------------------------------------------------
SERPER_ENDPOINT        = "https://google.serper.dev/search"
SERPER_RPS             = 20           # well below the 300 rps published limit
SERPER_RESULTS_PER_QUERY = 10
SERPER_TIMEOUT_SEC     = 15
SEARCH_WORKERS         = 20           # parallel Serper threads; shared limiter enforces RPS

PER_DOMAIN_DELAY_SEC   = 2.0          # polite delay between hits to the same domain
FETCH_TIMEOUT_SEC      = 20
FETCH_MAX_PAGE_BYTES   = 2_500_000    # 2.5 MB cap to avoid giant downloads
FETCH_WORKERS          = 8            # parallel across domains; per-domain lock enforces delay

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

PAYWALL_MIN_TEXT_LEN   = 500
PAYWALL_MARKERS = (
    "subscribe to continue",
    "to read this article",
    "subscribe to read",
    "become a subscriber",
    "sign in to read",
    "create a free account to continue",
)

# ---------------------------------------------------------------------------
# Context extraction (spaCy)
# ---------------------------------------------------------------------------
SPACY_MODEL            = "en_core_web_sm"
CONTEXT_WINDOW_SENTENCES = 3          # center sentence ± 1
CONTEXT_SHORT_SENT_CHARS = 40
CONTEXT_EXTENDED_WINDOW  = 5
MAX_CONTEXTS_PER_PAGE    = 10

# ---------------------------------------------------------------------------
# Extraction LLM
# ---------------------------------------------------------------------------
DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5"
EXTRACTION_TEMPERATURE  = 0.0
EXTRACTION_MAX_TOKENS   = 800
EXTRACT_WORKERS         = 8           # concurrent Claude calls
# Anthropic tier-1 accounts cap Haiku at 50 requests per minute. We stay
# just under (45) to leave headroom; workers share a single token bucket.
EXTRACT_MAX_RPM         = 45
EXTRACT_MAX_RETRIES     = 5           # on 429 / transient errors, with exp backoff

# ---------------------------------------------------------------------------
# Pre-LLM keyword gate (v2, 2026-04-22)
#
# A context is sent to Claude only if BOTH:
#   (a) RACE_IDENTITY_PATTERN  matches — an actual race/ethnicity/heritage
#       term is present, and
#   (b) FIRST_PERSON_PATTERN   matches — a first-person pronoun is present.
#
# Self-identification *requires* both, so both must appear in the ±3
# sentence window we already cut. Prior v1 gate matched either a race
# term OR a first-person "my X" pattern, which let through ~50% of game
# narratives ("my first touchdown", "black uniforms", etc.) and drove
# extract cost sky-high at scale. Tighter gate cuts LLM calls ~90% on
# smoke tests while losing only contexts that could not satisfy the
# extraction prompt's own requirements.
#
# If precision/recall ever needs retuning, tighten RACE_IDENTITY_PATTERN
# (drop generic terms like "white" if false positives dominate) or widen
# FIRST_PERSON_PATTERN (add contraction spellings). Err toward
# over-filtering: the same player is typically covered by multiple pages
# so one missed context rarely loses a label.
# ---------------------------------------------------------------------------
RACE_IDENTITY_PATTERN = (
    r"\b(?:"
    # core race / ethnicity identity terms
    r"black|white|latino|latina|latinx|hispanic|chicano|chicana|"
    r"asian[- ]american|asian|aapi|"
    r"african[- ]american|african|afro[- ]?(?:american|latino|latina|caribbean)?|"
    r"samoan|tongan|polynesian|hawaiian|fijian|maori|pacific[- ]islander|"
    r"korean|japanese|chinese|vietnamese|filipino|filipina|thai|"
    r"mexican|puerto[- ]rican|cuban|dominican|colombian|salvadoran|"
    r"guatemalan|honduran|venezuelan|peruvian|argentin(?:e|ian)|brazilian|"
    r"nigerian|ghanaian|kenyan|ethiopian|somali|eritrean|haitian|jamaican|"
    r"irish|italian|german|polish|scandinavian|scottish|english|welsh|"
    r"russian|ukrainian|greek|armenian|"
    r"biracial|multiracial|mixed[- ]race|half[- ](?:black|white|asian|"
    r"samoan|mexican|korean|japanese|tongan|filipino|latino|"
    r"african|european)|"
    # identity / lineage framing that on its own signals heritage talk
    r"heritage|ancestry|bloodline"
    r")\b"
)

FIRST_PERSON_PATTERN = (
    r"(?:\bI\b|\bI'?m\b|\bI'?ve\b|\bI'?ll\b|\bI'?d\b|"
    r"\bme\b|\bmy\b|\bmine\b|\bmyself\b|"
    r"\bwe\b|\bour\b|\bus\b|\bourselves\b)"
)

# Back-compat alias — some callers still import this.  The combined
# legacy pattern is kept as the OR of the two new patterns only for
# code that wants a single regex; the real gate uses both separately.
RACE_KEYWORDS_PATTERN = RACE_IDENTITY_PATTERN

# Per-million token pricing used to estimate costs in costs.log.
# Update as Anthropic pricing changes.
MODEL_PRICING_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.0,  "output": 5.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
}

# ---------------------------------------------------------------------------
# Cost caps (Serper)
# ---------------------------------------------------------------------------
SERPER_COST_PER_QUERY  = 0.0003       # $0.30 / 1K at volume
SERPER_COST_WARN_USD   = 50.0
SERPER_COST_HARDSTOP_USD = 100.0

# ---------------------------------------------------------------------------
# Domain tiering
# ---------------------------------------------------------------------------
TIER_1_DOMAINS = {
    "theplayerstribune.com",
    "andscape.com",
    "theundefeated.com",
    "uninterrupted.com",
}

TIER_2_DOMAINS = {
    "espn.com", "si.com", "nfl.com", "theathletic.com",
    "bleacherreport.com", "sbnation.com", "gq.com",
    "nytimes.com", "washingtonpost.com", "latimes.com",
    "chicagotribune.com", "bostonglobe.com",
    "houstonchronicle.com", "dallasnews.com", "sfchronicle.com",
    # podcast transcripts on YouTube — treat as tier 2
    "youtube.com",
}

DENYLIST_DOMAINS = {
    "wikipedia.org", "reddit.com", "quora.com",
    "twitter.com", "x.com", "facebook.com",
    "instagram.com", "tiktok.com",
    "draftkings.com", "fanduel.com", "actionnetwork.com",
    # AI content farms — grow this list as they surface
    "sportsbrief.com", "essentiallysports.com",
}


def classify_tier(domain: str) -> int | None:
    """Map a bare domain string to a source tier.

    Returns 1, 2, 3, or None (denylisted).
    The matcher uses substring containment so sub-domains inherit the tier of
    their registrable domain (e.g. `rams.com` matches `nfl.com`? no — only
    exact suffix logic below).
    """
    d = domain.lower()
    if d.startswith("www."):
        d = d[4:]

    for bad in DENYLIST_DOMAINS:
        if d == bad or d.endswith("." + bad):
            return None

    for tier1 in TIER_1_DOMAINS:
        if d == tier1 or d.endswith("." + tier1):
            return 1

    for tier2 in TIER_2_DOMAINS:
        if d == tier2 or d.endswith("." + tier2):
            return 2

    # College newspapers tend to live on .edu
    if d.endswith(".edu"):
        return 2

    return 3


# ---------------------------------------------------------------------------
# Query templates — content-discovery design (v1.3, 2026-03-31).
#
# v1.3 replaces the brand-anchored / exact-phrase templates of v1.2 with
# five natural-language queries that Google actually indexes against.
# The "NFL" token disambiguates players with common names without
# pre-selecting for identity content.
#
# Rationale (see diagnostic on the n=100 pilot in v1.2):
#   - Brand anchors ("Players Tribune", "Andscape") only cover ~1–2% of
#     NFL players → ~70% of random-sample players got 0 Serper results.
#   - Exact-phrase constraints ("growing up" hometown profile,
#     "{college}" profile feature, draft combine background) rarely
#     match verbatim in indexed pages.
#   - The five queries below each cover a distinct document type
#     (college/recruiting, biography, background, interview, heritage)
#     and collectively return ≥1 result for virtually every NFL player.
# ---------------------------------------------------------------------------
DISCOVERY_QUERIES: list[str] = [
    '{name} NFL {college}',
    '"{name}" NFL biography',
    '"{name}" background',
    '"{name}" NFL interview',
    '"{name}" NFL heritage',
]


def render_queries(name: str, college: str = "") -> list[tuple[str, str]]:
    """Expand DISCOVERY_QUERIES for a single player. All queries share the
    group label 'discovery' (the group column exists for back-compat and for
    analytics grouping — nothing downstream branches on its value).

    Templates that require a college are skipped when the player's college
    is unknown.
    """
    out: list[tuple[str, str]] = []
    for t in DISCOVERY_QUERIES:
        if "{college}" in t and not college.strip():
            continue
        q = t.format(name=name, college=college)
        out.append(("discovery", q))
    return out
