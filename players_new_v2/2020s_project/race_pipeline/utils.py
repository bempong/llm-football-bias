"""Shared helpers: JSONL/CSV I/O, timestamps, cost logging, text normalization.

Everything here is a small pure function. No abstractions — just things the
five stage scripts need to do the same way.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSONL append / iterate
# ---------------------------------------------------------------------------
def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# CSV append — manage header creation lazily
# ---------------------------------------------------------------------------
def append_csv_row(path: Path, row: dict, fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Cost logging
# ---------------------------------------------------------------------------
def log_cost(stage: str, model: str, input_tokens: int, output_tokens: int, usd: float, note: str = "") -> None:
    line = {
        "ts": utc_now_iso(),
        "stage": stage,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 6),
        "note": note,
    }
    config.COSTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with config.COSTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def estimate_claude_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = config.MODEL_PRICING_PER_MTOK.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


# ---------------------------------------------------------------------------
# Text normalization (used by verify.py and fetch.py paywall detection)
# ---------------------------------------------------------------------------
_SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2032": "'", "\u2033": '"',
}
_DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-"}
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFC-normalize, fold smart quotes / dashes, collapse whitespace, lowercase.
    Used on both sides of the verification substring check.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    for a, b in _SMART_QUOTES.items():
        s = s.replace(a, b)
    for a, b in _DASHES.items():
        s = s.replace(a, b)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------
_DOMAIN_RE = re.compile(r"^(?:https?://)?([^/:?#]+)", re.I)


def domain_of(url: str) -> str:
    m = _DOMAIN_RE.match(url or "")
    return (m.group(1) if m else "").lower().removeprefix("www.")
