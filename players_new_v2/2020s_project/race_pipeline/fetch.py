"""Stage 2 — fetch URLs, extract clean text, cut spaCy-bounded contexts.

Pipeline within this stage:

    for each unseen result in data/searches.jsonl:
        classify tier (denylist? tier 1 / 2 / 3)
        if denylisted -> record + skip
        fetch with httpx (per-domain 2s delay, shared across threads)
        extract clean text with trafilatura
        detect paywall (length < 500 OR marker substring)
        save text to data/pages/{fetch_id}.txt
        append fetches.csv row
        run context extraction (spaCy) -> append contexts.jsonl rows

Context extraction uses spaCy NER + sentence boundaries. For each PERSON
entity that matches the player's name we take the containing sentence ± 1
(or ± 2 if sentences are short). We also include sentences that contain a
pronoun immediately following a named mention, when no other PERSON entity
is closer — a cheap coreference heuristic sufficient for the pilot.

Resume semantics: we skip any search-result URL whose corresponding
fetch_id already exists in fetches.csv for the same (search_id, rank).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402
from race_pipeline.utils import (  # noqa: E402
    append_csv_row,
    append_jsonl,
    domain_of,
    iter_jsonl,
    utc_now_iso,
)

FETCH_FIELDS = [
    "fetch_id", "search_id", "result_rank", "player_id", "url", "domain",
    "source_tier", "status", "fetched_at", "text_path", "text_length",
]


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------
def _load_done_fetches() -> tuple[set[tuple[int, int]], int]:
    done: set[tuple[int, int]] = set()
    max_id = 0
    if not config.FETCHES_CSV.exists():
        return done, max_id
    with config.FETCHES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sid = int(row["search_id"])
                rk = int(row["result_rank"])
                fid = int(row["fetch_id"])
            except (KeyError, ValueError):
                continue
            done.add((sid, rk))
            max_id = max(max_id, fid)
    return done, max_id


def _load_url_cache() -> dict[str, dict[str, Any]]:
    """Map url -> {text_path, text_length, status, source_tier, domain}
    for every successful fetch already on disk. Used by URL-level dedup:
    when a URL reappears (often across players), we skip the HTTP + trafilatura
    round-trip and run only the spaCy context extraction for the new player.
    """
    cache: dict[str, dict[str, Any]] = {}
    if not config.FETCHES_CSV.exists():
        return cache
    with config.FETCHES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("url") or ""
            if not url or row.get("status") != "ok":
                continue
            tp = row.get("text_path") or ""
            # Resolve to absolute for re-reading
            abs_path = (config.PIPELINE_DIR / tp) if tp else None
            if abs_path and abs_path.exists():
                cache[url] = {
                    "text_path": tp,
                    "text_length": int(row.get("text_length") or 0),
                    "status": "ok",
                    "source_tier": row.get("source_tier") or "",
                    "domain": row.get("domain") or "",
                    "abs_path": abs_path,
                }
    return cache


def _load_done_contexts() -> tuple[set[int], int]:
    seen: set[int] = set()
    max_id = 0
    for rec in iter_jsonl(config.CONTEXTS_JSONL):
        try:
            seen.add(int(rec["fetch_id"]))
            max_id = max(max_id, int(rec["context_id"]))
        except (KeyError, ValueError):
            continue
    return seen, max_id


# ---------------------------------------------------------------------------
# Per-domain polite delay
# ---------------------------------------------------------------------------
class DomainGate:
    def __init__(self, delay_sec: float) -> None:
        self.delay = delay_sec
        self._last: dict[str, float] = defaultdict(float)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._dict_lock = threading.Lock()

    def wait(self, domain: str) -> None:
        with self._dict_lock:
            lock = self._locks[domain]
        with lock:
            now = time.monotonic()
            last = self._last[domain]
            gap = now - last
            if gap < self.delay:
                time.sleep(self.delay - gap)
            self._last[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# Fetch one URL
# ---------------------------------------------------------------------------
def _fetch_text(url: str, gate: DomainGate, client: httpx.Client) -> tuple[str, str]:
    """Return (status, clean_text). status in {ok, paywall, 404, timeout, blocked}."""
    import trafilatura

    d = domain_of(url)
    gate.wait(d)
    try:
        resp = client.get(url, timeout=config.FETCH_TIMEOUT_SEC, follow_redirects=True)
    except (httpx.TimeoutException,):
        return "timeout", ""
    except Exception:  # noqa: BLE001
        return "blocked", ""

    if resp.status_code == 404:
        return "404", ""
    if resp.status_code >= 400:
        return "blocked", ""
    if len(resp.content) > config.FETCH_MAX_PAGE_BYTES:
        return "blocked", ""

    html = resp.text
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    low = text.lower()
    if len(text) < config.PAYWALL_MIN_TEXT_LEN:
        return "paywall" if text else "blocked", text
    if any(m in low for m in config.PAYWALL_MARKERS):
        return "paywall", text
    return "ok", text


# ---------------------------------------------------------------------------
# spaCy context extraction
# ---------------------------------------------------------------------------
_nlp = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(config.SPACY_MODEL, disable=["lemmatizer"])
    return _nlp


def _name_variants(name: str) -> list[str]:
    """Lowercased full name plus last-name-only."""
    tokens = [t for t in name.split() if t]
    out = [name.lower()]
    if len(tokens) >= 2:
        out.append(tokens[-1].lower())
    return out


def _entity_matches_name(ent_text: str, variants: list[str]) -> bool:
    et = ent_text.lower()
    full = variants[0]
    if full and full in et:
        return True
    return any(et == v or et.endswith(" " + v) for v in variants[1:])


def _sentence_window(
    doc, center_sent_idx: int, sentences: list, short_char_thresh: int, extended: int, default: int
) -> tuple[int, int]:
    """Return (start_sent_idx, end_sent_idx) inclusive, padded when sentences are short."""
    s = sentences[center_sent_idx]
    pad = extended // 2 if len(s.text) < short_char_thresh else default // 2
    start = max(0, center_sent_idx - pad)
    end = min(len(sentences) - 1, center_sent_idx + pad)
    return start, end


def extract_contexts(
    text: str,
    player_name: str,
    max_contexts: int = config.MAX_CONTEXTS_PER_PAGE,
) -> list[dict[str, Any]]:
    """Return a list of context dicts with keys:
        context_text, player_mention_type, sentences_included, spacy_entity_label
    """
    if not text or len(text) < 50:
        return []
    nlp = _get_spacy()
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return []

    variants = _name_variants(player_name)

    # Index: for each sentence, list of PERSON ents it contains and the first
    # PERSON ent (if any).
    sent_person_ents: list[list] = [[] for _ in sentences]
    person_ents_all = [e for e in doc.ents if e.label_ == "PERSON"]
    for ent in person_ents_all:
        for i, s in enumerate(sentences):
            if ent.start >= s.start and ent.end <= s.end:
                sent_person_ents[i].append(ent)
                break

    seen_spans: set[tuple[int, int]] = set()
    contexts: list[dict[str, Any]] = []

    # 1) Named mentions
    for i, ents in enumerate(sent_person_ents):
        if len(contexts) >= max_contexts:
            break
        if not any(_entity_matches_name(e.text, variants) for e in ents):
            continue
        start, end = _sentence_window(
            doc, i, sentences,
            short_char_thresh=config.CONTEXT_SHORT_SENT_CHARS,
            extended=config.CONTEXT_EXTENDED_WINDOW,
            default=config.CONTEXT_WINDOW_SENTENCES,
        )
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        txt = " ".join(s.text.strip() for s in sentences[start : end + 1]).strip()
        if not txt:
            continue
        contexts.append(
            {
                "context_text": txt,
                "player_mention_type": "named",
                "sentences_included": end - start + 1,
                "spacy_entity_label": "PERSON",
            }
        )

    # 2) Pronoun coreference heuristic: a pronoun-bearing sentence that
    #    immediately follows a named-mention sentence, with no *closer*
    #    PERSON entity, is flagged as pronoun_resolved.
    for i, ents in enumerate(sent_person_ents):
        if len(contexts) >= max_contexts:
            break
        if i + 1 >= len(sentences):
            continue
        if not any(_entity_matches_name(e.text, variants) for e in ents):
            continue
        next_sent = sentences[i + 1]
        has_pron = any(tok.pos_ == "PRON" for tok in next_sent)
        if not has_pron:
            continue
        # Require the next sentence to have no other PERSON ent
        if sent_person_ents[i + 1]:
            continue
        start, end = i, i + 1
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        txt = " ".join(s.text.strip() for s in sentences[start : end + 1]).strip()
        contexts.append(
            {
                "context_text": txt,
                "player_mention_type": "pronoun_resolved",
                "sentences_included": end - start + 1,
                "spacy_entity_label": "PERSON",
            }
        )

    return contexts[:max_contexts]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _iter_search_results(players_index: dict[str, str]) -> Iterable[tuple[int, str, int, str]]:
    """Yield (search_id, player_id, rank, url) pairs from searches.jsonl,
    skipping denylisted and malformed entries."""
    for rec in iter_jsonl(config.SEARCHES_JSONL):
        sid = int(rec.get("search_id", 0))
        pid = str(rec.get("player_id", ""))
        for r in rec.get("results", []) or []:
            url = r.get("url") or ""
            if not url:
                continue
            rk = int(r.get("rank", 0))
            yield sid, pid, rk, url


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 — fetch + spaCy context extraction")
    parser.add_argument("--limit", type=int, default=0, help="Process first N fetches only")
    parser.add_argument("--resume", action="store_true", help="Skip (search_id, rank) already in fetches.csv")
    parser.add_argument("--workers", type=int, default=config.FETCH_WORKERS, help="Parallel fetch threads")
    args = parser.parse_args()

    if not config.PLAYERS_CSV.exists() or not config.SEARCHES_JSONL.exists():
        raise SystemExit("Need both data/players.csv and data/searches.jsonl to run fetch.")

    players_df = pd.read_csv(config.PLAYERS_CSV)
    players_df["player_id"] = players_df["player_id"].astype(str)
    pid_to_name = dict(zip(players_df["player_id"], players_df["name"]))

    done_fetches, max_fetch_id = _load_done_fetches() if args.resume else (set(), 0)
    done_contexts, max_context_id = _load_done_contexts() if args.resume else (set(), 0)
    url_cache = _load_url_cache()  # URL-level dedup (always on; harmless without --resume)

    work: list[tuple[int, str, int, str]] = []
    for sid, pid, rk, url in _iter_search_results(pid_to_name):
        if args.resume and (sid, rk) in done_fetches:
            continue
        if pid not in pid_to_name:
            continue
        work.append((sid, pid, rk, url))
        if args.limit and len(work) >= args.limit:
            break

    print(
        f"Fetches planned: {len(work)}  already done: {len(done_fetches)}  "
        f"url-cache entries: {len(url_cache)}  workers: {args.workers}"
    )

    gate = DomainGate(config.PER_DOMAIN_DELAY_SEC)
    client = httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        http2=False,
        follow_redirects=True,
    )
    next_fetch_id = max_fetch_id
    next_context_id = max_context_id
    id_lock = threading.Lock()
    cache_lock = threading.Lock()
    dedup_hits = 0

    def _process(job: tuple[int, str, int, str]) -> dict[str, Any]:
        nonlocal next_fetch_id, next_context_id, dedup_hits
        sid, pid, rk, url = job
        d = domain_of(url)
        tier = config.classify_tier(d)
        with id_lock:
            next_fetch_id += 1
            fid = next_fetch_id

        row = {
            "fetch_id": fid, "search_id": sid, "result_rank": rk, "player_id": pid,
            "url": url, "domain": d, "source_tier": tier if tier else "",
            "status": "", "fetched_at": utc_now_iso(),
            "text_path": "", "text_length": 0,
        }
        if tier is None:
            row["status"] = "denylisted"
            return {"fetch": row, "contexts": []}

        # URL-level dedup: if this exact URL has already been fetched
        # successfully (likely by a different player's search), skip the
        # HTTP round-trip and trafilatura extraction. Still run spaCy
        # against the cached page text for THIS player so the contexts
        # are tied to their player_id.
        cached = url_cache.get(url)
        if cached is not None:
            try:
                text = cached["abs_path"].read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print(f"[cache-read-error] fetch_id={fid} url={url[:80]}: {e}")
                text = ""
            if text:
                with cache_lock:
                    dedup_hits += 1
                row["status"] = "ok"
                row["text_path"] = cached["text_path"]
                row["text_length"] = len(text)

                ctxs: list[dict[str, Any]] = []
                try:
                    raw_ctxs = extract_contexts(text, pid_to_name[pid])
                except Exception as e:  # noqa: BLE001
                    print(f"[spacy-error] fetch_id={fid}: {e}")
                    raw_ctxs = []
                for c in raw_ctxs:
                    with id_lock:
                        next_context_id += 1
                        cid = next_context_id
                    ctxs.append({
                        "context_id": cid, "fetch_id": fid, "player_id": pid,
                        **c,
                    })
                return {"fetch": row, "contexts": ctxs}

        status, text = _fetch_text(url, gate, client)
        row["status"] = status
        if status == "ok":
            page_path = config.PAGES_DIR / f"{fid}.txt"
            page_path.write_text(text, encoding="utf-8")
            row["text_path"] = str(page_path.relative_to(config.PIPELINE_DIR))
            row["text_length"] = len(text)

            # Populate the cache so later jobs in this same run can dedup.
            with cache_lock:
                url_cache[url] = {
                    "text_path": row["text_path"],
                    "text_length": row["text_length"],
                    "status": "ok",
                    "source_tier": row["source_tier"],
                    "domain": d,
                    "abs_path": page_path,
                }

            ctxs: list[dict[str, Any]] = []
            try:
                raw_ctxs = extract_contexts(text, pid_to_name[pid])
            except Exception as e:  # noqa: BLE001
                print(f"[spacy-error] fetch_id={fid}: {e}")
                raw_ctxs = []
            for c in raw_ctxs:
                with id_lock:
                    next_context_id += 1
                    cid = next_context_id
                ctxs.append({
                    "context_id": cid, "fetch_id": fid, "player_id": pid,
                    **c,
                })
            return {"fetch": row, "contexts": ctxs}
        else:
            return {"fetch": row, "contexts": []}

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process, j) for j in work]
            for fut in as_completed(futures):
                try:
                    out = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[worker-error] {e}")
                    continue
                append_csv_row(config.FETCHES_CSV, out["fetch"], FETCH_FIELDS)
                for c in out["contexts"]:
                    append_jsonl(config.CONTEXTS_JSONL, c)
                completed += 1
                if completed % 50 == 0:
                    print(f"  fetched {completed}/{len(work)}")
    finally:
        client.close()

    print(f"Done. Wrote {completed} fetch rows  (url-dedup hits: {dedup_hits}).")


if __name__ == "__main__":
    main()
