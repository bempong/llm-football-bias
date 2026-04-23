"""Stage 4 — mechanically verify that each extracted quote appears at its URL.

For each quote in extractions.jsonl we:
    1. Load the clean text at data/pages/{fetch_id}.txt
    2. Unicode-normalize both the text and the quote (NFC, fold smart quotes
       and dashes, collapse whitespace, lowercase)
    3. Record whether the normalized quote is a substring of the normalized page text

One verification row per quote, written to data/verifications.csv. This is
the pipeline's hallucination check: only quotes with quote_found_at_url=True
flow to aggregation.

Attribution / speaker determination is NOT done here. It was handled by
spaCy during context extraction and by the extractor LLM via
attribution_confident. This stage is a mechanical string check only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402
from race_pipeline.utils import (  # noqa: E402
    append_csv_row,
    domain_of,
    iter_jsonl,
    normalize_text,
    utc_now_iso,
)

VERIFY_FIELDS = [
    "verification_id", "extraction_id", "player_id", "candidate_group",
    "exact_quote", "source_url", "domain", "source_tier",
    "quote_found_at_url", "verified_at",
]


def _load_fetch_index() -> dict[int, dict]:
    """Return {fetch_id: {url, domain, source_tier, text_path, status}}."""
    idx: dict[int, dict] = {}
    if not config.FETCHES_CSV.exists():
        return idx
    with config.FETCHES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fid = int(row["fetch_id"])
            except (KeyError, ValueError):
                continue
            idx[fid] = row
    return idx


def _load_context_to_fetch() -> dict[int, int]:
    out: dict[int, int] = {}
    for rec in iter_jsonl(config.CONTEXTS_JSONL):
        try:
            out[int(rec["context_id"])] = int(rec["fetch_id"])
        except (KeyError, ValueError):
            continue
    return out


def _load_done_and_max() -> tuple[set[int], int]:
    """Return (extraction_ids already verified, max verification_id)."""
    done: set[int] = set()
    max_id = 0
    if not config.VERIFICATIONS_CSV.exists():
        return done, max_id
    with config.VERIFICATIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                eid = int(row["extraction_id"])
                vid = int(row["verification_id"])
            except (KeyError, ValueError):
                continue
            done.add(eid)
            max_id = max(max_id, vid)
    return done, max_id


def _read_page_text(text_path_rel: str) -> str:
    if not text_path_rel:
        return ""
    p = config.PIPELINE_DIR / text_path_rel
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — verify quote exists at source URL")
    parser.add_argument("--limit", type=int, default=0, help="Process first N extractions only")
    parser.add_argument("--resume", action="store_true", help="Skip extraction_ids already verified")
    args = parser.parse_args()

    fetch_index = _load_fetch_index()
    context_to_fetch = _load_context_to_fetch()
    done, max_vid = _load_done_and_max() if args.resume else (set(), 0)

    # Ensure verifications.csv exists (even if empty) so downstream
    # aggregate.py doesn't error out on runs that produced zero quotes.
    if not config.VERIFICATIONS_CSV.exists():
        with config.VERIFICATIONS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=VERIFY_FIELDS).writeheader()

    page_cache: dict[int, str] = {}
    next_vid = max_vid
    n_processed = 0
    n_quotes = 0
    n_ok = 0

    for rec in iter_jsonl(config.EXTRACTIONS_JSONL):
        try:
            eid = int(rec["extraction_id"])
            cid = int(rec["context_id"])
            pid = str(rec["player_id"])
        except (KeyError, ValueError):
            continue
        if args.resume and eid in done:
            continue
        if args.limit and n_processed >= args.limit:
            break
        n_processed += 1

        extractions = rec.get("extractions") or []
        if not extractions:
            continue

        fid = context_to_fetch.get(cid)
        if fid is None:
            continue
        fetch_row = fetch_index.get(fid)
        if not fetch_row:
            continue

        if fid not in page_cache:
            page_cache[fid] = _read_page_text(fetch_row.get("text_path", ""))
        norm_page = normalize_text(page_cache[fid])
        url = fetch_row.get("url", "")
        d = fetch_row.get("domain", "") or domain_of(url)
        tier = fetch_row.get("source_tier", "")

        for ext in extractions:
            quote = ext.get("exact_quote", "")
            if not quote:
                continue
            norm_q = normalize_text(quote)
            found = bool(norm_q) and norm_q in norm_page
            next_vid += 1
            append_csv_row(
                config.VERIFICATIONS_CSV,
                {
                    "verification_id": next_vid,
                    "extraction_id": eid,
                    "player_id": pid,
                    "candidate_group": ext.get("candidate_group", ""),
                    "exact_quote": quote,
                    "source_url": url,
                    "domain": d,
                    "source_tier": tier,
                    "quote_found_at_url": "true" if found else "false",
                    "verified_at": utc_now_iso(),
                },
                VERIFY_FIELDS,
            )
            n_quotes += 1
            if found:
                n_ok += 1

    pct = (100 * n_ok / n_quotes) if n_quotes else 0.0
    print(f"Verified {n_quotes} quotes across {n_processed} extractions.")
    print(f"  quote_found_at_url=True: {n_ok} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
