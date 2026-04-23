"""Stage 3 — LLM extraction of verbatim self-ID quotes, per context.

Reads data/contexts.jsonl. For each context not yet processed, runs a cheap
regex pre-filter (config.RACE_KEYWORDS_PATTERN) to decide whether it's
even plausibly race-relevant. Contexts that fail the filter are recorded
with `extractions=[]` and `skipped_by_keyword_filter=true` — no LLM call.
Contexts that pass are sent to Claude in parallel (EXTRACT_WORKERS
concurrent calls).

Empty extractions=[] is always written so resume logic sees the context
as processed.

Every API call is logged to costs.log with token usage and estimated USD.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402
from race_pipeline.prompts import render_extraction_prompt  # noqa: E402
from race_pipeline.utils import (  # noqa: E402
    append_jsonl,
    estimate_claude_cost,
    iter_jsonl,
    log_cost,
    utc_now_iso,
)

_RACE_IDENTITY_RE = re.compile(config.RACE_IDENTITY_PATTERN, re.IGNORECASE)
_FIRST_PERSON_RE  = re.compile(config.FIRST_PERSON_PATTERN)  # case-sensitive "I"


def _passes_keyword_gate(text: str) -> bool:
    """Return True only if the context has BOTH a race/ethnicity identity
    term AND a first-person pronoun. Self-identification requires both —
    either alone is near-always a false positive."""
    if not text:
        return False
    if not _RACE_IDENTITY_RE.search(text):
        return False
    return bool(_FIRST_PERSON_RE.search(text))


class RateLimiter:
    """Shared token-bucket used by every worker thread.

    Pass max_rpm=45 to stay under Anthropic tier-1 Haiku's 50 req/min cap.
    """

    def __init__(self, max_rpm: float) -> None:
        self.min_gap = 60.0 / max_rpm if max_rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
            else:
                sleep_for = 0.0
            self._next_allowed = max(now, self._next_allowed) + self.min_gap
        if sleep_for > 0:
            time.sleep(sleep_for)


def _load_done_contexts() -> tuple[set[int], int]:
    seen: set[int] = set()
    max_id = 0
    for rec in iter_jsonl(config.EXTRACTIONS_JSONL):
        try:
            seen.add(int(rec["context_id"]))
            max_id = max(max_id, int(rec["extraction_id"]))
        except (KeyError, ValueError):
            continue
    return seen, max_id


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_extractions(raw: str) -> list[dict[str, Any]] | None:
    """Robust-ish JSON parsing. Returns list of extraction dicts, or None if
    the output is malformed beyond recovery."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    items = obj.get("extractions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append({
            "exact_quote": str(it.get("exact_quote", "")),
            "surrounding_sentence": str(it.get("surrounding_sentence", "")),
            "first_person": bool(it.get("first_person", False)),
            "attribution_confident": bool(it.get("attribution_confident", False)),
            "candidate_group": str(it.get("candidate_group", "")).lower(),
        })
    return out


def _call_claude(
    client, model: str, prompt: str,
    limiter: RateLimiter,
    max_retries: int = config.EXTRACT_MAX_RETRIES,
) -> tuple[str, int, int]:
    """Return (text, input_tokens, output_tokens).

    Honors the shared rate limiter before each attempt, and retries on
    transient errors (429 rate-limit, 5xx, connection drops) with
    exponential backoff starting at 2s.
    """
    attempt = 0
    while True:
        limiter.wait()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=config.EXTRACTION_MAX_TOKENS,
                temperature=config.EXTRACTION_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            retryable = (status == 429) or (status is not None and status >= 500) or status is None
            attempt += 1
            if not retryable or attempt > max_retries:
                raise
            backoff = min(60.0, 2.0 ** attempt)
            time.sleep(backoff)
            continue

        parts = []
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                parts.append(blk.text)
        usage = getattr(resp, "usage", None)
        in_toks = getattr(usage, "input_tokens", 0) or 0
        out_toks = getattr(usage, "output_tokens", 0) or 0
        return "".join(parts), in_toks, out_toks


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 — LLM verbatim-quote extraction")
    parser.add_argument("--limit", type=int, default=0, help="Process first N contexts only")
    parser.add_argument("--resume", action="store_true", help="Skip contexts already in extractions.jsonl")
    parser.add_argument("--model", type=str, default=config.DEFAULT_EXTRACTOR_MODEL, help="Claude model")
    parser.add_argument("--workers", type=int, default=config.EXTRACT_WORKERS,
                        help="Concurrent Claude calls")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable the race-keyword pre-filter (send every context to Claude)")
    args = parser.parse_args()

    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it to .env.")

    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise SystemExit("Install anthropic: pip install anthropic") from e

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    players_df = pd.read_csv(config.PLAYERS_CSV)
    players_df["player_id"] = players_df["player_id"].astype(str)
    pid_to_name = dict(zip(players_df["player_id"], players_df["name"]))

    done, max_id = _load_done_contexts() if args.resume else (set(), 0)

    contexts = []
    for rec in iter_jsonl(config.CONTEXTS_JSONL):
        try:
            cid = int(rec["context_id"])
        except (KeyError, ValueError):
            continue
        if args.resume and cid in done:
            continue
        contexts.append(rec)
        if args.limit and len(contexts) >= args.limit:
            break

    # Pre-filter: partition into LLM-bound and gate-rejected contexts.
    if args.no_filter:
        llm_contexts = contexts
        gated = []
    else:
        llm_contexts, gated = [], []
        for c in contexts:
            if _passes_keyword_gate(c.get("context_text", "")):
                llm_contexts.append(c)
            else:
                gated.append(c)

    print(
        f"Contexts considered: {len(contexts)}  "
        f"gate-rejected: {len(gated)}  "
        f"to LLM: {len(llm_contexts)}  "
        f"resume-skipped: {len(done)}  "
        f"workers: {args.workers}  model: {args.model}"
    )

    id_lock = threading.Lock()
    write_lock = threading.Lock()
    limiter = RateLimiter(config.EXTRACT_MAX_RPM)
    next_id = max_id
    running_cost = 0.0
    completed = 0

    # Record gate-rejected contexts first (no LLM call, no cost).
    for c in gated:
        with id_lock:
            next_id += 1
            eid = next_id
        rec = {
            "extraction_id": eid,
            "context_id": int(c["context_id"]),
            "player_id": str(c["player_id"]),
            "extracted_at": utc_now_iso(),
            "extractor_model": "keyword-gate",
            "skipped_by_keyword_filter": True,
            "extractions": [],
        }
        append_jsonl(config.EXTRACTIONS_JSONL, rec)

    def _process(c: dict[str, Any]) -> float:
        nonlocal next_id, completed, running_cost
        pid = str(c["player_id"])
        name = pid_to_name.get(pid, pid)
        prompt = render_extraction_prompt(
            player_name=name,
            mention_type=c.get("player_mention_type", "named"),
            context_text=c["context_text"],
        )
        try:
            raw, in_toks, out_toks = _call_claude(client, args.model, prompt, limiter)
        except Exception as e:  # noqa: BLE001
            print(f"[claude-error] context_id={c['context_id']}: {e}")
            return 0.0

        parsed = _parse_extractions(raw)
        usd = estimate_claude_cost(args.model, in_toks, out_toks)

        with id_lock:
            next_id += 1
            eid = next_id

        rec = {
            "extraction_id": eid,
            "context_id": int(c["context_id"]),
            "player_id": pid,
            "extracted_at": utc_now_iso(),
            "extractor_model": args.model,
            "extractions": parsed if parsed is not None else [],
        }
        if parsed is None:
            rec["parse_error"] = True
            rec["raw_output"] = raw[:2000]

        with write_lock:
            append_jsonl(config.EXTRACTIONS_JSONL, rec)
            log_cost(
                stage="extract",
                model=args.model,
                input_tokens=in_toks,
                output_tokens=out_toks,
                usd=usd,
                note=f"extraction_id={eid} context_id={c['context_id']} player={pid}",
            )
            running_cost += usd
            completed += 1
            if completed % 50 == 0:
                print(f"  extracted {completed}/{len(llm_contexts)}  running cost ${running_cost:.3f}")

        return usd

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process, c) for c in llm_contexts]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[worker-error] {e}")

    print(
        f"Done. LLM calls: {len(llm_contexts)}  gate-skipped: {len(gated)}  "
        f"total rows: {len(llm_contexts) + len(gated)}  "
        f"estimated cost ${running_cost:.3f}."
    )


if __name__ == "__main__":
    main()
