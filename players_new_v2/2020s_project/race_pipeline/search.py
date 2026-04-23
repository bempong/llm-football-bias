"""Stage 1 — Serper web searches for each (player, query) pair.

Reads data/players.csv. For every player, expands config.DISCOVERY_QUERIES
into concrete queries (substituting name and college), hits Serper, and
appends one JSON line per (player, query) to data/searches.jsonl.

The search stage is content-neutral: queries look for player biography,
profiles, podcasts, and college coverage. Race is extracted only later by
the LLM stage from whatever pages these queries surface. This avoids the
brittleness of enumerating race-specific quoted phrases and keeps the
retrieval set unbiased with respect to any particular group.

Resume semantics: on startup we load every (player_id, query) pair already
present in searches.jsonl into a set, and skip them on replay.

Rate limiting: SERPER_RPS queries per second, enforced with a simple
monotonic-clock gate.

Cost accounting: every query increments a running USD estimate logged to
costs.log. Above SERPER_COST_WARN_USD we print a warning; above
SERPER_COST_HARDSTOP_USD we abort unless --override-cost-cap is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402
from race_pipeline.utils import append_jsonl, log_cost, utc_now_iso  # noqa: E402


def _load_done_pairs() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not config.SEARCHES_JSONL.exists():
        return done
    with config.SEARCHES_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((str(rec.get("player_id", "")), rec.get("query", "")))
    return done


def _next_search_id() -> int:
    if not config.SEARCHES_JSONL.exists():
        return 1
    last = 0
    with config.SEARCHES_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = max(last, int(json.loads(line).get("search_id", 0)))
            except (json.JSONDecodeError, ValueError):
                continue
    return last + 1


class RateLimiter:
    """Shared token-bucket across all search worker threads."""

    def __init__(self, rps: float) -> None:
        self.min_gap = 1.0 / rps if rps > 0 else 0.0
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


def _serper_search(client: httpx.Client, query: str) -> list[dict[str, Any]]:
    """Call Serper and return its normalized top-N organic results."""
    payload = {"q": query, "num": config.SERPER_RESULTS_PER_QUERY}
    resp = client.post(
        config.SERPER_ENDPOINT,
        json=payload,
        headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
        timeout=config.SERPER_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    organic = data.get("organic") or []
    out = []
    for i, r in enumerate(organic[: config.SERPER_RESULTS_PER_QUERY], start=1):
        out.append(
            {
                "rank": i,
                "url": r.get("link", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — Serper searches per player")
    parser.add_argument("--limit", type=int, default=0, help="Process first N players only")
    parser.add_argument("--resume", action="store_true", help="Skip (player, query) pairs already in searches.jsonl")
    parser.add_argument("--override-cost-cap", action="store_true", help="Continue past SERPER_COST_HARDSTOP_USD")
    parser.add_argument("--workers", type=int, default=config.SEARCH_WORKERS, help="Parallel Serper threads")
    args = parser.parse_args()

    if not config.SERPER_API_KEY:
        raise SystemExit("SERPER_API_KEY not set. Add it to .env.")
    if not config.PLAYERS_CSV.exists():
        raise SystemExit(f"{config.PLAYERS_CSV} not found. Run build_players_csv.py first.")

    df = pd.read_csv(config.PLAYERS_CSV)
    if args.limit > 0:
        df = df.head(args.limit)

    done = _load_done_pairs() if args.resume else set()
    next_id = _next_search_id()

    work: list[tuple[str, str, str, str]] = []  # player_id, name, group, query
    for _, row in df.iterrows():
        pid = str(row["player_id"])
        name = str(row["name"])
        college = str(row.get("college", "") or "")
        for group, query in config.render_queries(name, college):
            if args.resume and (pid, query) in done:
                continue
            work.append((pid, name, group, query))

    print(
        f"players: {len(df)}  queries to run: {len(work)}  "
        f"already done: {len(done)}  workers: {args.workers}"
    )

    est_cost = len(done) * config.SERPER_COST_PER_QUERY
    limiter = RateLimiter(config.SERPER_RPS)
    warned = est_cost >= config.SERPER_COST_WARN_USD

    id_lock = threading.Lock()
    write_lock = threading.Lock()
    cost_lock = threading.Lock()
    abort = threading.Event()
    completed = 0

    def _run_one(item: tuple[str, str, str, str]) -> None:
        nonlocal est_cost, next_id, warned, completed
        if abort.is_set():
            return
        pid, name, group, query = item

        with cost_lock:
            if (not args.override_cost_cap) and est_cost >= config.SERPER_COST_HARDSTOP_USD:
                abort.set()
                return

        limiter.wait()
        try:
            results = _serper_search(client, query)
        except Exception as e:  # noqa: BLE001
            print(f"[serper-error] {pid} :: {query[:60]!r} :: {e}")
            results = []

        with id_lock:
            sid = next_id
            next_id += 1

        rec = {
            "search_id": sid,
            "player_id": pid,
            "query": query,
            "query_group": group,
            "retrieved_at": utc_now_iso(),
            "results": results,
        }
        with write_lock:
            append_jsonl(config.SEARCHES_JSONL, rec)
            log_cost(
                stage="search",
                model="serper",
                input_tokens=0,
                output_tokens=0,
                usd=config.SERPER_COST_PER_QUERY,
                note=f"search_id={sid} player={pid}",
            )
            completed += 1
            if completed % 50 == 0:
                print(f"  searched {completed}/{len(work)}")

        with cost_lock:
            est_cost += config.SERPER_COST_PER_QUERY
            if (not warned) and est_cost >= config.SERPER_COST_WARN_USD:
                print(f"[warning] Serper cost estimate has passed ${config.SERPER_COST_WARN_USD:.2f}")
                warned = True

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_run_one, item) for item in work]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[worker-error] {e}")

    if abort.is_set():
        raise SystemExit(
            f"Serper cost estimate ${est_cost:.2f} hit hardstop "
            f"${config.SERPER_COST_HARDSTOP_USD:.2f}. "
            "Re-run with --override-cost-cap to continue."
        )

    print(f"Done. Wrote {completed} searches. Cumulative Serper cost estimate: ${est_cost:.2f}")


if __name__ == "__main__":
    main()
