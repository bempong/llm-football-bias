"""Stage 5 — deterministic labeling from verified quotes.

Inputs:
    data/verifications.csv   (one row per quote; filter quote_found_at_url=True)
    data/extractions.jsonl   (provides attribution_confident flag per quote)
    data/fetches.csv         (for per-player non-denylisted page counts)
    data/players.csv         (for years_active / career-game approximation)

Outputs:
    data/labels.csv
    data/evidence.csv

Rules are pre-registered in LABELING_RULES.md. They are applied mechanically
here without any model call.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402
from race_pipeline.utils import append_csv_row, iter_jsonl, utc_now_iso  # noqa: E402

LABEL_FIELDS = [
    "player_id", "name", "final_label", "evidence_count",
    "tier1_count", "tier2_count", "tier3_count",
    "rule_applied", "has_low_footprint_flag", "labeled_at",
]
EVIDENCE_FIELDS = [
    "player_id", "candidate_group", "exact_quote", "surrounding_sentence",
    "source_url", "domain", "source_tier", "mention_type", "retrieved_at",
]

LOW_FOOTPRINT_GAMES_THRESH = 10  # games played threshold
LOW_FOOTPRINT_PAGES_THRESH = 5   # non-denylisted fetched pages threshold


def _years_active_to_estimated_games(years_active: str) -> int:
    """Rough proxy: 17 games/season. The real threshold uses career games if
    available; years_active is a best-effort stand-in from nflverse rosters."""
    if not isinstance(years_active, str) or not years_active:
        return 0
    ys = [y for y in years_active.split(",") if y.strip().isdigit()]
    return len(ys) * 17


def _load_extraction_index() -> dict[int, dict[str, Any]]:
    """Return {extraction_id: {context_id, player_id, extractions:[...], attribution per quote}}.
    Quotes are indexed by (extraction_id, exact_quote) downstream."""
    idx: dict[int, dict[str, Any]] = {}
    for rec in iter_jsonl(config.EXTRACTIONS_JSONL):
        try:
            eid = int(rec["extraction_id"])
        except (KeyError, ValueError):
            continue
        idx[eid] = rec
    return idx


def _load_context_index() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for rec in iter_jsonl(config.CONTEXTS_JSONL):
        try:
            out[int(rec["context_id"])] = rec
        except (KeyError, ValueError):
            continue
    return out


def _load_fetch_times() -> dict[int, str]:
    """Map fetch_id -> fetched_at ISO string (per-fetch retrieval timestamp)."""
    out: dict[int, str] = {}
    if not config.FETCHES_CSV.exists():
        return out
    with config.FETCHES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["fetch_id"])] = row.get("fetched_at", "")
            except (KeyError, ValueError):
                continue
    return out


def _pages_per_player() -> dict[str, int]:
    """Count of non-denylisted fetched pages per player (status=ok)."""
    counts: dict[str, int] = defaultdict(int)
    if not config.FETCHES_CSV.exists():
        return counts
    with config.FETCHES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if not row.get("source_tier"):
                continue
            counts[str(row.get("player_id", ""))] += 1
    return counts


def _group_counts(
    rows: list[dict], attribution_ok: dict[tuple[int, str], bool],
) -> dict[str, dict[str, Any]]:
    """Given a player's verified-True rows, compute per-group tier counts and
    per-group distinct-domain sets."""
    per_group: dict[str, dict[str, Any]] = {}
    for row in rows:
        g = row.get("candidate_group", "")
        if not g:
            continue
        eid = int(row["extraction_id"])
        quote = row.get("exact_quote", "")
        if not attribution_ok.get((eid, quote), False):
            continue
        tier = row.get("source_tier", "")
        try:
            tier_i = int(tier)
        except (TypeError, ValueError):
            continue
        d = row.get("domain", "")
        bucket = per_group.setdefault(g, {
            "tier1": 0, "tier2": 0, "tier3": 0,
            "tier1_domains": set(), "tier2_domains": set(), "tier3_domains": set(),
            "rows": [],
        })
        bucket[f"tier{tier_i}"] += 1
        bucket[f"tier{tier_i}_domains"].add(d)
        bucket["rows"].append(row)
    return per_group


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 — deterministic labeling")
    parser.add_argument("--limit", type=int, default=0, help="Process first N players only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite labels.csv / evidence.csv")
    args = parser.parse_args()

    for p in (config.LABELS_CSV, config.EVIDENCE_CSV):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p.name} already exists. Pass --overwrite to replace.")
        if p.exists():
            p.unlink()

    # Load players
    players_df = pd.read_csv(config.PLAYERS_CSV)
    players_df["player_id"] = players_df["player_id"].astype(str)
    pid_to_row = {r["player_id"]: r for _, r in players_df.iterrows()}

    if not config.VERIFICATIONS_CSV.exists():
        raise SystemExit("No verifications.csv — run verify.py first.")

    # Load attribution (from extractions.jsonl)
    ext_idx = _load_extraction_index()
    attribution_ok: dict[tuple[int, str], bool] = {}
    extraction_to_context: dict[int, int] = {}
    for eid, rec in ext_idx.items():
        extraction_to_context[eid] = int(rec.get("context_id", 0))
        for e in rec.get("extractions") or []:
            q = e.get("exact_quote", "")
            attribution_ok[(eid, q)] = bool(e.get("attribution_confident", False))

    ctx_idx = _load_context_index()
    fetch_times = _load_fetch_times()
    page_counts = _pages_per_player()

    # Group verifications by player_id
    by_player: dict[str, list[dict]] = defaultdict(list)
    with config.VERIFICATIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("quote_found_at_url", "").lower() != "true":
                continue
            by_player[str(row["player_id"])].append(row)

    n_labeled = 0
    n_total = 0
    for pid, p_row in pid_to_row.items():
        n_total += 1
        if args.limit and n_total > args.limit:
            break
        rows = by_player.get(pid, [])
        per_group = _group_counts(rows, attribution_ok)

        # Determine footprint
        games = _years_active_to_estimated_games(str(p_row.get("years_active", "")))
        pages = page_counts.get(pid, 0)
        low_footprint = games < LOW_FOOTPRINT_GAMES_THRESH and pages < LOW_FOOTPRINT_PAGES_THRESH

        # Evaluate each group independently under the tier rule
        passing: dict[str, str] = {}  # group -> rule_applied
        tier_counts_total = {"tier1": 0, "tier2": 0, "tier3": 0}
        for g, stats in per_group.items():
            tier_counts_total["tier1"] += stats["tier1"]
            tier_counts_total["tier2"] += stats["tier2"]
            tier_counts_total["tier3"] += stats["tier3"]
            if not low_footprint:
                if stats["tier1"] >= 1 or (stats["tier2"] >= 2 and len(stats["tier2_domains"]) >= 2):
                    passing[g] = "standard"
            else:
                if stats["tier1"] >= 1 or stats["tier2"] >= 1:
                    passing[g] = "relaxed"
                elif stats["tier3"] >= 2 and len(stats["tier3_domains"]) >= 2:
                    passing[g] = "relaxed"

        # Multi-group resolution
        final_label: str | None = None
        rule: str = "insufficient_evidence"
        if not passing:
            final_label, rule = None, "insufficient_evidence"
        elif len(passing) == 1:
            only_g = next(iter(passing))
            final_label, rule = only_g, passing[only_g]
            if only_g == "other_multiracial":
                rule = "multiracial"
        else:
            # Multi-group: prefer monoracial if any monoracial self-ID exists
            mono = [g for g in passing if g != "other_multiracial"]
            if len(mono) == 1:
                final_label, rule = mono[0], passing[mono[0]]
            elif len(mono) > 1:
                # Monoracial conflict: most recent source wins
                def _latest(g: str) -> str:
                    stats = per_group[g]
                    latest = ""
                    for r in stats["rows"]:
                        eid = int(r["extraction_id"])
                        cid = extraction_to_context.get(eid)
                        # Attempt to use fetch time from fetches.csv via the fetch_id in contexts
                        if cid and cid in ctx_idx:
                            fid = int(ctx_idx[cid].get("fetch_id", 0))
                            t = fetch_times.get(fid, "")
                            if t and t > latest:
                                latest = t
                    return latest
                winner = max(mono, key=_latest)
                final_label, rule = winner, passing[winner]
            else:
                # Only multiracial self-IDs
                final_label, rule = "other_multiracial", "multiracial"

        if final_label:
            n_labeled += 1

        append_csv_row(
            config.LABELS_CSV,
            {
                "player_id": pid,
                "name": p_row["name"],
                "final_label": final_label or "",
                "evidence_count": sum(tier_counts_total.values()),
                "tier1_count": tier_counts_total["tier1"],
                "tier2_count": tier_counts_total["tier2"],
                "tier3_count": tier_counts_total["tier3"],
                "rule_applied": rule,
                "has_low_footprint_flag": "true" if low_footprint else "false",
                "labeled_at": utc_now_iso(),
            },
            LABEL_FIELDS,
        )

        # Write evidence rows only for the passing label's group (to keep
        # evidence.csv focused on what actually supports the final label).
        if final_label and final_label in per_group:
            stats = per_group[final_label]
            for r in stats["rows"]:
                eid = int(r["extraction_id"])
                cid = extraction_to_context.get(eid)
                ctx = ctx_idx.get(cid, {}) if cid else {}
                ext_rec = ext_idx.get(eid, {})
                surrounding = ""
                for e in ext_rec.get("extractions") or []:
                    if e.get("exact_quote", "") == r.get("exact_quote", ""):
                        surrounding = e.get("surrounding_sentence", "")
                        break
                retrieved_at = ""
                if cid and cid in ctx_idx:
                    fid = int(ctx_idx[cid].get("fetch_id", 0))
                    retrieved_at = fetch_times.get(fid, "")
                append_csv_row(
                    config.EVIDENCE_CSV,
                    {
                        "player_id": pid,
                        "candidate_group": r.get("candidate_group", ""),
                        "exact_quote": r.get("exact_quote", ""),
                        "surrounding_sentence": surrounding,
                        "source_url": r.get("source_url", ""),
                        "domain": r.get("domain", ""),
                        "source_tier": r.get("source_tier", ""),
                        "mention_type": ctx.get("player_mention_type", ""),
                        "retrieved_at": retrieved_at,
                    },
                    EVIDENCE_FIELDS,
                )

    print(f"Labeled {n_labeled}/{n_total} players.")
    print(f"Labels:   {config.LABELS_CSV}")
    print(f"Evidence: {config.EVIDENCE_CSV}")


if __name__ == "__main__":
    main()
