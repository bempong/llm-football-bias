"""Helper — construct data/players.csv from the nflverse_raw roster dumps.

Deduplicates players across 2020–2025 seasons by gsis_id. For each unique
player the output row picks the most recent season as the source of
position / team, lists every season the player appeared in, and constructs
a Pro-Football-Reference URL from pfr_id when available.

Run once before search.py. Idempotent — overwrites data/players.csv each
time. If you pre-filter (e.g. drop players already labeled White in a prior
dataset), do it by editing the produced CSV before running search.py.

    python build_players_csv.py [--roster-dir PATH]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_pipeline import config  # noqa: E402

DEFAULT_ROSTER_DIR = Path(__file__).resolve().parents[1] / "nflverse_raw"


def _pfr_url(pfr_id: str) -> str:
    if not pfr_id or not isinstance(pfr_id, str):
        return ""
    pfr_id = pfr_id.strip()
    if not pfr_id or pfr_id.lower() == "nan":
        return ""
    initial = pfr_id[0].upper()
    return f"https://www.pro-football-reference.com/players/{initial}/{pfr_id}.htm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster-dir", type=Path, default=DEFAULT_ROSTER_DIR)
    parser.add_argument("--output", type=Path, default=config.PLAYERS_CSV)
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.roster_dir / "roster_*.csv")))
    if not files:
        raise SystemExit(f"No roster_*.csv files in {args.roster_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["_season"] = df["season"]
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Drop rows with no gsis_id (can't join later)
    all_df = all_df[all_df["gsis_id"].notna()].copy()
    all_df["gsis_id"] = all_df["gsis_id"].astype(str)

    # Deduplicate: one row per player at their most recent season.
    all_df = all_df.sort_values("_season")
    last = all_df.drop_duplicates(subset=["gsis_id"], keep="last").copy()

    # years_active as comma-separated list of seasons
    years_by_id = (
        all_df.groupby("gsis_id")["_season"]
        .apply(lambda s: ",".join(sorted({str(int(y)) for y in s})))
        .to_dict()
    )
    last["years_active"] = last["gsis_id"].map(years_by_id)

    last["pfr_url"] = last["pfr_id"].astype(str).map(_pfr_url)
    last["college"] = last["college"].fillna("").astype(str)
    last["full_name"] = last["full_name"].fillna("").astype(str)
    last["position"] = last["position"].fillna("").astype(str)
    last["team"] = last["team"].fillna("").astype(str)

    out = pd.DataFrame({
        "player_id": last["gsis_id"],
        "name": last["full_name"],
        "position": last["position"],
        "team": last["team"],
        "years_active": last["years_active"],
        "pfr_url": last["pfr_url"],
        "college": last["college"],
    })
    out = out[out["name"].str.strip() != ""].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} players to {args.output}")


if __name__ == "__main__":
    main()
