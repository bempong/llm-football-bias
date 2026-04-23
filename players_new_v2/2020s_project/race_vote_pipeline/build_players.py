"""Aggregate nflverse rosters 2020-2025 into a deduped players.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_vote_pipeline import config  # noqa: E402


def main() -> None:
    seen: dict[str, dict[str, str]] = {}
    for year in config.ROSTER_YEARS:
        path = config.ROSTERS_DIR / f"roster_{year}.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                pid  = (row.get("gsis_id") or "").strip()
                name = (row.get("full_name") or "").strip()
                url  = (row.get("headshot_url") or "").strip()
                if not pid or not name or not url:
                    continue
                # Keep the latest-year record for each player so we get the
                # freshest headshot and team/position.
                seen[pid] = {
                    "player_id":    pid,
                    "full_name":    name,
                    "position":     (row.get("position") or "").strip(),
                    "team":         (row.get("team") or "").strip(),
                    "season":       str(year),
                    "headshot_url": url,
                }

    fieldnames = ["player_id", "full_name", "position", "team", "season", "headshot_url"]
    with config.PLAYERS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in seen.values():
            w.writerow(row)

    print(f"wrote {config.PLAYERS_CSV} ({len(seen)} unique players across "
          f"{len(config.ROSTER_YEARS)} seasons)")


if __name__ == "__main__":
    main()
