"""
Merge 2020_2026_initial.csv with:
  - dataset_v2_1960_2019.csv  -> merged_1960_2026.csv
  - dataset_v2_2010_2019.csv  -> merged_2010_2026.csv

player_id is reset to be contiguous starting from 0 in each output.
"""

import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

def merge(base_path: Path, extra_path: Path, out_path: Path) -> None:
    base = pd.read_csv(base_path)
    extra = pd.read_csv(extra_path)
    merged = pd.concat([base, extra], ignore_index=True)
    merged["player_id"] = range(len(merged))
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows -> {out_path.name}")

if __name__ == "__main__":
    new_players = HERE / "2020_2026_initial.csv"

    merge(
        HERE / "dataset_v2_1960_2019.csv",
        new_players,
        HERE / "merged_1960_2026.csv",
    )

    merge(
        HERE / "dataset_v2_2010_2019.csv",
        new_players,
        HERE / "merged_2010_2026.csv",
    )
