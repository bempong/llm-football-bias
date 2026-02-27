"""
Shared utilities for prompt/position bias analysis scripts.
"""

import re
from pathlib import Path

import pandas as pd

from . import config
from .log_odds_scoring import compute_log_odds_by_race
from create_plots import generate_all_plots


def to_slug(label: str) -> str:
    """Convert a label like 'P1: Original' to a filename-safe slug 'p1_original'."""
    slug = label.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def load_filtered_data(completions_path: str, condition: str) -> pd.DataFrame:
    """Load CSV and filter to the requested condition."""
    df = pd.read_csv(completions_path)
    if condition != 'both':
        df = df[df['condition'] == condition]
        print(f"  Filtered to '{condition}' condition: {len(df)} rows")
    else:
        print(f"  Loaded {len(df)} rows (all conditions)")
    return df


def print_log_odds_summary(log_odds_df: pd.DataFrame, label: str, kind: str):
    """Print top-20 words associated with each race group."""
    print("\n" + "-" * 80)
    print(f"{kind.upper()} '{label}': TOP 20 WORDS — {config.RACE_GROUP_A.upper()}")
    print("-" * 80)
    for _, row in log_odds_df.head(20).iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")

    print("\n" + "-" * 80)
    print(f"{kind.upper()} '{label}': TOP 20 WORDS — {config.RACE_GROUP_B.upper()}")
    print("-" * 80)
    for _, row in log_odds_df.nsmallest(20, 'z_score').iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")
    print("\n" + "-" * 80)


def analyze_group(
    subset: pd.DataFrame,
    label: str,
    kind: str,
    output_dir: Path,
    text_col: str,
    race_col: str,
    no_plots: bool,
    top_n_words: int,
    z_threshold: float,
):
    """
    Run log-odds analysis on *subset* (already filtered to a single group).

    Parameters
    ----------
    label      : human-readable group name, e.g. "P1: Original" or "QB"
    kind       : 'prompt' or 'position' — used only for printed headers
    output_dir : directory to write tables/ and figures/ into
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING {kind.upper()}: {label}")
    print("=" * 80)
    print(f"\n  Total completions : {len(subset)}")
    print(f"  Race distribution :")
    for race, count in subset[race_col].value_counts().items():
        print(f"    {race}: {count}")

    if len(subset) < 10:
        print(f"\n  Warning: only {len(subset)} completions — skipping.")
        return
    if race_col not in subset.columns:
        print(f"\n  Warning: race column '{race_col}' not found — skipping.")
        return
    if subset[race_col].nunique() < 2:
        print(f"\n  Warning: only one race present — skipping.")
        return

    slug = to_slug(label)

    # Log-odds
    print(f"\n  Computing log-odds ratios...")
    try:
        log_odds_df = compute_log_odds_by_race(subset, text_col=text_col, race_col=race_col)
        log_odds_path = output_dir / "tables" / f"log_odds_tables_{slug}.csv"
        log_odds_path.parent.mkdir(parents=True, exist_ok=True)
        log_odds_df.to_csv(log_odds_path, index=False)
        print(f"  Saved: {log_odds_path}")
    except Exception as e:
        print(f"  Error computing log-odds: {e}")
        return

    if log_odds_df.empty:
        return

    print_log_odds_summary(log_odds_df, label, kind)

    # Plots
    if not no_plots:
        figures_dir = output_dir / "figures"
        try:
            generate_all_plots(
                log_odds_path=str(log_odds_path),
                output_dir=str(figures_dir),
                top_n_words=top_n_words,
                z_threshold=z_threshold,
                position=slug,
            )
            print(f"  Plots saved to: {figures_dir}")
        except Exception as e:
            print(f"  Error generating plots: {e}")


def save_race_distributions(
    df: pd.DataFrame,
    group_col: str,
    groups: list,
    race_col: str,
    output_dir: Path,
):
    """Write a combined race-distribution CSV for all groups."""
    distributions_dir = output_dir / "distributions"
    distributions_dir.mkdir(parents=True, exist_ok=True)

    all_distributions = []
    for grp in groups:
        grp_df = df[df[group_col] == grp]
        if not grp_df.empty:
            dist = grp_df[race_col].value_counts()
            dist.name = grp
            all_distributions.append(dist)

    if all_distributions:
        combined_df = pd.concat(all_distributions, axis=1).fillna(0)
        combined_path = distributions_dir / "combined_race_distributions.csv"
        combined_df.to_csv(combined_path)
        print(f"  Race distributions saved to: {combined_path}")
