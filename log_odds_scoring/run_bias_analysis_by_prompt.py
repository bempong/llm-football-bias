#!/usr/bin/env python3
"""
Prompt-Specific Bias Analysis

Runs bias analysis grouped by prompt_id so that bias patterns can be
compared across prompts.

Output structure
----------------
<output_dir>/
  tables/   log_odds_tables_<slug>.csv
  figures/
  distributions/  combined_race_distributions.csv
"""

import argparse
import re
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from . import config
from .log_odds_scoring import compute_log_odds_by_race
from create_plots import generate_all_plots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_slug(label: str) -> str:
    """Convert an arbitrary label to a filename-safe slug."""
    slug = label.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def _print_summary(log_odds_df: pd.DataFrame, label: str, kind: str):
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


# ---------------------------------------------------------------------------
# Core per-group analysis
# ---------------------------------------------------------------------------

def _analyze_group(
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
    label   : human-readable group name, e.g. "P1: Original" or "QB"
    kind    : 'prompt' or 'position' — used only for printed headers
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING {kind.upper()}: {label}")
    print("=" * 80)

    print(f"\n  Total completions: {len(subset)}")
    print(f"  Race distribution:")
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

    slug = _to_slug(label)

    # --- log-odds ---
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

    _print_summary(log_odds_df, label, kind)

    # --- plots ---
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


# ---------------------------------------------------------------------------
# Prompt-level analysis
# ---------------------------------------------------------------------------

def run_bias_analysis_by_prompt(
    df: pd.DataFrame,
    output_dir: Path,
    text_col: str,
    race_col: str,
    no_plots: bool,
    top_n_words: int,
    z_threshold: float,
):
    """Run log-odds analysis for every unique prompt_id in *df*."""
    prompt_dir = output_dir
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PROMPT-LEVEL ANALYSIS")
    print("=" * 80)

    prompt_counts = df['prompt_id'].value_counts()
    prompt_ids = prompt_counts.index.tolist()

    print(f"\nPrompt distribution ({len(prompt_ids)} prompts):")
    for pid in prompt_ids:
        print(f"  {pid}: {prompt_counts[pid]}")

    for prompt_id in prompt_ids:
        _analyze_group(
            subset=df[df['prompt_id'] == prompt_id].copy(),
            label=prompt_id,
            kind='prompt',
            output_dir=prompt_dir,
            text_col=text_col,
            race_col=race_col,
            no_plots=no_plots,
            top_n_words=top_n_words,
            z_threshold=z_threshold,
        )

    _save_race_distributions(df, 'prompt_id', prompt_ids, race_col, prompt_dir)
    print(f"\nPrompt-level results saved to: {prompt_dir}/")


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _save_race_distributions(
    df: pd.DataFrame,
    group_col: str,
    groups: list,
    race_col: str,
    output_dir: Path,
):
    """Save a combined race-distribution CSV for all groups."""
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


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_combined_analysis(
    completions_path: str,
    output_dir: str = 'output_combined',
    text_col: str = 'completion_text',
    race_col: str = 'true_race',
    condition: str = 'explicit',
    no_plots: bool = False,
    top_n_words: int = 15,
    z_threshold: float = 1.96,
):
    """Run prompt-level bias analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROMPT-LEVEL BIAS ANALYSIS: RACIAL BIAS IN FOOTBALL COMMENTARY")
    print("=" * 80)
    print(f"\nInput:      {completions_path}")
    print(f"Output:     {output_dir}/")
    print(f"Text col:   {text_col}")
    print(f"Race col:   {race_col}")
    print(f"Condition:  {condition}")

    # Load
    print("\nLoading completions...")
    df = pd.read_csv(completions_path)

    if condition != 'both':
        df = df[df['condition'] == condition]
        print(f"  Filtered to '{condition}' condition")

    print(f"  Loaded {len(df)} completions")
    print(f"\nOverall race distribution:")
    print(df[race_col].value_counts().to_string())

    run_bias_analysis_by_prompt(
        df=df,
        output_dir=output_dir,
        text_col=text_col,
        race_col=race_col,
        no_plots=no_plots,
        top_n_words=top_n_words,
        z_threshold=z_threshold,
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze racial bias in LLM football commentary by prompt"
    )
    parser.add_argument(
        '--completions-path', type=str,
        default="/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_new_generations/output/4o_mini_2010_2026_exp_01.csv",
        help='Path to CSV file with LLM completions'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output_combined',
        help='Root directory to save results (default: output_combined/)'
    )
    parser.add_argument(
        '--text-col', type=str, default='completion_text',
        help='Column name containing completion text'
    )
    parser.add_argument(
        '--race-col', type=str, default='true_race',
        help='Column name containing race labels'
    )
    parser.add_argument(
        '--condition', type=str, default='explicit',
        choices=['explicit', 'ablated', 'both'],
        help='Which condition to analyze (default: explicit)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots (only output CSV files)'
    )
    parser.add_argument(
        '--top-n-words', type=int, default=15,
        help='Number of top words to show in plots (default: 15)'
    )
    parser.add_argument(
        '--z-threshold', type=float, default=1.96,
        help='Minimum |z-score| for word plots (default: 1.96, p<0.05)'
    )
    args = parser.parse_args()

    run_combined_analysis(
        completions_path=args.completions_path,
        output_dir=args.output_dir,
        text_col=args.text_col,
        race_col=args.race_col,
        condition=args.condition,
        no_plots=args.no_plots,
        top_n_words=args.top_n_words,
        z_threshold=args.z_threshold,
    )


if __name__ == "__main__":
    main()


# Prompt Command
"""
python -m log_odds_scoring.run_bias_analysis_by_prompt \
   --completions-path "llm_new_generations/output/4o_mini_2010_2026_exp_01.csv" \
   --output-dir output_results/bias_analysis_v4/4o-mini/by_prompt \
   --top-n-words 20 \
   --z-threshold 0.01
"""