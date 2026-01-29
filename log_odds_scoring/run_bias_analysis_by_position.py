#!/usr/bin/env python3
"""
Position-Specific Bias Analysis

Runs bias analysis separately for each position (QB, RB, WR, DEF).
This allows us to see if bias patterns differ by position.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from . import config
from .log_odds_scoring import compute_log_odds_by_race
from .adjective_categories import (
    compute_adjective_category_stats,
    create_category_pivot_table
)
from create_plots import generate_all_plots


def print_position_header(position: str):
    """Print a header for position analysis."""
    print("\n" + "=" * 80)
    print(f"ANALYZING POSITION: {position}")
    print("=" * 80)


def print_summary(log_odds_df: pd.DataFrame, category_stats_df: pd.DataFrame, position: str):
    """Print a brief summary of results for a position."""

    print("\n" + "-" * 80)
    print(f"POSITION {position}: TOP 15 WORDS MOST ASSOCIATED WITH {config.RACE_GROUP_A.upper()}")
    print("-" * 80)
    top_a = log_odds_df[log_odds_df['z_score'] > 0].head(15)
    for _, row in top_a.iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")

    print("\n" + "-" * 80)
    print(f"POSITION {position}: TOP 15 WORDS MOST ASSOCIATED WITH {config.RACE_GROUP_B.upper()}")
    print("-" * 80)
    top_b = log_odds_df[log_odds_df['z_score'] < 0].head(15)
    for _, row in top_b.iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")

    # Category analysis summary
    print("\n" + "-" * 80)
    print(f"POSITION {position}: ADJECTIVE CATEGORY PROPORTIONS BY RACE")
    print("-" * 80)
    pivot = create_category_pivot_table(category_stats_df)
    print(pivot.to_string())

    print("\n" + "-" * 80)


def analyze_position(
    df: pd.DataFrame,
    position: str,
    output_dir: Path,
    text_col: str,
    race_col: str,
    no_plots: bool,
    top_n_words: int,
    z_threshold: float
):
    """Run bias analysis for a specific position."""

    print_position_header(position)

    # Filter to this position
    position_df = df[df['position'] == position].copy()

    # Print race distribution
    print(f"\nPosition: {position}")
    print(f"  Total completions: {len(position_df)}")
    print(f"\n  Race distribution:")
    race_counts = position_df[race_col].value_counts()
    for race, count in race_counts.items():
        print(f"    {race}: {count}")

    # Check if we have enough data
    if len(position_df) < 10:
        print(f"\n⚠️  Warning: Only {len(position_df)} completions for {position}. Skipping analysis.")
        return

    if race_col not in position_df.columns:
        print(f"\n⚠️  Warning: Race column '{race_col}' not found. Skipping {position}.")
        return

    # Check if we have both races
    unique_races = position_df[race_col].unique()
    if len(unique_races) < 2:
        print(f"\n⚠️  Warning: Only one race found in {position} data. Skipping analysis.")
        return

    # Create position-specific output directory
    position_output_dir = output_dir / position.lower()
    position_output_dir.mkdir(parents=True, exist_ok=True)

    # Compute log-odds ratios
    print(f"\nComputing log-odds ratios for {position}...")

    try:
        log_odds_df = compute_log_odds_by_race(
            position_df,
            text_col=text_col,
            race_col=race_col
        )

        log_odds_path = position_output_dir / "log_odds_results.csv"
        log_odds_df.to_csv(log_odds_path, index=False)
        print(f"✓ Saved log-odds results to: {log_odds_path}")
    except Exception as e:
        print(f"✗ Error computing log-odds for {position}: {e}")
        return

    # Compute adjective category statistics
    print(f"\nComputing adjective category distributions for {position}...")

    try:
        category_stats_df = compute_adjective_category_stats(
            position_df,
            text_col=text_col,
            race_col=race_col
        )

        category_stats_path = position_output_dir / "adjective_category_stats.csv"
        category_stats_df.to_csv(category_stats_path, index=False)
        print(f"✓ Saved category statistics to: {category_stats_path}")
    except Exception as e:
        print(f"✗ Error computing categories for {position}: {e}")
        category_stats_df = pd.DataFrame()

    # Print summary
    if not log_odds_df.empty:
        print_summary(log_odds_df, category_stats_df, position)

    # Generate plots (unless --no-plots flag is set)
    if not no_plots and not log_odds_df.empty:
        figures_dir = position_output_dir / "figures"
        try:
            generate_all_plots(
                str(log_odds_path),
                str(category_stats_path) if not category_stats_df.empty else None,
                str(figures_dir),
                top_n_words=top_n_words,
                z_threshold=z_threshold
            )
            print(f"✓ Generated plots in: {figures_dir}")
        except Exception as e:
            print(f"✗ Error generating plots for {position}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze racial bias in LLM-generated football commentary by position"
    )
    parser.add_argument(
        '--completions-path',
        type=str,
        default="/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_output/llm_generations.csv",
        help='Path to CSV file with LLM completions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_by_position',
        help='Directory to save results (default: output_by_position/)'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        default='completion_text',
        help='Column name containing completion text'
    )
    parser.add_argument(
        '--race-col',
        type=str,
        default='true_race',
        help='Column name containing race labels'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='explicit',
        choices=['explicit', 'ablated', 'both'],
        help='Which condition to analyze (default: explicit)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (only output CSV files)'
    )
    parser.add_argument(
        '--top-n-words',
        type=int,
        default=20,
        help='Number of top words to show in plots (default: 15)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=1.96,
        help='Minimum |z-score| for word plots (default: 1.96, p<0.05)'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("POSITION-SPECIFIC BIAS ANALYSIS: RACIAL BIAS IN FOOTBALL COMMENTARY")
    print("=" * 80)
    print(f"\nInput: {args.completions_path}")
    print(f"Output: {output_dir}/")
    print(f"Text column: {args.text_col}")
    print(f"Race column: {args.race_col}")
    print(f"Condition: {args.condition}")

    # Load completions
    print("\nLoading completions...")
    df = pd.read_csv(args.completions_path)

    # Filter by condition if specified
    if args.condition != 'both':
        df = df[df['condition'] == args.condition]
        print(f"  Filtered to '{args.condition}' condition")

    print(f"  Loaded {len(df)} completions")

    # Define positions to analyze
    positions = ['QB', 'RB', 'WR', 'DEF']

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"\nTotal completions: {len(df)}")
    print(f"\nPosition distribution:")
    position_counts = df['position'].value_counts()
    for pos in positions:
        count = position_counts.get(pos, 0)
        print(f"  {pos}: {count}")

    print(f"\nOverall race distribution:")
    print(df[args.race_col].value_counts().to_string())

    # Analyze each position
    for position in positions:
        analyze_position(
            df=df,
            position=position,
            output_dir=output_dir,
            text_col=args.text_col,
            race_col=args.race_col,
            no_plots=args.no_plots,
            top_n_words=args.top_n_words,
            z_threshold=args.z_threshold
        )

    # Final summary
    print("\n" + "=" * 80)
    print("POSITION-SPECIFIC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    for position in positions:
        pos_dir = output_dir / position.lower()
        if pos_dir.exists():
            print(f"\n{position}:")
            print(f"  {pos_dir}/log_odds_results.csv")
            print(f"  {pos_dir}/adjective_category_stats.csv")
            if not args.no_plots:
                print(f"  {pos_dir}/figures/")


if __name__ == "__main__":
    main()
