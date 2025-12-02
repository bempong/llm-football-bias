#!/usr/bin/env python3
"""
Main Script for Bias Analysis V2

Orchestrates log-odds scoring and adjective category analysis on LLM completions.
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .log_odds_scoring import compute_log_odds_by_race
from .adjective_categories import (
    compute_adjective_category_stats,
    create_category_pivot_table
)
from .create_plots import generate_all_plots


def print_summary(log_odds_df: pd.DataFrame, category_stats_df: pd.DataFrame):
    """Print a brief summary of results to stdout."""
    
    print("\n" + "=" * 80)
    print("BIAS ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Log-odds summary
    print("\n" + "-" * 80)
    print(f"TOP 20 WORDS MOST ASSOCIATED WITH {config.RACE_GROUP_A.upper()}")
    print("-" * 80)
    top_a = log_odds_df[log_odds_df['z_score'] > 0].head(20)
    for _, row in top_a.iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")
    
    print("\n" + "-" * 80)
    print(f"TOP 20 WORDS MOST ASSOCIATED WITH {config.RACE_GROUP_B.upper()}")
    print("-" * 80)
    top_b = log_odds_df[log_odds_df['z_score'] < 0].head(20)
    for _, row in top_b.iterrows():
        print(f"  {row['word']:20s} z={row['z_score']:6.2f}  "
              f"({config.RACE_GROUP_A}: {row['count_a']:3d}, "
              f"{config.RACE_GROUP_B}: {row['count_b']:3d})")
    
    # Category analysis summary
    print("\n" + "-" * 80)
    print("ADJECTIVE CATEGORY PROPORTIONS BY RACE")
    print("-" * 80)
    pivot = create_category_pivot_table(category_stats_df)
    print(pivot.to_string())
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze racial bias in LLM-generated football commentary"
    )
    parser.add_argument(
        '--completions-path',
        type=str,
        required=True,
        help='Path to CSV file with LLM completions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save results (default: output/)'
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
        '--no-plots',
        action='store_true',
        help='Skip generating plots (only output CSV files)'
    )
    parser.add_argument(
        '--top-n-words',
        type=int,
        default=15,
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
    print("BIAS ANALYSIS V2: RACIAL BIAS IN FOOTBALL COMMENTARY")
    print("=" * 80)
    print(f"\nInput: {args.completions_path}")
    print(f"Output: {output_dir}/")
    print(f"Text column: {args.text_col}")
    print(f"Race column: {args.race_col}")
    
    # Load completions
    print("\nLoading completions...")
    df = pd.DataFrame(pd.read_csv(args.completions_path))
    
    print(f"  Loaded {len(df)} completions")
    print(f"  Race distribution:")
    print(df[args.race_col].value_counts().to_string())
    
    # Compute log-odds ratios
    print("\n" + "=" * 80)
    print("COMPUTING LOG-ODDS RATIOS (Monroe et al., 2009)")
    print("=" * 80)
    
    log_odds_df = compute_log_odds_by_race(
        df,
        text_col=args.text_col,
        race_col=args.race_col
    )
    
    log_odds_path = output_dir / "log_odds_results.csv"
    log_odds_df.to_csv(log_odds_path, index=False)
    print(f"\n✓ Saved log-odds results to: {log_odds_path}")
    
    # Compute adjective category statistics
    print("\n" + "=" * 80)
    print("COMPUTING ADJECTIVE CATEGORY DISTRIBUTIONS")
    print("=" * 80)
    
    category_stats_df = compute_adjective_category_stats(
        df,
        text_col=args.text_col,
        race_col=args.race_col
    )
    
    category_stats_path = output_dir / "adjective_category_stats.csv"
    category_stats_df.to_csv(category_stats_path, index=False)
    print(f"\n✓ Saved category statistics to: {category_stats_path}")
    
    # Print summary
    print_summary(log_odds_df, category_stats_df)
    
    # Generate plots (unless --no-plots flag is set)
    if not args.no_plots:
        figures_dir = output_dir / "figures"
        generate_all_plots(
            str(log_odds_path),
            str(category_stats_path),
            str(figures_dir),
            top_n_words=args.top_n_words,
            z_threshold=args.z_threshold
        )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - log_odds_results.csv")
    print(f"  - adjective_category_stats.csv")
    if not args.no_plots:
        print(f"  - figures/")
        print(f"      - distinctive_words_by_race.png")
        print(f"      - category_distribution_by_race.png")
        print(f"      - category_bias_ratios.png")


if __name__ == "__main__":
    main()

