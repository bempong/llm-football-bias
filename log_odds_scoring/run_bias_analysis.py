#!/usr/bin/env python3
"""
Main Script for Bias Analysis

Computes log-odds ratios to identify words associated with racial groups in LLM completions.
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .log_odds_scoring import compute_log_odds_by_race
from .create_plots import generate_all_plots


def print_summary(log_odds_df: pd.DataFrame):
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

    ablated_df = df[df['condition'] == 'explicit']

    df = ablated_df
    
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
    
    # Print summary
    print_summary(log_odds_df)
    
    # Generate plots
    figures_dir = output_dir / "figures"
    generate_all_plots(
        str(log_odds_path),
        str(figures_dir),
        top_n_words=15,
        z_threshold=1.96
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - log_odds_results.csv")
    print(f"  - figures/")
    print(f"      - distinctive_words_by_race.png")


if __name__ == "__main__":
    main()


# python -m log_odds_scoring.run_bias_analysis \
#     --completions-path "mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_output/llm_generations.csv"
#     --output-dir output_results/bias_analysis_v2 \


# python log_odds_scoring.run_bias_analysis --completions-path "/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_output/llm_generations.csv" --output-dir output_ablated


# python -m log_odds_scoring.run_bias_analysis --completions-path "/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_new_generations/output/n500_gpt-5-mini_generations_explicit.csv" --output-dir output_results/bias_analysis_v2/gpt-5-mini_explicit_generations


# python -m log_odds_scoring.run_bias_analysis --completions-path "/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_completions/output/n500_gpt5mini_completions/llm_completions_explicit.csv" --output-dir output_results/bias_analysis_v2/gpt-5-mini_explicit_completions
