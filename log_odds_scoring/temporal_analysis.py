#!/usr/bin/env python3
"""
Temporal Analysis: Log-odds scoring by decade (1990s, 2000s, 2010s).

Analyzes how racial bias patterns in LLM-generated commentary may differ
based on the era of the player data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .log_odds_scoring import compute_log_odds_by_race, tokenize
from . import config


def assign_decade(year: int) -> str:
    """Assign a year to its decade label."""
    if 1990 <= year <= 1999:
        return "1990s"
    elif 2000 <= year <= 2009:
        return "2000s"
    elif 2010 <= year <= 2019:
        return "2010s"
    else:
        return "other"


def run_temporal_analysis(
    completions_path: str,
    output_dir: str,
    condition: str = "explicit"
) -> dict:
    """
    Run log-odds analysis separately for each decade.
    
    Args:
        completions_path: Path to the LLM completions CSV
        output_dir: Directory to save results
        condition: Which condition to analyze ('explicit', 'ablated', or 'both')
    
    Returns:
        Dictionary with DataFrames for each decade's results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load completions
    print(f"Loading completions from {completions_path}...")
    df = pd.read_csv(completions_path)
    
    # Filter by condition if specified
    if condition != "both":
        df = df[df['condition'] == condition].copy()
        print(f"  Filtered to {condition} condition: {len(df)} rows")
    
    # Assign decades
    df['decade'] = df['example_year'].apply(assign_decade)
    
    # Show distribution
    print("\n=== Player Distribution by Decade ===")
    decade_counts = df.groupby(['decade', 'true_race']).size().unstack(fill_value=0)
    print(decade_counts)
    print()
    
    decades = ["1990s", "2000s", "2010s"]
    results = {}
    
    for decade in decades:
        decade_df = df[df['decade'] == decade].copy()
        
        if len(decade_df) == 0:
            print(f"\n--- {decade}: No data ---")
            continue
            
        n_white = len(decade_df[decade_df['true_race'] == 'white'])
        n_nonwhite = len(decade_df[decade_df['true_race'] == 'nonwhite'])
        
        print(f"\n{'='*60}")
        print(f"=== {decade} (n={len(decade_df)}, white={n_white}, nonwhite={n_nonwhite}) ===")
        print(f"{'='*60}")
        
        # Run log-odds analysis
        log_odds_df = compute_log_odds_by_race(
            decade_df,
            text_col='completion_text',
            race_col='true_race'
        )
        
        # Save results
        log_odds_path = output_path / f"log_odds_{decade}.csv"
        log_odds_df.to_csv(log_odds_path, index=False)
        
        results[decade] = {
            'log_odds': log_odds_df,
            'n_samples': len(decade_df),
            'n_white': n_white,
            'n_nonwhite': n_nonwhite
        }
        
        # Print top distinctive words
        print(f"\nTop 10 WHITE-associated words ({decade}):")
        white_words = log_odds_df.nlargest(10, 'z_score')
        for _, row in white_words.iterrows():
            print(f"  {row['word']:20s} z={row['z_score']:+6.2f}  (white:{row['count_a']:4d}, nonwhite:{row['count_b']:4d})")
        
        print(f"\nTop 10 NONWHITE-associated words ({decade}):")
        nonwhite_words = log_odds_df.nsmallest(10, 'z_score')
        for _, row in nonwhite_words.iterrows():
            print(f"  {row['word']:20s} z={row['z_score']:+6.2f}  (white:{row['count_a']:4d}, nonwhite:{row['count_b']:4d})")
    
    return results


def create_temporal_comparison_plots(results: dict, output_dir: str):
    """Create plots comparing bias patterns across decades."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    decades = [d for d in ["1990s", "2000s", "2010s"] if d in results]
    
    if len(decades) < 2:
        print("Need at least 2 decades for comparison plots")
        return
    
    # --- Plot 1: Key Words Across Decades ---
    # Find words that appear significantly in at least one decade
    key_words = set()
    for decade, data in results.items():
        significant = data['log_odds'][data['log_odds']['z_score'].abs() >= 1.96]
        top_words = set(significant.nlargest(10, 'z_score')['word'].tolist())
        bottom_words = set(significant.nsmallest(10, 'z_score')['word'].tolist())
        key_words.update(top_words)
        key_words.update(bottom_words)
    
    # Select a subset of the most interesting words
    interest_words = [
        'poise', 'accuracy', 'precision', 'composure', 'pocket', 'decision',
        'speed', 'athletic', 'explosive', 'agility', 'instincts', 'burst',
        'smart', 'intelligent', 'quick', 'powerful', 'natural'
    ]
    tracked_words = [w for w in interest_words if w in key_words][:12]
    
    if tracked_words:
        # Build data for heatmap
        heatmap_data = []
        for word in tracked_words:
            row = {'word': word}
            for decade in decades:
                word_row = results[decade]['log_odds'][results[decade]['log_odds']['word'] == word]
                if len(word_row) > 0:
                    row[decade] = word_row.iloc[0]['z_score']
                else:
                    row[decade] = 0
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('word')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_df, 
            annot=True, 
            fmt='.1f', 
            cmap='RdBu_r', 
            center=0,
            vmin=-6, 
            vmax=6,
            cbar_kws={'label': 'Z-score (+ = white-associated, - = nonwhite-associated)'},
            ax=ax
        )
        ax.set_title('Racial Word Associations Across Decades\n(z-scores from log-odds analysis)', fontsize=14)
        ax.set_xlabel('Decade')
        ax.set_ylabel('Word')
        plt.tight_layout()
        plt.savefig(output_path / 'temporal_word_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path / 'temporal_word_heatmap.png'}")
    
    # --- Plot 2: Overall Bias Trend ---
    # Calculate mean absolute z-score for significant words per decade
    trend_data = []
    for decade in decades:
        log_odds_df = results[decade]['log_odds']
        significant = log_odds_df[log_odds_df['z_score'].abs() >= 1.96]
        
        trend_data.append({
            'decade': decade,
            'n_significant_words': len(significant),
            'mean_abs_z': significant['z_score'].abs().mean() if len(significant) > 0 else 0,
            'max_white_z': log_odds_df['z_score'].max(),
            'max_nonwhite_z': log_odds_df['z_score'].min(),
            'n_samples': results[decade]['n_samples']
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of significant words
    ax1 = axes[0]
    bars = ax1.bar(trend_df['decade'], trend_df['n_significant_words'], color='#5D6D7E')
    ax1.set_xlabel('Decade')
    ax1.set_ylabel('Number of Significant Words (|z| >= 1.96)')
    ax1.set_title('Vocabulary Differentiation by Race Over Time')
    
    # Add sample size annotations
    for bar, n in zip(bars, trend_df['n_samples']):
        ax1.annotate(f'n={n}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    # Maximum z-scores (most extreme associations)
    ax2 = axes[1]
    x = np.arange(len(decades))
    width = 0.35
    ax2.bar(x - width/2, trend_df['max_white_z'], width, label='Most White-associated', color='#4A90A4')
    ax2.bar(x + width/2, trend_df['max_nonwhite_z'].abs(), width, label='Most Nonwhite-associated', color='#E07B54')
    ax2.set_xlabel('Decade')
    ax2.set_ylabel('Maximum |Z-score|')
    ax2.set_title('Strength of Most Extreme Word Associations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(decades)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'temporal_bias_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'temporal_bias_trend.png'}")
    
    # --- Summary table ---
    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("="*70)
    print(trend_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Run temporal log-odds analysis by decade"
    )
    parser.add_argument(
        '--completions-path',
        type=str,
        required=True,
        help='Path to the LLM completions CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: log_odds_scoring/output/temporal_<study_name>)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='explicit',
        choices=['explicit', 'ablated', 'both'],
        help='Which condition to analyze'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        study_name = Path(args.completions_path).parent.name
        output_dir = f"log_odds_scoring/output/temporal_{study_name}"
    
    print(f"\n{'='*70}")
    print("TEMPORAL LOG-ODDS ANALYSIS BY DECADE")
    print(f"{'='*70}")
    print(f"Completions: {args.completions_path}")
    print(f"Condition: {args.condition}")
    print(f"Output: {output_dir}")
    
    # Run analysis
    results = run_temporal_analysis(
        args.completions_path,
        output_dir,
        args.condition
    )
    
    # Create comparison plots
    print("\n--- Creating temporal comparison plots ---")
    create_temporal_comparison_plots(results, output_dir)
    
    print(f"\n✓ Temporal analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

