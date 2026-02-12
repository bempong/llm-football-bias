#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_distinctive_words_plot(
    log_odds_df: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
    z_threshold: float = 1.96
):
    """
    Create a horizontal bar plot of distinctive words by race.
    
    Args:
        log_odds_df: DataFrame with log_odds results
        output_path: Path to save figure
        top_n: Number of top words to show for each group
        z_threshold: Minimum |z-score| for statistical significance (default 1.96 = p<0.05)
    """
    # Filter for statistically significant words
    significant = log_odds_df[log_odds_df['z_score'].abs() >= z_threshold].copy()
    
    if len(significant) == 0:
        print(f"  [Warning] No words meet z-score threshold {z_threshold}")
        return
    
    # Get top N for each group
    white_words = significant.nlargest(top_n, 'z_score')
    nonwhite_words = significant.nsmallest(top_n, 'z_score')
    
    # Combine and sort by z-score
    plot_df = pd.concat([nonwhite_words, white_words]).sort_values('z_score')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.25)))
    
    # Color bars by group
    colors = ['#d62728' if z < 0 else '#1f77b4' for z in plot_df['z_score']]
    
    # Horizontal bar plot
    bars = ax.barh(plot_df['word'], plot_df['z_score'], color=colors, alpha=0.8)
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    
    # Add significance threshold lines
    ax.axvline(-z_threshold, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label=f'p < 0.05 (|z| = {z_threshold})')
    ax.axvline(z_threshold, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Log-Odds Z-Score', fontweight='bold')
    ax.set_ylabel('Word', fontweight='bold')
    ax.set_title('Distinctive Words by Race (Log-Odds Ratio)', fontweight='bold', pad=15)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='White-associated'),
        Patch(facecolor='#d62728', alpha=0.8, label='Nonwhite-associated')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved word-level plot: {output_path}")


def generate_all_plots(
    log_odds_path: str,
    output_dir: str,
    top_n_words: int = 15,
    z_threshold: float = 1.96,
    position: str = None
):
    """
    Generate publication-quality plots for log-odds analysis.
    
    Args:
        log_odds_path: Path to log_odds_results.csv
        output_dir: Directory to save figures
        top_n_words: Number of top words to show in word plot
        z_threshold: Minimum |z-score| threshold for word plot
    """
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Create output directory
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading results...")
    log_odds_df = pd.read_csv(log_odds_path)
    print(f"  Loaded {len(log_odds_df)} words")
    
    # Generate plots
    print("\nGenerating figures...")
    
    # Distinctive words plot
    create_distinctive_words_plot(
        log_odds_df,
        str(figures_dir / f"distinctive_words_by_race_{position}.png"),
        top_n=top_n_words,
        z_threshold=z_threshold
    )
    
    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate bias analysis plots")
    parser.add_argument('log_odds_path', help='Path to log_odds_results.csv')
    parser.add_argument('output_dir', help='Directory to save figures')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top words to show')
    parser.add_argument('--z-threshold', type=float, default=1.96, help='Z-score threshold')
    
    args = parser.parse_args()
    
    generate_all_plots(
        args.log_odds_path,
        args.output_dir,
        top_n_words=args.top_n,
        z_threshold=args.z_threshold
    )




