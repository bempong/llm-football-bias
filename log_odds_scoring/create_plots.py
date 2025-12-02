#!/usr/bin/env python3
"""
Generate publication-quality figures from log-odds bias analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def create_log_odds_word_plot(
    log_odds_df: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
    z_threshold: float = 1.96
):
    """
    Create a horizontal bar plot of top distinctive words by z-score.
    
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


def create_category_distribution_plot(
    category_df: pd.DataFrame,
    output_path: str
):
    """
    Create a grouped bar plot of adjective category proportions by race.
    
    Args:
        category_df: DataFrame with adjective_category_stats
        output_path: Path to save figure
    """
    # Pivot for plotting
    pivot_df = category_df.pivot(index='category', columns='race', values='proportion')
    
    # Reorder categories by total usage
    category_order = pivot_df.sum(axis=1).sort_values(ascending=False).index
    pivot_df = pivot_df.loc[category_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot grouped bars
    x = range(len(pivot_df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], pivot_df['white'], width, 
                    label='White', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], pivot_df['nonwhite'], width,
                    label='Nonwhite', color='#d62728', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Labels and title
    ax.set_xlabel('Adjective Category', fontweight='bold')
    ax.set_ylabel('Proportion of Adjectives', fontweight='bold')
    ax.set_title('Adjective Category Distribution by Race', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in pivot_df.index], 
                       rotation=45, ha='right')
    ax.set_ylim(0, max(pivot_df.max()) * 1.15)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved category distribution plot: {output_path}")


def create_category_ratio_plot(
    category_df: pd.DataFrame,
    output_path: str
):
    """
    Create a horizontal bar plot showing white:nonwhite ratio for each category.
    
    Args:
        category_df: DataFrame with adjective_category_stats
        output_path: Path to save figure
    """
    # Pivot for plotting
    pivot_df = category_df.pivot(index='category', columns='race', values='proportion')
    
    # Compute ratio (white / nonwhite)
    # Avoid division by zero
    pivot_df['ratio'] = pivot_df.apply(
        lambda row: row['white'] / row['nonwhite'] if row['nonwhite'] > 0 else float('inf'),
        axis=1
    )
    
    # Handle infinity (white-only categories)
    pivot_df['ratio_display'] = pivot_df['ratio'].replace(float('inf'), pivot_df['ratio'][pivot_df['ratio'] != float('inf')].max() * 1.5)
    
    # Sort by ratio
    pivot_df = pivot_df.sort_values('ratio_display', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by direction
    colors = ['#d62728' if r < 1 else '#1f77b4' for r in pivot_df['ratio_display']]
    
    # Horizontal bar plot
    bars = ax.barh(range(len(pivot_df)), pivot_df['ratio_display'], color=colors, alpha=0.8)
    
    # Add vertical line at 1 (equal)
    ax.axvline(1, color='black', linewidth=1, linestyle='-', alpha=0.5, label='Equal (1:1)')
    
    # Add labels
    for i, (idx, row) in enumerate(pivot_df.iterrows()):
        if row['ratio'] == float('inf'):
            label = '∞ (white only)'
        else:
            label = f'{row["ratio"]:.2f}x'
        ax.text(row['ratio_display'] + 0.05, i, label, va='center', fontsize=8)
    
    # Labels and title
    ax.set_xlabel('White : Nonwhite Ratio', fontweight='bold')
    ax.set_ylabel('Category', fontweight='bold')
    ax.set_title('Adjective Category Bias Ratios', fontweight='bold', pad=15)
    ax.set_yticks(range(len(pivot_df)))
    ax.set_yticklabels([cat.replace('_', ' ').title() for cat in pivot_df.index])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='More for white players (>1)'),
        Patch(facecolor='#d62728', alpha=0.8, label='More for nonwhite players (<1)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved category ratio plot: {output_path}")


def create_condition_comparison_plot(
    log_odds_df: pd.DataFrame,
    category_df: pd.DataFrame,
    output_path: str,
    top_n: int = 10
):
    """
    Create comparison plots for explicit vs ablated race conditions (if available).
    
    Args:
        log_odds_df: DataFrame with log_odds results (with 'condition' column)
        category_df: DataFrame with category stats (with 'condition' column)
        output_path: Path to save figure
        top_n: Number of top words to show
    """
    if 'condition' not in category_df.columns:
        print("  [Info] No 'condition' column found - skipping condition comparison plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Category distributions by condition
    conditions = category_df['condition'].unique()
    pivot_df = category_df.pivot_table(
        index='category',
        columns=['race', 'condition'],
        values='proportion'
    )
    
    # Plot grouped bars for first condition
    if len(conditions) >= 2:
        cond1, cond2 = conditions[0], conditions[1]
        
        categories = pivot_df.index
        x = range(len(categories))
        width = 0.2
        
        # Plot 4 bars: white explicit, white ablated, nonwhite explicit, nonwhite ablated
        for i, (race, color) in enumerate([('white', '#1f77b4'), ('nonwhite', '#d62728')]):
            for j, cond in enumerate([cond1, cond2]):
                offset = (i * 2 + j - 1.5) * width
                values = pivot_df[(race, cond)] if (race, cond) in pivot_df.columns else [0] * len(categories)
                ax1.bar([xi + offset for xi in x], values, width,
                       label=f'{race.capitalize()} ({cond})',
                       color=color, alpha=0.8 - j*0.3)
        
        ax1.set_xlabel('Category', fontweight='bold')
        ax1.set_ylabel('Proportion', fontweight='bold')
        ax1.set_title('Category Distribution by Race & Condition', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                           rotation=45, ha='right')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=8)
        ax1.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Right plot: Effect of race ablation on bias
    # Show how much the white:nonwhite ratio changes when race is ablated
    if 'condition' in log_odds_df.columns:
        # Get top distinctive words for explicit condition
        explicit_words = log_odds_df[log_odds_df['condition'] == 'explicit'].nlargest(top_n, 'z_score')['word'].tolist()
        
        # Compare z-scores across conditions
        comparison_data = []
        for word in explicit_words:
            for cond in conditions:
                row = log_odds_df[(log_odds_df['word'] == word) & (log_odds_df['condition'] == cond)]
                if not row.empty:
                    comparison_data.append({
                        'word': word,
                        'condition': cond,
                        'z_score': row['z_score'].iloc[0]
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            pivot = comp_df.pivot(index='word', columns='condition', values='z_score')
            
            x2 = range(len(pivot))
            width2 = 0.35
            
            for i, cond in enumerate(conditions):
                if cond in pivot.columns:
                    ax2.bar([xi + i*width2 - width2/2 for xi in x2], pivot[cond], width2,
                           label=cond.capitalize(), alpha=0.8)
            
            ax2.set_xlabel('Word', fontweight='bold')
            ax2.set_ylabel('Z-Score', fontweight='bold')
            ax2.set_title('Effect of Race Condition on Distinctive Words', fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(pivot.index, rotation=45, ha='right')
            ax2.axhline(0, color='black', linewidth=0.8, alpha=0.3)
            ax2.legend(loc='upper right', frameon=True, fancybox=True)
            ax2.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved condition comparison plot: {output_path}")


def generate_all_plots(
    log_odds_csv: str,
    category_csv: str,
    output_dir: str,
    top_n_words: int = 15,
    z_threshold: float = 1.96
):
    """
    Generate all publication-quality plots from log-odds analysis results.
    
    Args:
        log_odds_csv: Path to log_odds_results.csv
        category_csv: Path to adjective_category_stats.csv
        output_dir: Directory to save figures
        top_n_words: Number of top words to show in plots
        z_threshold: Minimum |z-score| for word plots
    """
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading results...")
    log_odds_df = pd.read_csv(log_odds_csv)
    category_df = pd.read_csv(category_csv)
    print(f"  Loaded {len(log_odds_df)} words and {len(category_df)} category statistics")
    
    # Generate plots
    print("\nGenerating figures...")
    
    # 1. Word-level log-odds plot
    create_log_odds_word_plot(
        log_odds_df,
        output_path / "distinctive_words_by_race.png",
        top_n=top_n_words,
        z_threshold=z_threshold
    )
    
    # 2. Category distribution plot
    create_category_distribution_plot(
        category_df,
        output_path / "category_distribution_by_race.png"
    )
    
    # 3. Category ratio plot
    create_category_ratio_plot(
        category_df,
        output_path / "category_bias_ratios.png"
    )
    
    # 4. Condition comparison (if applicable)
    if 'condition' in category_df.columns:
        create_condition_comparison_plot(
            log_odds_df,
            category_df,
            output_path / "condition_comparison.png",
            top_n=10
        )
    
    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from log-odds bias analysis")
    parser.add_argument('--log-odds-csv', type=str, required=True,
                       help='Path to log_odds_results.csv')
    parser.add_argument('--category-csv', type=str, required=True,
                       help='Path to adjective_category_stats.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save figures')
    parser.add_argument('--top-n', type=int, default=15,
                       help='Number of top words to show (default: 15)')
    parser.add_argument('--z-threshold', type=float, default=1.96,
                       help='Minimum |z-score| for significance (default: 1.96)')
    
    args = parser.parse_args()
    
    generate_all_plots(
        args.log_odds_csv,
        args.category_csv,
        args.output_dir,
        top_n_words=args.top_n,
        z_threshold=args.z_threshold
    )

