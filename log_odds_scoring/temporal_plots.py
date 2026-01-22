#!/usr/bin/env python3
"""
Publication-quality plots for temporal bias analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def set_publication_style():
    """Set matplotlib style for clean, publication-ready plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_diverging_words_by_decade(output_dir: str):
    """
    Create diverging bar charts showing top white vs nonwhite associated words
    for each decade, side by side.
    """
    set_publication_style()
    output_path = Path(output_dir)
    
    decades = ['1990s', '2000s', '2010s']
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=False)
    
    colors = {'white': '#2E86AB', 'nonwhite': '#E94F37'}
    
    for ax, decade in zip(axes, decades):
        # Load data
        df = pd.read_csv(output_path / f'log_odds_{decade}.csv')
        
        # Get top 8 for each direction
        top_white = df.nlargest(8, 'z_score')[['word', 'z_score']]
        top_nonwhite = df.nsmallest(8, 'z_score')[['word', 'z_score']]
        
        # Combine and sort
        plot_df = pd.concat([top_nonwhite, top_white]).sort_values('z_score')
        
        # Create horizontal bar chart
        y_pos = np.arange(len(plot_df))
        bar_colors = [colors['nonwhite'] if z < 0 else colors['white'] for z in plot_df['z_score']]
        
        bars = ax.barh(y_pos, plot_df['z_score'], color=bar_colors, edgecolor='none', height=0.7)
        
        # Add word labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['word'], fontsize=10)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        # Add significance threshold lines
        ax.axvline(x=1.96, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(x=-1.96, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Z-score')
        ax.set_title(f'{decade}', fontweight='bold', fontsize=14)
        ax.set_xlim(-5, 8)
    
    # Add legend
    white_patch = mpatches.Patch(color=colors['white'], label='White-associated')
    nonwhite_patch = mpatches.Patch(color=colors['nonwhite'], label='Nonwhite-associated')
    fig.legend(handles=[white_patch, nonwhite_patch], loc='upper center', 
               ncol=2, bbox_to_anchor=(0.5, 0.02), frameon=False)
    
    fig.suptitle('Most Distinctive Words by Race Across Decades\n(LLM-Generated Football Commentary)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figures' / 'diverging_words_by_decade.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: diverging_words_by_decade.png")


def plot_word_trajectories(output_dir: str):
    """
    Slope/trajectory chart showing how key words' z-scores change across decades.
    """
    set_publication_style()
    output_path = Path(output_dir)
    
    decades = ['1990s', '2000s', '2010s']
    
    # Key words to track
    words_white = ['poise', 'accuracy', 'pocket', 'pressure', 'decision']
    words_nonwhite = ['speed', 'agility', 'instincts', 'burst', 'athletic']
    
    # Collect data
    data = {decade: {} for decade in decades}
    for decade in decades:
        df = pd.read_csv(output_path / f'log_odds_{decade}.csv')
        for _, row in df.iterrows():
            data[decade][row['word']] = row['z_score']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_positions = [0, 1, 2]
    
    # Plot white-associated words
    for word in words_white:
        y_values = [data[d].get(word, np.nan) for d in decades]
        if not all(np.isnan(y_values)):
            ax.plot(x_positions, y_values, 'o-', color='#2E86AB', linewidth=2, 
                   markersize=8, alpha=0.8)
            # Add label at end
            last_valid = next((i for i in reversed(range(len(y_values))) if not np.isnan(y_values[i])), None)
            if last_valid is not None:
                ax.annotate(word, (x_positions[last_valid] + 0.05, y_values[last_valid]),
                           fontsize=10, va='center', color='#2E86AB', fontweight='bold')
    
    # Plot nonwhite-associated words  
    for word in words_nonwhite:
        y_values = [data[d].get(word, np.nan) for d in decades]
        if not all(np.isnan(y_values)):
            ax.plot(x_positions, y_values, 'o-', color='#E94F37', linewidth=2,
                   markersize=8, alpha=0.8)
            # Add label at end
            last_valid = next((i for i in reversed(range(len(y_values))) if not np.isnan(y_values[i])), None)
            if last_valid is not None:
                ax.annotate(word, (x_positions[last_valid] + 0.05, y_values[last_valid]),
                           fontsize=10, va='center', color='#E94F37', fontweight='bold')
    
    # Styling
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=1.96, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(decades, fontsize=12)
    ax.set_ylabel('Z-score (log-odds ratio)', fontsize=12)
    ax.set_xlim(-0.3, 2.8)
    
    # Add region labels
    ax.text(-0.25, 4, 'WHITE\nASSOCIATED', fontsize=9, color='#2E86AB', 
            fontweight='bold', va='center', ha='left')
    ax.text(-0.25, -3, 'NONWHITE\nASSOCIATED', fontsize=9, color='#E94F37',
            fontweight='bold', va='center', ha='left')
    
    ax.set_title('Word Association Trajectories Across Decades\n(How racial language patterns evolved)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figures' / 'word_trajectories.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: word_trajectories.png")


def plot_bias_strength_summary(output_dir: str):
    """
    Summary visualization showing overall bias strength metrics across decades.
    """
    set_publication_style()
    output_path = Path(output_dir)
    
    decades = ['1990s', '2000s', '2010s']
    
    # Collect metrics
    metrics = []
    for decade in decades:
        df = pd.read_csv(output_path / f'log_odds_{decade}.csv')
        significant = df[df['z_score'].abs() >= 1.96]
        
        metrics.append({
            'decade': decade,
            'n_significant': len(significant),
            'mean_abs_z': significant['z_score'].abs().mean() if len(significant) > 0 else 0,
            'max_white_z': df['z_score'].max(),
            'max_nonwhite_z': abs(df['z_score'].min()),
            'vocab_differentiation': len(significant) / len(df) * 100  # % of vocab differentiated
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors
    decade_colors = ['#7FB069', '#FFB400', '#E94F37']
    
    # Plot 1: Number of significant words
    ax1 = axes[0, 0]
    bars = ax1.bar(decades, metrics_df['n_significant'], color=decade_colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Count')
    ax1.set_title('Statistically Significant Words\n(|z| ≥ 1.96)', fontweight='bold')
    for bar, val in zip(bars, metrics_df['n_significant']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(int(val)), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Mean effect size
    ax2 = axes[0, 1]
    bars = ax2.bar(decades, metrics_df['mean_abs_z'], color=decade_colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Mean |Z-score|')
    ax2.set_title('Average Bias Strength\n(among significant words)', fontweight='bold')
    for bar, val in zip(bars, metrics_df['mean_abs_z']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Maximum associations
    ax3 = axes[1, 0]
    x = np.arange(len(decades))
    width = 0.35
    bars1 = ax3.bar(x - width/2, metrics_df['max_white_z'], width, 
                    label='White', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, metrics_df['max_nonwhite_z'], width,
                    label='Nonwhite', color='#E94F37', edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Maximum |Z-score|')
    ax3.set_title('Most Extreme Associations', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(decades)
    ax3.legend()
    
    # Add value labels
    for bar, val in zip(bars1, metrics_df['max_white_z']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, metrics_df['max_nonwhite_z']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Key takeaway text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    takeaway_text = """"""
    ax4.text(0.1, 0.9, takeaway_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))
    
    fig.suptitle('Temporal Analysis: Racial Bias in LLM Football Commentary', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figures' / 'bias_strength_summary.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: bias_strength_summary.png")


def create_all_temporal_plots(output_dir: str):
    """Generate all temporal analysis plots."""
    output_path = Path(output_dir) / 'figures'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating publication-quality temporal plots...")
    plot_diverging_words_by_decade(output_dir)
    plot_word_trajectories(output_dir)
    plot_bias_strength_summary(output_dir)
    print("✓ All plots generated!")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "log_odds_scoring/output/temporal_n500_gpt4o_completion"
    create_all_temporal_plots(output_dir)



