#!/usr/bin/env python3
"""
Create Publication-Quality Plots for Bias Experiment

Run after bias_scoring.py to visualize perplexity and atypicality by race/condition.

Usage:
  python create_plots.py scored_completions.csv [output_dir]
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

# Color palette
COLORS = {
    'white': '#4a90d9',
    'nonwhite': '#e55934',
    'explicit': '#2d7d46',
    'ablated': '#9b59b6'
}


def load_scored_data(path: str) -> pd.DataFrame:
    """Load scored completions CSV."""
    print(f"Loading scored data from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Overall stats
    print("\n--- Perplexity by Race ---")
    print(df.groupby('true_race')['llm_ppl'].agg(['mean', 'std', 'count']))
    
    print("\n--- Perplexity by Condition ---")
    if 'condition' in df.columns:
        print(df.groupby('condition')['llm_ppl'].agg(['mean', 'std', 'count']))
    
    print("\n--- Perplexity by Race × Condition ---")
    if 'condition' in df.columns:
        print(df.groupby(['true_race', 'condition'])['llm_ppl'].agg(['mean', 'std', 'count']))
    
    print("\n--- Atypicality by Race ---")
    if 'llm_atypicality' in df.columns:
        print(df.groupby('true_race')['llm_atypicality'].agg(['mean', 'std', 'count']))
    
    # Statistical tests
    print("\n--- Statistical Tests ---")
    
    white_ppl = df[df['true_race'] == 'white']['llm_ppl'].dropna()
    nonwhite_ppl = df[df['true_race'] == 'nonwhite']['llm_ppl'].dropna()
    
    if len(white_ppl) > 0 and len(nonwhite_ppl) > 0:
        t_stat, p_value = stats.ttest_ind(white_ppl, nonwhite_ppl)
        print(f"\nPerplexity: White vs Nonwhite")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(white_ppl) - 1) * white_ppl.std()**2 + 
                             (len(nonwhite_ppl) - 1) * nonwhite_ppl.std()**2) / 
                            (len(white_ppl) + len(nonwhite_ppl) - 2))
        cohens_d = (white_ppl.mean() - nonwhite_ppl.mean()) / pooled_std
        print(f"  Cohen's d: {cohens_d:.4f}")


def plot_perplexity_by_race(df: pd.DataFrame, output_dir: str):
    """Plot perplexity by race (boxplot)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(
        data=df, 
        x='true_race', 
        y='llm_ppl',
        palette=[COLORS['white'], COLORS['nonwhite']],
        ax=ax
    )
    
    ax.set_xlabel('Player Race')
    ax.set_ylabel('Perplexity (lower = more typical)')
    ax.set_title('LLM Commentary Perplexity by Player Race')
    ax.set_xticklabels(['White', 'Nonwhite'])
    
    # Add mean markers
    means = df.groupby('true_race')['llm_ppl'].mean()
    for i, race in enumerate(['white', 'nonwhite']):
        if race in means.index:
            ax.scatter([i], [means[race]], marker='D', color='black', s=50, zorder=3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'perplexity_by_race.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_perplexity_by_race_condition(df: pd.DataFrame, output_dir: str):
    """Plot perplexity by race and condition (grouped boxplot)."""
    if 'condition' not in df.columns:
        print("Skipping race×condition plot (no 'condition' column)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x='true_race', 
        y='llm_ppl',
        hue='condition',
        palette=[COLORS['explicit'], COLORS['ablated']],
        ax=ax
    )
    
    ax.set_xlabel('Player Race')
    ax.set_ylabel('Perplexity (lower = more typical)')
    ax.set_title('LLM Commentary Perplexity by Race and Condition')
    ax.set_xticklabels(['White', 'Nonwhite'])
    ax.legend(title='Condition', loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'perplexity_by_race_condition.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_atypicality_by_race(df: pd.DataFrame, output_dir: str):
    """Plot atypicality by race (boxplot)."""
    if 'llm_atypicality' not in df.columns:
        print("Skipping atypicality plot (no 'llm_atypicality' column)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(
        data=df, 
        x='true_race', 
        y='llm_atypicality',
        palette=[COLORS['white'], COLORS['nonwhite']],
        ax=ax
    )
    
    ax.set_xlabel('Player Race')
    ax.set_ylabel('Atypicality (higher = more unusual)')
    ax.set_title('LLM Commentary Atypicality by Player Race')
    ax.set_xticklabels(['White', 'Nonwhite'])
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'atypicality_by_race.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_atypicality_by_race_condition(df: pd.DataFrame, output_dir: str):
    """Plot atypicality by race and condition."""
    if 'llm_atypicality' not in df.columns or 'condition' not in df.columns:
        print("Skipping atypicality×condition plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x='true_race', 
        y='llm_atypicality',
        hue='condition',
        palette=[COLORS['explicit'], COLORS['ablated']],
        ax=ax
    )
    
    ax.set_xlabel('Player Race')
    ax.set_ylabel('Atypicality (higher = more unusual)')
    ax.set_title('LLM Commentary Atypicality by Race and Condition')
    ax.set_xticklabels(['White', 'Nonwhite'])
    ax.legend(title='Condition', loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'atypicality_by_race_condition.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_by_position(df: pd.DataFrame, output_dir: str):
    """Create position-specific perplexity plots."""
    if 'position' not in df.columns:
        print("Skipping position plots (no 'position' column)")
        return
    
    positions = df['position'].dropna().unique()
    
    # Summary figure with all positions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, position in enumerate(positions[:4]):  # Max 4 positions
        df_pos = df[df['position'] == position]
        
        if len(df_pos) < 5:
            continue
        
        ax = axes[i]
        
        if 'condition' in df.columns:
            sns.boxplot(
                data=df_pos, 
                x='true_race', 
                y='llm_ppl',
                hue='condition',
                palette=[COLORS['explicit'], COLORS['ablated']],
                ax=ax
            )
            ax.legend(title='Condition', fontsize=8)
        else:
            sns.boxplot(
                data=df_pos, 
                x='true_race', 
                y='llm_ppl',
                palette=[COLORS['white'], COLORS['nonwhite']],
                ax=ax
            )
        
        ax.set_title(f'{position} Perplexity')
        ax.set_xlabel('Race')
        ax.set_ylabel('Perplexity')
        ax.set_xticklabels(['White', 'Nonwhite'])
    
    # Hide unused subplots
    for j in range(len(positions[:4]), 4):
        axes[j].set_visible(False)
    
    plt.suptitle('Perplexity by Position and Race', fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'perplexity_by_position.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_distributions(df: pd.DataFrame, output_dir: str):
    """Plot perplexity distributions as histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Perplexity distribution by race
    ax = axes[0]
    for race, color in [('white', COLORS['white']), ('nonwhite', COLORS['nonwhite'])]:
        data = df[df['true_race'] == race]['llm_ppl'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=race.title(), color=color, density=True)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title('Perplexity Distribution by Race')
    ax.legend()
    
    # Atypicality distribution by race
    ax = axes[1]
    if 'llm_atypicality' in df.columns:
        for race, color in [('white', COLORS['white']), ('nonwhite', COLORS['nonwhite'])]:
            data = df[df['true_race'] == race]['llm_atypicality'].dropna()
            ax.hist(data, bins=30, alpha=0.6, label=race.title(), color=color, density=True)
        ax.set_xlabel('Atypicality')
        ax.set_ylabel('Density')
        ax.set_title('Atypicality Distribution by Race')
        ax.legend()
    else:
        ax.set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_bar_summary(df: pd.DataFrame, output_dir: str):
    """Create bar plot summary (publication style)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perplexity bar plot
    ax = axes[0]
    
    if 'condition' in df.columns:
        summary = df.groupby(['true_race', 'condition'])['llm_ppl'].agg(['mean', 'std']).reset_index()
        
        x = np.arange(2)  # white, nonwhite
        width = 0.35
        
        explicit = summary[summary['condition'] == 'explicit']
        ablated = summary[summary['condition'] == 'ablated']
        
        bars1 = ax.bar(x - width/2, explicit['mean'], width, yerr=explicit['std'], 
                       label='Explicit', color=COLORS['explicit'], capsize=5)
        bars2 = ax.bar(x + width/2, ablated['mean'], width, yerr=ablated['std'], 
                       label='Ablated', color=COLORS['ablated'], capsize=5)
        
        ax.legend()
    else:
        summary = df.groupby('true_race')['llm_ppl'].agg(['mean', 'std']).reset_index()
        x = np.arange(2)
        colors = [COLORS['white'], COLORS['nonwhite']]
        ax.bar(x, summary['mean'], yerr=summary['std'], color=colors, capsize=5)
    
    ax.set_xlabel('Player Race')
    ax.set_ylabel('Mean Perplexity')
    ax.set_title('Mean Perplexity by Race')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['White', 'Nonwhite'])
    
    # Atypicality bar plot
    ax = axes[1]
    if 'llm_atypicality' in df.columns:
        if 'condition' in df.columns:
            summary = df.groupby(['true_race', 'condition'])['llm_atypicality'].agg(['mean', 'std']).reset_index()
            
            explicit = summary[summary['condition'] == 'explicit']
            ablated = summary[summary['condition'] == 'ablated']
            
            bars1 = ax.bar(x - width/2, explicit['mean'], width, yerr=explicit['std'], 
                           label='Explicit', color=COLORS['explicit'], capsize=5)
            bars2 = ax.bar(x + width/2, ablated['mean'], width, yerr=ablated['std'], 
                           label='Ablated', color=COLORS['ablated'], capsize=5)
            ax.legend()
        else:
            summary = df.groupby('true_race')['llm_atypicality'].agg(['mean', 'std']).reset_index()
            ax.bar(x, summary['mean'], yerr=summary['std'], color=colors, capsize=5)
        
        ax.set_xlabel('Player Race')
        ax.set_ylabel('Mean Atypicality')
        ax.set_title('Mean Atypicality by Race')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['White', 'Nonwhite'])
    else:
        ax.set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'summary_bar_plots.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    """Generate all plots."""
    if len(sys.argv) < 2:
        print("Usage: python create_plots.py <scored_completions.csv> [output_dir]")
        print("\nExample:")
        print("  python create_plots.py results/scored_completions.csv figures/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_scored_data(input_path)
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Generate all plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_perplexity_by_race(df, output_dir)
    plot_perplexity_by_race_condition(df, output_dir)
    plot_atypicality_by_race(df, output_dir)
    plot_atypicality_by_race_condition(df, output_dir)
    plot_by_position(df, output_dir)
    plot_distributions(df, output_dir)
    plot_bar_summary(df, output_dir)
    
    print("\n" + "=" * 80)
    print(f"DONE! All plots saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

