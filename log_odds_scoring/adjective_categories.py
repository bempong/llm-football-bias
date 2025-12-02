#!/usr/bin/env python3
"""
Adjective Category Analysis

Analyzes the distribution of adjectives from predefined categories
(cognitive, leadership, athleticism, etc.) across racial groups.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from . import config
from .log_odds_scoring import tokenize


def load_lexicon(lexicon_path: Path = None) -> Dict[str, List[str]]:
    """
    Load adjective category lexicon from YAML file.
    
    Args:
        lexicon_path: Path to lexicon file (default: config.LEXICONS_DIR / "adjective_categories.yml")
    
    Returns:
        Dictionary mapping category names to lists of adjectives
    """
    if lexicon_path is None:
        lexicon_path = config.LEXICONS_DIR / "adjective_categories.yml"
    
    with open(lexicon_path, 'r') as f:
        lexicon = yaml.safe_load(f)
    
    # Convert to lowercase and handle multi-word terms
    normalized_lexicon = {}
    for category, words in lexicon.items():
        normalized_lexicon[category] = [w.lower().replace('-', '') for w in words]
    
    return normalized_lexicon


def count_adjectives_in_text(text: str, lexicon: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Count occurrences of adjectives from each category in text.
    
    Args:
        text: Raw text string
        lexicon: Dictionary mapping categories to word lists
    
    Returns:
        Dictionary mapping category names to counts
    """
    if pd.isna(text) or not text:
        return {cat: 0 for cat in lexicon.keys()}
    
    # Tokenize
    tokens = tokenize(text)
    token_set = set(tokens)
    
    # Count by category
    counts = {}
    for category, words in lexicon.items():
        count = sum(1 for word in words if word in token_set)
        counts[category] = count
    
    return counts


def compute_adjective_category_stats(
    df: pd.DataFrame,
    text_col: str = 'completion_text',
    race_col: str = 'true_race',
    lexicon: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """
    Compute adjective category statistics by race.
    
    Args:
        df: DataFrame with LLM completions
        text_col: Column containing text
        race_col: Column containing race labels
        lexicon: Adjective lexicon (default: load from config)
    
    Returns:
        DataFrame with columns:
        - race: race label
        - category: adjective category
        - count: total occurrences
        - total_adjective_tokens: total adjectives across all categories
        - proportion: count / total_adjective_tokens
    """
    if lexicon is None:
        lexicon = load_lexicon()
    
    print(f"Analyzing adjective categories:")
    for category, words in lexicon.items():
        print(f"  {category}: {len(words)} words")
    
    # Count adjectives for each completion
    adjective_counts_list = []
    
    for idx, row in df.iterrows():
        counts = count_adjectives_in_text(row[text_col], lexicon)
        counts['race'] = row[race_col]
        counts['completion_id'] = idx
        adjective_counts_list.append(counts)
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(adjective_counts_list)
    
    # Aggregate by race and category
    results = []
    for race in df[race_col].unique():
        if pd.isna(race):
            continue
        
        race_data = counts_df[counts_df['race'] == race]
        
        # Sum across all completions for this race
        for category in lexicon.keys():
            total_count = race_data[category].sum()
            results.append({
                'race': race,
                'category': category,
                'count': total_count
            })
    
    results_df = pd.DataFrame(results)
    
    # Compute totals and proportions
    race_totals = results_df.groupby('race')['count'].sum().to_dict()
    results_df['total_adjective_tokens'] = results_df['race'].map(race_totals)
    results_df['proportion'] = results_df['count'] / results_df['total_adjective_tokens']
    
    # Sort by race, then category
    results_df = results_df.sort_values(['race', 'category'])
    
    return results_df


def create_category_pivot_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table for easier viewing.
    
    Args:
        stats_df: Output from compute_adjective_category_stats
    
    Returns:
        Pivot table with categories as rows, races as columns
    """
    pivot = stats_df.pivot(
        index='category',
        columns='race',
        values='proportion'
    )
    
    # Add a column for the ratio
    if 'white' in pivot.columns and 'nonwhite' in pivot.columns:
        pivot['ratio_white_nonwhite'] = pivot['white'] / (pivot['nonwhite'] + 1e-10)
    
    return pivot


if __name__ == "__main__":
    # Test with dummy data
    test_df = pd.DataFrame({
        'completion_text': [
            'He is a smart intelligent cerebral player with great leadership',
            'An athletic explosive powerful runner with raw natural talent',
            'Smart tactical thinker, very disciplined and polished technique',
            'Fast athletic gifted player, instinctive and natural ability'
        ],
        'true_race': ['white', 'nonwhite', 'white', 'nonwhite']
    })
    
    stats = compute_adjective_category_stats(test_df)
    print("\nCategory statistics:")
    print(stats)
    
    pivot = create_category_pivot_table(stats)
    print("\nPivot table (proportions):")
    print(pivot)

