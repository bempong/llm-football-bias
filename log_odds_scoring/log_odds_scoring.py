#!/usr/bin/env python3
"""
Log-Odds Ratio Scoring with Informative Dirichlet Prior

Implements the method from Monroe et al. (2009) for comparing word usage
between two groups (white vs nonwhite players in our case).
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config


def tokenize(text: str, extra_filter: set = None) -> List[str]:
    """
    Simple tokenization for commentary text.
    
    Args:
        text: Raw text string
        extra_filter: Additional words to exclude (e.g. player names, team names)
    
    Returns:
        List of lowercase tokens
    """
    if pd.isna(text) or not text:
        return []
    
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    
    filter_set = config.STOPWORDS
    if extra_filter:
        filter_set = filter_set | extra_filter
    
    tokens = [
        t for t in tokens 
        if len(t) >= config.MIN_TOKEN_LENGTH and t not in filter_set
    ]
    
    return tokens


def build_name_team_filter(df: pd.DataFrame) -> set:
    """Build a combined filter set of player name tokens and NFL team tokens."""
    extra = set(config.NFL_TEAM_TOKENS)
    if 'player_name' in df.columns:
        extra |= config.build_player_name_tokens(df, 'player_name')
    return extra


def count_words_by_group(
    df: pd.DataFrame,
    text_col: str,
    race_col: str,
    race_a: str,
    race_b: str,
    extra_filter: set = None
) -> Tuple[Counter, Counter, int, int]:
    """
    Count word frequencies for each racial group.
    Filters out player names and NFL team names.
    
    Args:
        df: DataFrame with completions
        text_col: Column name containing text
        race_col: Column name containing race labels
        race_a: Label for group A (e.g., "white")
        race_b: Label for group B (e.g., "nonwhite")
        extra_filter: Pre-built set of tokens to exclude (names, teams).
                      If None, builds from df automatically.
    
    Returns:
        (counts_a, counts_b, total_a, total_b)
    """
    if extra_filter is None:
        extra_filter = build_name_team_filter(df)
    print(f"  Filtering {len(extra_filter)} name/team tokens")

    df_a = df[df[race_col] == race_a]
    df_b = df[df[race_col] == race_b]
    
    text_a = ' '.join(df_a[text_col].fillna('').astype(str))
    text_b = ' '.join(df_b[text_col].fillna('').astype(str))
    
    tokens_a = tokenize(text_a, extra_filter)
    tokens_b = tokenize(text_b, extra_filter)
    
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    
    total_a = len(tokens_a)
    total_b = len(tokens_b)
    
    return counts_a, counts_b, total_a, total_b


def compute_log_odds_with_prior(
    word: str,
    count_a: int,
    count_b: int,
    total_a: int,
    total_b: int,
    count_background: int,
    total_background: int,
    alpha_0: float
) -> Tuple[float, float, float]:
    """
    Compute log-odds ratio with informative Dirichlet prior for a single word.
    
    Args:
        word: The word
        count_a: Count in group A (white)
        count_b: Count in group B (nonwhite)
        total_a: Total tokens in group A
        total_b: Total tokens in group B
        count_background: Count in pooled background
        total_background: Total tokens in background
        alpha_0: Dirichlet prior strength parameter
    
    Returns:
        (delta, z_score, variance) where:
        - delta: log-odds ratio (positive = more associated with white players)
        - z_score: standardized score (|z| > 1.96 = statistical significance)
        - variance: estimated variance of delta
    """
    # Compute informative prior weight for this word
    alpha_w = alpha_0 * (count_background / total_background)
    
    # Smoothed probabilities
    p_a = (count_a + alpha_w) / (total_a + alpha_0)
    p_b = (count_b + alpha_w) / (total_b + alpha_0)
    
    # Avoid log(0) by clipping probabilities away from 0 and 1
    eps = 1e-10
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    
    # Log-odds (logit) for each group
    logit_a = np.log(p_a / (1 - p_a))
    logit_b = np.log(p_b / (1 - p_b))
    
    # Log-odds ratio
    delta = logit_a - logit_b
    
    # Variance approximation
    variance = 1 / (count_a + alpha_w) + 1 / (count_b + alpha_w)
    
    # Z-score
    z_score = delta / np.sqrt(variance)
    
    return delta, z_score, variance


def compute_log_odds_by_race(
    df: pd.DataFrame,
    text_col: str = 'completion_text',
    race_col: str = 'true_race',
    race_a: str = None,
    race_b: str = None,
    min_count: int = None,
    alpha_0: float = None,
    extra_filter: set = None
) -> pd.DataFrame:
    """
    Compute log-odds ratios for all words comparing two racial groups.
    
    Args:
        df: DataFrame with LLM completions
        text_col: Column containing text
        race_col: Column containing race labels
        race_a: Label for group A (default: config.RACE_GROUP_A)
        race_b: Label for group B (default: config.RACE_GROUP_B)
        min_count: Minimum total count to include word (default: config.LOG_ODDS_MIN_COUNT)
        alpha_0: Prior strength (default: config.LOG_ODDS_ALPHA_0)
        extra_filter: Pre-built set of tokens to exclude (names, teams).
                      Built from the full dataset before subsetting by position/prompt.
    
    Returns:
        DataFrame with columns:
        - word: the word
        - count_a: count in group A
        - count_b: count in group B
        - total_count: total across both groups
        - log_odds: delta (positive = more group A, negative = more group B)
        - z_score: standardized score
        - variance: estimated variance
    """
    if race_a is None:
        race_a = config.RACE_GROUP_A
    if race_b is None:
        race_b = config.RACE_GROUP_B
    if min_count is None:
        min_count = config.LOG_ODDS_MIN_COUNT
    if alpha_0 is None:
        alpha_0 = config.LOG_ODDS_ALPHA_0
    
    print(f"Computing log-odds ratios: {race_a} vs {race_b}")
    print(f"  Min count: {min_count}")
    print(f"  Prior strength (alpha_0): {alpha_0}")
    
    counts_a, counts_b, total_a, total_b = count_words_by_group(
        df, text_col, race_col, race_a, race_b, extra_filter=extra_filter
    )
    
    print(f"\nGroup A ({race_a}): {len(counts_a)} unique words, {total_a} total tokens")
    print(f"Group B ({race_b}): {len(counts_b)} unique words, {total_b} total tokens")
    
    # Build background counts (pooled)
    counts_background = counts_a + counts_b
    total_background = total_a + total_b
    
    # Get all words that meet minimum count threshold
    words_to_analyze = {
        word for word, count in counts_background.items()
        if count >= min_count
    }
    
    print(f"\nAnalyzing {len(words_to_analyze)} words (after min_count={min_count} filter)")
    
    # Compute log-odds for each word
    results = []
    for word in words_to_analyze:
        c_a = counts_a[word]
        c_b = counts_b[word]
        c_bg = counts_background[word]
        
        delta, z_score, variance = compute_log_odds_with_prior(
            word, c_a, c_b, total_a, total_b,
            c_bg, total_background, alpha_0
        )
        
        results.append({
            'word': word,
            'count_a': c_a,
            'count_b': c_b,
            'total_count': c_bg,
            'log_odds': delta,
            'z_score': z_score,
            'variance': variance
        })
    
    # Create DataFrame and sort by absolute z-score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('z_score', key=abs, ascending=False)
    
    return results_df


if __name__ == "__main__":
    # Test with dummy data
    test_df = pd.DataFrame({
        'completion_text': [
            'smart intelligent player makes great decisions',
            'athletic explosive powerful runner',
            'smart tactical thinker with good awareness',
            'fast athletic gifted natural talent'
        ],
        'true_race': ['white', 'nonwhite', 'white', 'nonwhite']
    })
    
    results = compute_log_odds_by_race(test_df, min_count=1, alpha_0=0.01)
    print("\nTop words by z-score:")
    print(results.head(10))

