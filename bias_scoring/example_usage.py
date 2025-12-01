#!/usr/bin/env python3
"""
Example: Using the Bias Scoring Engine

This script demonstrates how to use bias_scoring.py to score
LLM-generated football commentary for racial bias.
"""

import os
import pandas as pd
from bias_scoring import score_completions, summarize_by_group

# ============================================================================
# EXAMPLE 1: Quick Scoring
# ============================================================================

def example_quick_scoring():
    """Score QB completions in one function call."""
    
    print("=" * 80)
    print("EXAMPLE 1: Quick Scoring")
    print("=" * 80)
    
    scored_df = score_completions(
        commentary_path="../tagged_transcripts.json",
        completions_path="../qb_commentary_race.json",
        kenlm_train_txt="models/kenlm_train.txt",
        kenlm_model_path="models/football_commentary.bin",
        scored_output_path="results/scored_qb.csv"
    )
    
    print("\n" + "=" * 80)
    print("RESULTS PREVIEW")
    print("=" * 80)
    print(scored_df[['name', 'true_race', 'llm_ppl', 'llm_atypicality']].head(10))
    
    return scored_df


# ============================================================================
# EXAMPLE 2: Manual Step-by-Step
# ============================================================================

def example_manual_scoring():
    """Score completions step-by-step for more control."""
    
    from bias_scoring import (
        load_commentary_corpus,
        build_kenlm_training_text,
        train_bigram_kenlm,
        load_kenlm_model,
        build_idf,
        load_llm_completions,
        add_perplexity_column,
        add_atypicality_column
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Manual Step-by-Step Scoring")
    print("=" * 80)
    
    # Step 1: Load corpus
    print("\nStep 1: Loading commentary corpus...")
    corpus = load_commentary_corpus("../tagged_transcripts.json")
    
    # Step 2: Build training text (if not exists)
    print("\nStep 2: Building KenLM training text...")
    if not os.path.exists("models/kenlm_train.txt"):
        build_kenlm_training_text(corpus, "models/kenlm_train.txt")
    
    # Step 3: Train model (if not exists)
    print("\nStep 3: Training KenLM model...")
    if not os.path.exists("models/football_commentary.bin"):
        train_bigram_kenlm("models/kenlm_train.txt", "models/football_commentary.bin")
    
    # Step 4: Load model
    print("\nStep 4: Loading KenLM model...")
    model = load_kenlm_model("models/football_commentary.bin")
    
    # Step 5: Build IDF
    print("\nStep 5: Building IDF dictionary...")
    idf = build_idf(corpus)
    
    # Step 6: Load completions
    print("\nStep 6: Loading LLM completions...")
    df = load_llm_completions("../qb_commentary_race.json")
    
    # Step 7: Score
    print("\nStep 7: Computing scores...")
    df = add_perplexity_column(df, model)
    df = add_atypicality_column(df, idf)
    
    # Step 8: Save
    print("\nStep 8: Saving results...")
    df.to_csv("results/scored_qb_manual.csv", index=False)
    
    return df


# ============================================================================
# EXAMPLE 3: Analysis
# ============================================================================

def example_analysis(scored_df):
    """Analyze scored completions for bias patterns."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Bias Analysis")
    print("=" * 80)
    
    # Basic stats by race
    print("\n--- Perplexity by Race ---")
    ppl_by_race = scored_df.groupby('true_race')['llm_ppl'].agg(['mean', 'std', 'count'])
    print(ppl_by_race)
    
    print("\n--- Atypicality by Race ---")
    atyp_by_race = scored_df.groupby('true_race')['llm_atypicality'].agg(['mean', 'std', 'count'])
    print(atyp_by_race)
    
    # Statistical test
    from scipy import stats
    
    white_ppl = scored_df[scored_df['true_race'] == 'white']['llm_ppl']
    nonwhite_ppl = scored_df[scored_df['true_race'] == 'nonwhite']['llm_ppl']
    
    t_stat, p_value = stats.ttest_ind(white_ppl, nonwhite_ppl)
    
    print("\n--- T-Test (White vs Nonwhite Perplexity) ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Summary for plotting
    print("\n--- Summary Table (for plotting) ---")
    summary = summarize_by_group(scored_df, ['true_race'])
    print(summary)
    summary.to_csv("results/summary_by_race.csv", index=False)
    
    # Find examples of high/low perplexity
    print("\n--- Examples: Lowest Perplexity (Most Typical) ---")
    lowest = scored_df.nsmallest(3, 'llm_ppl')[['name', 'true_race', 'llm_ppl', 'completion_text']]
    for idx, row in lowest.iterrows():
        print(f"\n{row['name']} ({row['true_race']}) - Perplexity: {row['llm_ppl']:.2f}")
        print(f"  Text: {row['completion_text'][:100]}...")
    
    print("\n--- Examples: Highest Perplexity (Most Atypical) ---")
    highest = scored_df.nlargest(3, 'llm_ppl')[['name', 'true_race', 'llm_ppl', 'completion_text']]
    for idx, row in highest.iterrows():
        print(f"\n{row['name']} ({row['true_race']}) - Perplexity: {row['llm_ppl']:.2f}")
        print(f"  Text: {row['completion_text'][:100]}...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run examples
    print("\n" + "=" * 80)
    print("BIAS SCORING ENGINE - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Quick scoring
    scored_df = example_quick_scoring()
    
    # Example 3: Analysis
    example_analysis(scored_df)
    
    print("\n" + "=" * 80)
    print("DONE! Check results/ directory for output files")
    print("=" * 80)

