#!/usr/bin/env python3
"""
Prompt × Position Bias Analysis

For every prompt_id, runs log-odds analysis for each position within that prompt.

Output structure
----------------
<output_dir>/
  <prompt_slug>/
    tables/     log_odds_tables_<position>.csv
    figures/
    distributions/  combined_race_distributions.csv
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ._bias_utils import (
    analyze_group,
    load_filtered_data,
    save_race_distributions,
    to_slug,
)


def _run_prompt_x_position(df, prompt_ids, output_dir, text_col, race_col,
                            no_plots, top_n_words, z_threshold):
    """Position breakdown within each prompt."""
    px_root = output_dir
    px_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("POSITIONAL BREAKDOWN PER PROMPT")
    print("=" * 80)

    for prompt_id in prompt_ids:
        prompt_df = df[df['prompt_id'] == prompt_id].copy()
        prompt_slug = to_slug(prompt_id)
        prompt_pos_dir = px_root / prompt_slug
        prompt_pos_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"PROMPT: {prompt_id}  ({len(prompt_df)} completions)")
        print("=" * 80)

        if 'position' not in prompt_df.columns:
            print("  Warning: 'position' column not found — skipping positional breakdown.")
            continue

        position_counts = prompt_df['position'].value_counts()
        positions = position_counts.index.tolist()
        print(f"  Positions: {', '.join(positions)}")

        for position in positions:
            analyze_group(
                subset=prompt_df[prompt_df['position'] == position].copy(),
                label=position,
                kind='position',
                output_dir=prompt_pos_dir,
                text_col=text_col,
                race_col=race_col,
                no_plots=no_plots,
                top_n_words=top_n_words,
                z_threshold=z_threshold,
            )

        save_race_distributions(
            prompt_df, 'position', positions, race_col, prompt_pos_dir
        )

    print(f"\nPrompt × position results saved to: {px_root}/")


def run_bias_analysis_by_prompt_and_position(
    completions_path: str,
    output_dir: str = 'output_by_prompt_and_position',
    text_col: str = 'completion_text',
    race_col: str = 'true_race',
    condition: str = 'explicit',
    no_plots: bool = False,
    top_n_words: int = 15,
    z_threshold: float = 1.96,
):
    """Run per-prompt positional breakdown analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROMPT × POSITION BIAS ANALYSIS: RACIAL BIAS IN FOOTBALL COMMENTARY")
    print("=" * 80)
    print(f"\nInput:     {completions_path}")
    print(f"Output:    {output_dir}/")
    print(f"Condition: {condition}")

    print("\nLoading completions...")
    df = load_filtered_data(completions_path, condition)

    prompt_counts = df['prompt_id'].value_counts()
    prompt_ids = prompt_counts.index.tolist()

    print(f"\nOverall race distribution:")
    print(df[race_col].value_counts().to_string())
    print(f"\nPrompts ({len(prompt_ids)}):")
    for pid in prompt_ids:
        print(f"  {pid}: {prompt_counts[pid]}")

    _run_prompt_x_position(
        df=df,
        prompt_ids=prompt_ids,
        output_dir=output_dir,
        text_col=text_col,
        race_col=race_col,
        no_plots=no_plots,
        top_n_words=top_n_words,
        z_threshold=z_threshold,
    )

    print("\n" + "=" * 80)
    print("PROMPT × POSITION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze racial bias in LLM football commentary by prompt and position")
    parser.add_argument(
        '--completions-path', type=str,
        default="/mnt/c/Users/hallj/GitHub/CS-329R-Project/llm-football-bias/llm_new_generations/output/4o_mini_2010_2026_exp_01.csv",
    )
    parser.add_argument('--output-dir', type=str, default='output_by_prompt_and_position')
    parser.add_argument('--text-col', type=str, default='completion_text')
    parser.add_argument('--race-col', type=str, default='true_race')
    parser.add_argument('--condition', type=str, default='explicit',
                        choices=['explicit', 'ablated', 'both'])
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--top-n-words', type=int, default=15)
    parser.add_argument('--z-threshold', type=float, default=1.96)

    args = parser.parse_args()
    run_bias_analysis_by_prompt_and_position(
        completions_path=args.completions_path,
        output_dir=args.output_dir,
        text_col=args.text_col,
        race_col=args.race_col,
        condition=args.condition,
        no_plots=args.no_plots,
        top_n_words=args.top_n_words,
        z_threshold=args.z_threshold,
    )


if __name__ == "__main__":
    main()

# Example for prompt + position
"""
python -m log_odds_scoring.run_bias_analysis_by_prompt_and_position \
   --completions-path "llm_new_generations/output/4o_mini_2010_2026_exp_01.csv" \
   --output-dir output_results/bias_analysis_v4/4o-mini/by_prompt_and_position \
   --top-n-words 20 \
   --z-threshold 0.01
"""