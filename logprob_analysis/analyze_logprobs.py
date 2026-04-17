"""
Token-level attribution analysis: compare model confidence on racially
distinctive words across white vs. nonwhite player conditions.

Position-stratified and word-controlled to remove confounds.
Uses position-specific distinctive word lists from the log-odds analysis.

Usage:
    python logprob_analysis/analyze_logprobs.py
"""

import json
import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = Path("logprob_analysis/output")

Z_THRESHOLD = 1.96
TOP_N_WORDS = 20
MIN_WORD_COUNT = 5

POSITION_LOG_ODDS_DIR = Path("output_results/bias_analysis_v4/4o-mini_3k/by_position/tables")
OVERALL_LOG_ODDS = Path("output_results/bias_analysis_v4/4o-mini_3k/by_prompt/tables/log_odds_tables_p1_original.csv")

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OL', 'DB', 'LB', 'DL', 'ST']

POSITION_LABELS = {
    'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE',
    'OL': 'OL', 'DB': 'DB', 'LB': 'LB', 'DL': 'DL', 'ST': 'ST',
}


def load_distinctive_words(log_odds_csv: str, top_n: int = TOP_N_WORDS):
    """Load top significant distinctive words from a log-odds table."""
    df = pd.read_csv(log_odds_csv)
    sig = df[df['z_score'].abs() >= Z_THRESHOLD].copy()

    white_words = set(
        sig[sig['z_score'] > 0].nlargest(top_n, 'z_score')['word'].str.lower()
    )
    nonwhite_words = set(
        sig[sig['z_score'] < 0].nsmallest(top_n, 'z_score')['word'].str.lower()
    )
    return white_words, nonwhite_words


def extract_token_logprobs(token_json_str: str):
    try:
        data = json.loads(token_json_str)
        return [(d['token'].strip().lower(), d['logprob']) for d in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def match_tokens_to_words(token_logprobs, distinctive_words):
    matches = []
    tokens = token_logprobs
    for i, (tok, lp) in enumerate(tokens):
        clean = re.sub(r'[^a-z]', '', tok)
        if not clean:
            continue
        if clean in distinctive_words:
            matches.append((clean, lp))
            continue
        if i + 1 < len(tokens):
            next_clean = re.sub(r'[^a-z]', '', tokens[i + 1][0])
            combined = clean + next_clean
            if combined in distinctive_words:
                avg_lp = (lp + tokens[i + 1][1]) / 2
                matches.append((combined, avg_lp))
    return matches


def compute_word_controlled_means(match_df, min_count=MIN_WORD_COUNT):
    """
    For each word, compute mean logprob under white and nonwhite conditions.
    Return the average of per-word means (each word weighted equally).
    """
    results = {}
    for wg in ['white-associated', 'nonwhite-associated']:
        sub = match_df[match_df['word_group'] == wg]
        w_word_means = []
        nw_word_means = []
        for word in sub['word'].unique():
            wsub = sub[sub['word'] == word]
            w_vals = wsub[wsub['true_race'] == 'white']['logprob']
            nw_vals = wsub[wsub['true_race'] == 'nonwhite']['logprob']
            if len(w_vals) >= min_count and len(nw_vals) >= min_count:
                w_word_means.append(w_vals.mean())
                nw_word_means.append(nw_vals.mean())

        if not w_word_means:
            results[wg] = None
            continue

        w_arr = np.array(w_word_means)
        nw_arr = np.array(nw_word_means)
        diffs = nw_arr - w_arr

        try:
            _, wilcoxon_p = stats.wilcoxon(diffs)
        except ValueError:
            wilcoxon_p = np.nan
        t_stat, t_p = stats.ttest_1samp(diffs, 0.0) if len(diffs) > 1 else (np.nan, np.nan)

        results[wg] = {
            'white_prob': np.exp(w_arr).mean() * 100,
            'nonwhite_prob': np.exp(nw_arr).mean() * 100,
            'white_prob_ci': 1.96 * np.exp(w_arr).std(ddof=1) / np.sqrt(len(w_arr)) * 100 if len(w_arr) > 1 else 0,
            'nonwhite_prob_ci': 1.96 * np.exp(nw_arr).std(ddof=1) / np.sqrt(len(nw_arr)) * 100 if len(nw_arr) > 1 else 0,
            'n_words': len(w_word_means),
            'mean_diff_logprob': diffs.mean(),
            'wilcoxon_p': wilcoxon_p,
            't_p': t_p,
        }
    return results


def build_match_df(completions_df, white_words, nonwhite_words):
    """Extract all distinctive-word token matches from completions."""
    all_distinctive = white_words | nonwhite_words
    records = []
    for _, row in completions_df.iterrows():
        token_lps = extract_token_logprobs(row['token_logprobs_json'])
        matches = match_tokens_to_words(token_lps, all_distinctive)
        for word, logprob in matches:
            if logprob < -20:
                continue
            records.append({
                'player_id': row['player_id'],
                'true_race': row['true_race'],
                'position': row['position'],
                'word': word,
                'logprob': logprob,
                'word_group': 'white-associated' if word in white_words else 'nonwhite-associated',
            })
    return pd.DataFrame(records)


def plot_grouped_bars(results, title, ax, show_stats=True):
    """Plot grouped bar chart on probability scale."""
    groups = ['white-associated', 'nonwhite-associated']
    x = np.arange(len(groups))
    width = 0.32

    w_probs, nw_probs = [], []
    w_cis, nw_cis = [], []
    valid = True

    for wg in groups:
        r = results.get(wg)
        if not r:
            valid = False
            break
        w_probs.append(r['white_prob'])
        nw_probs.append(r['nonwhite_prob'])
        w_cis.append(r['white_prob_ci'])
        nw_cis.append(r['nonwhite_prob_ci'])

    if not valid:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title(title, fontsize=11)
        return

    ax.bar(x - width / 2, w_probs, width, yerr=w_cis,
           label='White players', color='#4C72B0', capsize=3, alpha=0.85)
    ax.bar(x + width / 2, nw_probs, width, yerr=nw_cis,
           label='Nonwhite players', color='#DD8452', capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(['White-\nAssociated', 'Nonwhite-\nAssociated'], fontsize=9)
    ax.set_ylabel('Avg Token Probability (%)', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    if show_stats:
        for i, wg in enumerate(groups):
            r = results[wg]
            p = r['wilcoxon_p'] if not np.isnan(r['wilcoxon_p']) else r['t_p']
            if np.isnan(p):
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            n_w = r['n_words']
            y_max = max(w_probs[i] + w_cis[i], nw_probs[i] + nw_cis[i])
            ax.text(i, y_max + 1.0, f"p={p:.2e} {sig}\n({n_w} words)",
                    ha='center', fontsize=7, style='italic')


def run_analysis(completions_csv: str, top_n: int = TOP_N_WORDS):
    print("=" * 70)
    print("TOKEN-LEVEL ATTRIBUTION ANALYSIS")
    print("Position-stratified, word-controlled, probability scale")
    print("=" * 70)

    df = pd.read_csv(completions_csv)
    print(f"\nCompletions: {len(df)}")
    print(f"  By race: {df['true_race'].value_counts().to_dict()}")
    print(f"  By position: {df['position'].value_counts().to_dict()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Overall analysis (using P1 overall word list) ──
    print("\n" + "─" * 70)
    print("OVERALL (word-controlled)")
    print("─" * 70)

    white_words, nonwhite_words = load_distinctive_words(str(OVERALL_LOG_ODDS), top_n)
    match_df = build_match_df(df, white_words, nonwhite_words)
    print(f"  Matches: {len(match_df)} ({(match_df['word_group'] == 'white-associated').sum()} white-assoc, "
          f"{(match_df['word_group'] == 'nonwhite-associated').sum()} nonwhite-assoc)")

    overall_results = compute_word_controlled_means(match_df)
    for wg, r in overall_results.items():
        if r:
            print(f"\n  {wg} ({r['n_words']} words):")
            print(f"    White players:    {r['white_prob']:.1f}%")
            print(f"    Nonwhite players: {r['nonwhite_prob']:.1f}%")
            p = r['wilcoxon_p'] if not np.isnan(r['wilcoxon_p']) else r['t_p']
            print(f"    Wilcoxon p={p:.4e}")

    # ── Per-position analysis (using position-specific word lists) ──
    print("\n" + "─" * 70)
    print("PER-POSITION (position-specific words, word-controlled)")
    print("─" * 70)

    position_results = {}
    position_match_counts = {}
    for pos in POSITIONS:
        log_odds_file = POSITION_LOG_ODDS_DIR / f"log_odds_tables_{pos.lower()}.csv"
        if not log_odds_file.exists():
            print(f"\n  {pos}: no log-odds table found, skipping")
            continue

        pos_df = df[df['position'] == pos]
        if len(pos_df) == 0:
            print(f"\n  {pos}: no completions, skipping")
            continue

        w_race = (pos_df['true_race'] == 'white').sum()
        nw_race = (pos_df['true_race'] == 'nonwhite').sum()

        pos_white, pos_nonwhite = load_distinctive_words(str(log_odds_file), top_n)
        pos_match = build_match_df(pos_df, pos_white, pos_nonwhite)
        position_match_counts[pos] = len(pos_match)

        if len(pos_match) == 0:
            print(f"\n  {pos}: no token matches")
            continue

        res = compute_word_controlled_means(pos_match, min_count=3)
        position_results[pos] = res

        print(f"\n  {pos} ({len(pos_df)} players: {w_race}W/{nw_race}NW, {len(pos_match)} matches):")
        for wg, r in res.items():
            if r:
                p = r['wilcoxon_p'] if not np.isnan(r['wilcoxon_p']) else r['t_p']
                print(f"    {wg} ({r['n_words']} words): "
                      f"W={r['white_prob']:.1f}% NW={r['nonwhite_prob']:.1f}% p={p:.3e}")

    # ── Figure 1: Overall word-controlled ──
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_grouped_bars(overall_results, 'Overall (Word-Controlled)', ax)
    ax.legend(fontsize=9)
    plt.suptitle('Model Confidence on Racially Distinctive Words\n(GPT-4o-mini, P1 Prompt)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "logprob_overall.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'logprob_overall.png'}")
    plt.close()

    # ── Figure 2: Per-position panels ──
    positions_with_data = [p for p in POSITIONS if p in position_results
                           and position_results[p].get('white-associated')
                           and position_results[p].get('nonwhite-associated')]

    if positions_with_data:
        n_cols = 3
        n_rows = (len(positions_with_data) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, pos in enumerate(positions_with_data):
            ax = axes_flat[i]
            pos_df_size = len(df[df['position'] == pos])
            plot_grouped_bars(position_results[pos], f'{POSITION_LABELS[pos]} (n={pos_df_size})', ax,
                              show_stats=True)
            if i == 0:
                ax.legend(fontsize=7)

        for j in range(len(positions_with_data), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle('Model Confidence by Position\n(Position-Specific Distinctive Words, Word-Controlled)',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "logprob_by_position.png", dpi=200, bbox_inches='tight')
        print(f"  Saved: {OUTPUT_DIR / 'logprob_by_position.png'}")
        plt.close()

    # ── Save summary tables ──
    overall_rows = []
    for wg, r in overall_results.items():
        if r:
            overall_rows.append({'word_group': wg, 'position': 'overall', **r})
    for pos, res in position_results.items():
        for wg, r in res.items():
            if r:
                overall_rows.append({'word_group': wg, 'position': pos, **r})
    if overall_rows:
        pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / "logprob_results.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'logprob_results.csv'}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--completions', default='logprob_analysis/output/logprob_completions.csv')
    parser.add_argument('--top-n', type=int, default=TOP_N_WORDS)
    args = parser.parse_args()

    run_analysis(args.completions, top_n=args.top_n)
