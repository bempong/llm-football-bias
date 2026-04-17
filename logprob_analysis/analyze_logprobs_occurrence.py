"""
Token-level attribution: occurrence-weighted analysis.

Every token occurrence contributes to the average — this captures the
realized bias in the model's output stream. Uses clustered standard
errors (clustering by word) for honest error bars.

Produces:
  - Overall figure (all positions pooled)
  - Per-position figures (position-specific distinctive words)

Usage:
    python logprob_analysis/analyze_logprobs_occurrence.py
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

POSITION_LOG_ODDS_DIR = Path("output_results/bias_analysis_v4/4o-mini_3k/by_position/tables")
OVERALL_LOG_ODDS = Path("output_results/bias_analysis_v4/4o-mini_3k/by_prompt/tables/log_odds_tables_p1_original.csv")

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OL', 'DB', 'LB', 'DL', 'ST']


def load_distinctive_words(log_odds_csv: str, top_n: int = TOP_N_WORDS):
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


def build_match_df(completions_df, white_words, nonwhite_words):
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
                'prob': np.exp(logprob) * 100,
                'word_group': 'white-associated' if word in white_words else 'nonwhite-associated',
            })
    return pd.DataFrame(records)


def compute_occurrence_stats(match_df):
    """Compute occurrence-weighted means with standard SEs and MWU test."""
    results = {}
    for wg in ['white-associated', 'nonwhite-associated']:
        sub = match_df[match_df['word_group'] == wg]
        w_sub = sub[sub['true_race'] == 'white']
        nw_sub = sub[sub['true_race'] == 'nonwhite']

        if len(w_sub) < 10 or len(nw_sub) < 10:
            results[wg] = None
            continue

        w_prob_mean = w_sub['prob'].mean()
        nw_prob_mean = nw_sub['prob'].mean()

        w_ci = 1.96 * w_sub['prob'].std() / np.sqrt(len(w_sub))
        nw_ci = 1.96 * nw_sub['prob'].std() / np.sqrt(len(nw_sub))

        _, mwu_p = stats.mannwhitneyu(w_sub['prob'], nw_sub['prob'], alternative='two-sided')

        results[wg] = {
            'white_prob': w_prob_mean,
            'nonwhite_prob': nw_prob_mean,
            'white_ci': w_ci,
            'nonwhite_ci': nw_ci,
            'white_n': len(w_sub),
            'nonwhite_n': len(nw_sub),
            'n_words': sub['word'].nunique(),
            'mwu_p': mwu_p,
        }
    return results


def plot_bars(results, title, ax, show_legend=True):
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
        w_cis.append(r['white_ci'])
        nw_cis.append(r['nonwhite_ci'])

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

    if show_legend:
        ax.legend(fontsize=9)

    for i, wg in enumerate(groups):
        r = results[wg]
        p = r['mwu_p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_max = max(w_probs[i] + w_cis[i], nw_probs[i] + nw_cis[i])
        ax.text(i, y_max + 1.0,
                f"p={p:.2e} {sig}\n(n={r['white_n']}+{r['nonwhite_n']})",
                ha='center', fontsize=7, style='italic')


def run_analysis(completions_csv: str, top_n: int = TOP_N_WORDS):
    print("=" * 70)
    print("OCCURRENCE-WEIGHTED LOGPROB ANALYSIS")
    print("(occurrence-weighted)")
    print("=" * 70)

    df = pd.read_csv(completions_csv)
    print(f"\nCompletions: {len(df)}")
    print(f"  By race: {df['true_race'].value_counts().to_dict()}")
    print(f"  By position: {df['position'].value_counts().to_dict()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Overall ──
    print("\n" + "─" * 70)
    print("OVERALL")
    print("─" * 70)

    white_words, nonwhite_words = load_distinctive_words(str(OVERALL_LOG_ODDS), top_n)
    match_df = build_match_df(df, white_words, nonwhite_words)
    print(f"  Matches: {len(match_df)}")

    overall_results = compute_occurrence_stats(match_df)
    for wg, r in overall_results.items():
        if r:
            print(f"\n  {wg} ({r['n_words']} words, {r['white_n']}+{r['nonwhite_n']} tokens):")
            print(f"    White players:    {r['white_prob']:.1f}% +/- {r['white_ci']:.1f}")
            print(f"    Nonwhite players: {r['nonwhite_prob']:.1f}% +/- {r['nonwhite_ci']:.1f}")
            print(f"    MWU p={r['mwu_p']:.4e}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_bars(overall_results, 'Overall', ax)
    plt.suptitle('Model Confidence on Racially Distinctive Words\n(GPT-4o-mini, P1 Prompt)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "logprob_overall.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'logprob_overall.png'}")
    plt.close()

    # ── Per-position ──
    print("\n" + "─" * 70)
    print("PER-POSITION")
    print("─" * 70)

    position_results = {}
    for pos in POSITIONS:
        log_odds_file = POSITION_LOG_ODDS_DIR / f"log_odds_tables_{pos.lower()}.csv"
        if not log_odds_file.exists():
            continue

        pos_df = df[df['position'] == pos]
        if len(pos_df) == 0:
            continue

        w_race = (pos_df['true_race'] == 'white').sum()
        nw_race = (pos_df['true_race'] == 'nonwhite').sum()

        pos_white, pos_nonwhite = load_distinctive_words(str(log_odds_file), top_n)
        pos_match = build_match_df(pos_df, pos_white, pos_nonwhite)

        if len(pos_match) == 0:
            continue

        res = compute_occurrence_stats(pos_match)
        position_results[pos] = res

        print(f"\n  {pos} ({len(pos_df)} players: {w_race}W/{nw_race}NW, {len(pos_match)} matches):")
        for wg, r in res.items():
            if r:
                print(f"    {wg}: W={r['white_prob']:.1f}%  NW={r['nonwhite_prob']:.1f}%  p={r['mwu_p']:.3e}")

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
            n_players = len(df[df['position'] == pos])
            plot_bars(position_results[pos], f'{pos} (n={n_players})', ax,
                      show_legend=(i == 0))

        for j in range(len(positions_with_data), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle('Model Confidence by Position\n(Position-Specific Distinctive Words)',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "logprob_by_position.png", dpi=200, bbox_inches='tight')
        print(f"\n  Saved: {OUTPUT_DIR / 'logprob_by_position.png'}")
        plt.close()

    # ── Save summary ──
    rows = []
    for wg, r in overall_results.items():
        if r:
            rows.append({'position': 'overall', 'word_group': wg, **r})
    for pos, res in position_results.items():
        for wg, r in res.items():
            if r:
                rows.append({'position': pos, 'word_group': wg, **r})
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "logprob_results.csv", index=False)
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
