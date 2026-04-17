#!/usr/bin/env python3
"""
Overlay bar chart: top-20 words per prompt for each racial group.

Produces a single horizontal bar chart in the same style as the individual
prompt figures, but with 80 bars per group (top-20 from each of the four
prompts), each prompt rendered in a distinct color.

Colors:  P1 Original = red   P2 Immersive = blue
         P3 Critical Fan = green   P4 Professional = yellow

Usage (from project root):
    python -m log_odds_scoring.plot_prompt_overlay
    python log_odds_scoring/plot_prompt_overlay.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

BY_PROMPT_DIR = (
    _PROJECT_ROOT
    / "output_results"
    / "bias_analysis_v4"
    / "4o-mini_3k"
    / "by_prompt"
)

PROMPTS = [
    ("p1_original",     "P1: Original",      "#d62728"),  # red
    ("p2_immersive",    "P2: Immersive",      "#1f77b4"),  # blue
    ("p3_critical_fan", "P3: Critical Fan",   "#2ca02c"),  # green
    ("p4_professional", "P4: Professional",   "#FFD700"),  # yellow
]

TOP_N      = 10
Z_THRESHOLD = 1.96

OUTPUT_PATH = BY_PROMPT_DIR / "figures" / "overlay_top20_by_prompt.png"

# ---------------------------------------------------------------------------
# Build bar data
# ---------------------------------------------------------------------------

def load_prompt_data(slug: str) -> pd.DataFrame:
    path = BY_PROMPT_DIR / "tables" / f"log_odds_tables_{slug}.csv"
    return pd.read_csv(path)


def collect_bars(top_n: int = TOP_N) -> pd.DataFrame:
    """Return a DataFrame of (word, z_score, prompt_label, color) for all bars."""
    rows = []
    for slug, label, color in PROMPTS:
        df = load_prompt_data(slug)
        top_pos = df.nlargest(top_n,  "z_score")[["word", "z_score"]]
        top_neg = df.nsmallest(top_n, "z_score")[["word", "z_score"]]
        for _, r in pd.concat([top_pos, top_neg]).iterrows():
            rows.append({"word": r["word"], "z_score": r["z_score"],
                         "label": label, "color": color})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_panel(ax, group_df: pd.DataFrame, title: str, x_min: float, x_max: float) -> None:
    """Draw a single horizontal-bar panel for one racial group."""
    # Sort so the most extreme bar is at the top
    group_df = group_df.sort_values("z_score", ascending=True).reset_index(drop=True)

    ax.barh(range(len(group_df)), group_df["z_score"],
            color=group_df["color"], alpha=0.8)

    ax.axvline(0,            color="black", linewidth=0.8, linestyle="-",  alpha=0.3)
    ax.axvline(-Z_THRESHOLD, color="gray",  linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline( Z_THRESHOLD, color="gray",  linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(range(len(group_df)))
    ax.set_yticklabels(group_df["word"], fontsize=7)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Log-Odds Z-Score", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3, linestyle=":")


def plot_overlay() -> None:
    bars = collect_bars()

    pos = bars[bars["z_score"] > 0].copy()
    neg = bars[bars["z_score"] < 0].copy()

    # Shared x-axis range so both panels are comparable
    abs_max = bars["z_score"].abs().max() * 1.05
    x_min_pos, x_max_pos =  0,        abs_max
    x_min_neg, x_max_neg = -abs_max,  0

    n_bars = max(len(pos), len(neg))
    fig, (ax_neg, ax_pos) = plt.subplots(
        1, 2,
        figsize=(18, max(7, n_bars * 0.18)),
        sharey=False,
    )

    _draw_panel(ax_neg, neg, "Nonwhite-Associated Words (Negative Z)", x_min_neg, x_max_neg)
    _draw_panel(ax_pos, pos, "White-Associated Words (Positive Z)",    x_min_pos, x_max_pos)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=color, alpha=0.8, label=label)
        for _, label, color in PROMPTS
    ] + [
        plt.Line2D([0], [0], color="gray", linewidth=0.9, linestyle="--",
                   label=f"p < 0.05  (|z| = {Z_THRESHOLD})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               frameon=True, fancybox=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"Distinctive Words by Race — Top {TOP_N} per Prompt, All Prompts Overlaid",
        fontweight="bold", fontsize=13, y=1.01,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_overlay()
