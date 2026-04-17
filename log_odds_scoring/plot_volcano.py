#!/usr/bin/env python3
"""
Volcano-style scatter plot replicating the Monroe et al. (2008) style figure.

For each prompt, produces a scatter of:
    X-axis : total word frequency (log scale)
    Y-axis : log-odds ratio (positive = white-associated, negative = nonwhite-associated)

Top words by |z_score| are highlighted black with labels; the rest are gray.
A ranked legend of the top words is printed to the right margin.

Usage (from project root):
    python -m log_odds_scoring.plot_volcano
    python log_odds_scoring/plot_volcano.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

BASE_DIR = _PROJECT_ROOT / "output_results" / "bias_analysis_v4" / "4o-mini_3k"

BY_PROMPT_DIR    = BASE_DIR / "by_prompt"
BY_POSITION_DIR  = BASE_DIR / "by_position"

PROMPTS = [
    ("p1_original",     "P1: Original"),
    ("p2_immersive",    "P2: Immersive"),
    ("p3_critical_fan", "P3: Critical Fan"),
    ("p4_professional", "P4: Professional"),
]

POSITIONS = [
    ("qb", "Quarterback (QB)"),
    ("rb", "Running Back (RB)"),
    ("wr", "Wide Receiver (WR)"),
    ("te", "Tight End (TE)"),
    ("ol", "Offensive Line (OL)"),
    ("dl", "Defensive Line (DL)"),
    ("lb", "Linebacker (LB)"),
    ("db", "Defensive Back (DB)"),
    ("st", "Special Teams (ST)"),
]

TOP_N       = 20   # words to highlight / list in the legend
Z_THRESHOLD = 1.96


# ---------------------------------------------------------------------------
# Core plot
# ---------------------------------------------------------------------------

def plot_volcano(slug: str, title_label: str, tables_dir: Path, output_dir: Path) -> Path:
    path = tables_dir / f"log_odds_tables_{slug}.csv"
    df   = pd.read_csv(path)

    # Only keep words with a usable frequency
    df = df[df["total_count"] > 0].copy()

    # Split into significant / not
    sig = df[df["z_score"].abs() >= Z_THRESHOLD].copy()
    bg  = df[df["z_score"].abs() <  Z_THRESHOLD].copy()

    # Top words by |z| for labelling
    top = df.nlargest(TOP_N, "z_score")   # white-associated
    bot = df.nsmallest(TOP_N, "z_score")  # nonwhite-associated
    highlighted = pd.concat([top, bot]).drop_duplicates("word")

    # -----------------------------------------------------------------------
    # Figure layout: main scatter + narrow right panel for the word legend
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_axes([0.08, 0.10, 0.65, 0.82])   # main scatter
    ax_legend = fig.add_axes([0.75, 0.10, 0.22, 0.82])  # legend panel
    ax_legend.axis("off")

    # --- background (insignificant) dots ---
    ax.scatter(
        bg["total_count"], bg["log_odds"],
        s=8, color="lightgray", alpha=0.5, linewidths=0, zorder=1,
    )

    # --- significant but not top-N dots ---
    sig_not_top = sig[~sig["word"].isin(highlighted["word"])]
    ax.scatter(
        sig_not_top["total_count"], sig_not_top["log_odds"],
        s=14, color="gray", alpha=0.6, linewidths=0, zorder=2,
    )

    # --- highlighted top words ---
    ax.scatter(
        highlighted["total_count"], highlighted["log_odds"],
        s=highlighted["z_score"].abs() * 4,   # size encodes |z|
        color="black", alpha=0.85, linewidths=0, zorder=3,
    )

    # Labels for highlighted words — simple offset, no overlap avoidance
    for _, row in highlighted.iterrows():
        ax.text(
            row["total_count"] * 1.08, row["log_odds"],
            row["word"], fontsize=7, va="center", ha="left", zorder=4,
        )

    # --- zero line ---
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)

    # --- quadrant labels ---
    x_lo = df["total_count"].min() * 0.8
    y_hi = df["log_odds"].max()
    y_lo = df["log_odds"].min()
    ax.text(x_lo, y_hi * 0.85, "White-Associated",   fontsize=10,
            color="dimgray", alpha=0.7, va="top")
    ax.text(x_lo, y_lo * 0.85, "Nonwhite-Associated", fontsize=10,
            color="dimgray", alpha=0.7, va="bottom")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"
    ))
    ax.set_xlabel("Frequency of Word within Group", fontweight="bold")
    ax.set_ylabel("Log-Odds Ratio  (White − Nonwhite)", fontweight="bold")
    ax.set_title(
        f"Race-Predicted Words — {title_label}  (4o-mini, 3k players)",
        fontweight="bold", fontsize=11, pad=10,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # -----------------------------------------------------------------------
    # Right legend: two ranked columns (white top / nonwhite bottom)
    # -----------------------------------------------------------------------
    top_sorted = top.sort_values("z_score", ascending=False).reset_index(drop=True)
    bot_sorted = bot.sort_values("z_score", ascending=True).reset_index(drop=True)

    lines  = []
    lines += [("White-associated", True)]
    for _, r in top_sorted.iterrows():
        lines.append((r["word"], False))
    lines += [("", False), ("Nonwhite-associated", True)]
    for _, r in bot_sorted.iterrows():
        lines.append((r["word"], False))

    y_start = 1.0
    step    = 1.0 / (len(lines) + 1)
    for i, (text, bold) in enumerate(lines):
        weight = "bold" if bold else "normal"
        size   = 8.5   if bold else 7.5
        ax_legend.text(
            0, y_start - (i + 1) * step, text,
            transform=ax_legend.transAxes,
            fontsize=size, fontweight=weight, va="center",
        )

    out_path = output_dir / f"volcano_{slug}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== By Prompt ===")
    for slug, label in PROMPTS:
        out = plot_volcano(slug, label,
                           BY_PROMPT_DIR / "tables",
                           BY_PROMPT_DIR / "figures")
        print(f"Saved: {out}")

    print("\n=== By Position ===")
    for slug, label in POSITIONS:
        out = plot_volcano(slug, label,
                           BY_POSITION_DIR / "tables",
                           BY_POSITION_DIR / "figures")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
