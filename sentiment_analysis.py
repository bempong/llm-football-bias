"""
Sentiment analysis summary for P1: Original prompt (3k player list).

Outputs:
  1. Table: positive/neutral/negative counts by race (white vs nonwhite)
  2. Table: positive/neutral/negative counts by position group x race
  3. Plot:  top 20 highest positive sentiment scores per racial group
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = pathlib.Path(__file__).parent
INPUT_CSV = HERE / "sentiment_results.csv"
OUT_DIR = HERE / "sentiment_output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load & filter to P1 only
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df = df[df["prompt_id"] == "P1: Original"].copy()
print(f"P1 rows: {len(df):,}")

POSITIONS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST"]
positive_df = df[df["sentiment_label"] == "positive"]

# ---------------------------------------------------------------------------
# Table 1: sentiment counts + avg positive score by race
# ---------------------------------------------------------------------------
race_table = (
    df.groupby(["true_race", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
)
SENT_COLS = [c for c in ["positive", "neutral", "negative"] if c in race_table.columns]
race_table = race_table[SENT_COLS]
race_table["total"] = race_table.sum(axis=1)
for col in SENT_COLS:
    race_table[f"pct_{col}"] = (race_table[col] / race_table["total"] * 100).round(1)
race_table["avg_positive_score"] = (
    positive_df.groupby("true_race")["sentiment_score"].mean().round(4)
)

print("\n=== Table 1: Sentiment by Race ===")
print(race_table.to_string())
race_table.to_csv(OUT_DIR / "sentiment_by_race.csv")

# ---------------------------------------------------------------------------
# Table 2: sentiment counts + avg positive score by position x race
# ---------------------------------------------------------------------------
pos_race_table = (
    df.groupby(["position", "true_race", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
)
pos_race_table = pos_race_table[SENT_COLS]
pos_race_table["total"] = pos_race_table.sum(axis=1)
for col in SENT_COLS:
    pos_race_table[f"pct_{col}"] = (pos_race_table[col] / pos_race_table["total"] * 100).round(1)
pos_race_table["avg_positive_score"] = (
    positive_df.groupby(["position", "true_race"])["sentiment_score"].mean().round(4)
)

# Reindex so positions appear in a sensible order
pos_race_table = pos_race_table.reindex(
    pd.MultiIndex.from_product(
        [POSITIONS, ["white", "nonwhite"]],
        names=["position", "true_race"],
    ),
    fill_value=0,
)

print("\n=== Table 2: Sentiment by Position x Race ===")
print(pos_race_table.to_string())
pos_race_table.to_csv(OUT_DIR / "sentiment_by_position_race.csv")

# ---------------------------------------------------------------------------
# Plot: top 20 highest positive sentiment scores per racial group
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
fig.suptitle("Top 20 Highest Positive Sentiment Scores by Race\n(P1: Original, 3k players)", fontsize=14)

colors = {"white": "#4C72B0", "nonwhite": "#DD8452"}

for ax, race in zip(axes, ["white", "nonwhite"]):
    top20 = (
        positive_df[positive_df["true_race"] == race]
        .nlargest(20, "sentiment_score")
        .reset_index(drop=True)
    )
    top20["label"] = top20["player_name"].str.title() + " (" + top20["position"] + ")"

    bars = ax.barh(
        top20["label"][::-1],
        top20["sentiment_score"][::-1],
        color=colors[race],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Positive Sentiment Score")
    ax.set_title(f"{race.capitalize()} players")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, score in zip(bars[::-1], top20["sentiment_score"]):
        ax.text(
            score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=8,
        )

plt.tight_layout()
plot_path = OUT_DIR / "top20_positive_scores_by_race.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {plot_path}")
print(f"\nAll outputs written to {OUT_DIR}/")
