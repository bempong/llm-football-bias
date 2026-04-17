"""
Run FinBERT sentiment analysis on completion_text for every row in
4o_mini_2010_2026_exp_3k.csv and write the results (plus two new columns)
to sentiment_results.csv in the same directory as this script.

Runs locally via the transformers pipeline — no API calls, no rate limits.
Install deps: pip install transformers torch tqdm pandas
"""

import pathlib
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = pathlib.Path(__file__).parent / (
    "llm_new_generations/output/4o_mini_2010_2026_exp_3k.csv"
)
OUTPUT_CSV = pathlib.Path(__file__).parent / "sentiment_results.csv"
MODEL = "ProsusAI/finbert"
BATCH_SIZE = 64          # tune up if you have a GPU, down if OOM
TEXT_MAX_CHARS = 512     # FinBERT is BERT-based, ~512 token limit

device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
classifier = pipeline(
    "text-classification",
    model=MODEL,
    device=device,
    truncation=True,
    max_length=512,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} rows from {INPUT_CSV.name}")

texts = df["completion_text"].fillna("").str[:TEXT_MAX_CHARS].tolist()

# ---------------------------------------------------------------------------
# Sentiment inference
# ---------------------------------------------------------------------------
labels: list[str] = []
scores: list[float] = []

for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Sentiment"):
    batch = texts[start : start + BATCH_SIZE]
    results = classifier(batch)  # returns [{"label": ..., "score": ...}, ...]
    for r in results:
        labels.append(r["label"])
        scores.append(round(r["score"], 6))

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
df["sentiment_label"] = labels
df["sentiment_score"] = scores

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df):,} rows to {OUTPUT_CSV}")
