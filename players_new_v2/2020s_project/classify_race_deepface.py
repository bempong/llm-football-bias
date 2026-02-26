"""
Classify player race (white/nonwhite) using deepface on NFL headshot images.

Prerequisites:
    pip install deepface opencv-python-headless tf-keras requests pandas

Usage:
    python classify_race_deepface.py

Reads sampled_players_2020s.csv and headshot_lookup.json from the same directory.
Only processes rows where the 'race' column is empty/NaN.
Saves results back to sampled_players_2020s.csv, plus a detailed log CSV
(deepface_classification_details.csv) for verification.
"""

import json
import os
import tempfile
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYERS_CSV = os.path.join(SCRIPT_DIR, "sampled_players_2020s.csv")
HEADSHOT_JSON = os.path.join(SCRIPT_DIR, "headshot_lookup.json")
DETAILS_CSV = os.path.join(SCRIPT_DIR, "deepface_classification_details.csv")
FAILURES_TXT = os.path.join(SCRIPT_DIR, "deepface_failures.txt")

WHITE_THRESHOLD = 50.0
CHECKPOINT_INTERVAL = 200


def load_data():
    df = pd.read_csv(PLAYERS_CSV)
    with open(HEADSHOT_JSON) as f:
        headshots = json.load(f)
    return df, headshots


def classify_player(url, analyze_fn):
    """Download headshot and return (classification, white_score, dominant_race)."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        result = analyze_fn(
            img_path=tmp_path,
            actions=["race"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]

        race_scores = result.get("race", {})
        white_score = float(race_scores.get("white", 0))
        dominant_race = result.get("dominant_race", "unknown")
        classification = "white" if white_score > WHITE_THRESHOLD else "nonwhite"

        return classification, white_score, dominant_race
    finally:
        os.unlink(tmp_path)


def main():
    print("Loading data...")
    df, headshots = load_data()

    unlabeled_mask = df["race"].isna() | (df["race"] == "")
    has_url = df["player_name"].apply(
        lambda n: n in headshots and headshots[n] is not None
    )
    to_process_mask = unlabeled_mask & has_url
    to_process = df[to_process_mask].index.tolist()

    print(f"Total players: {len(df)}")
    print(f"Already labeled: {(~unlabeled_mask).sum()}")
    print(f"Unlabeled with headshot: {len(to_process)}")
    print(f"Unlabeled without headshot: {(unlabeled_mask & ~has_url).sum()}")

    if not to_process:
        print("Nothing to classify. Exiting.")
        return

    print("Loading deepface model...")
    from deepface import DeepFace

    details = []
    failed = []
    t0 = time.time()

    print(f"\nStarting classification for {len(to_process)} players...")

    for i, idx in enumerate(to_process):
        name = df.loc[idx, "player_name"]
        url = headshots[name]

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i) / rate / 60 if rate > 0 else 0
            print(
                f"  [{i}/{len(to_process)}] {i * 100 // len(to_process)}%  "
                f"({rate:.1f} players/s, ~{eta:.0f} min remaining)"
            )

        try:
            classification, white_score, dominant_race = classify_player(
                url, DeepFace.analyze
            )
            df.loc[idx, "race"] = classification
            details.append({
                "player_name": name,
                "classification": classification,
                "white_score": round(white_score, 2),
                "dominant_race": dominant_race,
                "headshot_url": url,
            })
        except Exception as e:
            failed.append((name, str(e)[:120]))

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            df.to_csv(PLAYERS_CSV, index=False)
            pd.DataFrame(details).to_csv(DETAILS_CSV, index=False)
            print(f"    (checkpoint saved at {i + 1})")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed / 60:.1f} minutes.")
    print(f"Classified: {len(details)}, Failed: {len(failed)}")

    df.to_csv(PLAYERS_CSV, index=False)
    print(f"Saved {PLAYERS_CSV}")

    if details:
        pd.DataFrame(details).to_csv(DETAILS_CSV, index=False)
        print(f"Saved {DETAILS_CSV}")

    if failed:
        with open(FAILURES_TXT, "w") as f:
            for name, err in failed:
                f.write(f"{name}: {err}\n")
        print(f"Saved {len(failed)} failures to {FAILURES_TXT}")

    print(f"\nRace distribution:")
    print(df["race"].value_counts().to_string())
    still_missing = df["race"].isna().sum()
    if still_missing:
        print(f"Still unlabeled: {still_missing}")


if __name__ == "__main__":
    main()
