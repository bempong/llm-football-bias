"""Config for the three-model vision majority-vote pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
for _d in (DATA_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_REPO_ROOT = PIPELINE_DIR.parents[2]  # cs329r/
for env_path in (_REPO_ROOT / ".env", PIPELINE_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

ROSTERS_DIR  = PIPELINE_DIR.parent / "nflverse_raw"
ROSTER_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

PLAYERS_CSV = DATA_DIR / "players.csv"
VOTES_CSV   = DATA_DIR / "votes.csv"

CATEGORIES = ["white", "black", "latino", "asian", "other"]

OPENAI_MODEL    = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GEMINI_MODEL    = "gemini-2.0-flash"

VOTE_WORKERS   = 10
HTTP_TIMEOUT   = 20
MODEL_TIMEOUT  = 30

PROMPT = (
    "Classify the apparent race or ethnicity of the person in this image "
    "into exactly one of these categories: white, black, latino, asian, other. "
    "This is for academic research on demographic representation in professional "
    "sports. Respond with only one word from the list above. No explanation, "
    "no punctuation, no additional text."
)
