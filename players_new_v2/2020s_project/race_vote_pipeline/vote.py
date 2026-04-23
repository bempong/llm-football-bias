"""Three-model vision majority-vote race classifier.

For every player in data/players.csv, downloads the headshot, asks GPT-4o,
Claude, and Gemini to classify the image into one of
{white, black, latino, asian, other}, and records per-model labels plus
a majority label in data/votes.csv. Resumable: rows already in votes.csv
are skipped.
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from race_vote_pipeline import config  # noqa: E402

from openai import OpenAI  # noqa: E402
import anthropic  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402


_openai    = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
_anthropic = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None
_gemini    = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None


_ALIASES = {
    "hispanic":          "latino",
    "latinx":            "latino",
    "latina":            "latino",
    "latino/a":          "latino",
    "caucasian":         "white",
    "european":          "white",
    "african-american":  "black",
    "african":           "black",
    "afro":              "black",
    "african_american":  "black",
    "pacific":           "other",
    "islander":          "other",
    "native":            "other",
    "middle":            "other",
    "mixed":             "other",
    "biracial":          "other",
    "multiracial":       "other",
}


def normalize(label: str) -> str:
    if not label:
        return ""
    s = label.strip().lower()
    s = s.strip(".,!?:;\"' \n\t")
    first = s.split()[0] if s else ""
    if first in config.CATEGORIES:
        return first
    if first in _ALIASES:
        return _ALIASES[first]
    return ""


def download_image(url: str) -> tuple[bytes, str]:
    r = httpx.get(url, timeout=config.HTTP_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    mime = (r.headers.get("content-type") or "image/jpeg").split(";")[0].strip().lower()
    if mime not in {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}:
        mime = "image/jpeg"
    if mime == "image/jpg":
        mime = "image/jpeg"
    return r.content, mime


def ask_openai(img: bytes, mime: str) -> str:
    if _openai is None:
        return ""
    b64 = base64.b64encode(img).decode()
    r = _openai.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": config.PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=10,
        timeout=config.MODEL_TIMEOUT,
    )
    return r.choices[0].message.content or ""


def ask_anthropic(img: bytes, mime: str) -> str:
    if _anthropic is None:
        return ""
    b64 = base64.b64encode(img).decode()
    r = _anthropic.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": config.PROMPT},
            ],
        }],
        timeout=config.MODEL_TIMEOUT,
    )
    return "".join(getattr(b, "text", "") for b in r.content)


def ask_gemini(img: bytes, mime: str) -> str:
    if _gemini is None:
        return ""
    r = _gemini.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=img, mime_type=mime),
            config.PROMPT,
        ],
    )
    return r.text or ""


def majority_vote(labels: list[str]) -> str:
    valid = [l for l in labels if l in config.CATEGORIES]
    if not valid:
        return "inconclusive"
    counts = Counter(valid).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return "tie"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            done.add(row["player_id"])
    return done


_write_lock = threading.Lock()


def process_player(player: dict, writer, fh) -> str:
    pid  = player["player_id"]
    name = player["full_name"]
    url  = player["headshot_url"]

    try:
        img, mime = download_image(url)
    except Exception as e:
        row = {
            "player_id": pid, "full_name": name, "headshot_url": url,
            "gpt4o": "", "claude": "", "gemini": "",
            "majority": "download_error",
            "gpt4o_raw": "", "claude_raw": "", "gemini_raw": "",
            "error": f"download:{e}"[:300],
        }
        with _write_lock:
            writer.writerow(row)
            fh.flush()
        return "download_error"

    raw: dict[str, str] = {"gpt4o": "", "claude": "", "gemini": ""}
    errors: dict[str, str] = {}

    def _run(fn, key):
        try:
            raw[key] = fn(img, mime)
        except Exception as e:
            errors[key] = f"{type(e).__name__}:{e}"[:200]

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(_run, ask_openai,    "gpt4o"),
            ex.submit(_run, ask_anthropic, "claude"),
            ex.submit(_run, ask_gemini,    "gemini"),
        ]
        for f in as_completed(futs):
            f.result()

    labels = {k: normalize(v) for k, v in raw.items()}
    maj = majority_vote(list(labels.values()))

    row = {
        "player_id":    pid,
        "full_name":    name,
        "headshot_url": url,
        "gpt4o":        labels["gpt4o"],
        "claude":       labels["claude"],
        "gemini":       labels["gemini"],
        "majority":     maj,
        "gpt4o_raw":    raw["gpt4o"].strip()[:60],
        "claude_raw":   raw["claude"].strip()[:60],
        "gemini_raw":   raw["gemini"].strip()[:60],
        "error":        "; ".join(f"{k}:{v}" for k, v in errors.items())[:300],
    }
    with _write_lock:
        writer.writerow(row)
        fh.flush()
    return maj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N un-voted players (0 = all)")
    ap.add_argument("--workers", type=int, default=config.VOTE_WORKERS)
    args = ap.parse_args()

    if not config.PLAYERS_CSV.exists():
        raise SystemExit(
            f"{config.PLAYERS_CSV} not found. Run `python -m "
            f"race_vote_pipeline.build_players` first."
        )

    players = list(csv.DictReader(config.PLAYERS_CSV.open()))
    done = load_done(config.VOTES_CSV)
    todo = [p for p in players if p["player_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(players)} total, {len(done)} already voted, {len(todo)} to process")
    if not todo:
        return

    header = [
        "player_id", "full_name", "headshot_url",
        "gpt4o", "claude", "gemini", "majority",
        "gpt4o_raw", "claude_raw", "gemini_raw",
        "error",
    ]
    mode = "a" if config.VOTES_CSV.exists() else "w"
    fh = config.VOTES_CSV.open(mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=header)
    if mode == "w":
        writer.writeheader()
        fh.flush()

    t0 = time.time()
    tally: Counter = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_player, p, writer, fh) for p in todo]
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    res = fut.result()
                except Exception as e:
                    res = f"error:{type(e).__name__}"
                tally[res] += 1
                if i % 25 == 0 or i == len(futs):
                    dt = time.time() - t0
                    rate = i / dt if dt > 0 else 0
                    print(f"  [{i}/{len(futs)}]  {rate:.2f}/s  "
                          f"last={res}  tally={dict(tally)}")
    finally:
        fh.close()

    print(f"done. wrote {config.VOTES_CSV}  final tally={dict(tally)}")


if __name__ == "__main__":
    main()
