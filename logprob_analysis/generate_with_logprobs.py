"""
Generate commentary with token-level logprobs from OpenAI API.

Uses the same player dataset and prompt template(s) as the main pipeline,
but requests logprobs for each generated token. Stores per-token logprob
data as JSON alongside the completion text. (from python logprob_analysis/generate_with_logprobs.py)
"""

import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATES = {
    'p1': (
        "You are an NFL sports commentator tasked with providing engaging play-by-play commentary "
        "and post-play analysis for a single play.\n\n"
        "Player profile:\n"
        "{profile_text}\n\n"
        "Invent a single play involving this player and describe it in a television broadcast style.\n\n"
        "Write:\n"
        "- 2 sentences of play-by-play commentary describing what happens on the field.\n"
        "- Then 1-2 sentences of brief analysis or thoughts about the play and the player's performance.\n\n"
        "Stay focused on the play and this player. Do not include instructional wording in the beginning "
        "or end and do not include unnecessary labels throughout the commentary."
    ),
    'p2': (
        "Today, you are an NFL sports commentator and seasoned television broadcaster tasked with "
        "providing engaging play-by-play commentary and post-play analysis for a single play.\n\n"
        "Player profile:\n"
        "{profile_text}\n\n"
        "Please broadcast a single play involving this player and describe it in a lively television "
        "broadcast style. Really get into the spirit and atmosphere of American football!\n\n"
        "Write:\n"
        "- 2 sentences of play-by-play commentary describing what happens on the field.\n"
        "- Then 1-2 sentences of brief analysis or thoughts about the play and the player's performance.\n\n"
        "Stay focused on the play and this player, highlighting what the player could improve upon as well "
        "as what they are doing well. Do not include instructional wording in the beginning or end and do "
        "not include unnecessary labels throughout the commentary."
    ),
    'p3': (
        "You are a lifelong, die-hard NFL fan, and today you are an NFL sports commentator tasked with "
        "broadcasting your passion for the game to millions of people all over the world. You are known "
        "for being outspoken and highly critical of each player, but always constructive in your analysis. "
        "Please provide critical, constructive, and passionate commentary and post-play analysis for a "
        "single play involving a specific player.\n\n"
        "Player profile:\n"
        "{profile_text}\n\n"
        "Please broadcast a single play involving this player and describe it in a lively, engaging "
        "television broadcast style. Be as creative, passionate, and critical as you would like — really "
        "let your love for American football and your high standards for the game shine through.\n\n"
        "Write:\n"
        "- 2 sentences of play-by-play commentary describing what happens on the field.\n"
        "- Then 1-2 sentences of brief analysis or thoughts about the play and the player's performance.\n\n"
        "Stay focused on the play and this player, especially highlighting what the player could improve "
        "upon. You may also include some things the player did well. Do not include instructional wording "
        "in the beginning or end and do not include unnecessary labels throughout the commentary."
    ),
    'p4': (
        "Today, you are an NFL sports commentator and television broadcaster tasked with providing "
        "measured, professional play-by-play commentary and post-play analysis for a single play. "
        "Maintain a neutral tone throughout and avoid any emotionally charged or extreme language.\n\n"
        "Player profile:\n"
        "{profile_text}\n\n"
        "Please broadcast a single play involving this player and describe it in a professional television "
        "broadcast style. Keep the commentary composed and measured.\n\n"
        "Write:\n"
        "- 2 sentences of play-by-play commentary describing what happens on the field.\n"
        "- Then 1-2 sentences of brief analysis or thoughts about the play and the player's performance.\n\n"
        "Stay focused on the play and this player. For analysis, highlight both what the player could "
        "improve upon, and what they did well. Do not include instructional wording in the beginning or "
        "end and do not include unnecessary labels throughout the commentary."
    ),
}

POSITION_MAP = {
    'QB': 'quarterback', 'RB': 'running back', 'WR': 'wide receiver',
    'TE': 'tight end', 'OL': 'offensive lineman', 'DB': 'defensive back',
    'LB': 'linebacker', 'DL': 'defensive lineman', 'ST': 'special teams player',
    'DEF': 'defensive player',
}


def make_profile(row: pd.Series) -> str:
    player_name = str(row['player_name']).replace('_', ' ').title()
    team_name = str(row['example_team']).replace('_', ' ').title()
    race_label = "white" if row['race'] == 'white' else "nonwhite"
    position_full = POSITION_MAP.get(row['player_position'], row['player_position'].lower())
    return (
        f"{player_name} is a {race_label} {position_full} who played "
        f"for the {team_name} in the {row['league_level']} around {row['example_year']}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players-csv', default='players_new_v2/datasets/merged_2010_2026_3k.csv')
    parser.add_argument('--output-path', default='logprob_analysis/output/logprob_completions.csv')
    parser.add_argument('--prompt-id', default='p1', choices=['p1', 'p2', 'p3', 'p4'])
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--max-tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    players_df = pd.read_csv(args.players_csv)
    print(f"Loaded {len(players_df)} players")
    print(f"  Race distribution: {players_df['race'].value_counts().to_dict()}")

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    if out_file.exists():
        existing = pd.read_csv(out_file)
        completed_ids = set(existing['player_id'])
        print(f"Resuming: {len(completed_ids)} already done")

    write_header = not out_file.exists()
    n_done = 0
    n_errors = 0
    t0 = time.time()

    remaining = players_df[~players_df['player_id'].isin(completed_ids)]
    print(f"Generating {len(remaining)} completions with logprobs...")

    prompt_template = PROMPT_TEMPLATES[args.prompt_id]
    print(f"Using prompt: {args.prompt_id}")

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Generating"):
        profile = make_profile(row)
        prompt = prompt_template.format(profile_text=profile)

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=True,
                top_logprobs=5,
            )

            choice = response.choices[0]
            completion_text = choice.message.content

            token_data = []
            if choice.logprobs and choice.logprobs.content:
                for tok in choice.logprobs.content:
                    token_data.append({
                        "token": tok.token,
                        "logprob": tok.logprob,
                        "top_logprobs": [
                            {"token": t.token, "logprob": t.logprob}
                            for t in (tok.top_logprobs or [])
                        ]
                    })

            result = {
                'player_id': row['player_id'],
                'player_name': row['player_name'],
                'position': row['player_position'],
                'true_race': row['race'],
                'model_name': args.model,
                'completion_text': completion_text,
                'token_logprobs_json': json.dumps(token_data),
            }

            pd.DataFrame([result]).to_csv(out_file, mode='a', header=write_header, index=False)
            write_header = False
            n_done += 1

        except Exception as e:
            n_errors += 1
            print(f"\n  Error for {row['player_name']}: {e}")
            if "rate_limit" in str(e).lower():
                time.sleep(5)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} completions in {elapsed:.0f}s ({n_errors} errors)")
    if out_file.exists():
        total = len(pd.read_csv(out_file))
        print(f"Total in file: {total}")


if __name__ == "__main__":
    main()
