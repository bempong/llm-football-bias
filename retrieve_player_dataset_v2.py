#!/usr/bin/env python3
"""
Player Dataset Retrieval for Racial Bias Experiment (v2)

Same as retrieve_player_dataset.py, but fixes team assignment by looking up
each player in NFL_team_rosters.json using (player_name, year) instead of
using the mode for the game they appear in.

Sources:
  - player_name, race, position  <- transcript <person> tags
  - year                         <- game filename (e.g. 2007-steelers-ravens.txt)
  - team                         <- roster lookup (player_name, year) -> team
                                    falls back to teams[0] if not found
"""

import json
import re
import os
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from html.parser import HTMLParser

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

random.seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'kaggle_path': 'tagged_transcripts.json',
    'roster_path': 'NFL_team_rosters.json',
    'year_start': 2010,
    'year_end': 2019,
    'qb_n': 100,
    'rb_n': 100,
    'wr_n': 100,
    'te_n': 100,
    'ol_n': 100,
    'db_n': 100,
    'lb_n': 100,
    'dl_n': 100,
    'st_n': 100,
    'extract_all_players': False
}

DEFENSIVE_BACKS  = {'DB', 'CB', 'S'}
LINEBACKERS      = {'LB'}
DEFENSIVE_LINE   = {'DE', 'DT', 'UT', 'NT', 'DL'}
OFFENSIVE_LINE   = {'OL', 'C', 'OG', 'OT', 'G', 'T'}
SPECIAL_TEAMS    = {'K', 'P', 'LS', 'PK', 'KR', 'PR'}
RUNNING_BACK     = {'RB', 'HB', 'FB', 'SB'}
WIDE_RECEIVER    = {'WR', 'SE'}
TIGHT_END        = {'TE'}
QUARTERBACK      = {'QB'}

# ============================================================================
# DATA LOADING
# ============================================================================

class PlayerTagParser(HTMLParser):
    """Extract player mentions from tagged transcripts."""

    def __init__(self):
        super().__init__()
        self.players = []
        self.current_player = None

    def handle_starttag(self, tag, attrs):
        if tag == 'person':
            attr_dict = dict(attrs)
            self.current_player = {
                'player':   attr_dict.get('player'),
                'race':     attr_dict.get('race'),
                'position': attr_dict.get('position'),
            }

    def handle_endtag(self, tag):
        if tag == 'person' and self.current_player:
            self.players.append(self.current_player)
            self.current_player = None

    def handle_data(self, data):
        pass


def extract_year_from_filename(filename: str) -> Optional[int]:
    match = re.match(r'(\d{4})-', filename)
    return int(match.group(1)) if match else None


def build_roster_lookup(roster_path: str) -> Dict[Tuple[str, int], str]:
    """
    Preprocess NFL_team_rosters.json into a fast O(1) lookup.

    Structure of roster file:
        { team: { year: { player_name: {position, race} } } }

    Returns:
        dict keyed by (player_name, year_int) -> team name.
        If a player appears on multiple teams in the same year (rare),
        the first team encountered is kept.
    """
    print(f"Building roster lookup...")
    with open(roster_path, 'r') as f:
        roster_data = json.load(f)

    lookup: Dict[Tuple[str, int], str] = {}
    for team, years in roster_data.items():
        for year_str, players in years.items():
            year = int(year_str)
            for player_name in players:
                key = (player_name, year)
                if key not in lookup:   # first team wins on collision
                    lookup[key] = team

    print(f"  Roster lookup built: {len(lookup):,} (player, year) entries")
    return lookup


def save_roster_lookup(lookup: Dict[Tuple[str, int], str], output_path: str) -> None:
    """
    Save the roster lookup to a readable JSON file.

    Converts the internal {(player_name, year_int): team} dict into a
    nested {player_name: {year: team}} structure so it's JSON-serializable
    and easy to inspect (e.g. search for "tom brady" to see all his years).
    """
    nested: Dict[str, Dict[str, str]] = {}
    for (player_name, year), team in sorted(lookup.items()):
        if player_name not in nested:
            nested[player_name] = {}
        nested[player_name][str(year)] = team

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(nested, f, indent=2)
    print(f"  Roster lookup saved to {output_path}")


def load_kaggle_data(
    path: str,
    roster_lookup: Dict[Tuple[str, int], str],
    year_start: int = 1990,
    year_end: int = 2019
) -> pd.DataFrame:
    """
    Load Kaggle football dataset and filter to the given year range.

    Team is assigned via roster lookup (player_name, year) -> team.
    Falls back to teams[0] from the game file when not found in the roster.

    Returns DataFrame with columns:
        player_name, player_position, race, year, team, league_level
    """
    print(f"Loading Kaggle dataset from {path}...")

    with open(path, 'r') as f:
        data = json.load(f)

    nfl_indicators = [
        '_49ers', '_bears', '_bengals', '_bills', '_broncos',
        '_browns', '_buccaneers', '_cardinals', '_chargers',
        '_chiefs', '_colts', '_cowboys', '_dolphins', '_eagles',
        '_falcons', '_giants', '_jaguars', '_jets', '_lions',
        '_packers', '_panthers', '_patriots', '_raiders', '_rams',
        '_ravens', '_redskins', '_saints', '_seahawks', '_steelers',
        '_texans', '_titans', '_vikings'
    ]

    rows = []

    for game_file, game_data in data.items():
        year = extract_year_from_filename(game_file)
        if not year or year < year_start or year > year_end:
            continue

        teams = game_data.get('teams', [])
        transcript = game_data.get('transcript', '')
        league_level = 'NFL' if any(ind in game_file for ind in nfl_indicators) else 'College'

        parser = PlayerTagParser()
        try:
            parser.feed(transcript)
        except Exception:
            continue

        for player in parser.players:
            if player['race'] not in ['white', 'nonwhite']:
                continue
            if not player['player'] or not player['position']:
                continue

            player_name = player['player']

            # --- Team lookup (the fix) ---
            roster_team = roster_lookup.get((player_name, year))
            if roster_team is not None:
                team = roster_team
            else:
                team = teams[0] if teams else 'Unknown'

            rows.append({
                'player_name':     player_name,
                'player_position': player['position'].upper(),
                'race':            player['race'],
                'year':            year,
                'team':            team,
                'league_level':    league_level,
            })

    df = pd.DataFrame(rows)

    print(f"\nLoaded {len(df)} player-mention records from {year_start}-{year_end}")
    print(f"  Unique players:   {df['player_name'].nunique()}")
    print(f"  Race distribution:\n{df['race'].value_counts()}")
    print(f"  All positions:\n{df['player_position'].value_counts()}")

    return df


# ============================================================================
# PLAYER SAMPLING
# ============================================================================

def canonicalize_position(position: str) -> str:
    position = position.upper().strip()
    if position in QUARTERBACK:    return 'QB'
    elif position in RUNNING_BACK: return 'RB'
    elif position in WIDE_RECEIVER: return 'WR'
    elif position in TIGHT_END:    return 'TE'
    elif position in OFFENSIVE_LINE: return 'OL'
    elif position in DEFENSIVE_BACKS: return 'DB'
    elif position in LINEBACKERS:  return 'LB'
    elif position in DEFENSIVE_LINE: return 'DL'
    elif position in SPECIAL_TEAMS: return 'ST'
    else: return 'OTHER'


def sample_player_position_combos(
    df: pd.DataFrame,
    qb_n: int = 100,
    rb_n: int = 100,
    wr_n: int = 100,
    te_n: int = 100,
    ol_n: int = 100,
    db_n: int = 100,
    lb_n: int = 100,
    dl_n: int = 100,
    st_n: int = 100,
    extract_all_players: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Collapse per-game player mentions into one row per unique player,
    then sample by position group.

    For team: takes the mode across all appearances. Now that team comes
    from the roster lookup, this correctly picks the team a player was
    with most often during their appearances in the dataset.
    """
    print("\nSampling player/position combos...")

    df['canonical_position'] = df['player_position'].apply(canonicalize_position)
    df = df[df['canonical_position'].isin(['QB', 'RB', 'WR', 'TE', 'OL', 'DB', 'LB', 'DL', 'ST'])]
    df = df[df['league_level'] == 'NFL']

    # Drop name collisions: player names where the same (name, year) pair appears
    # with conflicting race labels. This means two different people shared a name
    # AND played in the same year — corrupting both the race label and the roster
    # lookup (which can only store one team per name+year key).
    # Players who share a name but played in completely different years are fine —
    # their (name, year) keys are distinct and don't collide.
    ambiguous_names = (
        df.groupby(['player_name', 'year'])['race']
        .nunique()
        .pipe(lambda s: s[s > 1])
        .index.get_level_values('player_name')
        .unique()
    )
    if len(ambiguous_names) > 0:
        print(f"\n  Dropping {len(ambiguous_names)} ambiguous names (same name + year, conflicting races):")
        for name in sorted(ambiguous_names):
            print(f"    {name}")
        df = df[~df['player_name'].isin(ambiguous_names)]

    # Compute (team, year) together so they always refer to the same season.
    # Taking mode of each independently can produce impossible combos
    # (e.g. mode year = 2000, mode team = houston_texans which didn't exist yet).
    df['team_year'] = list(zip(df['team'], df['year']))

    player_combos = df.groupby(['player_name', 'canonical_position', 'race']).agg(
        team_year     = ('team_year', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),
        league_level  = ('league_level', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),
    ).reset_index().rename(columns={'canonical_position': 'player_position'})

    player_combos['example_team'] = player_combos['team_year'].apply(lambda x: x[0])
    player_combos['example_year'] = player_combos['team_year'].apply(lambda x: x[1])
    player_combos = player_combos.drop(columns=['team_year'])

    print(f"  Found {len(player_combos)} unique player/position combos")
    print(f"  By position:\n{player_combos['player_position'].value_counts()}")

    if extract_all_players:
        print("\nExtracting all players without sampling...")
        target_counts = {pos: len(player_combos[player_combos['player_position'] == pos])
                         for pos in ['QB', 'RB', 'WR', 'TE', 'OL', 'DB', 'LB', 'DL', 'ST']}
    else:
        print("\nSampling players by position:")
        target_counts = {
            'QB': qb_n, 'RB': rb_n, 'WR': wr_n, 'TE': te_n, 'OL': ol_n,
            'DB': db_n, 'LB': lb_n, 'DL': dl_n, 'ST': st_n
        }

    sampled = []
    for position, target_n in target_counts.items():
        pos_players = player_combos[player_combos['player_position'] == position]
        available_n = len(pos_players)
        actual_n = min(target_n, available_n)
        if actual_n < target_n:
            print(f"  Warning: {position} - requested {target_n}, only {available_n} available")
        if actual_n > 0:
            sampled.append(pos_players.sample(n=actual_n, random_state=random_state))

    players_df = pd.concat(sampled, ignore_index=True)
    players_df['player_id'] = range(len(players_df))

    print(f"\nSampled {len(players_df)} players:")
    print(f"  By position:\n{players_df['player_position'].value_counts()}")
    print(f"  By race:\n{players_df['race'].value_counts()}")

    return players_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrieve player dataset with accurate team assignment from rosters")
    parser.add_argument('--kaggle-path',  type=str, default=DEFAULT_CONFIG['kaggle_path'])
    parser.add_argument('--roster-path',  type=str, default=DEFAULT_CONFIG['roster_path'])
    parser.add_argument('--output-path',  type=str, default='players_new_v2/sampled_players_v2_recent.csv')
    parser.add_argument('--year-start',   type=int, default=DEFAULT_CONFIG['year_start'])
    parser.add_argument('--year-end',     type=int, default=DEFAULT_CONFIG['year_end'])
    parser.add_argument('--qb-n',  type=int, default=DEFAULT_CONFIG['qb_n'])
    parser.add_argument('--rb-n',  type=int, default=DEFAULT_CONFIG['rb_n'])
    parser.add_argument('--wr-n',  type=int, default=DEFAULT_CONFIG['wr_n'])
    parser.add_argument('--te-n',  type=int, default=DEFAULT_CONFIG['te_n'])
    parser.add_argument('--ol-n',  type=int, default=DEFAULT_CONFIG['ol_n'])
    parser.add_argument('--db-n',  type=int, default=DEFAULT_CONFIG['db_n'])
    parser.add_argument('--lb-n',  type=int, default=DEFAULT_CONFIG['lb_n'])
    parser.add_argument('--dl-n',  type=int, default=DEFAULT_CONFIG['dl_n'])
    parser.add_argument('--st-n',  type=int, default=DEFAULT_CONFIG['st_n'])
    parser.add_argument('--extract-all-players', action='store_true')
    parser.add_argument('--save-roster-lookup', type=str, default=None,
                        help='If set, saves the roster lookup dict as a readable JSON file at this path')
    args = parser.parse_args()

    print("=" * 80)
    print("Retrieving Player Dataset (v2 — roster-backed team assignment)")
    print("=" * 80)

    # Step 1: Build roster lookup
    roster_lookup = build_roster_lookup(args.roster_path)
    if args.save_roster_lookup:
        save_roster_lookup(roster_lookup, args.save_roster_lookup)

    # Step 2: Load transcripts, assign teams via roster
    df_raw = load_kaggle_data(
        args.kaggle_path,
        roster_lookup,
        year_start=args.year_start,
        year_end=args.year_end
    )

    # Step 3: Sample
    players_df = sample_player_position_combos(
        df_raw,
        qb_n=args.qb_n, rb_n=args.rb_n, wr_n=args.wr_n, te_n=args.te_n,
        ol_n=args.ol_n, db_n=args.db_n, lb_n=args.lb_n, dl_n=args.dl_n,
        st_n=args.st_n, extract_all_players=args.extract_all_players
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    players_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()


# Example (sampled_players_v2.csv):
# python3 retrieve_player_dataset_v2.py \
#   --kaggle-path "/mnt/c/Users/hallj/Downloads/nfl_kaggle/dataset/tagged_transcripts.json" \
#   --roster-path "NFL_team_rosters.json" \
#   --year-start 1960 --year-end 2019   \
#   --extract-all-players
#   --save-roster-lookup "players_new_v2/roster_lookup.json"

# Example (sampled_recent_players_v2.csv):
# python3 retrieve_player_dataset_v2.py \
#   --kaggle-path "/mnt/c/Users/hallj/Downloads/nfl_kaggle/dataset/tagged_transcripts.json" \
#   --roster-path "NFL_team_rosters.json" \
#   --year-start 2010 --year-end 2019   \
#   --extract-all-players
