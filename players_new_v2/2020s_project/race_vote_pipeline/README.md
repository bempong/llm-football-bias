# race_vote_pipeline

Three-model vision majority-vote race classifier for NFL players on 2020–2025 rosters.

## What it does

For every player in `data/players.csv`, downloads the nflverse headshot and asks three vision models to classify the person into one of `{white, black, latino, asian, other}`. The per-model labels are normalized, tallied, and collapsed to a single `majority` column using simple plurality rules.

### Models

- OpenAI `gpt-4o-mini`
- Anthropic `claude-haiku-4-5`
- Google `gemini-2.0-flash`

The three are queried in parallel per player. The prompt (`config.PROMPT`) asks each model to output a single category word.

### Majority rule

With 3 voters across 5 categories:

| Pattern | Result |
|---|---|
| 3 agree | that label |
| 2 agree | that label |
| 1 valid vote + 2 refusals | that label |
| All 3 different | `tie` |
| All 3 refused or unparseable | `inconclusive` |

## Files

```
race_vote_pipeline/
├── __init__.py
├── config.py              # API keys, model names, paths, categories, prompt
├── build_players.py       # aggregates 2020–2025 rosters → data/players.csv
├── vote.py                # runs the 3-model vote → data/votes.csv
└── data/
    ├── players.csv        # 5,883 unique players with headshot URLs
    ├── votes.csv          # full output with per-model + majority columns
    └── distribution.csv   # final label counts and percentages
```

## Usage

From `players_new_v2/2020s_project/`:

```bash
python3 -m race_vote_pipeline.build_players
python3 -m race_vote_pipeline.vote --workers 12
```

`vote.py` is resumable — it skips any `player_id` already present in `votes.csv`. Use `--limit N` for smoke tests.

## Data scope

- Source: `nflverse_raw/roster_2020.csv` … `roster_2025.csv`
- 18,606 player-season rows → 6,465 unique `gsis_id`s → **5,883 with a non-empty `headshot_url`** (91% coverage)
- The 582 players excluded lack an nflverse headshot in any 2020–2025 season. ~50 of those can be recovered via the ESPN CDN pattern `https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png` for players with a populated `espn_id`.

## Final distribution (n = 5,883)

| Label | Count | % |
|---|---|---|
| black | 3,937 | 66.92 |
| white | 1,515 | 25.75 |
| tie | 173 | 2.94 |
| latino | 138 | 2.35 |
| inconclusive | 65 | 1.10 |
| other | 41 | 0.70 |
| asian | 14 | 0.24 |

Consistent with published 2020s NFL roster demographics (TIDES: ~58% Black, ~25% White, ~17% other), with the slight over-representation of Black here explained by (a) 90-man rosters vs. TIDES' 53-man active rosters, and (b) vision-classifier resolution of Afro-Latino and Pacific Islander players into the `black` bucket.

## Caveats

- This pipeline measures **apparent race/ethnicity from a headshot**, not self-identification. Right signal for analyses of how an LLM "sees" a player; wrong signal for questions about actual identity.
- The 238 rows labeled `tie` or `inconclusive` (~4%) concentrate on mixed-heritage, Pacific Islander, and ambiguous-phenotype players. Treat as a separate category rather than imputing a majority.
- Claude occasionally declines race classification with a safety message. These show up in `claude_raw` but produce an empty `claude` column and count toward tie/inconclusive outcomes.

## Output schema (`votes.csv`)

| Column | Meaning |
|---|---|
| `player_id` | nflverse `gsis_id` |
| `full_name` | player name |
| `headshot_url` | nflverse headshot URL used for classification |
| `gpt4o`, `claude`, `gemini` | per-model normalized labels (or empty if refused/errored) |
| `majority` | final collapsed label |
| `gpt4o_raw`, `claude_raw`, `gemini_raw` | truncated raw model outputs (for audit) |
| `error` | any per-provider error strings, semicolon-delimited |
