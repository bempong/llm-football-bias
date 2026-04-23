# race_pipeline

Deterministic pipeline that assigns racial / ethnic self-identification
labels to NFL players based on verbatim quoted self-ID in documented public
sources. Built for the LLM-bias-in-NFL-commentary paper (ACL / FAccT).

Five flat scripts, five flat output files. No database, no web UI, no agent
loop.

## Pipeline at a glance

```
players.csv  -->  search.py  -->  searches.jsonl
                                       |
                                       v
                    fetch.py  -->  fetches.csv + pages/*.txt + contexts.jsonl
                                       |
                                       v
                    extract.py --> extractions.jsonl
                                       |
                                       v
                    verify.py  --> verifications.csv
                                       |
                                       v
                    aggregate.py -> labels.csv + evidence.csv
```

Labeling rules are pre-registered in `LABELING_RULES.md` and implemented
mechanically in `aggregate.py` — no LLM call at the labeling step.

## Search design (content-neutral)

Stage 1 does **not** iterate over race-specific phrases. For each player we
run ~18 biographical / profile / podcast / college queries
(`config.DISCOVERY_QUERIES`) and let the LLM extract any first-person
self-identification it finds on the retrieved pages. This avoids the
brittleness of enumerating phrasings of "I am Black" / "my Samoan heritage"
/ etc., cuts Serper cost ~5× vs. the v1.0 race-cycling design, and keeps
the retrieval pool unbiased with respect to any particular group. See the
v1.1 changelog entry in `LABELING_RULES.md` for the full rationale.

## Setup

```bash
cd players_new_v2/2020s_project/race_pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Put `SERPER_API_KEY` and `ANTHROPIC_API_KEY` in a `.env` file at either the
repo root or this directory.

## Building the input

```bash
python build_players_csv.py
```

Reads `../nflverse_raw/roster_2020.csv … roster_2025.csv` and writes
`data/players.csv` with one row per unique `gsis_id`. Columns:
`player_id, name, position, team, years_active, pfr_url, college`.

If you want to exclude players already labeled (e.g. prior white-player
dataset), filter `data/players.csv` before running `search.py`.

## Running

### Smoke test (2 players, all stages)

```bash
python pipeline_run.py --limit 2
```

### Full pilot (50 players end-to-end)

```bash
# One stage at a time so you can inspect intermediate output
python search.py   --limit 50
python fetch.py    --limit 50 --workers 8   # --limit N fetches, not players
python extract.py  --limit 50
python verify.py
python aggregate.py --overwrite
```

Then manually inspect every row of `data/evidence.csv` before scaling up.
The spec says to iterate 2–3 times on the extraction prompt / spaCy rules
before the full run.

### Full run

```bash
python pipeline_run.py --resume
```

`--resume` is additive at every stage: it skips records already in the
respective output file, so a killed run can be restarted without re-doing
work.

## Cost controls

- Serper: `config.SERPER_COST_WARN_USD = 50`, `SERPER_COST_HARDSTOP_USD = 100`.
  Hitting the hardstop aborts unless you pass `--override-cost-cap` to
  `search.py`.
- Claude: every call is logged to `costs.log` with token counts and USD.
  Swap `--model claude-sonnet-4-5` for higher accuracy at ~3× cost.

## Files

| File                | Role                                                  |
| ------------------- | ----------------------------------------------------- |
| `config.py`         | Paths, API keys, queries, tier domains, rate limits   |
| `prompts.py`        | Claude extraction prompt                              |
| `utils.py`          | Shared helpers (JSONL/CSV I/O, normalization)         |
| `search.py`         | Stage 1 — Serper searches                             |
| `fetch.py`          | Stage 2 — HTTP fetch, trafilatura, spaCy contexts     |
| `extract.py`        | Stage 3 — Claude verbatim quote extraction            |
| `verify.py`         | Stage 4 — quote-exists-at-URL check                   |
| `aggregate.py`      | Stage 5 — pre-registered labeling rules               |
| `pipeline_run.py`   | Thin wrapper that runs all five in order              |
| `build_players_csv.py` | Helper to produce `data/players.csv` from nflverse |
| `LABELING_RULES.md` | The rules implemented by `aggregate.py`               |

## Notes for future iteration

- **spaCy NER recall.** `en_core_web_sm` is fast but misses some player
  names, especially non-English or unusually capitalized ones. If the 50-
  player pilot shows low context-extraction recall, swap to
  `en_core_web_trf` — one-line change in `config.SPACY_MODEL`.
- **Coreference.** The current pronoun heuristic (a pronoun in the sentence
  immediately following a named mention, with no closer PERSON entity)
  catches most inline attributions. If the pilot shows the extractor
  missing pronoun-attributed quotes like "He said, 'As a Black man, I…'",
  add `fastcoref` in `fetch.py` only then. Don't add it up front.
