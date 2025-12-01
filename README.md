# LLM Racial Bias in Football Commentary

Measuring racial bias in LLM-generated football commentary using perplexity scoring.

## Quick Start

### 1. Set up API key 

```bash
export TOGETHER_API_KEY="your_key_here"
```

### 2. Generate LLM completions

```bash
cd llm_completions
pip install -r requirements.txt
python generate_llm_commentary.py --use-api --qb-n 5 --rb-n 5 --wr-n 5 --def-n 5
```

### 3. Score completions (Perplexity, Atypicality)

```bash
cd ../bias_scoring
pip install -r requirements.txt
python bias_scoring.py ../tagged_transcripts.json ../llm_completions/output/llm_completions.csv results/scored_completions.csv
```

### 4. Generate plots

```bash
python create_plots.py results/scored_completions.csv figures/
```

---

## Preliminary Results (20 players)

| Race | Perplexity | Atypicality |
|------|------------|-------------|
| White | 238.72 | 4.71 |
| Nonwhite | 248.21 | 4.71 |

- **Perplexity**: Lower = more typical of real 1990-2019 commentary
- **Atypicality**: Higher = more unusual words (IDF-based)

### Interpretation

White players received slightly lower perplexity scores, suggesting LLM commentary for white players is marginally more "typical" of real human commentary patterns.

**⚠️ Not statistically significant** — only 20 players (40 completions). Full experiment requires 600 players for meaningful results.

---

## Full Experiment

For publication-quality results, run with default settings (600 players, probably will take a couple hours):

```bash
cd llm_completions
python generate_llm_commentary.py --use-api
```
---

## Project Structure

```
cs329r/
├── tagged_transcripts.json     # Dataset (1990-2019 commentary)
├── llm_completions/            # LLM generation
│   ├── generate_llm_commentary.py
│   └── output/llm_completions.csv
└── bias_scoring/               # Perplexity scoring + plots
    ├── bias_scoring.py
    ├── create_plots.py
    ├── results/scored_completions.csv
    └── figures/
```


