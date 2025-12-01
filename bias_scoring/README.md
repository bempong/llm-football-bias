
# Bias Scoring Engine

A focused Python module for scoring racial bias in LLM-generated football commentary using **KenLM perplexity** and **IDF-based atypicality**.

## Purpose

This module **only** handles scoring. It does NOT:
- Generate LLM completions (done elsewhere)
- Create plots (done in separate notebooks)
- Run experiments

It provides a clean API to:
1. Train a KenLM bigram language model on 1990-2019 football commentary
2. Score LLM completions with perplexity (how typical text is)
3. Score completions with atypicality (how unusual words are)
4. Return/save scored DataFrames for downstream analysis

## Installation

```bash
cd bias_scoring

# Install dependencies
pip install -r requirements.txt

# Install KenLM
pip install https://github.com/kpu/kenlm/archive/master.zip

# Or with conda
conda install -c conda-forge kenlm
```

## Quick Start

### Command Line

```bash
python bias_scoring.py \
  ../tagged_transcripts.json \
  ../qb_commentary_race.json \
  scored_qb.csv
```

### Python API

```python
from bias_scoring import score_completions

scored_df = score_completions(
    commentary_path="../tagged_transcripts.json",
    completions_path="../qb_commentary_race.json",
    kenlm_train_txt="models/kenlm_train.txt",
    kenlm_model_path="models/football_commentary.bin",
    scored_output_path="scored_qb.csv"
)

# scored_df now has llm_ppl and llm_atypicality columns
print(scored_df.groupby('true_race')[['llm_ppl', 'llm_atypicality']].mean())
```

## Workflow

### Step 1: Load Commentary Corpus

```python
from bias_scoring import load_commentary_corpus

corpus = load_commentary_corpus(
    "../tagged_transcripts.json",
    year_start=1990,
    year_end=2019
)
# Returns list of ~6.8M commentary text strings
```

### Step 2: Train KenLM Model

```python
from bias_scoring import build_kenlm_training_text, train_bigram_kenlm

# Build training text
build_kenlm_training_text(corpus, "models/kenlm_train.txt")

# Train bigram model
train_bigram_kenlm("models/kenlm_train.txt", "models/football_commentary.bin")
```

This creates:
- `kenlm_train.txt` - Tokenized training text (one line per commentary)
- `football_commentary.bin` - Binary KenLM model

### Step 3: Load Model & Compute Scores

```python
from bias_scoring import (
    load_kenlm_model,
    compute_perplexity,
    build_idf,
    compute_atypicality
)

# Load model
model = load_kenlm_model("models/football_commentary.bin")

# Build IDF
idf = build_idf(corpus)

# Score a single text
text = "He shows explosive speed and natural athleticism."
ppl = compute_perplexity(model, text)
atyp = compute_atypicality(text, idf)

print(f"Perplexity: {ppl:.2f}")  # Lower = more typical
print(f"Atypicality: {atyp:.4f}")  # Higher = more unusual words
```

### Step 4: Score LLM Completions

```python
from bias_scoring import load_llm_completions, add_perplexity_column, add_atypicality_column

# Load completions
df = load_llm_completions("../qb_commentary_race.json")

# Add scores
df = add_perplexity_column(df, model)
df = add_atypicality_column(df, idf)

# Save
df.to_csv("scored_completions.csv", index=False)
```

## Input Data Formats

### Commentary Corpus

`tagged_transcripts.json` format:
```json
{
  "2010-team1-team2.txt": {
    "teams": ["team1", "team2"],
    "year": 2010,
    "transcript": "...commentary with <person player=\"Name\" race=\"white\">...</person> tags..."
  }
}
```

### LLM Completions

`qb_commentary_race.json` format:
```json
[
  {
    "name": "Tom Brady",
    "position": "QB",
    "race": "White",
    "response": "Brady drops back to pass...",
    "prompt_type": "race_included"
  }
]
```

## Output Format

Scored DataFrame columns:
- `base_id` - Unique ID
- `model_name` - LLM model used
- `condition` - "explicit", "ablated", or "counterfactual"
- `true_race` - "white" or "nonwhite"
- `position` - Player position
- `name` - Player name
- `completion_text` - LLM-generated text
- **`llm_ppl`** - Perplexity score (lower = more typical)
- **`llm_atypicality`** - Atypicality score (higher = more unusual)

## API Reference

### Core Functions

#### `load_commentary_corpus(path, year_start=1990, year_end=2019) -> List[str]`
Load real commentary texts from tagged transcripts.

#### `tokenize(text: str) -> List[str]`
Tokenize text (lowercase, alphanumeric only). Used consistently for training, IDF, and scoring.

#### `build_kenlm_training_text(corpus, out_path)`
Write tokenized corpus to file for KenLM training.

#### `train_bigram_kenlm(train_txt_path, model_path)`
Train KenLM bigram model using command-line tools.

#### `load_kenlm_model(model_path) -> kenlm.LanguageModel`
Load trained binary KenLM model.

#### `compute_perplexity(model, text) -> float`
Compute perplexity of text under model. **Lower = more typical of real commentary.**

#### `build_idf(corpus) -> Dict[str, float]`
Build IDF dictionary from corpus (one text = one document).

#### `compute_atypicality(text, idf, stopwords=None) -> float`
Compute mean IDF of content words. **Higher = more unusual words.**

#### `score_completions(...) -> pd.DataFrame`
Main pipeline: loads data, trains/loads model, scores completions, returns DataFrame.

### Helper Functions

#### `add_perplexity_column(df, model, text_col="completion_text") -> pd.DataFrame`
Add `llm_ppl` column to DataFrame.

#### `add_atypicality_column(df, idf, text_col="completion_text") -> pd.DataFrame`
Add `llm_atypicality` column to DataFrame.

#### `summarize_by_group(df, group_cols=None) -> pd.DataFrame`
Aggregate scores by race/condition/position for downstream plotting.

## Example: Full Scoring Pipeline

```python
from bias_scoring import score_completions

# Score QB completions
scored_df = score_completions(
    commentary_path="../tagged_transcripts.json",
    completions_path="../qb_commentary_race.json",
    kenlm_train_txt="models/kenlm_train.txt",
    kenlm_model_path="models/football_commentary.bin",
    scored_output_path="results/scored_qb.csv",
    year_start=1990,
    year_end=2019
)

# Analyze by race
print("\nPerplexity by Race:")
print(scored_df.groupby('true_race')['llm_ppl'].agg(['mean', 'std']))

print("\nAtypicality by Race:")
print(scored_df.groupby('true_race')['llm_atypicality'].agg(['mean', 'std']))

# Prepare for plotting
from bias_scoring import summarize_by_group

summary = summarize_by_group(scored_df, ['true_race', 'condition'])
summary.to_csv("results/summary_stats.csv", index=False)
```

## Interpretation

### Perplexity
- **Lower perplexity** = text is more typical/expected given the training corpus
- **Higher perplexity** = text is more atypical/unexpected

**Expected pattern if bias exists:**
- Physical language (e.g., "explosive speed") → lower perplexity (common in corpus)
- Cognitive language (e.g., "reads defense") → higher perplexity (less common)

### Atypicality (IDF-based)
- **Lower atypicality** = more common words
- **Higher atypicality** = more unusual/rare words

**Expected pattern:**
- Typical commentary → lower atypicality
- Unusual descriptors → higher atypicality

## Next Steps

After scoring, use the scored DataFrame in a separate notebook/script to:

1. **Statistical tests:**
   ```python
   from scipy import stats
   white_ppl = scored_df[scored_df['true_race']=='white']['llm_ppl']
   nonwhite_ppl = scored_df[scored_df['true_race']=='nonwhite']['llm_ppl']
   t_stat, p_value = stats.ttest_ind(white_ppl, nonwhite_ppl)
   ```

2. **Plotting:**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   sns.boxplot(data=scored_df, x='true_race', y='llm_ppl')
   plt.title('Perplexity by Race')
   plt.savefig('perplexity_by_race.png')
   ```

3. **Detailed analysis:**
   - Position-specific patterns
   - Temporal trends
   - Model comparisons

## File Structure

```
bias_scoring/
├── bias_scoring.py       # Main module
├── requirements.txt      # Dependencies
├── README.md            # This file
├── models/              # Created automatically
│   ├── kenlm_train.txt  # Training text
│   └── football_commentary.bin  # Trained model
└── results/             # Created by user
    ├── scored_qb.csv    # Scored completions
    └── summary_stats.csv # Aggregated stats
```

## Troubleshooting

### "kenlm not installed"
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### "lmplz command not found"
KenLM command-line tools need to be in PATH. Install from source:
```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### Training is slow
- Training on 6.8M texts takes ~5-10 minutes
- Binary model is ~500MB-1GB
- Model is cached, so subsequent runs are fast

### Out of memory
Reduce corpus size by sampling:
```python
import random
corpus_sample = random.sample(corpus, 1000000)  # 1M instead of 6.8M
```

## Citation

Based on Tie-Breaker methodology:
```
Fu, L., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). 
Tie-breaker: Using language models to quantify gender bias in sports journalism. 
arXiv preprint arXiv:1607.03895.
```

## License

MIT License - see parent directory for details.

