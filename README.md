# Evaluating LLM Racial Bias in American Football Commentary

## Abstract
Football commentary has a history of racial bias dating back to its inception, and this bias has become increasingly controversial in recent times. However, this bias has not been adequately explored in the context of Large Language Models (LLMs). To examine this, we first analyze NFL and NCAA game transcripts
from the Kaggle Football Commentary Dataset,
confirming significant racial disparities in descriptor usage. We then prompt gpt-4o-mini
for 500 player profiles across race-included and
race-ablated conditions, evaluating racial disparities via log-odds ratio with Dirichlet priors. Our results reveal that LLMs reproduce
stereotypical racial framing even when conditioned only on player race. These findings raise concerns for automated content generation in sports media, demonstrating that LLMs can amplify patterns and racial stereotypes from human commentary.

We analyze the extent to which LLMs exhibit racial bias when tasked with sports commentary. We take advantage of Log-Odds Ratio Analysis (Monroe et al., 2008) to identify which words are statistically over-represented in commentary about white vs nonwhite players. Overall, we explore racial bias in LLM-generated football commentary using statistical text analysis.

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy scipy matplotlib tqdm python-dotenv openai google-generativeai
```

### 2. Set up API key

```bash
export OPENAI_API_KEY="your_key_here"
# OR 
cp .env.example .env
# ...and add your own OPENAI_API_KEY
```

### 3. Generate LLM commentary

We use `gpt-4o-mini` to generate commentary under two conditions:
- **Completion**: LLM continues real commentary from the dataset
- **Generation**: LLM generates commentary from scratch

```bash
# COMPLETION task: Continue real commentary (Section 5.2)
cd llm_completions
python generate_llm_commentary.py \
    --use-api \
    --api-provider openai \
    --model-name gpt-4o-mini \
    --players-csv ../players/sampled_players.csv \
    --output-path output/n500_gpt4o_completion/llm_completions.csv

# Optional: Sample new players by position (instead of using pre-sampled CSV)
# python generate_llm_commentary.py \
#     --use-api --api-provider openai --model-name gpt-4o-mini \
#     --qb-n 50 --rb-n 50 --wr-n 50 --def-n 50 \
#     --output-path output/custom_sample/llm_completions.csv

# GENERATION ("from scratch") task: Generate from scratch (Section 5.3)
cd ../llm_new_generations
python generation_api_script.py \
    --model-name gpt-4o-mini \
    --output-path output/llm_generations.csv
```

### 4. Run log-odds bias analysis

```bash
cd ../log_odds_scoring

# Analyze completion data
python run_bias_analysis.py \
    --completions-path ../llm_completions/output/n500_gpt4o_completion/llm_completions.csv \
    --output-dir output/n500_gpt4o_completion

# Analyze generation data
python run_bias_analysis.py \
    --completions-path ../llm_new_generations/output/llm_generations.csv \
    --output-dir output/n500_gpt4o_generation
```

This generates:
- `log_odds_results.csv` - z-scores, log-odds ratios
- `figures/distinctive_words_by_race.png` - Visualization

---
## Map

```
cs329r/
├── README.md
├── tagged_transcripts.json                 # Kaggle dataset 
├── players/
│   └── sampled_players.csv                 # 500 sampled players
├── llm_completions/
│   ├── generate_llm_commentary.py          # Completion generation script
│   └── output/
│       └── n500_gpt4o_completion/          # Completion task data (Section 5.2)
├── llm_new_generations/
│   ├── generation_api_script.py            # From-scratch generation script
│   └── output/
│       └── llm_generations.csv             # Generation task data (Section 5.3)
└── log_odds_scoring/
    ├── run_bias_analysis.py                # Main analysis script
    ├── log_odds_scoring.py                 # Log-odds computation
    ├── create_plots.py                     # Visualization
    └── output/
        ├── n500_gpt4o_completion/          # Results for completion task
```


