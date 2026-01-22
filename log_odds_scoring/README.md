# Log-Odds Ratios with Informative Dirichlet Prior (Monroe et al., 2008)

Implements the log-odds ratio method for detecting racial bias in LLM-generated football commentary, computing which words are statistically over-represented in commentary about white vs nonwhite players.

- **Log-odds ratio (δ)**: Measures association with each racial group
  - Positive value → more associated with white players
  - Negative value → more associated with nonwhite players
- **Z-score**: Standardized measure accounting for word frequency
  - |z| > 1.96 indicates statistical significance (p < 0.05)
- **Informative prior**: Uses corpus-wide word frequencies to stabilize estimates for rare words

## Usage

```bash
cd log_odds_scoring

python run_bias_analysis.py \
    --completions-path ../llm_completions/output/n500_gpt4o_completion/llm_completions.csv \
    --output-dir output/n500_gpt4o_completion
```

### Output Files

**log_odds_results.csv**: Word-level statistics
   - `word`: The word
   - `count_a`, `count_b`: Counts in each group
   - `log_odds`: δ (positive = more white, negative = more nonwhite)
   - `z_score`: Standardized score


## References

Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict. *Political Analysis*, 16(4), 372-403.
