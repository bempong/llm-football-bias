#!/usr/bin/env python3
"""
Bias Scoring Engine for Football Commentary

This module computes perplexity and atypicality scores for LLM-generated 
football commentary completions using a KenLM bigram model trained on 
real 1990-2019 commentary.

Main workflow:
1. Train KenLM on real commentary corpus
2. Compute perplexity of LLM completions under this model
3. Compute IDF-based atypicality scores
4. Return scored DataFrame for downstream analysis/plotting
"""

import json
import re
import math
import os
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional, Tuple
from html.parser import HTMLParser

import pandas as pd
import numpy as np

# Pure Python bigram model (no external tools required)


# ============================================================================
# COMMENTARY CORPUS LOADING
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
                'player': attr_dict.get('player'),
                'race': attr_dict.get('race'),
                'position': attr_dict.get('position'),
                'text': ''
            }
    
    def handle_endtag(self, tag):
        if tag == 'person' and self.current_player:
            self.players.append(self.current_player)
            self.current_player = None
    
    def handle_data(self, data):
        if self.current_player is not None:
            self.current_player['text'] = data


def clean_xml_tags(text: str) -> str:
    """Remove XML tags from text."""
    text = re.sub(r'<person[^>]*>.*?</person>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename like '2010-team1-team2.txt'."""
    match = re.match(r'(\d{4})-', filename)
    return int(match.group(1)) if match else None


def load_commentary_corpus(path: str, year_start: int = 1990, year_end: int = 2019) -> List[str]:
    """
    Load real football commentary texts from tagged transcripts.
    
    Args:
        path: Path to tagged_transcripts.json
        year_start: Start year (inclusive)
        year_end: End year (inclusive)
    
    Returns:
        List of commentary text strings (one per player mention context)
    """
    print(f"Loading commentary corpus from {path}...")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    corpus = []
    
    for game_file, game_data in data.items():
        year = extract_year_from_filename(game_file)
        
        # Filter to year range
        if not year or year < year_start or year > year_end:
            continue
        
        transcript = game_data.get('transcript', '')
        
        # Parse player tags
        parser = PlayerTagParser()
        try:
            parser.feed(transcript)
        except:
            continue
        
        # Extract context around each player mention
        for player in parser.players:
            # Use the surrounding context (cleaned of XML)
            pattern = f'<person[^>]*player="{re.escape(player["player"])}"[^>]*>.*?</person>'
            for match in re.finditer(pattern, transcript):
                start_pos = max(0, match.start() - 150)
                end_pos = min(len(transcript), match.end() + 150)
                context = transcript[start_pos:end_pos]
                context_clean = clean_xml_tags(context)
                
                if len(context_clean) >= 50:  # Minimum length
                    corpus.append(context_clean)
    
    print(f"Loaded {len(corpus):,} commentary texts from {year_start}-{year_end}")
    return corpus


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words.
    
    Used consistently for:
    - KenLM training text
    - IDF computation
    - Perplexity scoring
    
    Args:
        text: Raw text string
    
    Returns:
        List of lowercase tokens
    """
    text = text.lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 0]


# ============================================================================
# KENLM TRAINING & LOADING
# ============================================================================

def build_kenlm_training_text(corpus: List[str], out_path: str) -> None:
    """
    Write tokenized corpus to text file for KenLM training.
    
    Each line = one tokenized commentary text.
    
    Args:
        corpus: List of raw commentary strings
        out_path: Where to write training text file
    """
    print(f"Building KenLM training text at {out_path}...")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for text in corpus:
            tokens = tokenize(text)
            if tokens:  # Skip empty
                f.write(' '.join(tokens) + '\n')
    
    print(f"Wrote {len(corpus):,} lines to {out_path}")


class BigramLanguageModel:
    """
    Pure Python bigram language model with add-k smoothing.
    No external dependencies required.
    """
    
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.total_tokens = 0
        self.vocab_size = 0
        self.k = 0.01  # Smoothing parameter
    
    def train(self, corpus: List[str]):
        """Train on list of tokenized texts."""
        print("Training bigram language model...")
        
        for i, text in enumerate(corpus):
            tokens = text.split()
            tokens = ['<s>'] + tokens + ['</s>']
            
            for j, token in enumerate(tokens):
                self.unigram_counts[token] += 1
                self.total_tokens += 1
                
                if j > 0:
                    bigram = (tokens[j-1], token)
                    self.bigram_counts[bigram] += 1
            
            if (i + 1) % 1000000 == 0:
                print(f"  Processed {i+1:,} texts...")
        
        self.vocab_size = len(self.unigram_counts)
        print(f"Model trained: {self.vocab_size:,} vocab, {self.total_tokens:,} tokens, {len(self.bigram_counts):,} bigrams")
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'unigram_counts': dict(self.unigram_counts),
                'bigram_counts': {f"{k[0]}|||{k[1]}": v for k, v in self.bigram_counts.items()},
                'total_tokens': self.total_tokens,
                'vocab_size': self.vocab_size,
                'k': self.k
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.unigram_counts = Counter(data['unigram_counts'])
        self.bigram_counts = Counter({tuple(k.split('|||')): v for k, v in data['bigram_counts'].items()})
        self.total_tokens = data['total_tokens']
        self.vocab_size = data['vocab_size']
        self.k = data.get('k', 0.01)
        print(f"Model loaded: {self.vocab_size:,} vocab, {self.total_tokens:,} tokens")
    
    def log_prob(self, text: str) -> float:
        """Compute log probability of text."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        tokens = ['<s>'] + tokens + ['</s>']
        log_prob = 0.0
        
        for i in range(1, len(tokens)):
            prev_token = tokens[i-1]
            curr_token = tokens[i]
            
            # Bigram probability with backoff
            bigram = (prev_token, curr_token)
            bigram_count = self.bigram_counts.get(bigram, 0)
            prev_count = self.unigram_counts.get(prev_token, 0)
            
            if prev_count > 0:
                # Add-k smoothed bigram probability
                prob = (bigram_count + self.k) / (prev_count + self.k * self.vocab_size)
            else:
                # Backoff to unigram
                curr_count = self.unigram_counts.get(curr_token, 0)
                prob = (curr_count + self.k) / (self.total_tokens + self.k * self.vocab_size)
            
            log_prob += math.log(prob + 1e-10)
        
        return log_prob
    
    def perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        tokens = text.lower().split()
        if not tokens:
            return float('nan')
        
        log_prob = self.log_prob(text)
        n_tokens = len(tokens) + 1  # +1 for </s>
        
        # Perplexity = exp(-log_prob / n)
        return math.exp(-log_prob / n_tokens)


def train_bigram_model(corpus: List[str], model_path: str) -> BigramLanguageModel:
    """
    Train a pure Python bigram model on the corpus.
    
    Args:
        corpus: List of tokenized text strings
        model_path: Where to save the model (.pkl)
    
    Returns:
        Trained BigramLanguageModel
    """
    model = BigramLanguageModel()
    model.train(corpus)
    model.save(model_path)
    return model


def load_bigram_model(model_path: str) -> BigramLanguageModel:
    """
    Load a trained bigram model from file.
    
    Args:
        model_path: Path to .pkl model file
    
    Returns:
        BigramLanguageModel object
    """
    print(f"Loading bigram model from {model_path}...")
    model = BigramLanguageModel()
    model.load(model_path)
    return model


# ============================================================================
# PERPLEXITY COMPUTATION
# ============================================================================

def compute_perplexity(model: BigramLanguageModel, text: str) -> float:
    """
    Compute perplexity of text under bigram model.
    
    Lower perplexity = more typical of training corpus
    Higher perplexity = more atypical
    
    Args:
        model: BigramLanguageModel
        text: Text to score
    
    Returns:
        Perplexity (float)
    """
    tokens = tokenize(text)
    if not tokens:
        return float('inf')
    
    # Join tokens with spaces
    tokenized_text = ' '.join(tokens)
    
    return model.perplexity(tokenized_text)


def add_perplexity_column(df: pd.DataFrame, model, text_col: str = "completion_text") -> pd.DataFrame:
    """
    Add perplexity scores to DataFrame.
    
    Args:
        df: DataFrame with LLM completions
        model: kenlm.LanguageModel
        text_col: Column name containing text to score
    
    Returns:
        DataFrame with new 'llm_ppl' column
    """
    print(f"Computing perplexity for {len(df)} completions...")
    
    df['llm_ppl'] = df[text_col].apply(lambda text: compute_perplexity(model, text))
    
    print(f"  Mean perplexity: {df['llm_ppl'].mean():.2f}")
    print(f"  Std perplexity: {df['llm_ppl'].std():.2f}")
    
    return df


# ============================================================================
# IDF & ATYPICALITY COMPUTATION
# ============================================================================

def build_idf(corpus: List[str]) -> Dict[str, float]:
    """
    Build inverse document frequency (IDF) scores.
    
    Each commentary text = one document.
    IDF(token) = log(N_docs / df(token))
    
    Args:
        corpus: List of commentary strings
    
    Returns:
        Dictionary mapping token -> IDF score
    """
    print(f"Building IDF dictionary from {len(corpus):,} documents...")
    
    # Count document frequency
    doc_freq = Counter()
    n_docs = len(corpus)
    
    for text in corpus:
        unique_tokens = set(tokenize(text))
        doc_freq.update(unique_tokens)
    
    # Compute IDF
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log(n_docs / df)
    
    print(f"  Built IDF for {len(idf):,} unique tokens")
    return idf


# Default stopwords (common words to exclude from atypicality)
DEFAULT_STOPWORDS = {
    'the', 'and', 'to', 'a', 'of', 'in', 'is', 'it', 'that', 'for', 'on', 'with',
    'as', 'was', 'at', 'be', 'this', 'by', 'from', 'or', 'an', 'are', 'but', 'not',
    'you', 'all', 'can', 'had', 'her', 'has', 'have', 'he', 'his', 'if', 'its',
    'my', 'no', 'so', 'up', 'out', 'there', 'when', 'who', 'will', 'would', 'they',
    'them', 'their', 'what', 'were', 'been', 'than', 'more', 'now', 'one', 'two',
    's', 't', 're', 've', 'll', 'd', 'm'
}


def compute_atypicality(text: str, idf: Dict[str, float], stopwords: Optional[Set[str]] = None) -> float:
    """
    Compute atypicality score as mean IDF of content words.
    
    Higher score = more unusual/atypical words
    
    Args:
        text: Text to score
        idf: IDF dictionary
        stopwords: Words to exclude (default: DEFAULT_STOPWORDS)
    
    Returns:
        Mean IDF score (float), or NaN if no valid tokens
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    
    tokens = tokenize(text)
    # Filter stopwords and unknown tokens
    content_tokens = [t for t in tokens if t not in stopwords and t in idf]
    
    if not content_tokens:
        return float('nan')
    
    # Mean IDF
    mean_idf = sum(idf[t] for t in content_tokens) / len(content_tokens)
    return mean_idf


def add_atypicality_column(df: pd.DataFrame, idf: Dict[str, float], 
                          text_col: str = "completion_text") -> pd.DataFrame:
    """
    Add atypicality scores to DataFrame.
    
    Args:
        df: DataFrame with LLM completions
        idf: IDF dictionary
        text_col: Column name containing text to score
    
    Returns:
        DataFrame with new 'llm_atypicality' column
    """
    print(f"Computing atypicality for {len(df)} completions...")
    
    df['llm_atypicality'] = df[text_col].apply(
        lambda text: compute_atypicality(text, idf)
    )
    
    print(f"  Mean atypicality: {df['llm_atypicality'].mean():.4f}")
    print(f"  Std atypicality: {df['llm_atypicality'].std():.4f}")
    
    return df


# ============================================================================
# LLM COMPLETIONS LOADING
# ============================================================================

def load_llm_completions(path: str) -> pd.DataFrame:
    """
    Load LLM completions from JSON or CSV file.
    
    Supports two formats:
    
    1. JSON format (from qb_commentary_race.json):
    [
      {
        "name": "Player Name",
        "position": "QB",
        "race": "White" or "Nonwhite",
        "response": "LLM-generated text",
        "prompt_type": "race_included" or similar
      },
      ...
    ]
    
    2. CSV format (from generate_llm_commentary.py):
    Columns: base_id, player_name, position, true_race, condition, 
             model_name, completion_text, ...
    
    Args:
        path: Path to JSON or CSV file
    
    Returns:
        DataFrame with standardized columns:
        - base_id: unique ID
        - model_name: extracted from filename or set to "unknown"
        - condition: mapped from prompt_type
        - true_race: standardized race label
        - position: position
        - completion_text: the LLM response
    """
    print(f"Loading LLM completions from {path}...")
    
    path_lower = path.lower()
    
    # Load based on file extension
    if path_lower.endswith('.csv'):
        # CSV format (from generate_llm_commentary.py)
        df = pd.read_csv(path)
        
        # Standardize column names (CSV already has good names)
        if 'player_name' in df.columns and 'name' not in df.columns:
            df['name'] = df['player_name']
        
        # Ensure base_id exists
        if 'base_id' not in df.columns:
            df['base_id'] = range(len(df))
        
        # Ensure model_name exists
        if 'model_name' not in df.columns:
            df['model_name'] = Path(path).stem
        
        # Ensure true_race is lowercase
        if 'true_race' in df.columns:
            df['true_race'] = df['true_race'].str.lower()
        elif 'race' in df.columns:
            df['true_race'] = df['race'].str.lower()
            
    else:
        # JSON format (legacy qb_commentary_race.json format)
        with open(path, 'r') as f:
            completions = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(completions)
        
        # Standardize columns
        df['base_id'] = range(len(df))
        df['model_name'] = Path(path).stem
        
        # Map prompt_type to condition
        condition_map = {
            'race_included': 'explicit',
            'race_excluded': 'ablated',
            'race_swapped': 'counterfactual'
        }
        if 'prompt_type' in df.columns:
            df['condition'] = df['prompt_type'].map(
                lambda x: condition_map.get(x, 'explicit')
            )
        else:
            df['condition'] = 'explicit'
        
        # Standardize race
        if 'race' in df.columns:
            df['true_race'] = df['race'].str.lower()
        
        # Rename response to completion_text
        if 'response' in df.columns:
            df = df.rename(columns={'response': 'completion_text'})
    
    # Final column selection (keep what we have)
    required_cols = ['base_id', 'model_name', 'condition', 'true_race', 'position', 'completion_text']
    optional_cols = ['name', 'player_name', 'example_year', 'example_team', 'league_level', 'sample_id', 'prompt_text']
    
    available = [c for c in required_cols + optional_cols if c in df.columns]
    df = df[available]
    
    print(f"Loaded {len(df)} completions")
    print(f"  Races: {df['true_race'].value_counts().to_dict()}")
    if 'condition' in df.columns:
        print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    
    return df


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def score_completions(
    commentary_path: str,
    completions_path: str,
    model_train_txt: str,
    model_path: str,
    scored_output_path: str,
    year_start: int = 1990,
    year_end: int = 2019,
    force_retrain: bool = False
) -> pd.DataFrame:
    """
    Main scoring pipeline: trains bigram LM, computes perplexity & atypicality.
    
    Workflow:
    1. Load commentary corpus (1990-2019)
    2. Build training text (if needed)
    3. Train bigram model (if needed)
    4. Load model
    5. Build IDF dictionary
    6. Load LLM completions
    7. Add perplexity scores
    8. Add atypicality scores
    9. Save scored DataFrame
    10. Return scored DataFrame
    
    Args:
        commentary_path: Path to tagged_transcripts.json
        completions_path: Path to LLM completions CSV/JSON
        model_train_txt: Where to save training text
        model_path: Where to save/load bigram model (.pkl)
        scored_output_path: Where to save scored completions
        year_start: Commentary corpus start year
        year_end: Commentary corpus end year
        force_retrain: If True, retrain even if model exists
    
    Returns:
        DataFrame with llm_ppl and llm_atypicality columns added
    """
    print("=" * 80)
    print("BIAS SCORING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load commentary corpus
    corpus = load_commentary_corpus(commentary_path, year_start, year_end)
    
    # Tokenize corpus for model training
    print("Tokenizing corpus for model training...")
    tokenized_corpus = [' '.join(tokenize(text)) for text in corpus]
    
    # Step 2: Train or load bigram model
    # Use .pkl extension for pure Python model
    model_path_pkl = model_path.replace('.bin', '.pkl')
    
    if not os.path.exists(model_path_pkl) or force_retrain:
        model = train_bigram_model(tokenized_corpus, model_path_pkl)
    else:
        print(f"Model already exists: {model_path_pkl}")
        model = load_bigram_model(model_path_pkl)
    
    # Step 3: Build IDF dictionary
    idf = build_idf(corpus)
    
    # Step 5: Load LLM completions
    df = load_llm_completions(completions_path)
    
    # Step 6: Add perplexity scores
    df = add_perplexity_column(df, model, text_col='completion_text')
    
    # Step 7: Add atypicality scores
    df = add_atypicality_column(df, idf, text_col='completion_text')
    
    # Step 8: Save scored DataFrame
    print(f"\nSaving scored completions to {scored_output_path}...")
    df.to_csv(scored_output_path, index=False)
    print(f"Saved {len(df)} scored completions")
    
    print("\n" + "=" * 80)
    print("SCORING COMPLETE")
    print("=" * 80)
    
    return df


# ============================================================================
# GROUPING/SUMMARY HELPERS (for downstream plotting)
# ============================================================================

def summarize_by_group(
    df: pd.DataFrame,
    group_cols: List[str] = None
) -> pd.DataFrame:
    """
    Summarize scored completions by grouping variables.
    
    Useful for downstream plotting scripts to get aggregated stats.
    
    Args:
        df: Scored DataFrame (with llm_ppl and llm_atypicality)
        group_cols: Columns to group by (default: model, race, condition, position)
    
    Returns:
        Summary DataFrame with mean/std for perplexity and atypicality
    """
    if group_cols is None:
        group_cols = ['model_name', 'true_race', 'condition', 'position']
    
    # Filter to columns that exist
    group_cols = [c for c in group_cols if c in df.columns]
    
    if not group_cols:
        print("Warning: No valid grouping columns found")
        return df
    
    summary = df.groupby(group_cols).agg({
        'llm_ppl': ['mean', 'std', 'count'],
        'llm_atypicality': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in summary.columns]
    
    return summary


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python bias_scoring.py <commentary_path> <completions_path> [output_path]")
        print("\nExample:")
        print("  python bias_scoring.py tagged_transcripts.json qb_commentary_race.json scored_qb.csv")
        sys.exit(1)
    
    commentary_path = sys.argv[1]
    completions_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "scored_completions.csv"
    
    # Paths for model artifacts
    model_train_txt = "models/train_corpus.txt"
    model_path = "models/football_commentary.pkl"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Run scoring
    scored_df = score_completions(
        commentary_path=commentary_path,
        completions_path=completions_path,
        model_train_txt=model_train_txt,
        model_path=model_path,
        scored_output_path=output_path
    )
    
    # Print summary
    print("\nSummary by race:")
    print(scored_df.groupby('true_race')[['llm_ppl', 'llm_atypicality']].mean())

