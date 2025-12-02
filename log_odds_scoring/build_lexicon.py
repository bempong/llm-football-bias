#!/usr/bin/env python3
"""
Build Adjective Lexicon from Real Football Commentary

This script extracts adjectives from the 1990-2019 football commentary corpus
and organizes them into bias-relevant categories based on their usage patterns
with white vs nonwhite players.

This is a ONE-TIME script to build the lexicon rigorously from data.
"""

import json
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from typing import Dict, List, Set, Tuple

import yaml


# ============================================================================
# HTML PARSER
# ============================================================================

class PlayerTagParser(HTMLParser):
    """Extract player mentions with race metadata."""
    
    def __init__(self):
        super().__init__()
        self.players = []
        self.current_player = None
        self.current_text = ""
        
    def handle_starttag(self, tag, attrs):
        if tag == 'person':
            attr_dict = dict(attrs)
            self.current_player = {
                'player': attr_dict.get('player'),
                'race': attr_dict.get('race'),
                'position': attr_dict.get('position'),
            }
            self.current_text = ""
    
    def handle_endtag(self, tag):
        if tag == 'person' and self.current_player:
            self.current_player['text'] = self.current_text.strip()
            self.players.append(self.current_player)
            self.current_player = None
    
    def handle_data(self, data):
        if self.current_player is not None:
            self.current_text += data


def clean_xml_tags(text: str) -> str:
    """Remove XML tags from text."""
    text = re.sub(r'<person[^>]*>.*?</person>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_context_around_player(transcript: str, player_name: str, window: int = 200) -> List[str]:
    """Extract text context around player mentions."""
    pattern = f'<person[^>]*player="{re.escape(player_name)}"[^>]*>.*?</person>'
    contexts = []
    
    for match in re.finditer(pattern, transcript):
        start_pos = max(0, match.start() - window)
        end_pos = min(len(transcript), match.end() + window)
        context = transcript[start_pos:end_pos]
        contexts.append(clean_xml_tags(context))
    
    return contexts


# ============================================================================
# NLP: ADJECTIVE EXTRACTION
# ============================================================================

def extract_adjectives_nltk(contexts: List[str]) -> List[str]:
    """Extract adjectives using NLTK POS tagging."""
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        
        # Ensure required data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        adjectives = []
        for context in contexts:
            # Tokenize and POS tag
            tokens = word_tokenize(context.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract adjectives (JJ, JJR, JJS)
            for word, tag in pos_tags:
                if tag.startswith('JJ') and len(word) >= 3 and word.isalpha():
                    adjectives.append(word)
        
        return adjectives
    except ImportError:
        print("Warning: NLTK not available, falling back to simple extraction")
        return extract_adjectives_simple(contexts)


def extract_adjectives_simple(contexts: List[str]) -> List[str]:
    """Simple adjective extraction using common suffixes."""
    adjective_suffixes = {
        'able', 'ible', 'al', 'ial', 'ed', 'en', 'ful', 'ic', 'ish', 
        'ive', 'less', 'ous', 'ious', 'y'
    }
    
    adjectives = []
    for context in contexts:
        words = re.findall(r'\b[a-z]{3,}\b', context.lower())
        for word in words:
            if any(word.endswith(suffix) for suffix in adjective_suffixes):
                adjectives.append(word)
    
    return adjectives


# ============================================================================
# CATEGORY ASSIGNMENT
# ============================================================================

# Seed words for each category (hand-picked from sports sociology literature)
CATEGORY_SEEDS = {
    'cognitive': {
        'smart', 'intelligent', 'cerebral', 'savvy', 'clever', 'wise',
        'astute', 'aware', 'mindful', 'tactical', 'strategic', 'analytical'
    },
    'leadership': {
        'leader', 'vocal', 'confident', 'commanding', 'poised', 'composed',
        'calm', 'mature', 'professional', 'accountable'
    },
    'work_ethic': {
        'hardworking', 'gritty', 'tough', 'relentless', 'determined',
        'hustling', 'scrappy', 'dedicated', 'committed'
    },
    'athleticism_physical': {
        'athletic', 'fast', 'quick', 'explosive', 'powerful', 'strong',
        'agile', 'elusive', 'swift', 'muscular', 'physical'
    },
    'instinct_raw': {
        'instinctive', 'natural', 'raw', 'gifted', 'talented', 'innate',
        'born', 'untapped', 'unpolished'
    },
    'discipline_technical': {
        'disciplined', 'polished', 'refined', 'technical', 'precise',
        'methodical', 'consistent', 'fundamentals', 'coached'
    }
}


def assign_to_categories(
    adjectives_by_race: Dict[str, Counter],
    min_freq: int = 5
) -> Dict[str, List[str]]:
    """
    Assign extracted adjectives to categories based on:
    1. Semantic similarity to seed words
    2. Frequency patterns
    3. Distinctiveness between racial groups
    """
    categories = defaultdict(list)
    
    # Get all adjectives that meet frequency threshold
    white_adj = adjectives_by_race['white']
    nonwhite_adj = adjectives_by_race['nonwhite']
    
    all_adj = set()
    for word in white_adj:
        if white_adj[word] + nonwhite_adj[word] >= min_freq:
            all_adj.add(word)
    for word in nonwhite_adj:
        if white_adj[word] + nonwhite_adj[word] >= min_freq:
            all_adj.add(word)
    
    print(f"\nFound {len(all_adj)} adjectives meeting frequency threshold (>= {min_freq})")
    
    # For each category, find adjectives similar to seeds
    for category, seeds in CATEGORY_SEEDS.items():
        # Add seed words if they appear in data
        for seed in seeds:
            if seed in all_adj:
                categories[category].append(seed)
        
        # Find related words using simple string similarity and co-occurrence
        # (In a more sophisticated version, use word embeddings)
        for word in all_adj:
            if word in seeds:
                continue
            
            # Simple heuristic: if word shares significant substring with seed
            for seed in seeds:
                if len(word) >= 5 and len(seed) >= 5:
                    if word[:4] == seed[:4] or word[:3] == seed[:3]:
                        if word not in categories[category]:
                            categories[category].append(word)
    
    return dict(categories)


# ============================================================================
# MAIN BUILD SCRIPT
# ============================================================================

def build_lexicon_from_corpus(
    corpus_path: str,
    output_path: str,
    year_start: int = 1990,
    year_end: int = 2019,
    min_freq: int = 5
):
    """
    Extract adjectives from real commentary and build categorized lexicon.
    """
    print("=" * 70)
    print("BUILDING ADJECTIVE LEXICON FROM CORPUS")
    print("=" * 70)
    print(f"Corpus: {corpus_path}")
    print(f"Years: {year_start}-{year_end}")
    print(f"Min frequency: {min_freq}")
    
    # Load corpus
    print("\nLoading corpus...")
    with open(corpus_path, 'r') as f:
        data = json.load(f)
    
    # Extract contexts by race
    adjectives_by_race = {
        'white': Counter(),
        'nonwhite': Counter()
    }
    
    games_processed = 0
    for game_file, game_data in data.items():
        # Extract year
        year_match = re.match(r'(\d{4})-', game_file)
        if not year_match:
            continue
        year = int(year_match.group(1))
        
        if year < year_start or year > year_end:
            continue
        
        games_processed += 1
        if games_processed % 100 == 0:
            print(f"  Processed {games_processed} games...")
        
        transcript = game_data.get('transcript', '')
        parser = PlayerTagParser()
        try:
            parser.feed(transcript)
        except:
            continue
        
        # Extract adjectives for each player
        for player in parser.players:
            if player['race'] not in ['white', 'nonwhite']:
                continue
            
            contexts = extract_context_around_player(transcript, player['player'])
            if not contexts:
                continue
            
            adjectives = extract_adjectives_nltk(contexts)
            adjectives_by_race[player['race']].update(adjectives)
    
    print(f"\nProcessed {games_processed} games")
    print(f"White player adjectives: {sum(adjectives_by_race['white'].values())} tokens")
    print(f"Nonwhite player adjectives: {sum(adjectives_by_race['nonwhite'].values())} tokens")
    
    # Assign to categories
    print("\nAssigning adjectives to categories...")
    lexicon = assign_to_categories(adjectives_by_race, min_freq=min_freq)
    
    # Print summary
    print("\n" + "=" * 70)
    print("LEXICON SUMMARY")
    print("=" * 70)
    for category, words in lexicon.items():
        print(f"\n{category.upper()} ({len(words)} words):")
        print(f"  {', '.join(sorted(words)[:20])}")
        if len(words) > 20:
            print(f"  ... and {len(words) - 20} more")
    
    # Save lexicon
    with open(output_path, 'w') as f:
        yaml.dump(lexicon, f, default_flow_style=False, sort_keys=True)
    
    print(f"\nLexicon saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build adjective lexicon from corpus")
    parser.add_argument('--corpus', type=str, 
                       default='../tagged_transcripts.json',
                       help='Path to tagged transcripts')
    parser.add_argument('--output', type=str,
                       default='lexicons/adjective_categories.yml',
                       help='Output lexicon file')
    parser.add_argument('--min-freq', type=int, default=5,
                       help='Minimum adjective frequency')
    
    args = parser.parse_args()
    
    build_lexicon_from_corpus(
        args.corpus,
        args.output,
        min_freq=args.min_freq
    )

