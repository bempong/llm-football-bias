#!/usr/bin/env python3
"""
Analyze word frequencies by race for 2010 football commentary.
Top 50 words for white players vs nonwhite players.
"""

import json
from collections import Counter
import re
from html.parser import HTMLParser

# Common English stopwords and metadata terms to filter out
STOPWORDS = {
    'the', 'and', 'to', 'a', 'of', 'in', 'is', 'it', 'that', 'for', 'on', 'with',
    'as', 'was', 'at', 'be', 'this', 'by', 'from', 'or', 'an', 'are', 'but', 'not',
    'you', 'all', 'can', 'had', 'her', 'has', 'have', 'he', 'his', 'if', 'its',
    'my', 'no', 'so', 'up', 'out', 'there', 'when', 'who', 'will', 'would', 'they',
    'them', 'their', 'what', 'were', 'been', 'than', 'more', 'now', 'one', 'two',
    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'some', 'could', 'about', 'into', 'only', 'other', 'just', 'those', 'these',
    'then', 'do', 'does', 'did', 'very', 'also', 'here', 'how', 'am', 'any', 'we',
    's', 't', 're', 've', 'll', 'd', 'm',
    'person', 'player', 'race', 'position', 'white', 'black', 'unk', 'nonwhite',
    'db', 'rb', 'qb', 'wr', 'lb', 'te', 'ol', 'dl'
}

class PlayerTagParser(HTMLParser):
    """Parser to extract player mentions from tagged transcripts."""
    
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

def extract_year_from_filename(filename):
    """Extract year from filename."""
    match = re.match(r'(\d{4})-', filename)
    return int(match.group(1)) if match else None

def tokenize_text(text):
    """Tokenize text into words, filtering stopwords."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS]

def clean_xml_tags(text):
    """Remove XML tags from text."""
    text = re.sub(r'<person[^>]*>.*?</person>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_context_around_player(transcript, player_name, context_window=100):
    """Extract text context around player mentions."""
    pattern = f'<person[^>]*player="{re.escape(player_name)}"[^>]*>.*?</person>'
    contexts = []
    
    for match in re.finditer(pattern, transcript):
        start_pos = max(0, match.start() - context_window)
        end_pos = min(len(transcript), match.end() + context_window)
        context = transcript[start_pos:end_pos]
        contexts.append(clean_xml_tags(context))
    
    return contexts

def analyze_players_by_race(data_file, race_filter, top_k=50):
    """Analyze top-k words for players of specified race in 2010."""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    word_counter = Counter()
    games_in_2010 = 0
    player_mentions = 0
    unique_players = set()
    
    # Process each game
    for game_file, game_data in data.items():
        if extract_year_from_filename(game_file) != 2010:
            continue
        
        games_in_2010 += 1
        transcript = game_data.get('transcript', '')
        
        parser = PlayerTagParser()
        try:
            parser.feed(transcript)
        except:
            continue
        
        # Process players matching race filter
        for player in parser.players:
            if player['race'] == race_filter:
                player_mentions += 1
                unique_players.add(player['player'])
                
                # Extract and tokenize context
                contexts = extract_context_around_player(transcript, player['player'])
                for context in contexts:
                    word_counter.update(tokenize_text(context))
    
    # Get top k
    top_words = word_counter.most_common(top_k)
    
    return {
        'games': games_in_2010,
        'mentions': player_mentions,
        'unique_players': len(unique_players),
        'total_words': sum(word_counter.values()),
        'unique_words': len(word_counter),
        'top_words': top_words
    }

def print_results(race_label, results):
    """Print analysis results."""
    print(f"\n{'=' * 70}")
    print(f"TOP 50 WORDS FOR {race_label.upper()} PLAYERS IN 2010")
    print(f"{'=' * 70}")
    print(f"\nDataset Statistics:")
    print(f"  - Games in 2010: {results['games']}")
    print(f"  - {race_label.capitalize()} player mentions: {results['mentions']}")
    print(f"  - Unique {race_label} players: {results['unique_players']}")
    print(f"  - Total words analyzed: {results['total_words']:,}")
    print(f"  - Unique words: {results['unique_words']:,}")
    print(f"\nTop 50 Words (excluding common stopwords):")
    print()
    
    for i, (word, count) in enumerate(results['top_words'], 1):
        print(f"  {i:2d}. {word:15s}  -  {count:,} occurrences")

if __name__ == "__main__":
    data_file = "/Users/aj/Documents/courses/cs329r/tagged_transcripts.json"
    
    print("Loading data...")
    
    # Analyze white players
    white_results = analyze_players_by_race(data_file, 'white', top_k=50)
    print_results('white', white_results)
    
    print("\n\n")
    
    # Analyze nonwhite players (Black and other non-white players)
    nonwhite_results = analyze_players_by_race(data_file, 'nonwhite', top_k=50)
    print_results('nonwhite', nonwhite_results)


