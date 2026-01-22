#!/usr/bin/env python3
"""
Analyze most distinctive words by race for 2000-2019 football commentary.
Focus on revealing characteristics that show different treatment.
"""

import json
from collections import Counter
import re
from html.parser import HTMLParser

# Revealing descriptor words to look for based on previous research
REVEALING_DESCRIPTORS = {
    # Cognitive
    'smart', 'intelligent', 'cerebral', 'savvy', 'knows', 'understands', 
    'thinks', 'reads', 'student', 'awareness', 'decision', 'decisions',
    'tactical', 'strategic', 'technical', 'calculated', 'clever', 'wise',
    'knowledge', 'mental', 'studying', 'preparation', 'prepared', 'heady',
    'brainy', 'genius', 'bright', 'sharp', 'thinking',
    
    # Physical
    'athletic', 'fast', 'quick', 'speed', 'explosive', 'powerful', 'strong',
    'agile', 'natural', 'gifted', 'raw', 'physical', 'physique', 'muscular',
    'freak', 'beast', 'specimen', 'instinctive', 'innate', 'born',
    'talented', 'ability', 'abilities', 'lightning', 'blazing', 'burst',
    
    # Work Ethic
    'hard', 'worker', 'works', 'effort', 'scrappy', 'gritty', 'tough',
    'toughness', 'determination', 'determined', 'relentless', 'hustle',
    'hustles', 'grind', 'grinder', 'workman', 'workhorse', 'lunch',
    'overachiever', 'overachieving', 'blue-collar', 'crafty', 'craftsman',
    
    # Leadership/Poise
    'leader', 'leadership', 'captain', 'commander', 'general', 'vocal',
    'confident', 'confidence', 'composed', 'composure', 'poise', 'poised',
    'mature', 'maturity', 'professional', 'calm', 'cool', 'collected',
    
    # Style/Character
    'flashy', 'flamboyant', 'showboat', 'cocky', 'arrogant', 'humble',
    'quiet', 'loud', 'aggressive', 'passive', 'emotional', 'fiery',
    'hot-headed', 'temperamental', 'brash', 'swagger', 'swag',
    
    # Natural talent vs learned skill
    'natural', 'naturally', 'instinct', 'instinctive', 'instincts',
    'learned', 'trained', 'coaching', 'coached', 'developed',
    'gifted', 'talent', 'talented',
    
    # Body/Physicality
    'body', 'build', 'frame', 'size', 'strength', 'muscle', 'muscles',
    
    # Intelligence/Skill framing
    'crafty', 'savvy', 'wily', 'clever', 'sneaky', 'slick', 'slippery'
}

# Common stopwords
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

def clean_xml_tags(text):
    """Remove XML tags from text."""
    text = re.sub(r'<person[^>]*>.*?</person>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_context_around_player(transcript, player_name, context_window=150):
    """Extract text context around player mentions."""
    pattern = f'<person[^>]*player="{re.escape(player_name)}"[^>]*>.*?</person>'
    contexts = []
    
    for match in re.finditer(pattern, transcript):
        start_pos = max(0, match.start() - context_window)
        end_pos = min(len(transcript), match.end() + context_window)
        context = transcript[start_pos:end_pos]
        contexts.append(clean_xml_tags(context))
    
    return contexts

def analyze_by_race(data_file, race_filter, year_range):
    """Analyze descriptor usage for players of specified race."""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    descriptor_counter = Counter()
    games_analyzed = 0
    player_mentions = 0
    unique_players = set()
    
    # Process each game
    for game_file, game_data in data.items():
        year = extract_year_from_filename(game_file)
        
        # Filter by year range
        if not year or year < year_range[0] or year > year_range[1]:
            continue
        
        games_analyzed += 1
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
                
                # Extract context
                contexts = extract_context_around_player(transcript, player['player'])
                
                for context in contexts:
                    # Tokenize
                    words = re.findall(r'\b[a-z]+\b', context.lower())
                    
                    # Only count revealing descriptors
                    for word in words:
                        if word in REVEALING_DESCRIPTORS:
                            descriptor_counter[word] += 1
    
    return {
        'games': games_analyzed,
        'mentions': player_mentions,
        'unique_players': len(unique_players),
        'descriptors': descriptor_counter
    }

def calculate_distinctive_words(white_results, nonwhite_results, min_count=10):
    """Calculate which words are most distinctive for each race."""
    
    white_mentions = white_results['mentions']
    nonwhite_mentions = nonwhite_results['mentions']
    
    # Get all descriptor words
    all_words = set(white_results['descriptors'].keys()) | set(nonwhite_results['descriptors'].keys())
    
    white_distinctive = []
    nonwhite_distinctive = []
    
    for word in all_words:
        white_count = white_results['descriptors'][word]
        nonwhite_count = nonwhite_results['descriptors'][word]
        
        # Calculate rates per 1000 mentions
        white_rate = (white_count / white_mentions * 1000) if white_mentions > 0 else 0
        nonwhite_rate = (nonwhite_count / nonwhite_mentions * 1000) if nonwhite_mentions > 0 else 0
        
        # Skip if total usage is too low
        if white_count + nonwhite_count < min_count:
            continue
        
        # Calculate ratio
        if white_count >= min_count:
            if nonwhite_count == 0:
                ratio = float('inf')
            else:
                ratio = white_rate / (nonwhite_rate if nonwhite_rate > 0 else 0.001)
            
            if ratio > 1.3:  # At least 30% more common
                white_distinctive.append((word, white_count, nonwhite_count, white_rate, nonwhite_rate, ratio))
        
        if nonwhite_count >= min_count:
            if white_count == 0:
                ratio = float('inf')
            else:
                ratio = nonwhite_rate / (white_rate if white_rate > 0 else 0.001)
            
            if ratio > 1.3:  # At least 30% more common
                nonwhite_distinctive.append((word, white_count, nonwhite_count, white_rate, nonwhite_rate, ratio))
    
    # Sort by ratio (most distinctive first)
    white_distinctive.sort(key=lambda x: x[5], reverse=True)
    nonwhite_distinctive.sort(key=lambda x: x[5], reverse=True)
    
    return white_distinctive, nonwhite_distinctive

def print_results(year_range, white_results, nonwhite_results, white_distinctive, nonwhite_distinctive):
    """Print analysis results."""
    
    print(f"\n{'=' * 80}")
    print(f"MOST DISTINCTIVE DESCRIPTIVE WORDS BY RACE ({year_range[0]}-{year_range[1]})")
    print('=' * 80)
    
    print(f"\nDataset Overview:")
    print(f"  Games analyzed: {white_results['games']}")
    print(f"  White player mentions: {white_results['mentions']:,}")
    print(f"  Nonwhite player mentions: {nonwhite_results['mentions']:,}")
    print(f"  Ratio (nonwhite:white): {nonwhite_results['mentions']/white_results['mentions']:.2f}:1")
    
    # Top 20 for white players
    print(f"\n{'=' * 80}")
    print(f"TOP 20 WORDS MORE ASSOCIATED WITH WHITE PLAYERS")
    print('=' * 80)
    print(f"{'Rank':<6}{'Word':<18}{'White':<10}{'Nonwhite':<10}{'W Rate':<12}{'NW Rate':<12}{'Ratio'}")
    print('-' * 80)
    
    for i, (word, w_count, nw_count, w_rate, nw_rate, ratio) in enumerate(white_distinctive[:20], 1):
        ratio_str = f"{ratio:.2f}:1" if ratio != float('inf') else "only W"
        print(f"{i:<6}{word:<18}{w_count:<10}{nw_count:<10}{w_rate:<12.2f}{nw_rate:<12.2f}{ratio_str}")
    
    # Top 20 for nonwhite players
    print(f"\n{'=' * 80}")
    print(f"TOP 20 WORDS MORE ASSOCIATED WITH NONWHITE PLAYERS")
    print('=' * 80)
    print(f"{'Rank':<6}{'Word':<18}{'White':<10}{'Nonwhite':<10}{'W Rate':<12}{'NW Rate':<12}{'Ratio'}")
    print('-' * 80)
    
    for i, (word, w_count, nw_count, w_rate, nw_rate, ratio) in enumerate(nonwhite_distinctive[:20], 1):
        ratio_str = f"{ratio:.2f}:1" if ratio != float('inf') else "only NW"
        print(f"{i:<6}{word:<18}{w_count:<10}{nw_count:<10}{w_rate:<12.2f}{nw_rate:<12.2f}{ratio_str}")
    
    # Summary interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print('=' * 80)
    print("\nRates shown are per 1,000 player mentions.")
    print("Ratio shows how much more common the word is for one race vs the other.")
    print("\nNote: This analysis reveals systematic differences in how commentators")
    print("describe players of different races, potentially reflecting unconscious bias.")

if __name__ == "__main__":
    data_file = "/Users/aj/Documents/courses/cs329r/tagged_transcripts.json"
    year_range = (1990, 2019)
    
    print("Loading data and analyzing...")
    print(f"Analyzing years {year_range[0]}-{year_range[1]}...")
    
    # Analyze both races
    white_results = analyze_by_race(data_file, 'white', year_range)
    nonwhite_results = analyze_by_race(data_file, 'nonwhite', year_range)
    
    # Calculate distinctive words
    white_distinctive, nonwhite_distinctive = calculate_distinctive_words(
        white_results, nonwhite_results, min_count=10
    )
    
    # Print results
    print_results(year_range, white_results, nonwhite_results, 
                 white_distinctive, nonwhite_distinctive)

