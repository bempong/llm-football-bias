"""
Configuration for Log-Odds Bias Analysis
"""

from pathlib import Path

# Directory structure
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Race group labels (must match the CSV column values)
RACE_GROUP_A = "white"
RACE_GROUP_B = "nonwhite"

# Log-odds scoring parameters
LOG_ODDS_ALPHA_0 = 0.01  # Dirichlet prior strength
LOG_ODDS_MIN_COUNT = 5   # Minimum word frequency to include in analysis

# Tokenization
MIN_TOKEN_LENGTH = 2
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
    # Football-specific stopwords that don't carry bias signal
    'player', 'game', 'play', 'yard', 'yards', 'ball', 'field', 'team', 'down'
}

# NFL team name tokens to filter (mascots, cities, nicknames)
NFL_TEAM_TOKENS = {
    # Mascots / team names
    'niners', 'bears', 'bengals', 'bills', 'broncos', 'browns', 'buccaneers',
    'bucs', 'cardinals', 'chargers', 'chiefs', 'colts', 'commanders', 'cowboys',
    'dolphins', 'eagles', 'falcons', 'giants', 'jaguars', 'jags', 'jets',
    'lions', 'packers', 'panthers', 'patriots', 'pats', 'raiders', 'rams',
    'ravens', 'redskins', 'saints', 'seahawks', 'hawks', 'steelers', 'texans',
    'titans', 'vikings',
    # Cities / regions
    'arizona', 'atlanta', 'baltimore', 'buffalo', 'carolina', 'chicago',
    'cincinnati', 'cleveland', 'dallas', 'denver', 'detroit', 'francisco',
    'green', 'bay', 'houston', 'indianapolis', 'indy', 'jacksonville',
    'kansas', 'las', 'vegas', 'angeles', 'los', 'miami', 'minnesota',
    'england', 'orleans', 'york', 'oakland', 'philadelphia', 'philly',
    'pittsburgh', 'diego', 'seattle', 'tampa', 'tennessee', 'washington',
    'nfl',
}


def build_player_name_tokens(df, name_col='player_name'):
    """Extract all individual name tokens from the player_name column."""
    tokens = set()
    for name in df[name_col].dropna().unique():
        for part in str(name).lower().replace('_', ' ').split():
            part = part.strip()
            if len(part) >= 2:
                tokens.add(part)
    return tokens

