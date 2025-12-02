"""
Configuration for Bias Analysis V2
"""

from pathlib import Path

# Directory structure
BASE_DIR = Path(__file__).parent
LEXICONS_DIR = BASE_DIR / "lexicons"
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

# Adjective category minimum frequency (for building lexicon from corpus)
ADJECTIVE_MIN_FREQ = 3  # Word must appear at least this many times to be considered
ADJECTIVE_MIN_DISTINCTIVENESS = 1.5  # Ratio threshold for considering a word distinctive

