"""
Lexicon-based cognitive vs. physical classifier for sports commentary.

Usage:
    from cog_phys_lexicon import score_text, score_mentions
    score_text("He has incredible instincts and reads the defense.")
    # -> {'cognitive': 2, 'physical': 0, 'work_ethic': 0, 'n_tokens': 8, ...}

Designed to be auditable: every match is logged so you can inspect what
drove a score. Swap in your own lexicon by editing LEXICONS below.
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. Lexicons. Grounded in Merullo et al. 2019 and the prior lit.
#    Kept as *lemmas* — we lemmatize the text before matching, so you do
#    NOT need to list every inflection here. "think" matches thinks/thinking.
# ---------------------------------------------------------------------------

LEXICONS: dict[str, set[str]] = {
    "cognitive": {
        # intelligence
        "smart", "intelligent", "intelligence", "savvy", "cerebral", "heady",
        "instinct", "instinctive", "iq", "sharp", "bright", "brilliant",
        "genius", "thinker", "studious", "mind", "brain", "brainy",
        # decision-making / processing
        "poise", "poised", "composure", "composed", "calm", "patient",
        "patience", "calculating", "disciplined", "discipline", "decisive",
        "read", "anticipate", "anticipation", "recognize", "recognition",
        "process", "processing", "aware", "awareness", "vision", "understand",
        "understanding", "know", "knowledge", "diagnose", "decipher",
        # leadership — consider splitting into its own category if you want
        "leader", "leadership", "captain", "veteran", "poised", "clutch",
    },
    "physical": {
        # athleticism
        "athletic", "athleticism", "athlete", "explosive", "explosiveness",
        "fast", "speed", "speedy", "quick", "quickness", "agile", "agility",
        "powerful", "power", "strong", "strength", "burst", "bursty",
        "twitchy", "elusive", "nimble", "fluid", "springy", "bouncy",
        # body / gifts — the ones most loaded in the bias literature
        "gifted", "natural", "raw", "freak", "freakish", "specimen",
        "physical", "physicality", "size", "length", "wingspan", "frame",
        "build", "stature",
        # visible traits
        "big", "tall", "huge", "massive", "imposing", "towering", "chiseled",
        "muscular", "lean", "thick",
    },
    # Report this separately. "Hard-working white guy / naturally gifted black
    # guy" is itself the stereotype under study, so don't fold it into either.
    "work_ethic": {
        "hardworking", "hard-working", "work", "worker", "dedication",
        "dedicated", "determined", "determination", "grit", "gritty",
        "grinder", "grind", "prepared", "preparation", "professional",
        "disciplined", "film", "study", "scrappy", "tough", "toughness",
        "effort", "motor", "relentless",
    },
}

# Negation cues within ±N tokens flip/zero a match.
NEGATION_CUES = {"not", "no", "never", "lacks", "lack", "without", "isn't",
                 "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
                 "can't", "cannot", "hardly", "barely"}
NEG_WINDOW = 3


# ---------------------------------------------------------------------------
# 2. Tokenization + lemmatization. spaCy if available, regex fallback if not.
# ---------------------------------------------------------------------------

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    def _tokens_and_lemmas(text: str) -> list[tuple[str, str]]:
        doc = _NLP(text.lower())
        return [(t.text, t.lemma_) for t in doc if not t.is_space]
except Exception:
    _NLP = None
    _WORD_RE = re.compile(r"[a-z][a-z\-']*")
    # Crude fallback lemmatizer — strip common suffixes. Good enough for a
    # first pass; swap in spaCy for any real run.
    def _crude_lemma(w: str) -> str:
        for suf in ("ingly", "edly", "ness", "ment", "ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[: -len(suf)]
        return w
    def _tokens_and_lemmas(text: str) -> list[tuple[str, str]]:
        toks = _WORD_RE.findall(text.lower())
        return [(t, _crude_lemma(t)) for t in toks]


# ---------------------------------------------------------------------------
# 3. Scoring.
# ---------------------------------------------------------------------------

@dataclass
class Score:
    cognitive: int = 0
    physical: int = 0
    work_ethic: int = 0
    n_tokens: int = 0
    hits: list[tuple[str, str, int]] = field(default_factory=list)  # (category, lemma, position)

    def normalized(self) -> dict[str, float]:
        n = max(self.n_tokens, 1)
        return {
            "cog_rate": self.cognitive / n,
            "phys_rate": self.physical / n,
            "work_rate": self.work_ethic / n,
            # signed axis score in [-1, 1]-ish; handy for plotting
            "cog_minus_phys": (self.cognitive - self.physical) / n,
        }


def _is_negated(tokens: list[tuple[str, str]], i: int) -> bool:
    lo = max(0, i - NEG_WINDOW)
    return any(tokens[j][0] in NEGATION_CUES for j in range(lo, i))


def score_text(text: str, lexicons: dict[str, set[str]] = LEXICONS) -> Score:
    tokens = _tokens_and_lemmas(text)
    s = Score(n_tokens=len(tokens))
    for i, (_surface, lemma) in enumerate(tokens):
        for category, vocab in lexicons.items():
            if lemma in vocab:
                if _is_negated(tokens, i):
                    continue  # skip negated praise/criticism
                setattr(s, category, getattr(s, category) + 1)
                s.hits.append((category, lemma, i))
    return s


# ---------------------------------------------------------------------------
# 4. Player-mention windowing. Score the sentence containing each mention
#    (plus the next sentence). Attribute the score to that player.
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    if _NLP is not None:
        return [s.text for s in _NLP(text).sents]
    return _SENT_SPLIT.split(text.strip())


def score_mentions(
    text: str,
    player_aliases: dict[str, list[str]],
    context_window: int = 2,
) -> dict[str, list[Score]]:
    """
    For each player, return a list of Scores — one per mention in `text`.

    player_aliases: {"Patrick Mahomes": ["Mahomes", "Patrick Mahomes", "Pat"]}
    context_window: how many sentences (including the one containing the
                    mention) to score as the mention's context.
    """
    sents = split_sentences(text)
    out: dict[str, list[Score]] = {p: [] for p in player_aliases}
    lower_sents = [s.lower() for s in sents]
    for player, aliases in player_aliases.items():
        patterns = [re.compile(rf"\b{re.escape(a.lower())}\b") for a in aliases]
        for idx, sent_low in enumerate(lower_sents):
            if any(p.search(sent_low) for p in patterns):
                window = " ".join(sents[idx : idx + context_window])
                out[player].append(score_text(window))
    return out


def aggregate_by_group(
    per_player: dict[str, Score],
    player_to_group: dict[str, str],
) -> dict[str, dict]:
    """
    Roll per-player scores up to a group (e.g., race, position).
    Returns totals and rates per group.
    """
    bucket: dict[str, Score] = {}
    for player, score in per_player.items():
        g = player_to_group.get(player)
        if g is None:
            continue
        b = bucket.setdefault(g, Score())
        b.cognitive += score.cognitive
        b.physical += score.physical
        b.work_ethic += score.work_ethic
        b.n_tokens += score.n_tokens
    return {g: {**s.__dict__, **s.normalized()} for g, s in bucket.items()}


# ---------------------------------------------------------------------------
# 5. Demo.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Mahomes shows incredible poise in the pocket. He reads the defense "
        "and anticipates the blitz. Hill is just a freak athlete — explosive "
        "speed, elusive in space. Kelce is a smart veteran who studies film, "
        "but he's not the most athletic tight end in the league."
    )
    aliases = {
        "Mahomes": ["Mahomes"],
        "Hill":    ["Hill"],
        "Kelce":   ["Kelce"],
    }
    groups = {"Mahomes": "black", "Hill": "black", "Kelce": "white"}

    mentions = score_mentions(sample, aliases)
    per_player = {}
    for p, scores in mentions.items():
        total = Score()
        for s in scores:
            total.cognitive += s.cognitive
            total.physical += s.physical
            total.work_ethic += s.work_ethic
            total.n_tokens += s.n_tokens
            total.hits.extend(s.hits)
        per_player[p] = total
        print(f"{p}: cog={total.cognitive} phys={total.physical} "
              f"work={total.work_ethic} hits={total.hits}")

    print("\nBy group:")
    for g, stats in aggregate_by_group(per_player, groups).items():
        print(f"  {g}: cog_rate={stats['cog_rate']:.3f} "
              f"phys_rate={stats['phys_rate']:.3f}")