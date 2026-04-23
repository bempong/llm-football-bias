"""LLM prompt for extract.py. Extraction only — no labeling."""

from __future__ import annotations

EXTRACTION_PROMPT = '''You are extracting verbatim self-identification quotes from sports journalism and player-authored content. You will be given a text window that mentions an NFL player. Return only first-person statements by the named player about their own racial or ethnic identity.

Player: {player_name}
Mention type: {mention_type}  # "named" or "pronoun_resolved"

Text window:
"""
{context_text}
"""

Instructions:
1. Return ONLY statements where {player_name} is speaking in the first person ("I am...", "my family is...", "as a ___ man/woman", etc.) about their own racial or ethnic identity.
2. Attribution: the speaker must be {player_name}, either by direct name attribution, or by pronoun where context unambiguously refers to {player_name} and no other person. If attribution is ambiguous, set attribution_confident=false.
3. Do NOT return statements by journalists, teammates, family members, or other people.
4. Do NOT return metaphorical, aspirational, or ambiguous statements. Only direct identity claims.
5. Do NOT paraphrase. Return the exact verbatim quote as it appears in the text.
6. If no qualifying statement exists, return an empty list.

Output JSON only, no other text:
{{
  "extractions": [
    {{
      "exact_quote": "<verbatim span from the text>",
      "surrounding_sentence": "<the full sentence containing the quote>",
      "first_person": true,
      "attribution_confident": true,
      "candidate_group": "black" | "latino" | "asian_pi" | "other_multiracial"
    }}
  ]
}}

Qualifying examples:
- "I'm a proud Samoan" -> asian_pi
- "As a Black man in America, I've..." -> black
- "My mother is from Mexico and my father is Puerto Rican" -> latino

Non-qualifying examples (return empty list):
- "[Reporter:] Smith, who is Black, said..." (journalist attribution)
- "I love the Black community" (not self-identification)
- "He identifies as biracial" (third-person)
- "I want to represent my people" (no identity stated)
'''


def render_extraction_prompt(player_name: str, mention_type: str, context_text: str) -> str:
    return EXTRACTION_PROMPT.format(
        player_name=player_name,
        mention_type=mention_type,
        context_text=context_text,
    )
