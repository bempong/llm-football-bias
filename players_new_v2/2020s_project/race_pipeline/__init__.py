"""race_pipeline — deterministic NFL-player racial self-identification extraction.

Five stages, each a plain script:
    1. search.py     — Serper queries per player
    2. fetch.py      — URL fetch + trafilatura + spaCy context extraction
    3. extract.py    — LLM verbatim-quote extraction per context
    4. verify.py     — mechanical quote-exists-at-URL check
    5. aggregate.py  — deterministic pre-registered labeling rules

Run pipeline_run.py to execute them in sequence, or each stage directly.
"""
