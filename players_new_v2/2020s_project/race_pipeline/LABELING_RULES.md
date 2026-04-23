# Pre-Registered Labeling Rules

This document specifies the deterministic rules used by `aggregate.py` to
convert verified, attribution-confident quotes into a final per-player label.
It is committed *before* the full run so that labeling decisions cannot be
retroactively tuned.

## 1. Inputs considered

For each player we consider every row in `data/verifications.csv` that meets
all of the following:

- `quote_found_at_url == True` (the exact quote appears verbatim at the
  source URL after unicode normalization)
- the corresponding extraction in `data/extractions.jsonl` has
  `attribution_confident == True`
- the source fetch is NOT denylisted (i.e. `source_tier in {1, 2, 3}`)

Every other row is discarded before rule application. No exceptions.

## 2. Source tiers

Defined in `config.py`. Summary:

- **Tier 1** — player-authored / player-controlled platforms
  (The Players' Tribune, Andscape, The Undefeated, Uninterrupted).
- **Tier 2** — major sports journalism outlets (ESPN, SI, NFL.com, The
  Athletic, Bleacher Report, SB Nation, GQ, NYT, WaPo, LA Times, Chicago
  Tribune, Boston Globe, Houston Chronicle, Dallas News, SF Chronicle,
  YouTube podcast transcripts) and `.edu` domains (student newspapers).
- **Tier 3** — everything else that is not denylisted.

## 3. Footprint classification

A player is classified as **low-footprint** iff BOTH hold:

- estimated career games `< 10` (approximated from `years_active` in
  `data/players.csv` at 17 games per listed season), AND
- non-denylisted OK-status fetched pages `< 5`.

Otherwise they are **standard-footprint**.

## 4. Per-group passing criterion

Apply these per candidate group G ∈ {black, latino, asian_pi, other_multiracial}:

### 4.1 Standard-footprint rule (applied to standard-footprint players)

Group G passes if EITHER:

- ≥ 1 verified, attribution-confident Tier 1 quote for G, OR
- ≥ 2 verified, attribution-confident Tier 2 quotes for G from ≥ 2 distinct
  domains.

### 4.2 Relaxed-footprint rule (applied to low-footprint players)

Group G passes if EITHER of the above holds, OR:

- ≥ 1 verified, attribution-confident Tier 1 OR Tier 2 quote for G (i.e.
  the two-distinct-domain requirement is waived), OR
- ≥ 2 verified, attribution-confident Tier 3 quotes for G from ≥ 2 distinct
  domains AND no passing quote exists for a conflicting group.

The relaxed rule exists because low-footprint players (practice-squad
churners, short-career UDFAs) will rarely clear the standard bar even when
self-identified evidence exists — biasing the dataset toward stars.

## 5. Multi-group resolution (pre-registered)

Let P = {G : G passes the per-group criterion for this player}.

- If `P` is empty → `final_label = None`, `rule_applied = insufficient_evidence`.
- If `|P| = 1` and the group is monoracial → label with that group,
  `rule_applied = standard` or `relaxed` depending on which criterion fired.
- If `|P| = 1` and the group is `other_multiracial` → label as
  `other_multiracial`, `rule_applied = multiracial`.
- If `|P| ≥ 2` and exactly one monoracial group passes → label with that
  monoracial group (monoracial self-ID preferred over coexistent multiracial
  self-ID).
- If `|P| ≥ 2` and multiple monoracial groups pass → most-recent-source-wins
  tiebreaker: the monoracial group whose latest supporting fetch has the
  greatest `fetched_at` timestamp is chosen.
- If all passing groups are `other_multiracial` → label as `other_multiracial`.

## 6. What this rule set excludes by design

- Any label not traceable to a verbatim quoted span with a live URL.
- Any label whose extractor flagged `attribution_confident == False`.
- Any label whose quote failed the mechanical substring check in
  `verify.py` (catches LLM hallucinations).
- Any label from a denylisted domain (Wikipedia, Reddit, social media,
  gambling sites, AI content farms).

## 7. Outputs

- `data/labels.csv` — one row per player in `data/players.csv`, with
  `final_label`, per-tier evidence counts, `rule_applied`, and
  `has_low_footprint_flag`.
- `data/evidence.csv` — one row per quote that supports a player's final
  label (quotes contributing to a non-labeled player are logged in
  `verifications.csv` but not duplicated here).

## 8. Change control

Any modification to tier membership, query set, passing criteria, or
resolution logic must be applied **before** the full 6,465-player run
begins. Changes discovered during the pilot should be appended here with a
dated changelog block and the rationale, and the pilot re-run on the new
rules.

### Changelog

- 2026-04-17 — initial version, committed alongside pipeline v1.
- 2026-04-22 — v1.1 search design change. `config.QUERIES` (5 race-keyed
  groups, ~99 queries/player) replaced with `config.DISCOVERY_QUERIES`
  (18 content-neutral queries/player). Retrieval now targets player
  biography/profile/podcast/college coverage only; no race-specific
  phrases appear in any Stage-1 query. Race is extracted exclusively at
  the LLM stage from whatever Google surfaces. Rationale: (a) quoted
  race-phrase queries are brittle — you cannot enumerate every phrasing
  of self-identification; the LLM reading an entire article catches them
  all; (b) the prior design was ~5× more expensive per player and did
  not materially improve recall for players covered by long-form
  journalism; (c) content-neutral retrieval removes any concern that
  race-conditioned queries biased the retrieval pool per group.
  Downstream labeling rules (tiers, footprint, per-group passing
  criteria, multi-group resolution) are unchanged. The `query_group`
  column in `searches.jsonl` is retained for back-compat but is now
  uniformly `"discovery"`.
