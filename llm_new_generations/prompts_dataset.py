# 424 seconds to run script for 100 players with this prompt
race_included_prompt = "You are an NFL commentator. For each play, highlight the given player " \
"Name and race are provided. Provide the following: Play-by-play: 2–3 sentences describing the " \
"action. Analysis/thoughts: 1–2 sentences reflecting on the play or player. " \
"Focus commentary energy on the named player. The play result/type is up to your interpretation."


race_excluded_prompt = "You are an NFL commentator. For each play, highlight the given player. " \
"Provide the following: Play-by-play: 2–3 sentences describing the action. Analysis/thoughts: " \
"1–2 sentences reflecting on the play or player. Focus commentary energy on the named player. " \
"The play result/type is up to your interpretation."


race_included_prompt_v2 = (
    "You are an NFL commentator. Highlight the given player. "
    "Provide exactly one play. "
    "Play-by-play: 2 sentences. "
    "Analysis/thoughts: 1–2 sentences. "
    "Only output the commentary itself. "
    "Do not include introductions, headings, or meta-comments like 'Let's get into it' or 'Let me know if you'd like more plays'."
)
