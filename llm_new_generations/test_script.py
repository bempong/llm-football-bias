import requests
import json
import time
from players_dataset import quarterbacks
from prompts_dataset import race_included_prompt, race_included_prompt_v2

# API endpoint
url = "http://localhost:11434/api/generate"

results = []

start_time = time.time()

for i, player in enumerate(quarterbacks):
    player_start = time.time()

    # Build prompt dynamically
    prompt = f"{race_included_prompt_v2} Highlight {player['name']}, a {player['race']} {player['position']}."

    payload = {
        "prompt": prompt,
        "model": "llama3",
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        entry = {
            "name": player["name"],
            "position": player["position"],
            "race": player["race"],
            "response": data.get("response", ""),
            "prompt_type": "race_included"
        }
        results.append(entry)

        # Save after every player
        with open("baseline_commentary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        player_end = time.time()
        elapsed_time = player_end - player_start
        print(f"Processed Player {i+1}, {player['name']} in {elapsed_time:.2f} seconds")

    else:
        print(f"Error for {player['name']}: {response.status_code}")

end_time = time.time()
total_elapsed_time = end_time - start_time
print(f"Total time for all players: {total_elapsed_time:.2f} seconds")
