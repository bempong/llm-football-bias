import pandas as pd

df = pd.read_csv("players_new_v2/datasets/merged_1960_2026.csv")

dupes = df[df.duplicated(subset="player_name", keep=False)].sort_values("player_name")

if dupes.empty:
    print("No duplicate names found.")
else:
    print(f"{dupes['player_name'].nunique()} names with duplicates ({len(dupes)} total rows):\n")
    print(dupes.to_string(index=True))
