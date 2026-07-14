"""
Build a season + round -> circuit name lookup table.

Run locally: python src/build_circuit_lookup.py
"""

import fastf1
import pandas as pd
import os

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

os.makedirs("data/processed", exist_ok=True)

SEASONS = range(2018, 2025)

rows = []
for year in SEASONS:
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule["EventFormat"] != "testing"]
    for _, event in races.iterrows():
        rows.append({
            "season": year,
            "round": event["RoundNumber"],
            "circuit_name": event["EventName"],  # matches circuit_metadata.py naming where possible
            "location": event["Location"],
            "country": event["Country"],
        })

lookup = pd.DataFrame(rows)
lookup.to_csv("data/processed/circuit_lookup.csv", index=False)

print(f"Saved {len(lookup)} season/round -> circuit mappings to data/processed/circuit_lookup.csv")
print("\nSample:")
print(lookup.head(10).to_string())