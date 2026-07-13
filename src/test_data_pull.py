"""
Phase 1: Verify FastF1 

Run: python src/test_data_pull.py

"""

import fastf1
import os

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

print("Pulling 2024 Monza Grand Prix...")
session = fastf1.get_session(2024, "Monza", "R")
session.load()

print("\nSession Info")
print(f"Event: {session.event['EventName']}")
print(f"Date: {session.event['EventDate']}")

print("\n Lap Data (first 5 rows) ")
print(session.laps[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]].head())

print("\n Race Results (top 5) ")
print(session.results[["Position", "Abbreviation", "TeamName", "GridPosition"]].head())

print("\n Weather Data (first 3 rows) ")
print(session.weather_data.head(3))

print("\nIf you see real data above with no errors, your FastF1 pipeline works.")
print("Next: try pulling a 2026 race the same way, to confirm current-season data is accessible too.")