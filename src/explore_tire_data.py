"""
Explore lap data before building tire degradation curves for the Pit-Stop
Simulator (System 2). Need to understand: compound distribution, TyreLife
range, and critically — how much of the data is "clean" racing laps vs.
anomalous laps (pit in/out laps, safety car laps, inaccurate laps) that
would corrupt a degradation curve if not filtered out.

Run locally:
    python src/explore_tire_data.py
"""

import pandas as pd

laps = pd.read_parquet("data/processed/all_laps.parquet")

print(f"Total laps: {len(laps)}")

print(f"\n--- Compound distribution ---")
print(laps["Compound"].value_counts())

print(f"\n--- TyreLife range ---")
print(laps["TyreLife"].describe())

print(f"\n--- LapTime dtype and sample ---")
print(f"Dtype: {laps['LapTime'].dtype}")
print(laps["LapTime"].dropna().head(3))

print(f"\n--- How many laps are 'clean' racing laps? ---")
print(f"IsAccurate True: {laps['IsAccurate'].sum()} / {len(laps)} "
      f"({laps['IsAccurate'].mean():.1%})")

has_pit_out = laps["PitOutTime"].notna()
has_pit_in = laps["PitInTime"].notna()
print(f"Pit out laps (first lap on new tires, often slower): {has_pit_out.sum()}")
print(f"Pit in laps (last lap of a stint, often slower): {has_pit_in.sum()}")

# TrackStatus '1' = green flag/clear track — anything else means SC/VSC/yellow
# affecting pace, which would corrupt a "pure pace" degradation curve
clean_track = laps["TrackStatus"] == "1"
print(f"Laps under fully green/clear track status: {clean_track.sum()} / {len(laps)} "
      f"({clean_track.mean():.1%})")

# Combine all the "clean lap" conditions together
truly_clean = (
    laps["IsAccurate"] &
    ~has_pit_out &
    ~has_pit_in &
    clean_track &
    laps["LapTime"].notna()
)
print(f"\nLaps meeting ALL clean criteria: {truly_clean.sum()} / {len(laps)} "
      f"({truly_clean.mean():.1%})")

print(f"\n--- Sample of clean laps for one compound (SOFT), one race ---")
sample = laps[truly_clean & (laps["Compound"] == "SOFT") &
              (laps["season"] == 2023) & (laps["round"] == 1)]
print(sample[["Driver", "LapNumber", "TyreLife", "LapTime", "Compound"]].head(10))