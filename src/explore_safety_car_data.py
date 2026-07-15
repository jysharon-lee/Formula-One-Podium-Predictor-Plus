"""
Explore how safety car / VSC events are encoded in the laps data, before
building the Safety Car Model. 

Run locally: python src/explore_safety_car_data.py
"""

import pandas as pd

laps = pd.read_parquet("data/processed/all_laps.parquet")

print("Columns available:", list(laps.columns))
print(f"\nTrackStatus dtype: {laps['TrackStatus'].dtype}")
print(f"\nUnique TrackStatus values (sample):")
print(laps["TrackStatus"].value_counts().head(20))

print(f"\n--- FastF1's TrackStatus code reference ---")
print("1 = Track clear, 2 = Yellow flag, 4 = Safety Car, 5 = Red flag,")
print("6 = Virtual Safety Car, 7 = VSC ending")
print("A lap's TrackStatus can contain MULTIPLE codes if status changed mid-lap")
print("(e.g. '14' means both code 1 and code 4 occurred during that lap)")

# Check one specific race we know had a safety car 
sample_race = laps[(laps["season"] == 2023) & (laps["round"] == 17)]
if len(sample_race) > 0:
    print(f"\n--- Sample: 2023 round 17 TrackStatus values ---")
    print(sample_race["TrackStatus"].value_counts())