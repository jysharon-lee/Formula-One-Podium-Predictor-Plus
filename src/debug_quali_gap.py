"""
Debug: inspect raw Q1/Q2/Q3 data for a single race to figure out why
calculate_team_pace_trend() is producing an unrealistic 7.5 second gap.

Run locally:
    python src/debug_quali_gap.py
"""

import pandas as pd
from feature_engineering import load_all_qualifying

quali = load_all_qualifying()

# Look at one specific race in detail — 2023 round 9 (the race right before
# our target of round 10), should show Red Bull near the top
race = quali[(quali["season"] == 2023) & (quali["round"] == 9)]

print("Columns available:", list(quali.columns))
print(f"\nDtype of Q1/Q2/Q3 columns:")
for col in ["Q1", "Q2", "Q3"]:
    if col in quali.columns:
        print(f"  {col}: {quali[col].dtype}")

print(f"\n--- Raw qualifying results, 2023 round 9 ---")
cols_to_show = ["TeamName", "Abbreviation", "Q1", "Q2", "Q3"]
cols_to_show = [c for c in cols_to_show if c in race.columns]
print(race[cols_to_show].sort_values("TeamName").to_string())

# Now specifically check Red Bull's rows
rb = race[race["TeamName"] == "Red Bull Racing"]
print(f"\n--- Red Bull Racing rows for 2023 round 9 ---")
print(rb[cols_to_show].to_string())

# Check how many rows across the WHOLE dataset have all-NaT Q2/Q3
# (would explain why the fallback to Q1 is happening more than expected)
print(f"\n--- NaT counts across entire qualifying dataset ---")
for col in ["Q1", "Q2", "Q3"]:
    if col in quali.columns:
        nat_count = quali[col].isna().sum()
        print(f"  {col}: {nat_count} / {len(quali)} rows are NaT ({nat_count/len(quali):.1%})")