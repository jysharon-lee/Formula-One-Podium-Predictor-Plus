"""
Quick sanity check before combining everything

Confirms: how many races got pulled per season, and flags any season that
looks suspiciously short (missing races due to a failed pull didn't
notice at the time).

Run locally: python src/check_pulled_data.py

"""

import glob
import re
from collections import defaultdict

files = glob.glob("data/raw/laps/*.parquet")

counts = defaultdict(int)
for f in files:
    match = re.search(r"(\d{4})_(\d{2})\.parquet", f)
    if match:
        season = match.group(1)
        counts[season] += 1

print("Races pulled per season:")
for season in sorted(counts):
    print(f"  {season}: {counts[season]} races")

print(f"\nTotal races pulled: {sum(counts.values())}")
print("\nExpected roughly: 21 (2018), 21 (2019), 17 (2020, COVID-shortened),")
print("22 (2021), 22 (2022), 22 (2023), 24 (2024)")
print("\nIf any season looks noticeably short vs. these expectations,")
print("rerun pull_season_data.py — it will skip completed races and only")
print("retry the missing ones.")