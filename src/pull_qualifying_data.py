"""
Phase 2: Pull qualifying ("Q") sessions across 2018-2024

Run locally: python src/pull_qualifying_data.py
"""

import fastf1
from fastf1.exceptions import RateLimitExceededError
import pandas as pd
import os
import time

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

os.makedirs("data/raw/qualifying", exist_ok=True)

SEASONS = range(2018, 2025)  # 2018-2024 inclusive
RATE_LIMIT_WAIT_SECONDS = 65 * 60


def call_with_rate_limit_retry(func, *args, **kwargs):
    """Same retry wrapper as pull_season_data.py — wait out rate limits
    instead of crashing."""
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitExceededError:
            print(f"    Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS // 60} "
                  f"minutes before retrying...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            print("    Resuming...")


def pull_qualifying_season(year):
    """Pull qualifying results for every race in a given season."""
    print(f"\n=== Season {year} (Qualifying) ===")
    schedule = call_with_rate_limit_retry(fastf1.get_event_schedule, year)
    races = schedule[schedule["EventFormat"] != "testing"]

    for _, event in races.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        quali_path = f"data/raw/qualifying/{year}_{round_num:02d}.parquet"

        if os.path.exists(quali_path):
            print(f"  Round {round_num} ({event_name}) — already pulled, skipping")
            continue

        try:
            session = call_with_rate_limit_retry(fastf1.get_session, year, round_num, "Q")
            call_with_rate_limit_retry(session.load)

            results = session.results.copy()
            results["season"] = year
            results["round"] = round_num

            # Calculate gap to pole in seconds — the actual signal we need.
            # Q3 time is the relevant one for drivers who made it to Q3;
            # for others, fall back to their best available qualifying time.
            # FastF1 gives Q1/Q2/Q3 as separate columns already in results.
            results.to_parquet(quali_path)

            print(f"  Round {round_num} ({event_name}) — pulled OK "
                  f"({len(results)} results)")

        except Exception as e:
            print(f"  Round {round_num} ({event_name}) — FAILED: {e}")

        time.sleep(2)


if __name__ == "__main__":
    for year in SEASONS:
        pull_qualifying_season(year)

    print("\nDone. Check data/raw/qualifying for output.")
    print("Next: combine these (add to combine_data.py) and build")
    print("calculate_team_pace_trend() using real gap-to-pole data.")