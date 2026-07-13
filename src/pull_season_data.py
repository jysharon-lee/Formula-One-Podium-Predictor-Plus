"""
Phase 2: Pull all races across the training window (2018-2024)

Run locally: python src/pull_season_data.py

"""

import fastf1
from fastf1.exceptions import RateLimitExceededError
import pandas as pd
import os
import time

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

os.makedirs("data/raw/laps", exist_ok=True)
os.makedirs("data/raw/results", exist_ok=True)
os.makedirs("data/raw/weather", exist_ok=True)

SEASONS = range(2018, 2025)  # 2018-2024 inclusive

RATE_LIMIT_WAIT_SECONDS = 65 * 60  # 65 minutes, a bit of buffer over the hour


def call_with_rate_limit_retry(func, *args, **kwargs):
    """Run a FastF1 call; if it hits the rate limit, wait an hour and retry
    automatically instead of crashing."""
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitExceededError:
            print(f"    Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS // 60} "
                  f"minutes before retrying...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            print("    Resuming...")


def pull_season(year):
    """Pull every race in a given season and save laps/results/weather separately."""
    print(f"\n=== Season {year} ===")
    schedule = call_with_rate_limit_retry(fastf1.get_event_schedule, year)
    # Skip testing events, keep only actual race weekends
    races = schedule[schedule["EventFormat"] != "testing"]

    for _, event in races.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        laps_path = f"data/raw/laps/{year}_{round_num:02d}.parquet"
        results_path = f"data/raw/results/{year}_{round_num:02d}.parquet"
        weather_path = f"data/raw/weather/{year}_{round_num:02d}.parquet"

        # Skip if already pulled 
        if os.path.exists(laps_path):
            print(f"  Round {round_num} ({event_name}) — already pulled, skipping")
            continue

        try:
            session = call_with_rate_limit_retry(fastf1.get_session, year, round_num, "R")
            call_with_rate_limit_retry(session.load)

            # Tag every row with year/round to identify it after combining files
            laps = session.laps.copy()
            laps["season"] = year
            laps["round"] = round_num

            results = session.results.copy()
            results["season"] = year
            results["round"] = round_num

            weather = session.weather_data.copy()
            weather["season"] = year
            weather["round"] = round_num

            laps.to_parquet(laps_path)
            results.to_parquet(results_path)
            weather.to_parquet(weather_path)

            print(f"  Round {round_num} ({event_name}) — pulled OK "
                  f"({len(laps)} laps, {len(results)} results)")

        except RateLimitExceededError:
            raise  
        except Exception as e:
            print(f"  Round {round_num} ({event_name}) — FAILED: {e}")

        time.sleep(2)  # be politer to F1's servers, avoid hammering the API and re-hitting the limit ：）


if __name__ == "__main__":
    for year in SEASONS:
        pull_season(year)

    print("\nDone. Check data/raw/laps, data/raw/results, data/raw/weather for output.")
    print("Next: combine these into one clean dataset (Phase 2, Step 7).")