"""
Phase 2: Feature engineering skeleton.

"""

import pandas as pd
import numpy as np


# STEP 1: Combine per-race parquet files into one big table

def load_all_laps(path="data/processed/all_laps.parquet"):
    """Load the combined laps dataset built by src/combine_data.py."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} lap rows across {df['season'].nunique()} seasons")
    return df


def load_all_results(path="data/processed/all_results.parquet"):
    """Load the combined results dataset built by src/combine_data.py."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} result rows across {df['season'].nunique()} seasons")
    return df


def load_all_weather(path="data/processed/all_weather.parquet"):
    """Load the combined weather dataset built by src/combine_data.py."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} weather rows across {df['season'].nunique()} seasons")
    return df


def assign_regulation_era(season):
    """Maps a season year to its regulation era"""
    if 2018 <= season <= 2021:
        return "2018-2021"
    elif 2022 <= season <= 2025:
        return "2022-2025"
    elif season >= 2026:
        return "2026+"
    else:
        return "pre-2018"


# STEP 2: Per-race features (one row per driver, per race)

def calculate_recent_form(results_df, driver, season, round_num, n_races=5):
    driver_races = results_df[results_df["Abbreviation"] == driver].copy()

    # Keep only races strictly b4 this one, ordered chronologically
    driver_races = driver_races.sort_values(["season", "round"])
    before_this_race = driver_races[
        (driver_races["season"] < season) |
        ((driver_races["season"] == season) & (driver_races["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan  # no prior races at all 

    recent = before_this_race.tail(n_races)
    # Position can be non-numeric for DNFs in some FastF1 result encodings
    positions = pd.to_numeric(recent["Position"], errors="coerce").dropna()

    if len(positions) == 0:
        return np.nan  # all recent races were DNFs with no position recorded

    return positions.mean()


def calculate_team_pace_trend(results_df, team, season, round_num, n_races=5):
    """TODO: average qualifying gap to pole for `team`, across their last
    n_races races before this one. Requires qualifying session data, not just
    race results — check whether FastF1 gives this from the 'R' session
    or whether needs to pull the 'Q' (qualifying) session separately.
    """
    raise NotImplementedError("Fill in once combined results data exists")


def calculate_circuit_history(results_df, driver, circuit_name):
    """TODO: driver's average finishing position at this specific circuit,
    across all past seasons in the training window.
    """
    raise NotImplementedError("Fill in once combined results data exists")


def get_weather_features(weather_df, season, round_num):
    """TODO: summarize weather_df for this race into simple features:
    e.g. was_wet (bool), avg_track_temp, avg_air_temp.
    """
    raise NotImplementedError("Fill in once combined weather data exists")


# STEP 3: Per-lap features (for the pit-stop simulator, built later)

def calculate_tire_degradation(laps_df, compound):
    """TODO: fit a simple regression of lap time vs. tire age (TyreLife),
    per compound
    """
    raise NotImplementedError("Build this after the race outcome model works")


# STEP 4: Assemble the final modeling dataset

def build_feature_table():
    """TODO: the main function that ties everything above together into one
    clean table.

    Suggested order once you're filling this in:
    1. Load combined laps/results/weather
    2. Add regulation_era column
    3. Loop through each race, calculate recent_form / team_pace_trend /
       circuit_history / weather features for every driver
    4. Save the final table to data/processed/features.parquet
    """
    raise NotImplementedError("Fill in once Steps A-C above are implemented")


if __name__ == "__main__":
    print("This is a skeleton file — functions are stubbed with NotImplementedError.")
    print("Fill in the logic once data/raw/ is fully populated from pull_season_data.py")