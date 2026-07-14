"""
Phase 2, Step 7: Feature engineering skeleton.

This file defines the SHAPE of the feature engineering pipeline, matching the
feature list already locked in PROJECT_SCOPE.md. Logic inside each function
is a placeholder/TODO — fill these in once data/raw/ is fully populated.

Having this structure written now (while the data pull runs) means once real
data lands, you're filling in logic, not designing from scratch.
"""

import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------------
# STEP A: Combine per-race parquet files into one big table
# ---------------------------------------------------------------------------

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


def load_all_qualifying(path="data/processed/all_qualifying.parquet"):
    """Load the combined qualifying dataset built by src/combine_data.py."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} qualifying rows across {df['season'].nunique()} seasons")
    return df


def load_circuit_lookup(path="data/processed/circuit_lookup.csv"):
    """Load the season+round -> circuit name lookup built by
    build_circuit_lookup.py. Needed since raw results/qualifying data only
    carries season/round, not which circuit that actually was."""
    df = pd.read_csv(path)
    print(f"Loaded circuit lookup: {len(df)} season/round mappings")
    return df


def assign_regulation_era(season):
    """Maps a season year to its regulation era, per PROJECT_SCOPE.md Section 5.
    This one's simple enough to write now, no data needed."""
    if 2018 <= season <= 2021:
        return "2018-2021"
    elif 2022 <= season <= 2025:
        return "2022-2025"
    elif season >= 2026:
        return "2026+"
    else:
        return "pre-2018"


# ---------------------------------------------------------------------------
# STEP B: Per-race features (one row per driver, per race)
# ---------------------------------------------------------------------------

def calculate_recent_form(results_df, driver, season, round_num, n_races=5):
    """Average finishing position for `driver` across their last n_races races
    BEFORE this one (not including this race — that would leak the answer
    into the features).

    Design decision: recent form looks ACROSS season boundaries, not resetting
    each season. Reasoning: a driver's form in the last race of 2023 is still
    meaningful signal for the first race of 2024 — car development doesn't
    reset overnight, and early-season races would otherwise have no "recent
    form" signal at all for the first n_races of every season.

    Returns np.nan if the driver has fewer than n_races prior races (e.g. a
    rookie's first race) — the model needs to handle missing values here,
    don't silently fill with 0 or a fake default.
    """
    driver_races = results_df[results_df["Abbreviation"] == driver].copy()

    # Keep only races strictly BEFORE this one, ordered chronologically
    driver_races = driver_races.sort_values(["season", "round"])
    before_this_race = driver_races[
        (driver_races["season"] < season) |
        ((driver_races["season"] == season) & (driver_races["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan  # no prior races at all — e.g. driver's debut race

    recent = before_this_race.tail(n_races)
    # Position can be non-numeric for DNFs in some FastF1 result encodings —
    # coerce and drop those before averaging
    positions = pd.to_numeric(recent["Position"], errors="coerce").dropna()

    if len(positions) == 0:
        return np.nan  # all recent races were DNFs with no position recorded

    return positions.mean()


def precompute_qualifying_gaps(qualifying_df):
    """Precompute gap-to-pole for every team, at every race, ONCE.

    This replaces the earlier approach of recalculating gap-to-pole from
    scratch inside calculate_team_pace_trend() every time it's called. That
    approach was correct but slow — recomputing across the whole qualifying
    dataset per function call doesn't scale once you're calling it for
    ~2900 driver-race rows while building the full feature table.

    Call this ONCE, then pass the result into calculate_team_pace_trend()
    for fast repeated lookups.
    """
    def best_time(row):
        times = [row.get(col) for col in ["Q1", "Q2", "Q3"]]
        times = [t for t in times if pd.notna(t)]
        return min(times) if times else np.nan

    df = qualifying_df.copy()
    df["best_quali_time"] = df.apply(best_time, axis=1)

    gap_rows = []
    for (s, r), race_group in df.groupby(["season", "round"]):
        pole_time = race_group["best_quali_time"].min()
        if pd.isna(pole_time):
            continue
        team_gaps = race_group.groupby("TeamName")["best_quali_time"].mean() - pole_time
        for team, gap in team_gaps.items():
            gap_rows.append({
                "season": s,
                "round": r,
                "TeamName": team,
                "gap_to_pole_seconds": gap.total_seconds() if pd.notna(gap) else np.nan,
            })

    gap_table = pd.DataFrame(gap_rows)
    print(f"Precomputed gap-to-pole for {len(gap_table)} team-race combinations")
    return gap_table


def calculate_team_pace_trend(gap_table, team, season, round_num, n_races=5):
    """Average qualifying gap to pole for `team`, across their last n_races
    races BEFORE this one — using the precomputed gap_table from
    precompute_qualifying_gaps(), not raw qualifying data. Fast lookup,
    not a recomputation, so this is safe to call per-row when building the
    full feature table.

    Returns np.nan if no prior data exists for this team.
    """
    team_gaps = gap_table[gap_table["TeamName"] == team].sort_values(["season", "round"])

    before_this_race = team_gaps[
        (team_gaps["season"] < season) |
        ((team_gaps["season"] == season) & (team_gaps["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan

    recent = before_this_race.tail(n_races)
    return recent["gap_to_pole_seconds"].mean()


def calculate_circuit_history(results_df, circuit_lookup_df, driver, circuit_name, season, round_num):
    """Driver's average finishing position at this specific circuit, across
    all past races in the training window BEFORE this one.

    results_df does NOT have a circuit name column directly (FastF1's raw
    results only carry season/round, not which circuit that was) — so we
    join against circuit_lookup_df (built by build_circuit_lookup.py) to
    identify which past races were actually at this circuit.

    Returns np.nan if the driver has never raced at this circuit before.
    """
    # Find every (season, round) that was this circuit
    circuit_rounds = circuit_lookup_df[
        circuit_lookup_df["circuit_name"] == circuit_name
    ][["season", "round"]]

    if len(circuit_rounds) == 0:
        return np.nan  # circuit name not found in lookup — check spelling matches exactly

    # Join results to only the races that happened at this circuit
    driver_results = results_df[results_df["Abbreviation"] == driver]
    circuit_races = driver_results.merge(circuit_rounds, on=["season", "round"], how="inner")

    before_this_race = circuit_races[
        (circuit_races["season"] < season) |
        ((circuit_races["season"] == season) & (circuit_races["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan

    positions = pd.to_numeric(before_this_race["Position"], errors="coerce").dropna()

    if len(positions) == 0:
        return np.nan  # all past races at this circuit were DNFs

    return positions.mean()


def get_weather_features(weather_df, season, round_num, wet_rainfall_threshold=0.0):
    """Summarize weather_df for a specific race into simple features:
    was_wet, avg_track_temp, avg_air_temp.

    FastF1's weather data is sampled many times per session (roughly every
    minute), so we aggregate across the whole race rather than using a
    single snapshot.

    was_wet is determined by checking if ANY rainfall was recorded during
    the race — a race can start dry and turn wet partway through, and any
    rain during the race is relevant signal, not just rain at the start.
    """
    race_weather = weather_df[
        (weather_df["season"] == season) & (weather_df["round"] == round_num)
    ]

    if len(race_weather) == 0:
        return {"was_wet": np.nan, "avg_track_temp": np.nan, "avg_air_temp": np.nan}

    # FastF1's Rainfall column is boolean per sample — True if rain was
    # detected at that timestamp. was_wet = True if it rained at ANY point.
    was_wet = bool(race_weather["Rainfall"].any()) if "Rainfall" in race_weather.columns else np.nan

    avg_track_temp = race_weather["TrackTemp"].mean() if "TrackTemp" in race_weather.columns else np.nan
    avg_air_temp = race_weather["AirTemp"].mean() if "AirTemp" in race_weather.columns else np.nan

    return {
        "was_wet": was_wet,
        "avg_track_temp": avg_track_temp,
        "avg_air_temp": avg_air_temp,
    }


# ---------------------------------------------------------------------------
# STEP C: Per-lap features (for the pit-stop simulator, built later)
# ---------------------------------------------------------------------------

def calculate_tire_degradation(laps_df, compound):
    """TODO: fit a simple regression of lap time vs. tire age (TyreLife),
    per compound. This is the foundation for the pit-stop simulator (System 3
    in the architecture) — not needed yet for the race outcome model, but
    good to stub out now.
    """
    raise NotImplementedError("Build this after the race outcome model works")


# ---------------------------------------------------------------------------
# STEP D: Assemble the final modeling dataset
# ---------------------------------------------------------------------------

def build_feature_table():
    """Build the final modeling dataset — one row per driver per race, every
    feature from PROJECT_SCOPE.md Section 4, ready to feed into the race
    outcome model.

    Saves the result to data/processed/features.parquet.
    """
    print("Loading combined datasets...")
    results = load_all_results()
    qualifying = load_all_qualifying()
    weather = load_all_weather()
    circuit_lookup = load_circuit_lookup()

    print("\nPrecomputing qualifying gaps (one-time cost)...")
    gap_table = precompute_qualifying_gaps(qualifying)

    # Build a season/round -> circuit_name lookup dict for fast access inside the loop
    round_to_circuit = circuit_lookup.set_index(["season", "round"])["circuit_name"].to_dict()

    print("\nBuilding feature rows...")
    feature_rows = []
    total = len(results)

    for i, row in results.iterrows():
        if i % 500 == 0:
            print(f"  Processing row {i}/{total}...")

        driver = row["Abbreviation"]
        team = row["TeamName"]
        season = row["season"]
        round_num = row["round"]
        grid_position = row.get("GridPosition", np.nan)
        finishing_position = pd.to_numeric(row.get("Position"), errors="coerce")

        circuit_name = round_to_circuit.get((season, round_num), None)

        recent_form = calculate_recent_form(results, driver, season, round_num)
        team_pace = calculate_team_pace_trend(gap_table, team, season, round_num)
        circuit_hist = (
            calculate_circuit_history(results, circuit_lookup, driver, circuit_name, season, round_num)
            if circuit_name else np.nan
        )
        weather_features = get_weather_features(weather, season, round_num)

        # The actual prediction target: did this driver finish in the top 3?
        podium = 1 if pd.notna(finishing_position) and finishing_position <= 3 else 0

        feature_rows.append({
            "season": season,
            "round": round_num,
            "driver": driver,
            "team": team,
            "circuit_name": circuit_name,
            "regulation_era": assign_regulation_era(season),
            "grid_position": grid_position,
            "recent_form": recent_form,
            "team_pace_trend": team_pace,
            "circuit_history": circuit_hist,
            "was_wet": weather_features["was_wet"],
            "avg_track_temp": weather_features["avg_track_temp"],
            "avg_air_temp": weather_features["avg_air_temp"],
            "finishing_position": finishing_position,
            "podium": podium,
        })

    feature_table = pd.DataFrame(feature_rows)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/features.parquet"
    feature_table.to_parquet(output_path)

    print(f"\nDone. Saved {len(feature_table)} rows to {output_path}")
    print(f"\nPodium rate: {feature_table['podium'].mean():.1%} (should be roughly 3/20 = 15%)")
    print(f"\nMissing value counts:")
    print(feature_table[["recent_form", "team_pace_trend", "circuit_history", "was_wet"]].isna().sum())

    return feature_table


if __name__ == "__main__":
    build_feature_table()