"""
Phase 2: Build features for the Safety Car Model.

Run locally: python src/build_safety_car_features.py
"""

import pandas as pd
import numpy as np
import os
from feature_engineering import load_all_weather, get_weather_features, assign_regulation_era


def has_safety_car_or_vsc(track_status_series):
    """Check if code '4' (Safety Car) or '6' (VSC) appears anywhere in this
    race's TrackStatus values. Codes can be concatenated in a single lap's
    string (e.g. '124'), so we check character membership, not exact match."""
    all_codes = "".join(track_status_series.dropna().astype(str))
    return ("4" in all_codes) or ("6" in all_codes)


def build_safety_car_target(laps_df):
    """One row per race: did a safety car/VSC happen, season, round."""
    rows = []
    for (season, round_num), race_group in laps_df.groupby(["season", "round"]):
        had_sc = has_safety_car_or_vsc(race_group["TrackStatus"])
        rows.append({"season": season, "round": round_num, "had_safety_car": int(had_sc)})
    return pd.DataFrame(rows)


def calculate_circuit_sc_history(target_df, circuit_lookup_df, circuit_name, season, round_num):
    """Real, empirically-computed safety car rate for this circuit, using
    only races BEFORE this one (no leakage) — this REPLACES the qualitative
    guesses in circuit_metadata.py with actual measured data.

    Returns np.nan if no prior races at this circuit exist.
    """
    circuit_rounds = circuit_lookup_df[
        circuit_lookup_df["circuit_name"] == circuit_name
    ][["season", "round"]]

    merged = target_df.merge(circuit_rounds, on=["season", "round"], how="inner")

    before_this_race = merged[
        (merged["season"] < season) |
        ((merged["season"] == season) & (merged["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan

    return before_this_race["had_safety_car"].mean()


def main():
    print("Loading laps data...")
    laps = pd.read_parquet("data/processed/all_laps.parquet")
    weather = load_all_weather()
    circuit_lookup = pd.read_csv("data/processed/circuit_lookup.csv")

    print("Building safety car target (one row per race)...")
    target_df = build_safety_car_target(laps)
    print(f"Built target for {len(target_df)} races")
    print(f"Overall safety car rate: {target_df['had_safety_car'].mean():.1%}")

    round_to_circuit = circuit_lookup.set_index(["season", "round"])["circuit_name"].to_dict()

    print("\nBuilding features per race...")
    feature_rows = []
    for _, row in target_df.iterrows():
        season = row["season"]
        round_num = row["round"]
        circuit_name = round_to_circuit.get((season, round_num), None)

        circuit_sc_history = (
            calculate_circuit_sc_history(target_df, circuit_lookup, circuit_name, season, round_num)
            if circuit_name else np.nan
        )
        weather_features = get_weather_features(weather, season, round_num)

        feature_rows.append({
            "season": season,
            "round": round_num,
            "circuit_name": circuit_name,
            "regulation_era": assign_regulation_era(season),
            "circuit_sc_history": circuit_sc_history,
            "was_wet": weather_features["was_wet"],
            "avg_track_temp": weather_features["avg_track_temp"],
            "avg_air_temp": weather_features["avg_air_temp"],
            "had_safety_car": row["had_safety_car"],
        })

    feature_table = pd.DataFrame(feature_rows)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/safety_car_features.parquet"
    feature_table.to_parquet(output_path)

    print(f"\nDone. Saved {len(feature_table)} rows to {output_path}")
    print(f"Missing circuit_sc_history: {feature_table['circuit_sc_history'].isna().sum()} "
          f"(expected — circuits' first appearance in the training window)")


if __name__ == "__main__":
    main()