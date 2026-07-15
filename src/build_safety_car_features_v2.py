"""
Phase 2, Step 9: Improved features for the Safety Car Model.

Adds three new features beyond the original weather/circuit-SC-history set:
1. grid_competitiveness — how tightly packed qualifying was this race (std
   dev of gap-to-pole across the grid). Tighter grids -> closer racing ->
   more incidents, a real and well-documented racing phenomenon.
2. circuit_dnf_history — historical DNF rate at this circuit (captures
   mechanical failures/incidents more directly than SC rate alone).
3. num_laps — race distance. More laps = more opportunity for something
   to go wrong.

Run locally: python src/build_safety_car_features_v2.py
"""

import pandas as pd
import numpy as np
import os
from feature_engineering import (
    load_all_weather, load_all_qualifying, load_all_results,
    get_weather_features, assign_regulation_era, precompute_qualifying_gaps
)
from build_safety_car_features import build_safety_car_target, calculate_circuit_sc_history


def calculate_grid_competitiveness(gap_table, season, round_num):
    race_gaps = gap_table[
        (gap_table["season"] == season) & (gap_table["round"] == round_num)
    ]
    if len(race_gaps) < 2:
        return np.nan
    return race_gaps["gap_to_pole_seconds"].std()


def calculate_circuit_dnf_history(results_df, circuit_lookup_df, circuit_name, season, round_num):
    """Historical DNF rate at this circuit, using only races BEFORE this one.
    DNF = driver did not finish (Status column isn't 'Finished' and isn't a
    lapped-but-classified result)."""
    circuit_rounds = circuit_lookup_df[
        circuit_lookup_df["circuit_name"] == circuit_name
    ][["season", "round"]]

    merged = results_df.merge(circuit_rounds, on=["season", "round"], how="inner")

    before_this_race = merged[
        (merged["season"] < season) |
        ((merged["season"] == season) & (merged["round"] < round_num))
    ]

    if len(before_this_race) == 0:
        return np.nan

    # A driver DNF'd if their Status isn't "Finished" and doesn't start with
    # "+" (which indicates classified but lapped, e.g. "+1 Lap" — still
    # finished, just not on the lead lap)
    dnf = ~(
        before_this_race["Status"].astype(str).str.startswith("Finished") |
        before_this_race["Status"].astype(str).str.startswith("+")
    )
    return dnf.mean()


def calculate_num_laps(laps_df, season, round_num):
    """Total laps completed in this race."""
    race_laps = laps_df[(laps_df["season"] == season) & (laps_df["round"] == round_num)]
    if len(race_laps) == 0:
        return np.nan
    return race_laps["LapNumber"].max()


def main():
    print("Loading data...")
    laps = pd.read_parquet("data/processed/all_laps.parquet")
    weather = load_all_weather()
    qualifying = load_all_qualifying()
    results = load_all_results()
    circuit_lookup = pd.read_csv("data/processed/circuit_lookup.csv")

    print("\nPrecomputing qualifying gaps...")
    gap_table = precompute_qualifying_gaps(qualifying)

    print("Building safety car target...")
    target_df = build_safety_car_target(laps)

    round_to_circuit = circuit_lookup.set_index(["season", "round"])["circuit_name"].to_dict()

    print("\nBuilding v2 features per race...")
    feature_rows = []
    for _, row in target_df.iterrows():
        season = row["season"]
        round_num = row["round"]
        circuit_name = round_to_circuit.get((season, round_num), None)

        circuit_sc_history = (
            calculate_circuit_sc_history(target_df, circuit_lookup, circuit_name, season, round_num)
            if circuit_name else np.nan
        )
        circuit_dnf_history = (
            calculate_circuit_dnf_history(results, circuit_lookup, circuit_name, season, round_num)
            if circuit_name else np.nan
        )
        grid_competitiveness = calculate_grid_competitiveness(gap_table, season, round_num)
        num_laps = calculate_num_laps(laps, season, round_num)
        weather_features = get_weather_features(weather, season, round_num)

        feature_rows.append({
            "season": season,
            "round": round_num,
            "circuit_name": circuit_name,
            "regulation_era": assign_regulation_era(season),
            "circuit_sc_history": circuit_sc_history,
            "circuit_dnf_history": circuit_dnf_history,
            "grid_competitiveness": grid_competitiveness,
            "num_laps": num_laps,
            "was_wet": weather_features["was_wet"],
            "avg_track_temp": weather_features["avg_track_temp"],
            "avg_air_temp": weather_features["avg_air_temp"],
            "had_safety_car": row["had_safety_car"],
        })

    feature_table = pd.DataFrame(feature_rows)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/safety_car_features_v2.parquet"
    feature_table.to_parquet(output_path)

    print(f"\nDone. Saved {len(feature_table)} rows to {output_path}")
    print(f"\nMissing value counts:")
    print(feature_table[["circuit_sc_history", "circuit_dnf_history",
                          "grid_competitiveness", "num_laps"]].isna().sum())


if __name__ == "__main__":
    main()