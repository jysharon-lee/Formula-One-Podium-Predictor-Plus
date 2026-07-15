"""
Phase 2, Step 10: Build tire degradation curves (foundation for System 2,
the Pit-Stop Simulator).

Two important design decisions, both confirmed necessary by exploring the
real data first:
1. Only use "clean" laps (accurate, no pit in/out, green track) — otherwise
   pit-stop laps and safety car laps would corrupt the curve with artificially
   slow times unrelated to actual tire wear.
2. Normalize lap times WITHIN each race before fitting across compounds —
   raw lap times differ hugely by circuit (Monaco ~75s vs Spa ~105s), and
   mixing them directly would swamp the actual degradation signal. We express
   each lap as a DELTA from that race's own best clean lap time.
3. Exclude legacy 2018-only compound names (HYPERSOFT/ULTRASOFT/SUPERSOFT) —
   these don't map cleanly onto today's SOFT/MEDIUM/HARD system, and forcing
   a mapping would introduce more noise than signal.

Run locally:
    python src/build_tire_degradation_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib

MODERN_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def filter_clean_laps(laps):
    """Same 'clean lap' criteria explored in explore_tire_data.py."""
    has_pit_out = laps["PitOutTime"].notna()
    has_pit_in = laps["PitInTime"].notna()
    clean_track = laps["TrackStatus"] == "1"

    clean = (
        laps["IsAccurate"] &
        ~has_pit_out &
        ~has_pit_in &
        clean_track &
        laps["LapTime"].notna() &
        laps["Compound"].isin(MODERN_COMPOUNDS)
    )
    return laps[clean].copy()


def normalize_lap_times(clean_laps):
    """Express each lap's time as a delta from that RACE's own best clean
    lap — isolates tire degradation from track-to-track pace differences."""
    clean_laps["lap_time_seconds"] = clean_laps["LapTime"].dt.total_seconds()

    race_best = clean_laps.groupby(["season", "round"])["lap_time_seconds"].transform("min")
    clean_laps["lap_time_delta"] = clean_laps["lap_time_seconds"] - race_best

    return clean_laps


def fit_degradation_curves(clean_laps, max_tyre_life=40):
    """Fit a simple quadratic regression per compound: lap_time_delta ~ TyreLife.
    Quadratic (not linear) because tire degradation often accelerates near
    end-of-life ("falling off a cliff"), not just a steady linear increase.

    Capped at max_tyre_life to avoid extreme outlier stints (very old tires,
    rare strategy choices) distorting the curve for the normal operating range.
    """
    curves = {}
    for compound in MODERN_COMPOUNDS:
        compound_laps = clean_laps[
            (clean_laps["Compound"] == compound) &
            (clean_laps["TyreLife"] <= max_tyre_life)
        ]

        if len(compound_laps) < 30:
            print(f"  {compound}: only {len(compound_laps)} laps — too few to fit reliably, skipping")
            continue

        # Fit lap_time_delta = a*TyreLife^2 + b*TyreLife + c
        coeffs = np.polyfit(compound_laps["TyreLife"], compound_laps["lap_time_delta"], deg=2)
        curves[compound] = {
            "coefficients": coeffs,  # [a, b, c] for a*x^2 + b*x + c
            "n_laps": len(compound_laps),
            "tyre_life_range": (compound_laps["TyreLife"].min(), compound_laps["TyreLife"].max()),
        }
        print(f"  {compound}: fit on {len(compound_laps)} laps "
              f"(TyreLife range {curves[compound]['tyre_life_range']})")

    return curves


def predict_lap_time_delta(curves, compound, tyre_life):
    """Given a compound and tire age, predict the expected lap time delta
    (seconds slower than that race's best lap) — this is what the pit-stop
    simulator will call to estimate degradation cost."""
    if compound not in curves:
        return np.nan
    a, b, c = curves[compound]["coefficients"]
    return a * tyre_life**2 + b * tyre_life + c


def main():
    print("Loading laps data...")
    laps = pd.read_parquet("data/processed/all_laps.parquet")
    print(f"Total laps: {len(laps)}")

    print("\nFiltering to clean laps + modern compounds...")
    clean_laps = filter_clean_laps(laps)
    print(f"Clean laps remaining: {len(clean_laps)} ({len(clean_laps)/len(laps):.1%} of total)")

    print("\nNormalizing lap times within each race...")
    clean_laps = normalize_lap_times(clean_laps)

    print("\nFitting degradation curves per compound...")
    curves = fit_degradation_curves(clean_laps)

    os.makedirs("models", exist_ok=True)
    joblib.dump(curves, "models/tire_degradation_curves.pkl")
    print(f"\nSaved degradation curves to models/tire_degradation_curves.pkl")

    print(f"\n--- Sanity check: predicted lap time delta at various tire ages ---")
    for compound in curves:
        print(f"\n{compound}:")
        for tyre_life in [1, 5, 10, 20, 30]:
            delta = predict_lap_time_delta(curves, compound, tyre_life)
            print(f"  Tire age {tyre_life} laps: +{delta:.3f}s vs. race best")


if __name__ == "__main__":
    main()