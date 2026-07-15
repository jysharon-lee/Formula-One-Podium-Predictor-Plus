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
    """Express each lap's time as a delta from the FIELD'S median lap time
    at that exact lap number, in that race — not just that race's single
    best lap.

    Why this is better than race-best + a linear lap_number correction:
    subtracting the field median at each specific lap automatically removes
    ANY shared track-level trend at that moment — fuel burn-off, track
    drying in wet conditions, rubber being laid down — regardless of what's
    causing it or whether it's linear. All cars experience roughly the same
    track conditions at the same lap, so this shared trend cancels out,
    leaving mostly car/tire-specific variation (tire age, compound, driver
    pace) in the residual.

    This is what fixed the wet/intermediate track-drying confound that a
    simple linear lap_number term couldn't fully separate (tire age and lap
    number are too collinear within a single wet stint, since cars rarely
    pit mid-chaos — not enough independent variation for a linear term to
    isolate the two effects).
    """
    clean_laps = clean_laps.copy()
    clean_laps["lap_time_seconds"] = clean_laps["LapTime"].dt.total_seconds()

    field_median_at_lap = clean_laps.groupby(
        ["season", "round", "LapNumber"]
    )["lap_time_seconds"].transform("median")

    clean_laps["lap_time_delta"] = clean_laps["lap_time_seconds"] - field_median_at_lap

    return clean_laps


def fit_degradation_curves(clean_laps, max_tyre_life=40):
    """Fit lap_time_delta ~ TyreLife (quadratic), per compound.

    No longer need a separate lap_number covariate — field-median-per-lap
    normalization already removed shared track-level trends (fuel burn,
    track drying), so TyreLife is now the primary remaining source of
    systematic variation in the residual.
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

        coeffs = np.polyfit(compound_laps["TyreLife"], compound_laps["lap_time_delta"], deg=2)
        curves[compound] = {
            "coefficients": coeffs,  # [a, b, c] for a*x^2 + b*x + c
            "n_laps": len(compound_laps),
            "tyre_life_range": (compound_laps["TyreLife"].min(), compound_laps["TyreLife"].max()),
        }
        print(f"  {compound}: fit on {len(compound_laps)} laps "
              f"(TyreLife range {curves[compound]['tyre_life_range']})")
        print(f"    Coefficients (a, b, c): ({coeffs[0]:.5f}, {coeffs[1]:.5f}, {coeffs[2]:.5f})")

    return curves


def predict_lap_time_delta(curves, compound, tyre_life):
    """Given a compound and tire age, predict the expected lap time delta
    vs. the field's typical pace at that point in the race.

    IMPORTANT: clips tyre_life to the range actually observed in the data
    for that compound. Querying a quadratic curve outside its fitted range
    is unreliable extrapolation, not a real prediction — this bit us
    directly with WET tires (only observed up to 24 laps old; querying age
    30 produced a nonsensical -5.2s "faster than fresh" result purely from
    extrapolation instability on a small, 372-lap sample).
    """
    if compound not in curves:
        return np.nan, False

    min_observed, max_observed = curves[compound]["tyre_life_range"]
    was_clipped = tyre_life > max_observed or tyre_life < min_observed
    clipped_tyre_life = max(min_observed, min(tyre_life, max_observed))

    a, b, c = curves[compound]["coefficients"]
    delta = a * clipped_tyre_life**2 + b * clipped_tyre_life + c
    return delta, was_clipped


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
    print(f"(delta is vs. the field's typical pace at that point in the race)")
    print(f"(ages beyond each compound's OBSERVED range are marked [CLIPPED] —")
    print(f" these are extrapolations and should not be trusted)")
    test_ages = [1, 5, 10, 20, 30, 40]
    for compound in curves:
        min_obs, max_obs = curves[compound]["tyre_life_range"]
        print(f"\n{compound} (observed range: {min_obs:.0f}-{max_obs:.0f} laps):")
        for tyre_life in test_ages:
            delta, was_clipped = predict_lap_time_delta(curves, compound, tyre_life)
            flag = " [CLIPPED — outside observed data, unreliable]" if was_clipped else ""
            print(f"  Tire age {tyre_life} laps: {delta:+.3f}s vs. field median{flag}")


if __name__ == "__main__":
    main()