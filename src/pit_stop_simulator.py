"""
Phase 2, Step 11: The actual Pit-Stop Simulator (System 2).

Uses the validated tire degradation curves (SOFT/MEDIUM/HARD reliable,
INTERMEDIATE usable with caution, WET excluded per PROJECT_SCOPE.md Section 10)
to answer the core strategic question: given a driver's current tire age and
race situation, what's the estimated cost/benefit of pitting NOW vs. waiting
N more laps?

This is deterministic simulation logic built ON TOP of the ML degradation
curves — not a model itself, but the "what if" calculator the dashboard's
Strategy Explorer page will call.

Run locally (demo):
    python src/pit_stop_simulator.py
"""

import joblib
import numpy as np

# Rough real-world pit stop cost: time lost entering pit lane, stationary
# time for the tire change, and exiting — varies by circuit, but ~20-25
# seconds is a reasonable general estimate for modern F1 pit lanes.
PIT_STOP_TIME_COST_SECONDS = 22.0

RELIABLE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
CAUTION_COMPOUNDS = ["INTERMEDIATE"]
EXCLUDED_COMPOUNDS = ["WET"]  # per PROJECT_SCOPE.md Section 10 — data too unreliable


def load_degradation_curves(path="models/tire_degradation_curves.pkl"):
    return joblib.load(path)


def predict_lap_time_delta(curves, compound, tyre_life):
    """Same logic as build_tire_degradation_model.py — clips to observed
    range, never extrapolates blindly."""
    if compound not in curves:
        return np.nan, False

    min_observed, max_observed = curves[compound]["tyre_life_range"]
    was_clipped = tyre_life > max_observed or tyre_life < min_observed
    clipped_tyre_life = max(min_observed, min(tyre_life, max_observed))

    a, b, c = curves[compound]["coefficients"]
    delta = a * clipped_tyre_life**2 + b * clipped_tyre_life + c
    return delta, was_clipped


def get_compound_confidence(compound):
    """Returns a confidence label — the dashboard should surface this
    directly, not just show a number as if all compounds were equally
    reliable."""
    if compound in RELIABLE_COMPOUNDS:
        return "high"
    elif compound in CAUTION_COMPOUNDS:
        return "medium"
    elif compound in EXCLUDED_COMPOUNDS:
        return "low — not recommended for strategic decisions"
    else:
        return "unknown compound"


def simulate_pit_decision(curves, compound, current_tyre_age, laps_to_simulate,
                           new_tyre_compound=None):
    """The core simulation: compare staying out on current tires for
    `laps_to_simulate` more laps vs. pitting now for a fresh set.

    Returns a dict with the estimated total time cost of each strategy over
    the simulated window, so they can be directly compared.

    This is intentionally a SIMPLE model — total accumulated degradation
    time on the current tire vs. (pit stop cost + accumulated degradation on
    a fresh tire) — not accounting for traffic, track position value, or
    undercut/overcut effects. That's an honest scope limitation, not
    something to hide from whoever's using this.
    """
    if new_tyre_compound is None:
        new_tyre_compound = compound  # default: same compound fresh

    warnings = []

    # Strategy A: stay out on current tires
    stay_out_cost = 0.0
    for lap_offset in range(laps_to_simulate):
        age = current_tyre_age + lap_offset
        delta, clipped = predict_lap_time_delta(curves, compound, age)
        if clipped:
            warnings.append(f"Stay-out lap {lap_offset+1}: tire age {age} exceeds "
                             f"observed data for {compound}, prediction clipped")
        stay_out_cost += delta

    # Strategy B: pit now, run laps_to_simulate laps on fresh tires.
    # Fresh tire's first completed lap is age 1, not 0 — our curves were
    # fit starting at TyreLife=1 (the true "age 0" pit-out lap was already
    # excluded during clean-lap filtering), so age=0 was never in the data.
    pit_now_cost = PIT_STOP_TIME_COST_SECONDS
    for lap_offset in range(laps_to_simulate):
        age = lap_offset + 1  # fresh tire, first completed lap is age 1
        delta, clipped = predict_lap_time_delta(curves, new_tyre_compound, age)
        if clipped:
            warnings.append(f"Pit-now lap {lap_offset+1}: tire age {age} exceeds "
                             f"observed data for {new_tyre_compound}, prediction clipped")
        pit_now_cost += delta

    time_difference = pit_now_cost - stay_out_cost
    recommendation = "PIT NOW" if time_difference < 0 else "STAY OUT"

    return {
        "stay_out_total_delta": round(stay_out_cost, 3),
        "pit_now_total_delta": round(pit_now_cost, 3),
        "time_difference_seconds": round(time_difference, 3),
        "recommendation": recommendation,
        "confidence": get_compound_confidence(compound),
        "warnings": warnings,
    }


if __name__ == "__main__":
    curves = load_degradation_curves()

    print("=== Demo: Pit-stop simulation ===\n")

    print("Scenario 1: Driver on 15-lap-old MEDIUM tires, considering next 10 laps")
    result = simulate_pit_decision(curves, compound="MEDIUM", current_tyre_age=15,
                                    laps_to_simulate=10)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nScenario 2: Driver on 25-lap-old HARD tires, considering next 15 laps")
    result = simulate_pit_decision(curves, compound="HARD", current_tyre_age=25,
                                    laps_to_simulate=15)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nScenario 3: Driver on 5-lap-old SOFT tires (should clearly favor staying out)")
    result = simulate_pit_decision(curves, compound="SOFT", current_tyre_age=5,
                                    laps_to_simulate=10)
    for k, v in result.items():
        print(f"  {k}: {v}")