"""
Scan across tire ages and simulation windows to find where (or whether)
the pit-stop simulator ever genuinely flips to recommending PIT NOW for a
same-compound tire refresh — rather than hand-picking a scenario that might
not reflect what the model can actually produce.

Run locally:
    python src/find_pit_crossover.py
"""

from pit_stop_simulator import load_degradation_curves, simulate_pit_decision, RELIABLE_COMPOUNDS

curves = load_degradation_curves()

print("Scanning for the tire age / window combination where PIT NOW first")
print("becomes the recommendation, per compound (same-compound refresh)...\n")

for compound in RELIABLE_COMPOUNDS:
    print(f"=== {compound} ===")
    found_crossover = False

    for current_age in [10, 20, 30, 39]:
        for window in [10, 20, 30, 40, 50, 60]:
            result = simulate_pit_decision(
                curves, compound=compound, current_tyre_age=current_age,
                laps_to_simulate=window
            )
            if result["recommendation"] == "PIT NOW":
                print(f"  CROSSOVER FOUND: tire age {current_age}, "
                      f"window {window} laps -> PIT NOW "
                      f"(would save {-result['time_difference_seconds']:.2f}s)")
                found_crossover = True

    if not found_crossover:
        print(f"  No crossover found within tested ranges (tire age up to 39, "
              f"window up to 60 laps) — same-compound refresh never beats "
              f"the ~22s pit stop cost for {compound} within observed data.")
    print()

print("--- What this tells us ---")
print("If no crossover was found for any compound, that's a genuine finding:")
print("these degradation curves are simply too flat (< 0.5s difference across")
print("the whole observed tire-age range) for same-compound wear alone to ever")
print("justify a pit stop's fixed ~22s cost. Real pit stops are typically")
print("driven by COMPOUND CHANGES (switching to a faster tire) or race")
print("position tactics (undercut/overcut), not pure same-tire degradation —")
print("which is actually a realistic and defensible result, not a model failure.")