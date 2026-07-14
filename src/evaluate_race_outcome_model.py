"""
Phase 3, Step 13-15: Proper evaluation matching PROJECT_SCOPE.md Section 6.

This checks the ACTUAL success criteria we locked in during Phase 1:
1. Top-3 accuracy PER RACE — was the real podium finisher in our model's
   top-3 predicted drivers for that specific race? (not just generic
   classification accuracy across all rows mixed together)
2. Calibration — if the model says "70% podium chance", do drivers in that
   probability range actually podium about 70% of the time?

Run locally:
    python src/evaluate_race_outcome_model.py
"""

import pandas as pd
import numpy as np
import joblib
from train_race_outcome_model import prepare_data, time_based_split


def evaluate_top3_per_race(df, model, feature_columns):
    """For each race, take the model's top-3 predicted drivers (by podium
    probability) and check how many actually podiumed."""
    X, y, _ = prepare_data(df)
    X = X.reindex(columns=feature_columns, fill_value=0)

    df = df.copy()
    df["predicted_proba"] = model.predict_proba(X)[:, 1]

    race_results = []
    for (season, round_num), race_group in df.groupby(["season", "round"]):
        predicted_top3 = set(
            race_group.nlargest(3, "predicted_proba")["driver"]
        )
        actual_top3 = set(
            race_group[race_group["podium"] == 1]["driver"]
        )

        overlap = len(predicted_top3 & actual_top3)
        exact_match = predicted_top3 == actual_top3

        race_results.append({
            "season": season,
            "round": round_num,
            "predicted_top3": predicted_top3,
            "actual_top3": actual_top3,
            "overlap": overlap,  # how many of the 3 predicted actually podiumed
            "exact_match": exact_match,
        })

    return pd.DataFrame(race_results)


def check_calibration(df, model, feature_columns, n_bins=5):
    """Bin predictions by probability, compare predicted vs actual podium
    rate in each bin. A well-calibrated model should have these roughly
    match — e.g. predictions in the 60-70% bin should actually podium
    roughly 60-70% of the time."""
    X, y, _ = prepare_data(df)
    X = X.reindex(columns=feature_columns, fill_value=0)

    predicted_proba = model.predict_proba(X)[:, 1]

    calibration_df = pd.DataFrame({
        "predicted_proba": predicted_proba,
        "actual_podium": y.values,
    })
    calibration_df["bin"] = pd.qcut(
        calibration_df["predicted_proba"], q=n_bins, duplicates="drop"
    )

    summary = calibration_df.groupby("bin", observed=True).agg(
        avg_predicted=("predicted_proba", "mean"),
        avg_actual=("actual_podium", "mean"),
        count=("actual_podium", "size"),
    )
    return summary


def main():
    print("Loading model and test data...")
    saved = joblib.load("models/race_outcome_model.pkl")
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    df = pd.read_parquet("data/processed/features.parquet")
    _, test_df = time_based_split(df, test_season=2024)

    print(f"Evaluating on {test_df['season'].nunique()} season, "
          f"{test_df.groupby(['season', 'round']).ngroups} races\n")

    print("=== Per-Race Top-3 Accuracy ===")
    race_eval = evaluate_top3_per_race(test_df, model, feature_columns)

    avg_overlap = race_eval["overlap"].mean()
    exact_match_rate = race_eval["exact_match"].mean()

    print(f"Average podium overlap: {avg_overlap:.2f} / 3 "
          f"(how many of the predicted top-3 actually podiumed, on average)")
    print(f"Exact match rate: {exact_match_rate:.1%} "
          f"(all 3 predicted drivers exactly matched actual podium)")
    print(f"\nAt-least-1-correct rate: {(race_eval['overlap'] >= 1).mean():.1%}")
    print(f"At-least-2-correct rate: {(race_eval['overlap'] >= 2).mean():.1%}")

    print(f"\nWorst predicted races (lowest overlap):")
    worst = race_eval.nsmallest(5, "overlap")[["season", "round", "overlap", "predicted_top3", "actual_top3"]]
    print(worst.to_string(index=False))

    print(f"\n=== Calibration Check ===")
    calibration = check_calibration(test_df, model, feature_columns)
    print(calibration.to_string())
    print("\n(avg_predicted and avg_actual should be reasonably close within each bin —")
    print(" large gaps mean the model's probabilities are over/under-confident)")


if __name__ == "__main__":
    main()