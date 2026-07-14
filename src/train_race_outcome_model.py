"""
Phase 2, Step 8: Train the Race Outcome Model (System 1).

Trains an XGBoost classifier to predict podium probability, using a
TIME-BASED train/test split — train on 2018-2023, test on 2024 — per
PROJECT_SCOPE.md Section 6. This is deliberate: randomly shuffling train/test
would let the model "see the future" (e.g. train on a mix of early and late
2024 races), which isn't the honest real-world scenario of predicting a race
you haven't seen yet.

Run locally:
    python src/train_race_outcome_model.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix
)
import os
import joblib

FEATURE_COLUMNS = [
    "grid_position", "recent_form", "team_pace_trend",
    "circuit_history", "was_wet", "avg_track_temp", "avg_air_temp",
]
CATEGORICAL_COLUMNS = ["team", "regulation_era"]  # circuit_name excluded — too many
                                                    # unique values relative to data size;
                                                    # circuit_history already captures
                                                    # circuit-specific signal per driver


def prepare_data(df):
    """Prepare features and target, encoding categoricals as XGBoost expects."""
    df = df.copy()

    # was_wet is currently True/False/NaN — convert to 1/0/NaN for XGBoost
    df["was_wet"] = df["was_wet"].astype(float)

    # Track columns BEFORE dummy encoding, so we can identify exactly which
    # columns get_dummies created — using a prefix check (e.g. "starts with
    # team_") is fragile: our own feature 'team_pace_trend' starts with
    # "team_" too and would get wrongly caught by that filter, creating a
    # duplicate column. Set difference avoids this entirely.
    columns_before = set(df.columns)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    new_dummy_columns = sorted(set(df.columns) - columns_before)

    feature_cols = FEATURE_COLUMNS + new_dummy_columns
    X = df[feature_cols]
    y = df["podium"]

    return X, y, feature_cols


def time_based_split(df, test_season=2024):
    """Split by season, not randomly — train on everything before test_season,
    test on test_season itself. This is the honest validation scenario:
    predicting a season the model has never seen."""
    train_df = df[df["season"] < test_season]
    test_df = df[df["season"] == test_season]
    return train_df, test_df


def main():
    print("Loading feature table...")
    df = pd.read_parquet("data/processed/features.parquet")
    print(f"Loaded {len(df)} rows")

    train_df, test_df = time_based_split(df, test_season=2024)
    print(f"\nTrain: {len(train_df)} rows (seasons < 2024)")
    print(f"Test: {len(test_df)} rows (season 2024)")

    X_train, y_train, feature_cols = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)

    # Align columns — test set might be missing a category that only
    # appeared in training data (e.g. a team's one-hot column), or vice versa
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\nTraining XGBoost classifier on {len(feature_cols)} features...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n=== Evaluation on 2024 (held-out season) ===")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC AUC: {auc:.3f}")
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No podium", "Podium"]))

    print(f"\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance — worth checking this makes intuitive sense
    # (grid_position and team_pace_trend should rank highly)
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\nTop 10 most important features:")
    print(importance.head(10).to_string(index=False))

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X_train.columns)},
                "models/race_outcome_model.pkl")
    print(f"\nModel saved to models/race_outcome_model.pkl")


if __name__ == "__main__":
    main()