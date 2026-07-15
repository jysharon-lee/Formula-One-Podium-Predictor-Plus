"""
Phase 2, Step 9 (cont.): Train the Safety Car Model (System 3).

Same time-based split philosophy as System 1 — train on 2018-2023, test on
2024 — for the same reason: honest evaluation on a season the model has
never seen.

Run locally:
    python src/train_safety_car_model.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os
import joblib

FEATURE_COLUMNS = ["circuit_sc_history", "circuit_dnf_history", "grid_competitiveness",
                   "num_laps", "was_wet", "avg_track_temp", "avg_air_temp"]
CATEGORICAL_COLUMNS = ["regulation_era"]


def prepare_data(df):
    df = df.copy()
    df["was_wet"] = df["was_wet"].astype(float)

    columns_before = set(df.columns)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    new_dummy_columns = sorted(set(df.columns) - columns_before)

    feature_cols = FEATURE_COLUMNS + new_dummy_columns
    X = df[feature_cols]
    y = df["had_safety_car"]

    return X, y, feature_cols


def time_based_split(df, test_season=2024):
    train_df = df[df["season"] < test_season]
    test_df = df[df["season"] == test_season]
    return train_df, test_df


def main():
    print("Loading safety car feature table (v2 — with added features)...")
    df = pd.read_parquet("data/processed/safety_car_features_v2.parquet")
    print(f"Loaded {len(df)} rows (races)")

    train_df, test_df = time_based_split(df, test_season=2024)
    print(f"\nTrain: {len(train_df)} races (seasons < 2024)")
    print(f"Test: {len(test_df)} races (season 2024)")

    X_train, y_train, feature_cols = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\nTraining XGBoost classifier on {len(feature_cols)} features...")
    print("Note: only ~118 training races total (much smaller than System 1's")
    print("~2900 driver-race rows) — keeping the model simple to avoid overfitting.")
    model = xgb.XGBClassifier(
        n_estimators=50,   # fewer trees than System 1 — small dataset, avoid overfitting
        max_depth=3,       # shallower — same reason
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n=== Evaluation on 2024 (held-out season) ===")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    # ROC AUC needs both classes present in y_test — guard against a season
    # where every single race had (or didn't have) a safety car
    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: undefined (only one class present in 2024 test data)")

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Baseline (always predict majority class): "
          f"{max(y_test.mean(), 1 - y_test.mean()):.2%}")

    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No SC", "Safety Car"], zero_division=0))

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\nFeature importance:")
    print(importance.to_string(index=False))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X_train.columns)},
                "models/safety_car_model.pkl")
    print(f"\nModel saved to models/safety_car_model.pkl")


if __name__ == "__main__":
    main()