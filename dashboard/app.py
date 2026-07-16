"""
F1 Podium Predictor Plus — Dashboard

Three pages, wiring together all three systems:
1. Race Predictor — System 1 (Race Outcome Model), explored against the
   historical feature dataset (2018-2024) we already built and validated.
2. Strategy Explorer — System 2 (Pit-Stop Simulator), interactive "what if"
   pit-lap simulation.
3. Chaos Meter — System 3 (Safety Car Model), presented HONESTLY given its
   ROC AUC ~0.49 — the historical base rate is shown as the primary,
   trustworthy number, with the ML prediction clearly caveated as unreliable
   per PROJECT_SCOPE.md Section 9.

Run locally:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Allow importing from src/ when running via `streamlit run dashboard/app.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pit_stop_simulator import (
    load_degradation_curves, simulate_pit_decision,
    RELIABLE_COMPOUNDS, CAUTION_COMPOUNDS, EXCLUDED_COMPOUNDS
)

st.set_page_config(page_title="F1 Podium Predictor Plus", page_icon="🏎️", layout="wide")


@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")


@st.cache_resource
def load_race_outcome_model():
    return joblib.load("models/race_outcome_model.pkl")


@st.cache_resource
def load_tire_curves():
    return load_degradation_curves()


@st.cache_data
def load_safety_car_features():
    return pd.read_parquet("data/processed/safety_car_features_v2.parquet")


def prepare_race_features(race_df, feature_columns):
    """Encode a race's features to match the model's training-time columns.

    IMPORTANT: does NOT use pd.get_dummies() directly on race_df. Reason:
    get_dummies' drop_first logic depends on which categories are PRESENT
    in the data it's called on — for a single race (one team, one
    regulation_era), it would drop whatever category happens to be the only
    one present, which often does NOT match what was dropped during
    training (which saw many races/categories at once). This would silently
    misencode regulation_era or team for single-race predictions. Instead,
    we explicitly check each known training-time dummy column against this
    race's actual category value.
    """
    df = race_df.copy()
    df["was_wet"] = df["was_wet"].astype(float)

    result = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col.startswith("team_"):
            team_name = col[len("team_"):]
            result[col] = (df["team"] == team_name).astype(int)
        elif col.startswith("regulation_era_"):
            era_name = col[len("regulation_era_"):]
            result[col] = (df["regulation_era"] == era_name).astype(int)
        elif col in df.columns:
            result[col] = df[col]
        else:
            result[col] = 0  # feature not present for this race, default to 0

    return result[feature_columns]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🏎️ F1 Podium Predictor Plus")
page = st.sidebar.radio("Navigate", ["Race Predictor", "Strategy Explorer", "Chaos Meter"])

st.sidebar.markdown("---")
sidebar_caption_text = (
    "Built on real FastF1 data, 2018-2024 (149 races). "
    "Predictions explore HISTORICAL races the model has already seen results "
    "for — this is a validation/exploration tool, not a live upcoming-race "
    "predictor yet."
)
st.sidebar.caption(sidebar_caption_text)

# ============================================================
# PAGE 1: RACE PREDICTOR (System 1)
# ============================================================
if page == "Race Predictor":
    st.title("Race Outcome Predictor")
    st.caption("System 1 — validated ROC AUC 0.928 on held-out 2024 season")

    features_df = load_features()
    saved_model = load_race_outcome_model()
    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]

    col1, col2 = st.columns(2)
    with col1:
        season = st.selectbox("Season", sorted(features_df["season"].unique(), reverse=True))
    with col2:
        available_rounds = sorted(
            features_df[features_df["season"] == season]["round"].unique()
        )
        round_num = st.selectbox("Round", available_rounds)

    race_data = features_df[
        (features_df["season"] == season) & (features_df["round"] == round_num)
    ].copy()

    if len(race_data) == 0:
        st.warning("No data for this race.")
    else:
        circuit_name = race_data["circuit_name"].iloc[0]
        st.subheader(f"{circuit_name} — {season}")

        X = prepare_race_features(race_data, feature_columns)
        race_data["predicted_proba"] = model.predict_proba(X)[:, 1]

        display_df = race_data[["driver", "team", "grid_position", "predicted_proba", "podium"]].copy()
        display_df = display_df.sort_values("predicted_proba", ascending=False)
        display_df["predicted_proba"] = (display_df["predicted_proba"] * 100).round(1)
        display_df.columns = ["Driver", "Team", "Grid", "Podium Probability (%)", "Actually Podiumed"]
        display_df["Actually Podiumed"] = display_df["Actually Podiumed"].map({1: "✅", 0: ""})

        st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)

        predicted_top3 = set(display_df.head(3)["Driver"])
        actual_top3 = set(race_data[race_data["podium"] == 1]["driver"])
        overlap = len(predicted_top3 & actual_top3)

        st.metric("Predicted vs. Actual Podium Overlap", f"{overlap} / 3")
        if season < 2024:
            training_data_warning_text = (
                "⚠️ This race was in the model's TRAINING data — treat this "
                "match as illustrative, not a fair test of accuracy. "
                "Select a 2024 race for an honest out-of-sample view."
            )
            st.caption(training_data_warning_text)

# ============================================================
# PAGE 2: STRATEGY EXPLORER (System 2)
# ============================================================
elif page == "Strategy Explorer":
    st.title("Pit-Stop Strategy Explorer")
    st.caption("System 2 — 'what if' simulation using validated tire degradation curves")

    curves = load_tire_curves()

    col1, col2, col3 = st.columns(3)
    with col1:
        compound = st.selectbox("Current Compound", list(curves.keys()))
    with col2:
        current_age = st.slider("Current Tire Age (laps)", 1, 40, 15)
    with col3:
        window = st.slider("Laps to Simulate", 5, 40, 15)

    if compound in EXCLUDED_COMPOUNDS:
        excluded_warning_text = (
            f"⚠️ {compound} tire data is too limited (only 372 clean laps across "
            f"7 seasons) to produce a reliable simulation. Results below should "
            f"NOT be trusted for strategic decisions — shown for exploration only."
        )
        st.error(excluded_warning_text)
    elif compound in CAUTION_COMPOUNDS:
        st.warning(f"⚠️ {compound} predictions are lower-confidence (smaller sample size than dry compounds).")

    result = simulate_pit_decision(curves, compound=compound, current_tyre_age=current_age,
                                    laps_to_simulate=window)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stay Out — Total Time Cost", f"{result['stay_out_total_delta']:+.2f}s")
    with col2:
        st.metric("Pit Now — Total Time Cost", f"{result['pit_now_total_delta']:+.2f}s")

    if result["recommendation"] == "PIT NOW":
        st.success(f"### Recommendation: {result['recommendation']} "
                   f"(saves {-result['time_difference_seconds']:.2f}s)")
    else:
        st.info(f"### Recommendation: {result['recommendation']} "
                f"(pitting would cost {result['time_difference_seconds']:.2f}s more)")

    st.caption(f"Confidence: {result['confidence']}")

    if result["warnings"]:
        with st.expander("⚠️ Extrapolation warnings"):
            for w in result["warnings"]:
                st.text(w)

    st.markdown("---")
    strategy_caption_text = (
        "Note: this simulator models tire degradation cost only — it does NOT "
        "account for traffic, track position value, or undercut/overcut "
        "race-craft effects. Real strategy calls weigh those factors too."
    )
    st.caption(strategy_caption_text)

# ============================================================
# PAGE 3: CHAOS METER (System 3)
# ============================================================
elif page == "Chaos Meter":
    st.title("Safety Car Chaos Meter")
    st.caption("System 3 — presented honestly given validated ROC AUC ~0.49")

    chaos_meter_warning_text = (
        "⚠️ **Honest limitation, documented in PROJECT_SCOPE.md Section 9:** "
        "our safety car prediction model performs no better than random "
        "guessing (ROC AUC 0.49-0.50) on held-out data, even after adding "
        "more features. This appears to be a genuine data limitation — safety "
        "cars are driven by in-race randomness (crashes, mechanical failures) "
        "that isn't visible in pre-race information. The historical base rate "
        "below is a more honest number than any per-race prediction we could "
        "currently offer."
    )
    st.error(chaos_meter_warning_text)

    sc_features = load_safety_car_features()
    overall_rate = sc_features["had_safety_car"].mean()

    st.metric("Overall Safety Car Rate (2018-2024)", f"{overall_rate:.1%}")

    st.subheader("Safety Car Rate by Circuit (Historical)")
    circuit_rates = sc_features.groupby("circuit_name")["had_safety_car"].agg(["mean", "count"])
    circuit_rates.columns = ["SC Rate", "Races Observed"]
    circuit_rates["SC Rate"] = (circuit_rates["SC Rate"] * 100).round(0).astype(int).astype(str) + "%"
    circuit_rates = circuit_rates.sort_values("Races Observed", ascending=False)
    st.dataframe(circuit_rates, use_container_width=True)

    chaos_caption_text = (
        "This table shows ACTUAL historical frequency — a more trustworthy "
        "reference point than a per-race ML prediction, given the documented "
        "model limitation above."
    )
    st.caption(chaos_caption_text)