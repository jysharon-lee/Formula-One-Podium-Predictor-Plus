# F1 Podium Predictor Plus

🏎️ An F1 podium predictor that doesn't just guess who wins — it simulates pit-stop strategy and safety car chaos to show *why*.

> **Status: v1 complete.** All three systems built, validated, and wired into a working dashboard. See `PROJECT_SCOPE.md` for the full technical record, including honestly-documented limitations.

## What this is

Three connected systems, wired into one dashboard, built on real FastF1 data across 149 races (2018-2024):

1. **Race Outcome Model** — predicts podium (top-3) probability per driver, per race. XGBoost classifier, validated on a held-out 2024 season with a proper time-based split (never trained on the data it's tested against). ROC AUC 0.928, average podium overlap 1.96/3 across the full held-out season.
2. **Pit-Stop Simulator** — tire degradation curves fit per compound (SOFT/MEDIUM/HARD validated as reliable; INTERMEDIATE usable with lower confidence; WET explicitly excluded as unreliable — see limitations below), powering a live "what if they pit now vs. wait N laps" comparison.
3. **Safety Car Model** — investigated honestly, found to perform no better than random guessing (ROC AUC ~0.49) even after two rounds of feature engineering. Documented as a genuine finding rather than hidden — see `PROJECT_SCOPE.md` Section 9.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Three pages:
- **Race Predictor** — pick any season/round from 2018-2024, see predicted podium probabilities vs. actual results
- **Strategy Explorer** — interactive pit-stop simulation with confidence labeling built in
- **Chaos Meter** — presents the safety car model's real limitation honestly, with historical base rates as the trustworthy reference instead

## Honest engineering, not just working code

This project treats a negative result as seriously as a positive one. A few things worth knowing before digging into the code:

- The Safety Car model doesn't work, and that's documented, not hidden — see `PROJECT_SCOPE.md` Section 9 for the investigation.
- Two real data confounds were found and fixed in the tire degradation modeling (fuel burn-off, then track-level effects like drying) — see Section 10.
- WET tire predictions are explicitly excluded from strategic use due to genuine data scarcity (372 clean laps across 7 seasons) — not a bug, a sample-size ceiling.
- A real off-by-one bug and a real categorical-encoding bug were caught during development via debugging scripts left in `src/` for reference (`debug_missing_race.py`, `debug_quali_gap.py`).

## Dataset

Live-pulled via [FastF1](https://docs.fastf1.dev) — no static dataset to download. FastF1 provides official F1 timing data (laps, tires, pit stops, weather, qualifying, safety car flags) going back to 2018.

```bash
pip install fastf1
```

See `src/pull_season_data.py` for the full multi-season pull (handles FastF1's rate limits automatically) or `src/test_data_pull.py` for a minimal single-race example.

## A note on the 2026 season

2026 introduced F1's biggest regulation overhaul in the sport's history — new power units, active aerodynamics, smaller/narrower cars, new tires, new fuel. Historical (pre-2026) performance data does not transfer directly to this season. This project's core models are trained and validated on the pre-2026 era (2018-2024) — see `PROJECT_SCOPE.md` Section 5 for how a future live-2026 extension would need to handle this.

## Project structure

```
f1-podium-predictor-plus/
├── PROJECT_SCOPE.md          # full technical record: decisions, findings, limitations
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                  # raw FastF1 pulls, gitignored (regenerate via src/pull_*.py)
│   └── processed/            # combined + feature-engineered datasets, gitignored
├── models/                   # trained models (race outcome, safety car, tire degradation curves)
├── dashboard/
│   └── app.py                # Streamlit dashboard, all 3 systems
├── src/
│   ├── pull_season_data.py             # multi-season race data pull (rate-limit-aware)
│   ├── pull_qualifying_data.py         # qualifying session data pull
│   ├── combine_data.py                 # merges per-race files into unified datasets
│   ├── build_circuit_lookup.py         # season/round to circuit name mapping
│   ├── circuit_metadata.py             # circuit type reference (street/permanent/etc.)
│   ├── feature_engineering.py          # core feature functions (recent form, team pace, etc.)
│   ├── train_race_outcome_model.py     # System 1
│   ├── evaluate_race_outcome_model.py  # per-race top-3 accuracy + calibration checks
│   ├── build_safety_car_features.py    # System 3 features (v1)
│   ├── build_safety_car_features_v2.py # System 3 features (v2, more predictors)
│   ├── train_safety_car_model.py       # System 3
│   ├── build_tire_degradation_model.py # System 2 - degradation curves
│   ├── pit_stop_simulator.py           # System 2 - the actual simulation logic
│   └── find_pit_crossover.py           # System 2 - crossover analysis
└── cache/                    # FastF1's local cache, gitignored
```

## Setup

```bash
git clone [repo-url]
cd f1-podium-predictor-plus
pip install -r requirements.txt

# Rebuild the data pipeline from scratch (takes several hours due to FastF1 rate limits):
python src/pull_season_data.py
python src/pull_qualifying_data.py
python src/combine_data.py
python src/build_circuit_lookup.py

# Build features and train models:
python src/feature_engineering.py
python src/train_race_outcome_model.py
python src/build_safety_car_features_v2.py
python src/train_safety_car_model.py
python src/build_tire_degradation_model.py

# Run the dashboard:
streamlit run dashboard/app.py
```

## Roadmap

- [x] Phase 1: Scope defined, project structure set up
- [x] Phase 2: Data pipeline - 149 races, 4 data types, across 2018-2024
- [x] Phase 2: Race outcome model - built and validated
- [x] Phase 2: Safety car model - built, investigated, honestly documented as unreliable
- [x] Phase 2: Pit-stop simulator - built, confounds fixed, crossover behavior validated
- [x] Phase 2: Dashboard - all 3 systems wired together, working
- [ ] Phase 3: Formal backtest report against 2024 season (informal validation done; see evaluate_race_outcome_model.py output)
- [ ] Future: Live validation against 2026 races as they complete (requires a new live-feature pipeline - 2026 is a new regulation era, see note above)

## License

MIT