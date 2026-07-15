# Project Scope — F1 Podium Predictor Plus

Written during Phase 1 (planning), before any modeling code. This file is the
source of truth for scope decisions — update it if a decision changes, don't
let scope drift silently through the notebooks.

## 1. What we are predicting

**Primary target:** Podium probability (top-3 finish) per driver, per race.

Not exact finishing position. Not win/no-win. Podium probability, because:
- It's more robust than exact position (real races have crashes/failures no model can predict)
- It matches the "Podium Predictor" framing directly
- It's naturally interpretable as a ranked list (top 3 by predicted probability)

## 2. Sub-systems and what each one outputs

| System | Predicts | Model type |
|---|---|---|
| Race outcome model | Podium probability per driver | Classifier (Random Forest / XGBoost) |
| Pit-stop simulator | Predicted finish under a given pit strategy | Regression + deterministic simulation |
| Safety car model | Probability of SC/VSC in a race/lap window | Classifier |

## 3. Training window decision

- **Training/testing data:** 2018–2024 seasons (pre-2026 regulation era)
- **2025:** held as buffer, not yet decided whether to include in training or hold out — revisit after Phase 2 data exploration
- **2026:** treated as a SEPARATE case, not blended with historical training data.
  Reason: 2026 introduced F1's biggest regulation overhaul in the sport's history
  (new power unit split, active aero, smaller/narrower cars, new tires, new fuel).
  Pre-2026 car performance data does not transfer. See Section 5.

## 4. Features planned (draft — refine during Phase 2)

**Per-lap features:**
- Tire compound, tire age (laps on current tire)
- Gap to car ahead / car behind
- Current position
- Sector time deltas vs. session best

**Per-race features:**
- Grid position
- Circuit type (street / permanent / high-speed) — circuit metadata, not FastF1 native
- Weather (dry/wet, track temp)
- Driver's recent form (avg finishing position, last 5 races)
- Team's recent car performance (avg qualifying gap to pole, last 5 races)
- Driver's historical performance at this specific circuit
- `regulation_era` — categorical: `2018-2021`, `2022-2025`, `2026+`

## 5. Handling the 2026 regulation change

Decision: do NOT train a single model blending 2018-2025 with 2026 data as if
they're equivalent. Instead:
- Historical model (2018-2024) is the validation/backtesting baseline — proves
  the modeling approach works within a stable era
- 2026 predictions come from a model trained ONLY on 2026 races as they
  accumulate week by week (a "cold start" problem, by design)
- Dashboard should show a confidence indicator that is explicitly lower early
  in 2026 and improves as more same-era races complete

## 6. Success criteria (how we'll know it's working)

- **Backtest metric:** top-3 accuracy on held-out 2024 races (was the actual
  podium finisher in our model's predicted top 3?)
- **Calibration check:** do predicted probabilities match real-world frequency
  (drivers predicted ~70% podium chance actually podium ~70% of the time)?
- Exact position match is NOT a success metric — too strict given real-world
  race randomness (crashes, mechanical failures, incidents)

## 7. Out of scope (for now)

- Qualifying result prediction (separate problem, not in v1)
- Constructor championship modeling
- Driver transfer/lineup change effects
- Multi-class exact-position prediction (podium probability only, v1)

## 8. Known data gaps

- **2018 Italian Grand Prix (round 14) is missing from the training data.**
  FastF1's source has broken/incomplete timing data for this specific session
  (confirmed via direct debugging — `session.load()` completes but lap timing
  data fails to reconstruct for all drivers, throwing `DataNotLoadedError`).
  This is an upstream data issue, not a bug in our pull script. Accepted as-is:
  147/148 races (99.3%) is more than sufficient for training, and this is not
  worth further investigation time.

## 9. System 3 (Safety Car Model) — a documented negative result

Race-level safety car occurrence (predicted from pre-race features: circuit
history, weather, grid competitiveness, DNF history, race distance) achieves
ROC AUC ~0.49 on held-out 2024 data — statistically indistinguishable from a
coin flip, and no better than always predicting the majority class.

**This was investigated, not just accepted blindly:**
- Started with 4 features (circuit SC history + weather) — AUC 0.490
- Added 3 more plausible features (grid competitiveness, circuit DNF
  history, race distance) — AUC 0.493, no meaningful improvement

**Conclusion:** with only 124 training races (a genuinely small sample) and
features that are all pre-race proxies for risk (not the actual triggering
events — a specific crash, a specific mechanical failure), race-level safety
car occurrence appears close to fundamentally unpredictable from information
available before the race starts. This is a legitimate finding, not a failed
implementation — it suggests SC events are dominated by in-race randomness
that pre-race data cannot capture.

**Path not taken (documented for future work):** reframing the target as
per-lap-window probability using the ~161,000-row lap-level dataset (vs. 148
race-level rows) might surface weak signal invisible at race-level
aggregation (e.g. early-lap incident risk, pit-window bunching). Not pursued
in this version — noted as a genuine v2 direction, not abandoned due to
difficulty.

**Impact on the overall system:** the dashboard's "chaos meter" (Safety Car
page) should honestly reflect this — show the ~72% base rate as a reference
point rather than presenting per-race predictions as if they were reliable,
OR clearly label the low confidence. Presenting an unreliable model's output
as confident would be misleading to anyone using the dashboard.

## 10. System 2 (Pit-Stop Simulator) — tire degradation data quality by compound

Tire degradation curves (lap_time_delta ~ TyreLife, quadratic, fit after
removing shared track-level trends via field-median-per-lap normalization)
have very different reliability by compound:

- **SOFT, MEDIUM, HARD: solid.** Clean, monotonically increasing degradation
  across the entire observed range (1-40 laps), 22,907 / 44,318 / 48,326
  laps respectively. These are the reliable foundation for the simulator.
- **INTERMEDIATE: usable, lower confidence.** Some noise mid-range (5,441
  laps), but overall trend is still correctly upward across the full range.
- **WET: documented as unreliable, not fixed further.** Only 372 clean laps
  across 7 seasons — even within the observed range (2-24 laps, no
  extrapolation involved), the fitted curve is non-monotonic (-0.03s → +0.62s
  → -0.60s), meaning the data itself can't support a stable curve. This is a
  genuine sample-size ceiling: wet races vary enormously race-to-race (rain
  intensity, drying speed, standing water location) in ways 372 laps can't
  average out. Investigated two rounds of confound-fixing (race-best
  normalization + lap_number covariate, then field-median-per-lap
  normalization) — neither resolved it, confirming this is a data limitation,
  not a fixable methodology issue.

**Decision:** build the pit-stop simulator on SOFT/MEDIUM/HARD (and
INTERMEDIATE with a lower-confidence flag). If WET tire strategy ever
surfaces in the dashboard, label it explicitly as low-confidence rather than
presenting it with the same authority as the dry-compound predictions.