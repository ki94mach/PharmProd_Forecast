# V3A neural forecasting experiment

Isolated package for **per-SKU** neural forecasting experiments. Does not modify
V1 (`pkg.sales_forecasting` / `pkg.forecast`) or V2 classical models
(`pkg.ts_v2`).

## Contract reuse (from V2)

V3A reuses the V2 forecasting contract wherever possible:

- `ForecastWindow` / `make_forecast_window` from `pkg.ts_v2`
- Explicit `forecast_origin` (first target month; Shamsi `YYYYMM`)
- Training history strictly `date < forecast_origin`
- Exactly 15 target months (`horizon = 15`)
- Expanding historical backtesting and horizon-level MAE (engine TBD)
- Final full-history refit (engine TBD)
- No quarterly smoothing
- Common non-negative forecast constraint (postprocess TBD)
- Immutable forecast artifacts (persistence TBD)

Callers should pass monthly series already cut to `date < forecast_origin`
(e.g. via `pkg.ts_v2.prepare_monthly_series`).

## Training infrastructure

Provides architecture/config abstractions (A0–A6), recursive and DIRECT/MIMO
window builders, unique-observation fold-local scaling, chronological internal
train/validation for early stopping, sample eligibility gates, a shared
`NeuralTrainer`, deterministic seed helpers, and training metadata.

**Implemented:** A1 `small_recursive_lstm` (shared `NeuralTrainer` + recursive rollout).
**Not implemented yet:** A0 / A2–A6 graphs, backtest engine, results writers.

## Architecture candidates

| ID | Name | Target mode |
|----|------|-------------|
| A0 | `legacy_recursive_lstm` | RECURSIVE |
| A1 | `small_recursive_lstm` | RECURSIVE |
| A2 | `mimo_lstm` | DIRECT/MIMO |
| A3 | `stacked_mimo_lstm` | DIRECT/MIMO |
| A4 | `encoder_decoder_lstm` | DIRECT/MIMO |
| A5 | `bidirectional_mimo_lstm` | DIRECT/MIMO |
| A6 | `attention_bidirectional_lstm` | DIRECT/MIMO |

## Fold-local scaling

Scaler is fitted **exactly once** on the unique chronological raw observations
belonging to the internal training period (series indices covered by train
windows). Overlapping windows do not re-weight observations. Never fit on the
outer CV test horizon, internal validation-only observations, or future data.
The same univariate scaler transforms train and validation; predictions are
inverse-transformed before evaluation. `fit_on_series` supports a later full
pre-origin refit.

## Sample eligibility

- `mathematically_constructible`: at least one supervised window exists
- `eligible_for_training`: internal train windows ≥ `min_internal_train_windows`
  (default 8) **and** validation windows ≥ `min_internal_validation_windows`
  (default 2)

NeuralTrainer refuses train-only fits when useful validation cannot be built.
