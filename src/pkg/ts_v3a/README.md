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

## This foundation step

Provides architecture/config abstractions (A0–A6), recursive and DIRECT/MIMO
window builders, fold-local scaling, chronological internal train/validation
for early stopping, deterministic seed helpers, and training metadata.

**Not implemented yet:** LSTM/Keras layers, backtest engine, results writers.

## Architecture candidates

| ID | Name | Target mode |
|----|------|-------------|
| A0 | `legacy_recursive_lstm` | RECURSIVE |
| A1 | `recursive_lstm` | RECURSIVE |
| A2 | `mimo_lstm` | DIRECT/MIMO |
| A3 | `stacked_mimo_lstm` | DIRECT/MIMO |
| A4 | `encoder_decoder_lstm` | DIRECT/MIMO |
| A5 | `bidirectional_mimo_lstm` | DIRECT/MIMO |
| A6 | `attention_bidirectional_lstm` | DIRECT/MIMO |

## Fold-local scaling

Scaler is fitted **only** on values inside the internal-train supervised
windows for the current historical fold. Never fit on full SKU history, the
outer CV test horizon, or future observations. Inverse-transform predictions
before evaluation.
