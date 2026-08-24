# V2 time-series forecasting baseline

Separate package from production V1 (`pkg.sales_forecasting` / `pkg.forecast`).
Scaffold only: config, types, and module boundaries. Models and CLI wiring come later.

## Principles

1. **Explicit forecast origin** — callers pass the first forecast month (Shamsi `YYYYMM`). Training uses months strictly before that origin. Origin is never inferred as `max(history) + 1`.
2. **No implicit last-month removal** — V1 drops `sale_series[:-1]` as an incomplete month. V2 does not silently discard the last available month; as-of cuts are explicit (`date < origin`).
3. **No preprocessing leakage** — transforms and scalers fit only on training history available at each origin, never on holdout months or post-origin data.
4. **Multi-origin / multi-horizon backtesting** — selection evaluates real forecast origins and horizons `1..H`, not a single scaled 1-step RMSE roll.
5. **Final full-history refit** — after selection, the winning model is refit on all history before the production origin, then used for the `H`-step forecast (V1 reuses the last selection fit).
6. **Raw forecast output** — emit monthly point forecasts without V1 quarterly `redistribute_smoothing`. Downstream packaging may still round or clip via config.
7. **V1 stays untouched** — do not migrate or mutate V1 modules, frozen benchmarks, or production CLI in this package’s early steps.

## Layout

| Module | Role |
|--------|------|
| `config.py` | `TSForecastConfig` defaults (`forecast_horizon`, `selection_metric`, …) |
| `types.py` | Origins, series, forecasts, selection / engine result types |
| `dates.py` | Origin parsing and Shamsi month helpers |
| `data.py` | Series construction as-of an origin (stub) |
| `models/` | Model registry / protocol (no implementations yet) |
| `backtest.py` | Multi-origin / multi-horizon evaluation (stub) |
| `selection.py` | Metric-based winner pick |
| `engine.py` | Orchestration: backtest → select → full-history refit (stub) |

## Defaults

```text
forecast_horizon = 15
selection_metric = "mae"
seasonal_period = 12
min_train_months = 12
nonnegative_forecasts = True
```
