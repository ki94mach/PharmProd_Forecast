# V2 time-series forecasting baseline

Separate package from production V1 (`pkg.sales_forecasting` / `pkg.forecast`).
Scaffold only: config, types, and module boundaries. Models and CLI wiring come later.

## Principles

1. **Explicit forecast origin** — callers pass the first forecast month (Shamsi `YYYYMM`). Training uses months strictly before that origin. Origin is never inferred as `max(history) + 1`.
2. **No implicit last-month removal** — V1 drops `sale_series[:-1]` as an incomplete month. V2 does not silently discard the last available month; as-of cuts are explicit (`date < origin`).
3. **No preprocessing leakage** — models see **raw sales units**. V2 does not apply MinMax, Yeo–Johnson, or ADF-triggered transforms (V1 fitted those, including on the incomplete last month). There is no global transform before backtest.
4. **Multi-origin / multi-horizon backtesting** — selection evaluates real forecast origins and horizons `1..H`, not a single scaled 1-step RMSE roll.
5. **Final full-history refit** — after selection, the winning model is refit on all history before the production origin, then used for the `H`-step forecast (V1 reuses the last selection fit).
6. **Raw forecast output** — emit monthly point forecasts without V1 quarterly `redistribute_smoothing`. Downstream packaging may still round or clip via config.
7. **V1 stays untouched** — do not migrate or mutate V1 modules, frozen benchmarks, or production CLI in this package’s early steps.

## Date contract

CLI/business origin is Shamsi `YYYYMM` (e.g. `140501`).

Use `make_forecast_window(140501)` → `ForecastWindow`:

| Field | Meaning |
|-------|---------|
| `forecast_origin` | First target month (`140501`) |
| `training_end` | Last inclusive train month (`140412` = origin − 1) |
| `target_dates` | Exactly `H` months: h1=`140501` … h15=`140603` |
| `horizons` | `(1, …, H)` |

**Training rule:** `date < forecast_origin` only. No `series[:-1]`. Models do not decide to skip a month. Shamsi ↔ pandas `+62100` / `-62100` lives only in `dates.py`.

## Series preparation

`prepare_monthly_series` returns `PreparedSeries` (raw units):

- sum duplicate product/month rows
- truncate `date < forecast_origin`
- optional **activity start**: first month with sales **> `activity_start_min_sales`** (default **5.0**, V1-compatible). V1’s docstring says “first non-zero sale” but the live code uses `sales > 5`, almost certainly to ignore tiny pre-launch / residual shipments. Set the option to `None` to disable.
- contiguous monthly Shamsi index from `first_active_month` through `last_training_month` (`origin − 1`)
- `missing_month_policy`: `"zero"` (V1 `asfreq.fillna(0)` for **values**) or `"missing"` (NaN in values). **`is_missing_month` is always set** so calendar gaps are not conceptually identical to explicit observed zeros.

## Layout

| Module | Role |
|--------|------|
| `config.py` | `TSForecastConfig` defaults (`forecast_horizon`, `selection_metric`, …) |
| `types.py` | Origins, `ForecastWindow`, series, forecasts, selection / engine result types |
| `dates.py` | `make_forecast_window`, Shamsi helpers, `+62100` / `-62100` |
| `data.py` | `prepare_monthly_series`: origin cut, monthly grid, gap flags |
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
activity_start_min_sales = 5.0   # V1 sales > 5; None disables
missing_month_policy = "zero"    # V1-compatible fill; gaps still flagged
```
