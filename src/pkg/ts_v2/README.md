# V2 time-series forecasting baseline

Separate package from production V1 (`pkg.sales_forecasting` / `pkg.forecast`).
Scaffold only: config, types, and module boundaries. Models and CLI wiring come later.

## Principles

1. **Explicit forecast start** — callers pass the first *delivered* forecast month (Shamsi `YYYYMM`). Origin is never inferred as `max(history) + 1`.
2. **Production partial / bridge month** — `current_partial_month = forecast_start − 1` is excluded from training (`last_complete_month = forecast_start − 2`). Models predict `H+1` internal steps; `run_model` discards the bridge and delivers `H` horizons from `forecast_start`.
3. **No preprocessing leakage** — models see **raw sales units**. V2 does not apply MinMax, Yeo–Johnson, or ADF-triggered transforms. There is no global transform before backtest.
4. **Multi-origin / multi-horizon backtesting** — selection evaluates real forecast origins and horizons `1..H`, not a single scaled 1-step RMSE roll.
5. **Final full-history refit** — after selection, the winning model is refit on all history through `last_complete_month`, then used for the production forecast.
6. **Raw forecast output** — emit monthly point forecasts without V1 quarterly `redistribute_smoothing`. Downstream packaging may still round or clip via config.
7. **V1 stays untouched** — do not migrate or mutate V1 modules, frozen benchmarks, or production CLI in this package’s early steps.

## Date contract

CLI/business start is Shamsi `YYYYMM` (e.g. `140501`).

Use `make_forecast_window(140501)` → production `ForecastWindow`:

| Field | Meaning |
|-------|---------|
| `forecast_origin` | First *delivered* target (`140501` = `forecast_start`) |
| `current_partial_month` | Bridge month (`140412` = start − 1) |
| `training_end` | Last inclusive train month (`140411` = start − 2) |
| `target_dates` | Exactly `H` delivered months: h1=`140501` … h15=`140603` |
| `internal_target_dates` | Bridge + delivered (`H+1`) |
| `horizons` | `(1, …, H)` |

**Training rule:** `date <= training_end`. Screening bake-offs for A2–A5 use `make_screening_forecast_window` (`training_end = origin − 1`, no bridge). Shamsi month math uses `shamsi_add_months` only; `+62100` / `-62100` lives only in `dates.py`.

## Series preparation

`prepare_monthly_series` returns `PreparedSeries` (raw units):

- sum duplicate product/month rows
- truncate `date <= training_end`
- optional **activity start**: first month with sales **> `activity_start_min_sales`** (default **5.0**, V1-compatible). V1’s docstring says “first non-zero sale” but the live code uses `sales > 5`, almost certainly to ignore tiny pre-launch / residual shipments. Set the option to `None` to disable.
- contiguous monthly Shamsi index from `first_active_month` through `last_training_month` (`training_end`)
- `missing_month_policy`: `"zero"` (V1 `asfreq.fillna(0)` for **values**) or `"missing"` (NaN in values). **`is_missing_month` is always set** so calendar gaps are not conceptually identical to explicit observed zeros.

## Model interface

Every candidate is called identically by backtest and engine:

```text
outcome = run_model(model, train_series, window)
```

`fit(train_series)` then `predict(horizon, target_dates)` must return a `ForecastResult`:

- `predictions` length equals requested horizon
- `target_dates` are copied unchanged (horizon 1 = origin; models must not skip a month)
- no rounding, quarterly smoothing, or ad-hoc bias inside the model
- failures become `ModelFailure` so one broken candidate does not abort the SKU

Register factories on `pkg.ts_v2.models.REGISTRY` and list names in `TSForecastConfig.candidate_models`.

Intermittent candidates `croston_sba` and `tsb` compete in backtesting like other models;
`PreparedSeries` exposes `zero_month_proportion` / `average_inter_demand_interval`.
V2 leaves them as diagnostics only; V2.1 uses them to gate `croston_sba` / `tsb`
(see `docs/ts_v21_intermittent_eligibility.md`).

## Layout

| Module | Role |
|--------|------|
| `config.py` | `TSForecastConfig` defaults (`forecast_horizon`, `selection_metric`, …) |
| `types.py` | Origins, `ForecastWindow`, series, forecasts, selection / engine result types |
| `dates.py` | `make_forecast_window`, Shamsi helpers, `+62100` / `-62100` |
| `data.py` | `prepare_monthly_series`: origin cut, monthly grid, gap flags |
| `models/` | Interface + baselines + library + croston_sba / tsb |
| `backtest.py` | Multi-origin / multi-horizon evaluation (stub) |
| `selection.py` | Metric-based winner pick |
| `engine.py` | Orchestration: backtest → select → full-history refit (stub) |

## Defaults

```text
forecast_horizon = 15
selection_metric = "mae"
seasonal_period = 12
seasonal_enable_after_months = 24   # seasonality iff n > 24 (V1 rule)
min_train_months = 12
nonnegative_forecasts = True
activity_start_min_sales = 5.0   # V1 sales > 5; None disables
missing_month_policy = "zero"    # V1-compatible fill; gaps still flagged
prophet_changepoint_prior_scale = 0.05
prophet_growth = "linear"
croston_alpha = 0.1
croston_beta = None   # defaults to croston_alpha
tsb_alpha = 0.1
tsb_beta = 0.1
candidate_models = (
  "naive", "seasonal_naive", "drift",
  "auto_arima", "ets", "prophet",
  "croston_sba", "tsb",
)
```

Library adapters train in raw units, forecast exactly the requested `target_dates`,
and use one shared seasonal threshold / Prophet CPS for CV and final refit.
They do not pad horizons, apply ×0.8, or use `freq="M"`.

Croston SBA / TSB use fixed config smoothing parameters (not tuned on future data)
and emit a constant per-period demand rate for every horizon.

## Gap / zero-month audit (no policy change yet)

``Flat_Fact_Sale`` rows are **present months**; absent months are **unknown**, not
proven zeros. Explicit ``sales == 0`` rows are observed zero shipments.

Audit helpers: ``pkg.ts_v2.gap_audit`` (`run_gap_audit`, `write_gap_audit_report`).
Write-up: [`docs/ts_v2_gap_audit.md`](../../docs/ts_v2_gap_audit.md).

**Do not change ``missing_month_policy`` until audit results are reviewed.**

## Product identity (audit pending)

V1 keys products on **`ProductTitleEN`**. V2 target key is **`Dim.Product.ID_INT`**
via **`Flat_Fact_Sale.FKProduct`** (see peer sales / F3C inventory SQL). Title
remains descriptive metadata only.

**Do not switch loaders or backtest keys until live DQ passes.** Audit helpers:
`pkg.ts_v2.product_identity`; proposed FK sales SQL: `pkg.db.query.sales_v2`;
write-up: [`docs/ts_v2_product_identity.md`](../../docs/ts_v2_product_identity.md).
