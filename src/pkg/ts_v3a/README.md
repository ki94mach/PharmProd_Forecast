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
- Expanding historical backtesting and horizon-level MAE (`run_outer_backtest`)
- Final full-history refit (engine TBD)
- No quarterly smoothing
- Common non-negative forecast constraint (postprocess TBD)
- Immutable screening experiment artifacts (`pkg.ts_v3a.persistence`)

Callers should pass monthly series already cut to `date < forecast_origin`
(e.g. via `pkg.ts_v2.prepare_monthly_series`).

## Outer expanding backtest

`run_outer_backtest` / `backtest_product_architectures` evaluate A0–A5 under
the V2 historical contract for every SKU × architecture × outer origin × seed:

1. history strictly `date < outer_origin` (V2 `prepare_monthly_series`)
2. architecture resolution from that history only (A0 adaptive tiers)
3. supervised windows → chronological internal train/val
4. `FoldScaler` on unique raw internal-train observations
5. fresh `NeuralTrainer` fit; forecast h1..h15; inverse-transform to raw units
6. score outer actuals only after prediction; discard the fitted model

Insufficient / ineligible history records a typed unavailable fold (no zero
forecasts).

### Multi-seed evaluation

Default seeds are `(41, 42, 43)`. Each seed is an independent training run; all
seed OOF rows are retained (`predictions` / `seed_predictions` with
`prediction_kind="seed"`).

A seed-ensemble forecast is then formed per architecture × origin:

`prediction_seed_ensemble = mean(prediction across successful seeds)`

Architecture screening requires `min_successful_seeds = 3` by default. If fewer
seeds succeed, that origin’s ensemble is marked unavailable
(`insufficient_successful_seeds`) rather than averaging a partial set.

Result tables:

- `seed_metrics` — per-seed horizon MAE / mean_horizon_MAE / RMSE / bias / WMAPE
- `ensemble_predictions` / `ensemble_metrics` — ensemble OOF and primary screening metrics
- `stability` — prediction std by horizon, MAE std/min/max/CV across seeds
- `metrics` — alias of `ensemble_metrics`

Primary architecture metric remains equal-weight `mean_horizon_MAE` on the
seed-ensemble forecasts (not row-weighted MAE).

### Screening persistence

Immutable screening outputs land under:

`src/data/ts_v3a/screening/{experiment_id}/`

with `manifest.json`, seed OOF parquet, fold metadata, architecture /
horizon / seed metrics CSVs, and `failures.csv`. Incomplete work uses
`src/data/ts_v3a/screening/.incomplete/{experiment_id}/` then promotes on finalize.

After each SKU finishes, the CLI writes a recoverable checkpoint
(`checkpoint.json` + cumulative parquets/CSVs) under `.incomplete/`. Re-run
with the same `--experiment-id` and `--resume` (or let auto-resume detect an
existing incomplete checkpoint) to skip finished products.

A deterministic `config_hash` covers scientific settings only. Completed
experiments refuse overwrite; incompatible hashes raise
`ExperimentConfigConflictError` (no silent append). Not wired to production
or backfill runners.

### Smoke / screening CLI

```bash
python -m pkg.ts_v3a.screen \
  --products product1,product2 \
  --origins 140401,140501 \
  --architectures a0,a1,a2,a3,a4,a5 \
  --seeds 41,42,43 \
  --output src/data/ts_v3a/screening
```

Runs a small product×origin set through A0–A5 under the same V2/V3A contract,
prints a per-fold runtime table, checkpoints after each product, and persists
immutable screening artifacts on completion. Short aliases `a0`–`a5` are
accepted. Does **not** select a winning architecture.
Use `--dry-run` to resolve inputs and print `config_hash` without training.
Optional `--max-epochs` / `--early-stopping-patience` bound smoke runtime.
Use `--resume` with the same `--experiment-id` after a crash or kill.

## Training infrastructure

Provides architecture/config abstractions (A0–A6), recursive and DIRECT/MIMO
window builders, unique-observation fold-local scaling, chronological internal
train/validation for early stopping, sample eligibility gates, a shared
`NeuralTrainer`, deterministic seed helpers, and training metadata.

**Implemented:** A0–A5, outer expanding CV with multi-seed ensemble evaluation,
immutable screening persistence (`V3A_VERSION = "v3a"`), smoke/screening CLI,
offline analysis (`python -m pkg.ts_v3a.analysis`).
**Not implemented yet:** A6 graph, architecture selection, full-history refit,
production/backfill integration.

### Offline analysis

```bash
python -m pkg.ts_v3a.analysis build-panel --target-n 20
python -m pkg.ts_v3a.analysis smoke-metrics --experiment-id <id> --output ...
python -m pkg.ts_v3a.analysis recigen --experiment-id <id> --output ...
python -m pkg.ts_v3a.analysis report --experiment-id <id>
```

Panel products: `src/pkg/benchmark/universes/v3a_architecture_validation_products.csv`
(MVP subset; frozen `src/data/benchmarks/v1/raw/sales.parquet` only).

## Architecture candidates

| ID | Name | Target mode |
|----|------|-------------|
| A0 | `legacy_adaptive_recursive_lstm` | RECURSIVE |
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
