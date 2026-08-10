# Residual Prediction Benchmark v0 (Legacy)

Frozen snapshot of the XGBoost residual-correction MVP as of the `residual_prediction.ipynb` experiment.

- **Active experiment:** [`../residual_prediction.ipynb`](../residual_prediction.ipynb)
- **Frozen notebook (with outputs):** [`residual_prediction_v0_benchmark.ipynb`](residual_prediction_v0_benchmark.ipynb)

Do not treat this snapshot as the active experiment. It exists so later methodology changes can be compared against a reproducible baseline.

---

## Methodology caveat (preliminary results)

These results are **preliminary**. The train/test split is by **target actualized month (`date`)**, not by **forecast origin**. The same forecast quarter file can therefore contribute rows to both train and test when its window spans the date cutoff. Subsequent work should evaluate origin-based (or other leakage-safe) splits before treating metrics as production-grade.

---

## Target

```text
residual = sales - forecast
forecast_adj = max(0, forecast + predicted_residual)
```

Horizon filter: `1 <= horizon <= 15` (`FORECAST_HORIZON_MONTHS = 15`).

---

## Train / test split

- Split key: distinct actualized Shamsi months in `date` (not forecast origin / `qrt`)
- `TEST_DATE_FRACTION = 0.2` → last 20% of sorted unique months
- Recorded run:
  - 25 distinct months → test last **5** months
  - `TEST_DATES`: `140412`, `140501`, `140502`, `140503`, `140504`
  - Train: **4228** rows (`140304` .. `140411`)
  - Test: **1041** rows (`140412` .. `140504`)
  - `feat_model`: **5269** rows (10 rows dropped for NaN residual/forecast/sales)

Incomplete Shamsi month `140505` is excluded from modeling.

---

## XGBoost parameters

```python
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
```

Two models:

- `xgb_baseline` on baseline features
- `xgb_baseline_plus_business` on baseline + business features

---

## Baseline feature set

1. `forecast`
2. `horizon`
3. `month` (`date % 100`)
4. `quarter` (`((month - 1) // 3) + 1`)
5. `sales_lag_1` (sales at origin − 1)
6. `sales_lag_2` (sales at origin − 2)
7. `sales_lag_3` (sales at origin − 3)
8. `sales_lag_12` (sales at origin − 12)
9. `sales_roll3` (mean of lag 1/2/3)
10. `model_enc`
11. `field_enc`
12. `form_enc`
13. `provider_enc`

Missing sales lags are filled with `0` before split.

---

## Business feature set

Window per origin: inclusive Shamsi months `origin-6` .. `origin-1` (`last_complete_6m`).

1. `event_count_6m`
2. `activity_count_6m`
3. `event_per_sales` = `event_count_6m / sales_roll3` (0 if roll3 is 0/NaN)
4. `activity_per_sales` = `activity_count_6m / sales_roll3` (same)

`ALL_FEATURES = BASELINE_FEATURES + BUSINESS_FEATURES` (17 features).

---

## Horizon sample weighting

Training only:

```python
sample_weight = 1.0 / horizon   # clipped at horizon >= 1
```

Examples: h=1 → 1.0, h=3 → 1/3, h=12 → 1/12, h=15 → 1/15.

---

## Evaluation metrics

Evaluated on **test sales** vs:

- `forecast` → `base_ts`
- `forecast_adj_base` → `xgb_baseline`
- `forecast_adj_biz` → `xgb_baseline_plus_business`

| Metric | Definition |
| --- | --- |
| RMSE | `sqrt(mean_squared_error(y, ŷ))` |
| MAE | `mean_absolute_error(y, ŷ)` |
| MAPE | mean `\|y−ŷ\|/\|y\| * 100` over rows with `y != 0` |
| WMAPE | `sum(\|y−ŷ\|) / sum(\|y\|) * 100` |

MAPE is inflated by near-zero sales months; WMAPE is the more interpretable percentage error.

---

## Overall metrics (recorded)

| model | RMSE | MAE | MAPE | WMAPE |
| --- | ---: | ---: | ---: | ---: |
| base_ts | 27034.75 | 9125.20 | 4124.96 | 65.42 |
| xgb_baseline | 18856.38 | 6800.40 | 3514.43 | 48.75 |
| xgb_baseline_plus_business | 18755.84 | 6730.82 | 3240.01 | 48.25 |

Best overall RMSE: `xgb_baseline_plus_business` (**18755.84**) vs `base_ts` (**27034.75**) → **~30.6%** RMSE reduction.

---

## Per-horizon metrics (RMSE, recorded)

Only RMSE was displayed in the notebook; MAE/MAPE/WMAPE are also computed in `by_horizon` inside the frozen notebook. Horizons **2** and **14** have no test rows in this run.

| horizon | base (TS) | xgb_base | xgb_biz |
| ---: | ---: | ---: | ---: |
| 1 | 27722.56 | 22097.40 | 21889.92 |
| 3 | 32631.97 | 9140.78 | 9437.59 |
| 4 | 36496.47 | 12732.46 | 12398.09 |
| 5 | 24494.40 | 14265.98 | 14127.11 |
| 6 | 21594.39 | 16109.59 | 15944.30 |
| 7 | 34003.69 | 22278.22 | 22150.88 |
| 8 | 19734.55 | 15433.10 | 15328.64 |
| 9 | 20996.75 | 16776.22 | 16632.64 |
| 10 | 25802.17 | 21759.19 | 21673.89 |
| 11 | 23507.31 | 17378.35 | 17719.86 |
| 12 | 28133.09 | 22506.16 | 22718.84 |
| 13 | 32137.67 | 29049.85 | 28643.84 |
| 15 | 19545.40 | 14584.84 | 14442.98 |

---

## Reproducibility notes

- Forecast inputs: allowed quarter CSVs under `src/data/results/{qrt}/{qrt}_total_forecast.csv` (`1403Q1`–`1403Q4`, `1404Q2`–`1404Q4`, `1405Q2`)
- Product scope: `TARGET_GENERIC_EN`
- This benchmark is notebook-only; it does not write models or predictions back to `results/`
