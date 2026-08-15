"""Locked constants for benchmark v1 (copied from residual_prediction.ipynb)."""
from __future__ import annotations

from pathlib import Path

# Incomplete Shamsi months excluded from sales, forecasts, and training
INCOMPLETE_SHAMSI_MONTHS = frozenset({140505})

# Products with misaligned / stale forecast windows
EXCLUDED_ODD_COVERAGE_PRODUCTS = frozenset(
    {
        "Nanojade 90 Old",
        "Nanojade 180 Old",
        "Nanojade 360 Old",
        "Kidi Mab",
        "Alvocade 3.5",
        "Aryoseven 2.4",
        "Reditux 500",
    }
)

ALLOWED_FORECAST_QRTS = (
    "1401Q1",
    "1401Q3",
    "1401Q4",
    "1402Q1",
    "1402Q2",
    "1402Q3",
    "1402Q4",
    "1403Q1",
    "1403Q2",
    "1403Q3",
    "1403Q4",
    "1404Q1",
    "1404Q2",
    "1404Q3",
    "1404Q4",
    "1405Q1",
    "1405Q2",
)
EXCLUDED_EMPTY_FORECAST_QRTS = ("1401Q2",)
FORECAST_HORIZON_MONTHS = 15

# PRIMARY eligibility (Analysis A/B)
MIN_TRAIN_ROWS = 500
MIN_HISTORY_MONTHS = 12
MIN_PRIOR_BUDGET_VINTAGES = 4

# Analysis B PRIMARY origins from the freeze run (n=1877)
PRIMARY_ORIGINS = (140404, 140407, 140410, 140501, 140504)

HORIZON_BUCKETS = (
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-12", 7, 12),
    ("13-15", 13, 15),
)

CLEAN_QUANT_FEATURES = [
    "horizon",
    "month",
    "quarter",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
    "model_enc",
    "field_enc",
    "form_enc",
    "provider_enc",
]
TS_RESID_FEATURES = ["ts_forecast"] + CLEAN_QUANT_FEATURES
BUDGET_RESID_FEATURES = ["budget_forecast"] + CLEAN_QUANT_FEATURES
INTEGRATED_FEATURES = [
    "budget_forecast",
    "ts_forecast",
    "human_adjustment",
] + CLEAN_QUANT_FEATURES

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

RIDGE_ALPHA = 1.0
BIAS_AF_EPS = 1.0

# Expected Analysis B PRIMARY WMAPE (freeze run)
EXPECTED_ANALYSIS_B_PRIMARY = {
    "ts": 43.883479,
    "human": 40.042928,
    "ts_xgb": 37.230140,
    "human_xgb": 36.694750,
    "integrated": 40.143204,
    "n": 1877,
    "n_origins": 5,
}

# Expected Analysis A PRIMARY (negative controls + Human+XGB)
EXPECTED_ANALYSIS_A_PRIMARY = {
    "human": 40.056187,
    "bias_global": 43.219084,
    "bias_product": 45.775277,
    "bias_product_horizon": 46.199196,
    "af_ratio": 42.863905,
    "ridge": 41.209985,
    "human_xgb": 36.707273,
    "n": 1923,
    "n_origins": 5,
}

BENCHMARK_VERSION = "v1"
PANEL_FILES = (
    "ts_universe.parquet",
    "budget_universe.parquet",
    "matched_universe.parquet",
)
RAW_FILES = (
    "sales.parquet",
    "line_budget.parquet",
    "product_attrs.parquet",
)


def default_benchmark_root() -> Path:
    """src/data/benchmarks/v1 relative to the installed pkg."""
    # pkg/benchmark/config.py -> pkg -> src -> data/benchmarks/v1
    src_dir = Path(__file__).resolve().parents[2]
    return src_dir / "data" / "benchmarks" / BENCHMARK_VERSION


def manifest_path() -> Path:
    return Path(__file__).resolve().parent / "v1_manifest.json"
