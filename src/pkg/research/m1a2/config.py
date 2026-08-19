"""M1A2 constants — fixed n_estimators=200, no early stopping."""
from __future__ import annotations

from pathlib import Path

from pkg.benchmark.config import PRIMARY_ORIGINS, TS_RESID_FEATURES

PRE_PRIMARY_CUTOFF = int(PRIMARY_ORIGINS[0])  # 140404
PRIMARY_ORIGINS_LOCKED = tuple(int(x) for x in PRIMARY_ORIGINS)
EXPECTED_N = 1877
EXPECTED_N_ORIGINS = 5

SEED = 42
OPTUNA_TRIALS = 40
OPTUNA_N_JOBS = 1
OPTUNA_STUDY_NAME = "m1a2_ts_fixed200_structural"

FIXED_N_ESTIMATORS = 200

# Same 9 inner origins as M1R — hard gate before tuning
EXPECTED_INNER_ORIGINS: tuple[int, ...] = (
    140201,
    140204,
    140207,
    140210,
    140304,
    140306,
    140307,
    140310,
    140401,
)

# M1R deterministic F0 reference (exploratory; STOP if predictions diverge)
M1R_F0_WMAPE_REF = 38.64273633337519
PREDICTION_REPEAT_TOL = 1e-9

XGB_DETERMINISTIC_FIXED = {
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": 1,
    "tree_method": "hist",
}

ANCHOR = "ts"
FORECAST_COL = "ts_forecast"
FEATURES = tuple(TS_RESID_FEATURES)
TRAIN_UNIVERSE = "ts"

M1R_RESULTS_DIR = Path(__file__).resolve().parents[3] / "data" / "results" / "m1r_reproducibility"


def output_dir() -> Path:
    p = Path(__file__).resolve().parents[3] / "data" / "results" / "m1a2_fixed200"
    p.mkdir(parents=True, exist_ok=True)
    return p


def optuna_db_url() -> str:
    return f"sqlite:///{output_dir() / 'optuna.db'}"


def docs_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "m1a2_fixed200_optuna.md"
