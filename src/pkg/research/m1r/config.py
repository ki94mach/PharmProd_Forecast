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
OPTUNA_STUDY_NAME = "m1r_ts_f0_deterministic"

INNER_N_ESTIMATORS = 3000
INNER_EARLY_STOPPING_ROUNDS = 75
INNER_EVAL_METRIC = "mae"

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


def output_dir() -> Path:
    p = Path(__file__).resolve().parents[3] / "data" / "results" / "m1r_reproducibility"
    p.mkdir(parents=True, exist_ok=True)
    return p


def optuna_db_url() -> str:
    return f"sqlite:///{output_dir() / 'optuna.db'}"


def docs_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "m1r_deterministic_optuna.md"

