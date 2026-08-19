"""M1 tuning constants.

Nothing here mutates pkg.benchmark constants or XGB_PARAMS.
"""
from __future__ import annotations

from pathlib import Path

from pkg.benchmark.config import (
    BUDGET_RESID_FEATURES,
    MIN_HISTORY_MONTHS,
    MIN_PRIOR_BUDGET_VINTAGES,
    MIN_TRAIN_ROWS,
    PRIMARY_ORIGINS,
    TS_RESID_FEATURES,
)

# ── Pre-PRIMARY cutoff ────────────────────────────────────────────────────────
# Inner tuning must use ONLY origins strictly before this value.
PRE_PRIMARY_CUTOFF: int = int(PRIMARY_ORIGINS[0])  # 140404

# ── Expected PRIMARY origins ──────────────────────────────────────────────────
EXPECTED_PRIMARY_ORIGINS: tuple[int, ...] = PRIMARY_ORIGINS
EXPECTED_N: int = 1877
EXPECTED_N_ORIGINS: int = 5

# ── Baseline sanity bounds — STOP if current F0 deviates beyond ───────────────
# These are the current-environment canonical F0 values (F3A config establishes
# 38.2848 / 36.5602 for this Python env; different from freeze-time 37.230 / 36.695).
# Use F3A-documented current-env values as reference.  Tolerance 0.10 absolute.
BASELINE_TS_WMAPE_REF: float = 38.2848
BASELINE_HUMAN_WMAPE_REF: float = 36.5602
BASELINE_STOP_TOL: float = 0.10  # |current_wmape - ref| > this → STOP

# ── Minimum inner folds to start a study ─────────────────────────────────────
MIN_INNER_FOLDS: int = 3

# ── Eligibility rules reused from benchmark.config ───────────────────────────
INNER_MIN_TRAIN_ROWS: int = MIN_TRAIN_ROWS          # 500
INNER_MIN_HISTORY_MONTHS: int = MIN_HISTORY_MONTHS  # 12
INNER_MIN_BUDGET_VINTAGES: int = MIN_PRIOR_BUDGET_VINTAGES  # 4 (Human only)

# ── Optuna settings ───────────────────────────────────────────────────────────
OPTUNA_SEED: int = 42
N_TRIALS: int = 40
N_JOBS_OPTUNA: int = 1
STUDY_NAME_TS: str = "m1_ts_f0_optuna"
STUDY_NAME_HUMAN: str = "m1_human_f0_optuna"

# ── Inner-fold XGB settings ───────────────────────────────────────────────────
INNER_N_ESTIMATORS: int = 3_000
INNER_EARLY_STOPPING_ROUNDS: int = 75
INNER_EVAL_METRIC: str = "mae"

# ── Fixed XGB params (never tuned, same as frozen F0 intent) ─────────────────
# IMPORTANT: this dict is NEW and is NOT XGB_PARAMS. It only captures the parts
# that M1 keeps fixed while Optuna searches the rest.
XGB_FIXED_PARAMS: dict = dict(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# ── F0 features (canonical, never modified) ───────────────────────────────────
F0_FEATURES: dict[str, tuple[str, ...]] = {
    "ts": tuple(TS_RESID_FEATURES),
    "human": tuple(BUDGET_RESID_FEATURES),
}

# ── Forecast columns by anchor ────────────────────────────────────────────────
FORECAST_COL: dict[str, str] = {
    "ts": "ts_forecast",
    "human": "budget_forecast",
}

# ── Train universes (matching benchmark Analysis B) ───────────────────────────
TRAIN_UNIVERSE: dict[str, str] = {
    "ts": "ts",
    "human": "budget",
}

# ── Inner universe origin columns ─────────────────────────────────────────────
INNER_ORIGIN_COL: dict[str, str] = {
    "ts": "ts_origin",
    "human": "budget_origin",
}


def m1_output_dir() -> Path:
    """src/data/results/m1_optuna/ — created on first access."""
    # parents[3] from src/pkg/research/tuning/config.py = src/
    p = Path(__file__).resolve().parents[3] / "data" / "results" / "m1_optuna"
    p.mkdir(parents=True, exist_ok=True)
    return p


def optuna_db_url() -> str:
    return f"sqlite:///{m1_output_dir() / 'optuna.db'}"


def docs_dir() -> Path:
    # parents[4] = project root
    return Path(__file__).resolve().parents[4] / "docs"
