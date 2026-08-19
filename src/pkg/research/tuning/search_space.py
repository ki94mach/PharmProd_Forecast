"""Exact search space for the M1 Optuna study.

Only the eight parameters specified in the design. Nothing else is tuned.
"""
from __future__ import annotations

from typing import Any

import optuna


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Return one hyperparameter configuration from the M1 search space.

    Exactly the eight parameters from the spec. No additions permitted.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 30.0, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.015, 0.15, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }


SEARCH_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "max_depth",
        "min_child_weight",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
    }
)
