"""Optuna objective for one anchor: pooled inner-fold WMAPE.

The objective is computed purely from PRE-PRIMARY folds. PRIMARY data never
enters this function.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import optuna

from pkg.research.tuning.config import FORECAST_COL, XGB_FIXED_PARAMS
from pkg.research.tuning.fit import fit_inner_fold, predict_inner_fold
from pkg.research.tuning.folds import InnerFold
from pkg.research.tuning.search_space import suggest_params


def _wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.abs(actual).sum())
    if denom == 0.0:
        return float("nan")
    return float(np.abs(actual - pred).sum() / denom * 100.0)


def _pooled_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _pooled_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _pooled_bias(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred - actual))


def make_objective(
    anchor: str,
    features: Sequence[str],
    folds: list[InnerFold],
):
    """Return a closure suitable for optuna.Study.optimize.

    Pooled WMAPE across ALL inner folds is the single minimization objective.
    Also records per-trial diagnostics as user attributes.
    """
    feat_cols = list(features)
    forecast_col = FORECAST_COL[anchor]

    def objective(trial: optuna.Trial) -> float:
        tuned = suggest_params(trial)

        all_actuals: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        wmape_by_origin: dict[int, float] = {}
        best_iter_by_origin: dict[int, int] = {}

        for fold in folds:
            model, best_iter = fit_inner_fold(
                anchor, feat_cols, fold.train, fold.val, tuned
            )
            preds = predict_inner_fold(anchor, feat_cols, model, fold.val)
            actual = fold.val["sales"].astype(float).to_numpy()

            all_actuals.append(actual)
            all_preds.append(preds)
            wmape_by_origin[fold.origin] = _wmape(actual, preds)
            best_iter_by_origin[fold.origin] = best_iter

        concat_actual = np.concatenate(all_actuals)
        concat_pred = np.concatenate(all_preds)
        pooled = _wmape(concat_actual, concat_pred)

        # Diagnostics stored as user attributes (never used for selection)
        origin_wmapes = list(wmape_by_origin.values())
        trial.set_user_attr("wmape_by_origin", json.dumps(wmape_by_origin))
        trial.set_user_attr(
            "median_origin_wmape",
            float(np.median(origin_wmapes)) if origin_wmapes else float("nan"),
        )
        trial.set_user_attr(
            "worst_origin_wmape",
            float(np.max(origin_wmapes)) if origin_wmapes else float("nan"),
        )
        trial.set_user_attr("pooled_mae", _pooled_mae(concat_actual, concat_pred))
        trial.set_user_attr("pooled_rmse", _pooled_rmse(concat_actual, concat_pred))
        trial.set_user_attr("pooled_bias", _pooled_bias(concat_actual, concat_pred))
        trial.set_user_attr(
            "best_iteration_by_origin", json.dumps(best_iter_by_origin)
        )
        best_iters = list(best_iter_by_origin.values())
        trial.set_user_attr(
            "median_best_iteration",
            float(np.median(best_iters)) if best_iters else float("nan"),
        )
        trial.set_user_attr("n_inner_origins", len(folds))
        trial.set_user_attr("n_validation_rows", int(len(concat_actual)))

        return pooled

    return objective
