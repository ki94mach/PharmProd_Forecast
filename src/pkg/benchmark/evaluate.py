"""Metrics and ``backtest(model, origins, products)`` for benchmark v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pkg.benchmark.config import (
    MIN_HISTORY_MONTHS,
    MIN_PRIOR_BUDGET_VINTAGES,
    MIN_TRAIN_ROWS,
)
from pkg.benchmark.dataset import (
    BenchmarkDataset,
    filter_products,
    horizon_bucket,
    load_benchmark,
    prep_lags,
    resolve_origins,
)
from pkg.benchmark.models import FROZEN_NAMES, TRAIN_UNIVERSE, ModelSpec, predict_frozen


def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100.0)


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def metrics_block(y_true, y_pred, model_name: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": model_name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


@dataclass
class BacktestResult:
    """Outputs of a single ``backtest`` call."""

    model_name: str
    overall: pd.DataFrame
    by_origin: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    origins: list = field(default_factory=list)
    universe: str = "matched"


def _train_slice(
    name_or_callable: ModelSpec,
    train_universe: Optional[str],
    O: int,
    ts_u: pd.DataFrame,
    bud_u: pd.DataFrame,
    matched_u: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if callable(name_or_callable):
        key = train_universe or "budget"
    elif isinstance(name_or_callable, str):
        key = TRAIN_UNIVERSE.get(name_or_callable)
        if key is None:
            return None
    else:
        raise TypeError(f"model must be str or callable, got {type(name_or_callable)}")

    if key == "ts":
        return ts_u.loc[ts_u["target_date"].astype(int) < O].copy()
    if key == "budget":
        return bud_u.loc[bud_u["target_date"].astype(int) < O].copy()
    if key == "matched":
        return matched_u.loc[matched_u["target_date"].astype(int) < O].copy()
    raise ValueError(f"Unknown train_universe: {key!r}")


def _fold_eligible_primary(train_bud: pd.DataFrame) -> bool:
    if train_bud.empty:
        return False
    months = int(train_bud["target_date"].nunique())
    vintages = int(train_bud["budget_origin"].nunique())
    rows = len(train_bud)
    return (
        months >= MIN_HISTORY_MONTHS
        and vintages >= MIN_PRIOR_BUDGET_VINTAGES
        and rows >= MIN_TRAIN_ROWS
    )


def backtest(
    model: ModelSpec,
    origins: Optional[Iterable[int]] = None,
    products: Optional[Sequence[str]] = None,
    *,
    dataset: Optional[BenchmarkDataset] = None,
    root: Optional[Path] = None,
    universe: str = "matched",
    train_universe: Optional[str] = None,
    eligibility: str = "primary",
) -> BacktestResult:
    """Rolling-origin OOS evaluation on the frozen benchmark.

    Parameters
    ----------
    model :
        Frozen name (``"ts"``, ``"human"``, ``"ts_xgb"``, ``"human_xgb"``,
        ``"integrated"``, bias/Ridge names) or a callable
        ``(train_df, test_df) -> forecasts`` aligned to ``test_df``.
    origins :
        Test origins. Default = PRIMARY origins from the v1 manifest.
    products :
        Optional product subset applied to **test** rows only. Training still
        uses the full frozen train universe (Analysis B recipe).
    universe :
        ``"matched"`` (Analysis B) or ``"budget"`` (Analysis A).
    train_universe :
        For custom callables only: ``"ts"`` | ``"budget"`` | ``"matched"``.
        Default ``"budget"`` (Human+XGB recipe).
    eligibility :
        ``"primary"`` applies the temporally mature filter on Budget train
        diagnostics; ``"all"`` keeps every fold with non-empty train/test.

    Returns
    -------
    BacktestResult with overall / by_origin / by_horizon metrics and row-level preds.
    """
    ds = dataset or load_benchmark(root)
    # Train universes stay full (frozen recipe). ``products`` filters TEST only.
    ts_u = prep_lags(ds.ts_universe)
    bud_u = prep_lags(ds.budget_universe)
    matched_u = prep_lags(ds.matched_universe)

    if universe == "matched":
        test_panel = filter_products(matched_u, products)
        origin_col = "origin"
    elif universe == "budget":
        test_panel = filter_products(bud_u, products)
        origin_col = "budget_origin"
        # ensure budget_forecast naming
        if "budget_forecast" not in test_panel.columns and "forecast" in test_panel.columns:
            test_panel = test_panel.rename(columns={"forecast": "budget_forecast"})
    else:
        raise ValueError("universe must be 'matched' or 'budget'")

    if isinstance(model, str):
        if model not in FROZEN_NAMES:
            raise ValueError(
                f"Unknown model {model!r}. Frozen names: {sorted(FROZEN_NAMES)}"
            )
        model_name = model
        # Anchors that need columns from matched panel
        if universe == "budget" and model in {"ts", "ts_xgb", "integrated"}:
            raise ValueError(
                f"Model {model!r} requires universe='matched' (needs TS columns)"
            )
    elif callable(model):
        model_name = getattr(model, "__name__", "custom")
    else:
        raise TypeError(f"model must be str or callable, got {type(model)}")

    use_primary = eligibility == "primary"
    requested_origins = resolve_origins(origins, ds, use_primary=use_primary)

    pred_parts = []
    fold_rows = []

    for O in sorted(int(o) for o in requested_origins):
        test = test_panel.loc[test_panel[origin_col].astype(int) == O].copy()
        if test.empty:
            continue

        # Always compute Budget train diagnostics for eligibility
        train_bud = bud_u.loc[bud_u["target_date"].astype(int) < O].copy()
        if eligibility == "primary" and not _fold_eligible_primary(train_bud):
            continue

        train = _train_slice(model, train_universe, O, ts_u, bud_u, matched_u)
        if train is not None and train.empty:
            continue
        if train is not None:
            assert int(train["target_date"].max()) < O, (
                f"leakage: train.target_date.max not < origin={O}"
            )

        if isinstance(model, str):
            preds = predict_frozen(model, train, test)
        else:
            if train is None:
                train = train_bud
            preds = np.asarray(model(train, test), dtype=float)
            if len(preds) != len(test):
                raise ValueError(
                    f"custom model returned {len(preds)} preds for {len(test)} test rows"
                )

        fold = test.copy()
        fold["prediction"] = preds
        fold["actual"] = fold["sales"].astype(float)
        fold["test_origin"] = int(O)
        fold["train_rows"] = 0 if train is None else len(train)
        fold["train_rows_budget"] = len(train_bud)
        fold["prior_budget_vintages"] = (
            int(train_bud["budget_origin"].nunique()) if len(train_bud) else 0
        )
        fold["historical_months_budget"] = (
            int(train_bud["target_date"].nunique()) if len(train_bud) else 0
        )
        pred_parts.append(fold)
        fold_rows.append(
            {
                "origin": int(O),
                "test_rows": len(test),
                "test_products": int(test["product"].nunique()),
                "train_rows": 0 if train is None else len(train),
                "train_rows_budget": len(train_bud),
                "prior_budget_vintages": int(train_bud["budget_origin"].nunique())
                if len(train_bud)
                else 0,
                "historical_months_budget": int(train_bud["target_date"].nunique())
                if len(train_bud)
                else 0,
            }
        )

    if not pred_parts:
        raise RuntimeError(
            f"No backtest folds for model={model_name!r} origins={requested_origins}"
        )

    predictions = pd.concat(pred_parts, ignore_index=True)
    if "horizon_bucket" not in predictions.columns:
        predictions["horizon_bucket"] = predictions["horizon"].map(horizon_bucket)

    overall = pd.DataFrame(
        [metrics_block(predictions["actual"], predictions["prediction"], model_name)]
    )
    by_origin_rows = []
    for o, g in predictions.groupby("test_origin"):
        m = metrics_block(g["actual"], g["prediction"], model_name)
        m["origin"] = int(o)
        by_origin_rows.append(m)
    by_horizon_rows = []
    for h, g in predictions.groupby("horizon"):
        m = metrics_block(g["actual"], g["prediction"], model_name)
        m["horizon"] = int(h)
        by_horizon_rows.append(m)

    return BacktestResult(
        model_name=model_name,
        overall=overall,
        by_origin=pd.DataFrame(by_origin_rows),
        by_horizon=pd.DataFrame(by_horizon_rows),
        predictions=predictions,
        fold_diagnostics=pd.DataFrame(fold_rows),
        origins=sorted(predictions["test_origin"].astype(int).unique().tolist()),
        universe=universe,
    )


def scoreboard(
    models: Optional[Sequence[ModelSpec]] = None,
    **backtest_kwargs,
) -> pd.DataFrame:
    """Run several frozen models and stack overall metrics."""
    if models is None:
        models = ["ts", "human", "ts_xgb", "human_xgb", "integrated"]
    rows = []
    for m in models:
        res = backtest(m, **backtest_kwargs)
        rows.append(res.overall.iloc[0].to_dict())
    return pd.DataFrame(rows)
