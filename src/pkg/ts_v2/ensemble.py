"""Ensemble strategy evaluation from out-of-fold backtest predictions.

Ranking and weighting at each historical origin use only OOF rows from strictly
earlier origins — no future actuals or later-origin performance enter the mix.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, SelectionStrategy, TSForecastConfig
from pkg.ts_v2.metrics import aggregate_metrics, metrics_summary_row
from pkg.ts_v2.selection import simplicity_rank
from pkg.ts_v2.types import BacktestResult, EnsembleComparisonReport, ForecastResult

# Internal analysis names (report evaluates all four).
STRATEGY_BEST_SINGLE = "best_single_model"
STRATEGY_MEAN_TOP3 = "mean_top_3"
STRATEGY_MEDIAN_TOP3 = "median_top_3"
STRATEGY_INVERSE_MAE_TOP3 = "inverse_mae_weighted_top_3"

ALL_ENSEMBLE_STRATEGIES: tuple[str, ...] = (
    STRATEGY_BEST_SINGLE,
    STRATEGY_MEAN_TOP3,
    STRATEGY_MEDIAN_TOP3,
    STRATEGY_INVERSE_MAE_TOP3,
)

CONFIG_TO_INTERNAL: dict[SelectionStrategy, str] = {
    "best_model": STRATEGY_BEST_SINGLE,
    "top3_mean": STRATEGY_MEAN_TOP3,
    "top3_median": STRATEGY_MEDIAN_TOP3,
    "top3_inverse_mae": STRATEGY_INVERSE_MAE_TOP3,
}

INTERNAL_TO_CONFIG: dict[str, SelectionStrategy] = {
    v: k for k, v in CONFIG_TO_INTERNAL.items()
}

ENSEMBLE_PREDICTION_COLUMNS = (
    "product",
    "strategy",
    "origin",
    "target_date",
    "horizon",
    "actual",
    "prediction",
    "contributing_models",
)


def strategy_from_config(config: Optional[TSForecastConfig] = None) -> str:
    """Map :attr:`TSForecastConfig.selection_strategy` to an internal strategy name."""
    cfg = config or DEFAULT_CONFIG
    return CONFIG_TO_INTERNAL[cfg.selection_strategy]


def _expanding_model_scores(
    predictions: pd.DataFrame,
    *,
    product: str,
    origin: int,
    config: TSForecastConfig,
) -> dict[str, float]:
    """Mean horizon-level MAE per model using origins strictly before ``origin``."""
    prior = predictions.loc[
        (predictions["product"] == product) & (predictions["origin"] < int(origin))
    ]
    scores: dict[str, float] = {}
    if prior.empty:
        return scores
    for model, group in prior.groupby("model"):
        m = aggregate_metrics(group, config=config)
        sel = m["selection_mae"]
        if sel is not None and math.isfinite(float(sel)):
            scores[str(model)] = float(sel)
    return scores


def _models_at_origin(
    predictions: pd.DataFrame,
    *,
    product: str,
    origin: int,
) -> set[str]:
    sub = predictions.loc[
        (predictions["product"] == product) & (predictions["origin"] == int(origin))
    ]
    if sub.empty:
        return set()
    return {str(m) for m in sub["model"].unique()}


def _rank_models_for_origin(
    expanding_scores: Mapping[str, float],
    available: set[str],
    *,
    config: TSForecastConfig,
) -> list[str]:
    """Rank models available at an origin; lower score is better."""
    order = config.selection_simplicity_order
    top_k = int(config.ensemble_top_k)

    def sort_key(name: str) -> tuple[float, int, str]:
        score = expanding_scores.get(name)
        if score is None or not math.isfinite(float(score)):
            score_key = float("inf")
        else:
            score_key = float(score)
        return (score_key, simplicity_rank(name, order), name)

    ranked = sorted(available, key=sort_key)
    return ranked[: max(1, top_k)]


def _combine_predictions(
    model_predictions: Mapping[str, float],
    ranked_models: Sequence[str],
    expanding_scores: Mapping[str, float],
    strategy: str,
) -> tuple[float, tuple[str, ...]]:
    """Combine candidate model forecasts for one origin/horizon cell."""
    used = [m for m in ranked_models if m in model_predictions]
    if not used:
        return float("nan"), tuple()

    if strategy == STRATEGY_BEST_SINGLE:
        return float(model_predictions[used[0]]), (used[0],)

    values = np.array([float(model_predictions[m]) for m in used], dtype=float)

    if strategy == STRATEGY_MEAN_TOP3:
        return float(np.mean(values)), tuple(used)

    if strategy == STRATEGY_MEDIAN_TOP3:
        return float(np.median(values)), tuple(used)

    if strategy == STRATEGY_INVERSE_MAE_TOP3:
        weights: list[float] = []
        for model in used:
            mae = expanding_scores.get(model)
            if mae is None or not math.isfinite(float(mae)) or float(mae) <= 0.0:
                weights.append(1.0)
            else:
                weights.append(1.0 / float(mae))
        w = np.array(weights, dtype=float)
        if float(w.sum()) <= 0.0:
            w = np.ones_like(w)
        w = w / w.sum()
        return float(np.dot(w, values)), tuple(used)

    raise ValueError(f"unknown ensemble strategy {strategy!r}")


def combine_model_forecast_results(
    model_results: Mapping[str, ForecastResult],
    ranked_models: Sequence[str],
    model_scores: Mapping[str, float],
    strategy: str,
    *,
    output_name: str,
) -> ForecastResult:
    """Combine fresh constituent forecasts into one production :class:`ForecastResult`."""
    if not model_results:
        raise ValueError("model_results must be non-empty")
    if strategy not in ALL_ENSEMBLE_STRATEGIES:
        raise ValueError(f"unknown ensemble strategy {strategy!r}")

    reference = next(iter(model_results.values()))
    horizon = len(reference.predictions)
    target_dates = reference.target_dates
    if horizon != len(target_dates):
        raise ValueError("constituent forecast length mismatch")

    combined: list[float] = []
    used_overall: list[str] = []
    for i in range(horizon):
        cell_preds = {
            name: float(model_results[name].predictions[i])
            for name in ranked_models
            if name in model_results and i < len(model_results[name].predictions)
        }
        pred, used = _combine_predictions(cell_preds, ranked_models, model_scores, strategy)
        combined.append(float(pred))
        if i == 0:
            used_overall = list(used)

    return ForecastResult(
        model_name=output_name,
        predictions=tuple(combined),
        target_dates=tuple(int(d) for d in target_dates),
        horizons=tuple(range(1, horizon + 1)),
        metadata={
            "strategy": strategy,
            "constituent_models": tuple(used_overall),
            "ranked_models": tuple(ranked_models),
        },
    )


def rank_models_for_production(
    candidate_scores: Mapping[str, float],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[str]:
    """Rank models by full-CV ``selection_mae`` for production ensemble constituents."""
    cfg = config or DEFAULT_CONFIG
    available = {str(k) for k in candidate_scores.keys()}
    finite_scores = {
        str(k): float(v)
        for k, v in candidate_scores.items()
        if v is not None and math.isfinite(float(v))
    }
    return _rank_models_for_origin(finite_scores, available, config=cfg)


def build_ensemble_predictions(
    backtest: BacktestResult,
    product: str,
    strategy: str,
    *,
    config: Optional[TSForecastConfig] = None,
) -> pd.DataFrame:
    """Construct leakage-safe ensemble OOF predictions for one SKU."""
    cfg = config or DEFAULT_CONFIG
    if strategy not in ALL_ENSEMBLE_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {ALL_ENSEMBLE_STRATEGIES!r}, got {strategy!r}"
        )

    preds = backtest.predictions
    if preds is None or preds.empty:
        return pd.DataFrame(columns=list(ENSEMBLE_PREDICTION_COLUMNS))

    product_s = str(product)
    sub = preds.loc[preds["product"] == product_s].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(ENSEMBLE_PREDICTION_COLUMNS))

    rows: list[dict] = []
    for origin in sorted(int(o) for o in sub["origin"].unique()):
        expanding_scores = _expanding_model_scores(
            preds, product=product_s, origin=origin, config=cfg
        )
        available = _models_at_origin(preds, product=product_s, origin=origin)
        ranked = _rank_models_for_origin(expanding_scores, available, config=cfg)

        origin_frame = sub.loc[sub["origin"] == origin]
        for (horizon, target_date), cell in origin_frame.groupby(
            ["horizon", "target_date"], sort=True
        ):
            actual_vals = pd.to_numeric(cell["actual"], errors="coerce").dropna()
            if actual_vals.empty:
                continue
            actual = float(actual_vals.iloc[0])
            model_preds = {
                str(row.model): float(row.prediction)
                for row in cell.itertuples(index=False)
                if pd.notna(row.prediction)
            }
            ensemble_pred, contributors = _combine_predictions(
                model_preds,
                ranked,
                expanding_scores,
                strategy,
            )
            if not math.isfinite(ensemble_pred):
                continue
            rows.append(
                {
                    "product": product_s,
                    "strategy": strategy,
                    "origin": int(origin),
                    "target_date": int(target_date),
                    "horizon": int(horizon),
                    "actual": actual,
                    "prediction": ensemble_pred,
                    "contributing_models": contributors,
                }
            )

    return pd.DataFrame(rows, columns=list(ENSEMBLE_PREDICTION_COLUMNS))


def _metrics_for_strategy_predictions(
    predictions: pd.DataFrame,
    *,
    product: str,
    strategy: str,
    config: TSForecastConfig,
) -> dict:
    if predictions is None or predictions.empty:
        return {
            "product": product,
            "strategy": strategy,
            "selection_mae": float("nan"),
            "number_of_origins": 0,
            "number_of_predictions": 0,
            "evaluated_horizons": (),
            "max_evaluated_horizon": 0,
        }

    n_origins = int(predictions["origin"].nunique())
    horizons = tuple(sorted(int(h) for h in predictions["horizon"].unique()))
    cov = {
        "number_of_origins": n_origins,
        "number_of_predictions": int(len(predictions)),
        "evaluated_horizons": horizons,
        "max_evaluated_horizon": int(max(horizons)) if horizons else 0,
    }
    row = metrics_summary_row(
        product,
        strategy,
        predictions.rename(columns={"strategy": "model"}).assign(model=strategy),
        cov,
        config=config,
    )
    row["strategy"] = strategy
    return row


def evaluate_product_strategies(
    backtest: BacktestResult,
    product: str,
    *,
    strategies: Sequence[str] = ALL_ENSEMBLE_STRATEGIES,
    config: Optional[TSForecastConfig] = None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Build predictions and metric rows for each ensemble strategy on one SKU."""
    cfg = config or DEFAULT_CONFIG
    pred_by_strategy: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict] = []
    for strategy in strategies:
        frame = build_ensemble_predictions(backtest, product, strategy, config=cfg)
        pred_by_strategy[strategy] = frame
        metric_rows.append(
            _metrics_for_strategy_predictions(
                frame, product=str(product), strategy=strategy, config=cfg
            )
        )
    return pred_by_strategy, metric_rows


def compare_ensemble_strategies(
    backtest: BacktestResult,
    products: Optional[Iterable[str]] = None,
    *,
    strategies: Sequence[str] = ALL_ENSEMBLE_STRATEGIES,
    config: Optional[TSForecastConfig] = None,
) -> EnsembleComparisonReport:
    """Compare ensemble strategies across SKUs (analysis report; no production change)."""
    cfg = config or DEFAULT_CONFIG
    if products is None:
        if backtest.predictions is None or backtest.predictions.empty:
            empty = pd.DataFrame()
            return EnsembleComparisonReport(
                strategy_predictions={s: empty.copy() for s in strategies},
                strategy_metrics=empty,
                sku_comparison=empty,
                summary=empty,
            )
        product_list = sorted(backtest.predictions["product"].astype(str).unique())
    else:
        product_list = [str(p) for p in products]

    all_preds: dict[str, list[pd.DataFrame]] = {s: [] for s in strategies}
    metric_rows: list[dict] = []

    for product in product_list:
        pred_map, rows = evaluate_product_strategies(
            backtest, product, strategies=strategies, config=cfg
        )
        for strategy, frame in pred_map.items():
            if not frame.empty:
                all_preds[strategy].append(frame)
        metric_rows.extend(rows)

    strategy_predictions = {
        s: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=list(ENSEMBLE_PREDICTION_COLUMNS)
        )
        for s, frames in all_preds.items()
    }
    strategy_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()

    sku_comparison = _build_sku_comparison(strategy_metrics, config=cfg)
    summary = _build_summary(strategy_metrics, sku_comparison, config=cfg)

    return EnsembleComparisonReport(
        strategy_predictions=strategy_predictions,
        strategy_metrics=strategy_metrics,
        sku_comparison=sku_comparison,
        summary=summary,
    )


def _strategy_rank(name: str) -> int:
    try:
        return ALL_ENSEMBLE_STRATEGIES.index(str(name))
    except ValueError:
        return len(ALL_ENSEMBLE_STRATEGIES)


def _build_sku_comparison(
    strategy_metrics: pd.DataFrame,
    *,
    config: TSForecastConfig,
) -> pd.DataFrame:
    if strategy_metrics is None or strategy_metrics.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for product in sorted(strategy_metrics["product"].astype(str).unique()):
        sub = strategy_metrics.loc[strategy_metrics["product"] == product]
        row: dict = {"product": product}
        for _, srow in sub.iterrows():
            strategy = str(srow["strategy"])
            row[f"{strategy}_selection_mae"] = float(srow["selection_mae"])
        finite = {
            str(srow["strategy"]): float(srow["selection_mae"])
            for _, srow in sub.iterrows()
            if math.isfinite(float(srow["selection_mae"]))
        }
        if finite:
            best = min(
                finite,
                key=lambda k: (finite[k], _strategy_rank(k), k),
            )
            row["best_strategy_by_mae"] = best
            row["best_selection_mae"] = finite[best]
        else:
            row["best_strategy_by_mae"] = None
            row["best_selection_mae"] = float("nan")
        rows.append(row)

    wide = pd.DataFrame(rows)
    current = CONFIG_TO_INTERNAL[config.selection_strategy]
    if f"{current}_selection_mae" in wide.columns:
        wide["current_default_strategy"] = current
        wide["current_default_selection_mae"] = wide[f"{current}_selection_mae"]
    return wide


def _build_summary(
    strategy_metrics: pd.DataFrame,
    sku_comparison: pd.DataFrame,
    *,
    config: TSForecastConfig,
) -> pd.DataFrame:
    if strategy_metrics is None or strategy_metrics.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for strategy in ALL_ENSEMBLE_STRATEGIES:
        sub = strategy_metrics.loc[strategy_metrics["strategy"] == strategy]
        maes = pd.to_numeric(sub["selection_mae"], errors="coerce").dropna()
        row = {
            "strategy": strategy,
            "config_name": INTERNAL_TO_CONFIG.get(strategy, strategy),
            "n_skus": int(len(sub)),
            "mean_selection_mae": float(maes.mean()) if not maes.empty else float("nan"),
            "median_selection_mae": float(maes.median()) if not maes.empty else float("nan"),
            "is_production_default": CONFIG_TO_INTERNAL[config.selection_strategy] == strategy,
        }
        if not sku_comparison.empty and "best_strategy_by_mae" in sku_comparison.columns:
            row["sku_win_count"] = int((sku_comparison["best_strategy_by_mae"] == strategy).sum())
        else:
            row["sku_win_count"] = 0
        rows.append(row)

    return pd.DataFrame(rows)


def assert_ensemble_no_future_ranking_leakage(
    backtest: BacktestResult,
    product: str,
    origin: int,
    *,
    config: Optional[TSForecastConfig] = None,
) -> None:
    """Diagnostic: ranking at ``origin`` ignores same-origin and future rows."""
    cfg = config or DEFAULT_CONFIG
    scores = _expanding_model_scores(
        backtest.predictions, product=str(product), origin=int(origin), config=cfg
    )
    if not scores:
        return
    current = backtest.predictions.loc[
        (backtest.predictions["product"] == str(product))
        & (backtest.predictions["origin"] == int(origin))
    ]
    if current.empty:
        return
    for model in scores:
        future = backtest.predictions.loc[
            (backtest.predictions["product"] == str(product))
            & (backtest.predictions["model"] == model)
            & (backtest.predictions["origin"] >= int(origin))
        ]
        if future.empty:
            continue
        prior_only = backtest.predictions.loc[
            (backtest.predictions["product"] == str(product))
            & (backtest.predictions["model"] == model)
            & (backtest.predictions["origin"] < int(origin))
        ]
        from pkg.ts_v2.metrics import aggregate_metrics

        expected = aggregate_metrics(prior_only, config=cfg)["selection_mae"]
        if not math.isfinite(float(expected)):
            continue
        if not math.isclose(float(scores[model]), float(expected), rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(
                f"ranking score for {model!r} at origin {origin} does not match prior-only MAE"
            )
