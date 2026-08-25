"""Model selection for V2 (metric-driven, leakage-safe).

Selection uses mean horizon-level MAE from out-of-fold backtest predictions
(:mod:`pkg.ts_v2.backtest` / :mod:`pkg.ts_v2.metrics`). It never uses in-sample
fit statistics (AIC, etc.) or RMSE for the final winner. After a winner is
chosen, the production path refits on full history via :mod:`pkg.ts_v2.engine`.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import BacktestResult, ForecastOrigin, ProductSelectionResult, SelectionResult


def simplicity_rank(
    model_name: str,
    order: Sequence[str],
) -> int:
    """Lower rank means simpler / preferred on ties."""
    try:
        return order.index(str(model_name))
    except ValueError:
        return len(order)


def pick_winner_with_tiebreak(
    scores: Mapping[str, float],
    *,
    tolerance: float,
    simplicity_order: Sequence[str],
) -> tuple[str, bool]:
    """Return ``(winner, tie_break_applied)`` from eligible selection scores."""
    eligible = {
        str(name): float(score)
        for name, score in scores.items()
        if score is not None and math.isfinite(float(score))
    }
    if not eligible:
        raise ValueError("scores must contain at least one finite value")

    best_score = min(eligible.values())
    tol = max(0.0, float(tolerance))
    near_best = [
        name
        for name, score in eligible.items()
        if score <= best_score + tol
    ]
    near_best.sort(key=lambda name: (simplicity_rank(name, simplicity_order), name))
    winner = near_best[0]
    tie_break = len(near_best) > 1
    return winner, tie_break


def select_best_model(
    scores: Mapping[str, float],
    *,
    product: str,
    origin: Optional[ForecastOrigin] = None,
    config: Optional[TSForecastConfig] = None,
) -> SelectionResult:
    """Pick the model with the lowest selection metric score (with tie-break)."""
    cfg = config or DEFAULT_CONFIG
    if not scores:
        raise ValueError("scores must be non-empty")
    best_name, _ = pick_winner_with_tiebreak(
        scores,
        tolerance=cfg.selection_tie_tolerance,
        simplicity_order=cfg.selection_simplicity_order,
    )
    return SelectionResult(
        product=product,
        origin=origin,
        best_model_name=best_name,
        scores={k: float(v) for k, v in scores.items() if math.isfinite(float(v))},
        metric=cfg.selection_metric,
    )


def _failure_reasons(
    failures: pd.DataFrame,
    *,
    product: str,
    model: str,
) -> str:
    if failures is None or failures.empty:
        return ""
    sub = failures.loc[
        (failures["product"] == product) & (failures["model"] == model)
    ]
    if sub.empty:
        return ""
    parts = []
    for row in sub.itertuples(index=False):
        reason = str(getattr(row, "reason", "") or "")
        err_type = str(getattr(row, "error_type", "") or "")
        if err_type and err_type not in reason:
            parts.append(f"{err_type}: {reason}" if reason else err_type)
        elif reason:
            parts.append(reason)
    # Stable dedupe preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "; ".join(uniq)


def _metrics_row(
    metrics: pd.DataFrame,
    *,
    product: str,
    model: str,
) -> Optional[pd.Series]:
    if metrics is None or metrics.empty:
        return None
    sub = metrics.loc[(metrics["product"] == product) & (metrics["model"] == model)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _horizon_maes_from_row(
    row: pd.Series,
    *,
    forecast_horizon: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for h in range(1, int(forecast_horizon) + 1):
        col = f"mae_h{h}"
        if col in row.index:
            val = row[col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                out[h] = float(val)
    return out


def _eligible_for_selection(
    row: pd.Series,
    *,
    config: TSForecastConfig,
) -> tuple[bool, str]:
    n_origins = int(row.get("number_of_origins", 0) or 0)
    n_preds = int(row.get("number_of_predictions", 0) or 0)
    sel = row.get("selection_mae")
    if n_origins < config.min_selection_origins:
        return False, (
            f"insufficient origins ({n_origins} < {config.min_selection_origins})"
        )
    if n_preds < config.min_selection_predictions:
        return False, (
            f"insufficient predictions ({n_preds} < {config.min_selection_predictions})"
        )
    if sel is None or (isinstance(sel, float) and np.isnan(sel)):
        return False, "missing selection_mae (no evaluable horizon MAEs)"
    if not math.isfinite(float(sel)):
        return False, "non-finite selection_mae"
    if config.selection_metric != "mae":
        return False, f"unsupported selection_metric={config.selection_metric!r}"
    return True, ""


def select_product_model(
    backtest: BacktestResult,
    product: str,
    *,
    config: Optional[TSForecastConfig] = None,
    candidate_models: Optional[Sequence[str]] = None,
) -> ProductSelectionResult:
    """Select the best candidate for one SKU from backtest output."""
    cfg = config or DEFAULT_CONFIG
    product_s = str(product)
    candidates = tuple(candidate_models) if candidate_models is not None else cfg.candidate_models

    eligible_scores: dict[str, float] = {}
    unavailable: dict[str, str] = {}

    winner_row: Optional[pd.Series] = None
    winner_name: Optional[str] = None
    tie_break = False

    for model in candidates:
        row = _metrics_row(backtest.metrics, product=product_s, model=str(model))
        if row is None:
            fail_reason = _failure_reasons(backtest.failures, product=product_s, model=str(model))
            unavailable[str(model)] = fail_reason or "no backtest predictions"
            continue

        ok, reason = _eligible_for_selection(row, config=cfg)
        if not ok:
            fail_reason = _failure_reasons(backtest.failures, product=product_s, model=str(model))
            if fail_reason:
                reason = f"{reason}; {fail_reason}" if reason else fail_reason
            unavailable[str(model)] = reason
            continue

        score = float(row["selection_mae"])
        eligible_scores[str(model)] = score

    if not eligible_scores:
        reasons = "; ".join(f"{m}: {r}" for m, r in unavailable.items()) or "no eligible models"
        raise ValueError(f"{product_s!r}: no eligible models for selection ({reasons})")

    winner_name, tie_break = pick_winner_with_tiebreak(
        eligible_scores,
        tolerance=cfg.selection_tie_tolerance,
        simplicity_order=cfg.selection_simplicity_order,
    )
    winner_row = _metrics_row(backtest.metrics, product=product_s, model=winner_name)
    assert winner_row is not None

    horizon_maes = _horizon_maes_from_row(
        winner_row,
        forecast_horizon=cfg.forecast_horizon,
    )
    eval_horizons = winner_row.get("evaluated_horizons", ())
    if isinstance(eval_horizons, float) and np.isnan(eval_horizons):
        eval_horizons = ()
    if not isinstance(eval_horizons, tuple):
        eval_horizons = tuple(int(h) for h in eval_horizons)

    return ProductSelectionResult(
        product=product_s,
        selected_model=winner_name,
        selection_mae=float(winner_row["selection_mae"]),
        horizon_maes=horizon_maes,
        number_of_origins=int(winner_row.get("number_of_origins", 0) or 0),
        evaluated_horizons=tuple(int(h) for h in eval_horizons),
        candidate_scores=dict(eligible_scores),
        unavailable=dict(unavailable),
        metric=cfg.selection_metric,
        tie_break_applied=tie_break,
    )


def select_models(
    backtest: BacktestResult,
    products: Optional[Iterable[str]] = None,
    *,
    config: Optional[TSForecastConfig] = None,
    candidate_models: Optional[Sequence[str]] = None,
) -> dict[str, ProductSelectionResult]:
    """Select a model for each SKU present in the backtest metrics table."""
    cfg = config or DEFAULT_CONFIG
    if products is None:
        if backtest.metrics is None or backtest.metrics.empty:
            return {}
        product_list = sorted(backtest.metrics["product"].astype(str).unique())
    else:
        product_list = [str(p) for p in products]

    out: dict[str, ProductSelectionResult] = {}
    for product in product_list:
        out[product] = select_product_model(
            backtest,
            product,
            config=cfg,
            candidate_models=candidate_models,
        )
    return out


def selection_results_to_frame(
    results: Mapping[str, ProductSelectionResult],
    *,
    config: Optional[TSForecastConfig] = None,
) -> pd.DataFrame:
    """Flatten per-SKU selection results for reporting."""
    cfg = config or DEFAULT_CONFIG
    rows: list[dict] = []
    for product in sorted(results.keys()):
        r = results[product]
        row = {
            "product": product,
            "selected_model": r.selected_model,
            "selection_mae": r.selection_mae,
            "number_of_origins": r.number_of_origins,
            "evaluated_horizons": r.evaluated_horizons,
            "metric": r.metric,
            "tie_break_applied": r.tie_break_applied,
            "candidate_scores": dict(r.candidate_scores),
            "unavailable": dict(r.unavailable),
        }
        for h in range(1, int(cfg.forecast_horizon) + 1):
            row[f"mae_h{h}"] = r.horizon_maes.get(h, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def score_candidates(
    backtest: BacktestResult,
    product: str,
    candidate_names: Sequence[str],
    *,
    config: Optional[TSForecastConfig] = None,
) -> ProductSelectionResult:
    """Score and select among ``candidate_names`` using backtest OOF metrics."""
    return select_product_model(
        backtest,
        product,
        config=config,
        candidate_models=tuple(candidate_names),
    )
