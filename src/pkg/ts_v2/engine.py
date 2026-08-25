"""V2 forecasting engine (orchestration).

Pipeline:

1. Resolve explicit :class:`~pkg.ts_v2.types.ForecastOrigin`.
2. Build history with ``date < origin`` (no implicit last-month removal).
3. Multi-origin / multi-horizon backtest + selection.
4. **Discard** CV-fitted model instances.
5. Instantiate fresh model(s) and refit on full eligible pre-origin history.
6. Emit raw monthly forecasts for ``h1..h15``.

Models are invoked only through :func:`pkg.ts_v2.models.run_model` (same as backtest).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from pkg.ts_v2.backtest import run_backtest
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.data import assert_training_before_origin, prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window, parse_origin
from pkg.ts_v2.ensemble import (
    STRATEGY_BEST_SINGLE,
    build_ensemble_predictions,
    combine_model_forecast_results,
    rank_models_for_production,
    strategy_from_config,
)
from pkg.ts_v2.metrics import aggregate_metrics
from pkg.ts_v2.models.base import ForecastModel, is_failure, is_success, run_model
from pkg.ts_v2.models.registry import get_model
from pkg.ts_v2.postprocess import apply_final_constraints
from pkg.ts_v2.selection import select_models
from pkg.ts_v2.types import (
    BacktestResult,
    ConstrainedForecastResult,
    EngineResult,
    ForecastOrigin,
    ForecastResult,
    ForecastWindow,
    HorizonForecast,
    ModelFailure,
    ModelOutcome,
    PreparedSeries,
    ProductFinalForecast,
    ProductSelectionResult,
)


def assert_final_forecast_contract(
    prepared: PreparedSeries,
    window: ForecastWindow,
    outcome: ConstrainedForecastResult | ForecastResult,
    *,
    config: Optional[TSForecastConfig] = None,
) -> None:
    """Enforce production forecast date/length contract."""
    cfg = config or DEFAULT_CONFIG
    assert_training_before_origin(prepared.values.dropna(), window, config=cfg)
    if prepared.dates:
        max_train = max(int(d) for d in prepared.dates)
        if not (max_train < int(window.forecast_origin)):
            raise AssertionError(
                f"max(final_training_date)={max_train} must be < "
                f"forecast_origin={window.forecast_origin}"
            )
    preds = (
        outcome.constrained_predictions
        if isinstance(outcome, ConstrainedForecastResult)
        else outcome.predictions
    )
    expected_h = int(cfg.forecast_horizon)
    if len(preds) != expected_h:
        raise AssertionError(
            f"len(final_forecast)={len(preds)} != {expected_h}"
        )
    if tuple(int(d) for d in outcome.target_dates) != tuple(window.target_dates):
        raise AssertionError(
            "final target_dates must equal ForecastWindow.target_dates: "
            f"{tuple(outcome.target_dates)!r} != {tuple(window.target_dates)!r}"
        )
    if tuple(int(h) for h in outcome.horizons) != tuple(window.horizons):
        raise AssertionError(
            "final horizons must equal ForecastWindow.horizons: "
            f"{tuple(outcome.horizons)!r} != {tuple(window.horizons)!r}"
        )


def _horizon_forecasts_from_result(
    product: str,
    origin: ForecastOrigin,
    raw: ForecastResult,
    constrained: ConstrainedForecastResult,
) -> tuple[HorizonForecast, ...]:
    out: list[HorizonForecast] = []
    for h, target, raw_v, con_v in zip(
        constrained.horizons,
        constrained.target_dates,
        raw.predictions,
        constrained.constrained_predictions,
    ):
        out.append(
            HorizonForecast(
                product=str(product),
                origin=origin,
                horizon=int(h),
                target_shamsi_yyyymm=int(target),
                raw_forecast=float(raw_v),
                constrained_forecast=float(con_v),
                model_name=str(constrained.model_name),
            )
        )
    return tuple(out)


def _cv_metadata_for_strategy(
    backtest: Optional[BacktestResult],
    product: str,
    selection: ProductSelectionResult,
    strategy_internal: str,
    *,
    config: TSForecastConfig,
) -> dict:
    if strategy_internal == STRATEGY_BEST_SINGLE:
        return {
            "cv_score": float(selection.selection_mae),
            "number_of_origins": int(selection.number_of_origins),
            "number_of_predictions": None,
            "evaluated_horizons": tuple(int(h) for h in selection.evaluated_horizons),
            "max_evaluated_horizon": (
                int(max(selection.evaluated_horizons))
                if selection.evaluated_horizons
                else 0
            ),
        }
    if backtest is None:
        raise ValueError("backtest is required for ensemble strategy CV metadata")
    frame = build_ensemble_predictions(
        backtest, product, strategy_internal, config=config
    )
    if frame is None or frame.empty:
        return {
            "cv_score": float("nan"),
            "number_of_origins": 0,
            "number_of_predictions": 0,
            "evaluated_horizons": (),
            "max_evaluated_horizon": 0,
        }
    metrics = aggregate_metrics(frame, config=config)
    horizons = tuple(sorted(int(h) for h in frame["horizon"].unique()))
    return {
        "cv_score": float(metrics["selection_mae"]),
        "number_of_origins": int(frame["origin"].nunique()),
        "number_of_predictions": int(len(frame)),
        "evaluated_horizons": horizons,
        "max_evaluated_horizon": int(max(horizons)) if horizons else 0,
    }


def _fresh_model(name: str) -> ForecastModel:
    """Instantiate a new registry model (never reuse a CV/backtest instance)."""
    return get_model(name)


def _run_fresh_model(
    model_name: str,
    train_series: pd.Series,
    window: ForecastWindow,
) -> tuple[ForecastResult, int]:
    model = _fresh_model(model_name)
    model_id = id(model)
    outcome = run_model(model, train_series, window)
    if is_failure(outcome):
        assert isinstance(outcome, ModelFailure)
        raise RuntimeError(
            f"final refit failed for {model_name!r}: {outcome.reason}"
        )
    assert is_success(outcome)
    assert isinstance(outcome, ForecastResult)
    return outcome, model_id


def refit_and_forecast_product(
    sales: pd.DataFrame,
    product: str,
    origin: ForecastOrigin,
    selection: ProductSelectionResult,
    *,
    backtest: Optional[BacktestResult] = None,
    config: Optional[TSForecastConfig] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> ProductFinalForecast:
    """Fresh refit + production forecast for one SKU (never reuses CV models)."""
    cfg = config or DEFAULT_CONFIG
    window = make_forecast_window(origin, config=cfg)
    prepared = prepare_monthly_series(
        sales,
        product,
        origin,
        config=cfg,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    if prepared.n_observations < cfg.min_train_months:
        raise ValueError(
            f"{product!r} at origin {window.forecast_origin}: "
            f"need >= {cfg.min_train_months} training months, got {prepared.n_observations}"
        )

    strategy_config = cfg.selection_strategy
    strategy_internal = strategy_from_config(cfg)
    cv_meta = _cv_metadata_for_strategy(
        backtest, product, selection, strategy_internal, config=cfg
    )
    refit_model_ids: list[int] = []

    if strategy_internal == STRATEGY_BEST_SINGLE:
        model_name = str(selection.selected_model)
        forecast, model_id = _run_fresh_model(
            model_name, prepared.values, window
        )
        refit_model_ids.append(model_id)
        constituents = (model_name,)
        selected_label = model_name
    else:
        ranked = rank_models_for_production(selection.candidate_scores, config=cfg)
        if not ranked:
            raise ValueError(
                f"{product!r}: no eligible ensemble constituents in candidate_scores"
            )
        model_results: dict[str, ForecastResult] = {}
        for name in ranked:
            result, model_id = _run_fresh_model(name, prepared.values, window)
            model_results[name] = result
            refit_model_ids.append(model_id)
        output_name = f"ensemble:{strategy_config}"
        forecast = combine_model_forecast_results(
            model_results,
            ranked,
            selection.candidate_scores,
            strategy_internal,
            output_name=output_name,
        )
        constituents = tuple(ranked)
        selected_label = output_name

    raw_forecast = forecast
    constrained_forecast = apply_final_constraints(raw_forecast, config=cfg)
    assert_final_forecast_contract(prepared, window, constrained_forecast, config=cfg)
    origin_parsed = parse_origin(window.forecast_origin)
    horizon_forecasts = _horizon_forecasts_from_result(
        product, origin_parsed, raw_forecast, constrained_forecast
    )

    training_start = (
        int(prepared.first_active_month)
        if prepared.first_active_month is not None
        else (int(prepared.dates[0]) if prepared.dates else None)
    )
    training_end = (
        int(prepared.last_training_month)
        if prepared.last_training_month is not None
        else None
    )

    return ProductFinalForecast(
        product=str(product),
        forecast_origin=int(window.forecast_origin),
        selected_strategy=strategy_config,
        selected_model=selected_label,
        constituent_models=constituents,
        training_start=training_start,
        training_end=training_end,
        n_training_observations=int(prepared.n_observations),
        cv_score=float(cv_meta["cv_score"]),
        cv_coverage={
            "number_of_origins": cv_meta["number_of_origins"],
            "number_of_predictions": cv_meta.get("number_of_predictions"),
            "evaluated_horizons": cv_meta["evaluated_horizons"],
            "max_evaluated_horizon": cv_meta["max_evaluated_horizon"],
            "candidate_scores": dict(selection.candidate_scores),
            "unavailable_models": dict(selection.unavailable),
        },
        horizon_forecasts=horizon_forecasts,
        raw_forecast=raw_forecast,
        constrained_forecast=constrained_forecast,
        metadata={
            "refit_model_ids": tuple(refit_model_ids),
            "selection_metric": selection.metric,
            "tie_break_applied": selection.tie_break_applied,
            "horizon_maes": dict(selection.horizon_maes),
            "nonneg_adjustment": constrained_forecast.metadata.get(
                "nonneg_adjustment", {}
            ),
        },
    )


def forecast_series(
    model: ForecastModel,
    train_series: pd.Series,
    window: ForecastWindow,
) -> ModelOutcome:
    """Fit/predict one model on a prepared training series (shared model API)."""
    return run_model(model, train_series, window)


def forecast_prepared(
    model: ForecastModel,
    prepared: PreparedSeries,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ModelOutcome:
    """Run ``model`` on a :class:`PreparedSeries` using its forecast origin window."""
    cfg = config or DEFAULT_CONFIG
    window = make_forecast_window(prepared.forecast_origin, config=cfg)
    return run_model(model, prepared.values, window)


def forecast_with_backtest(
    sales: pd.DataFrame,
    products: Iterable[str],
    final_origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> EngineResult:
    """Backtest, select, discard CV models, then fresh refit at ``final_origin``."""
    cfg = config or DEFAULT_CONFIG
    product_list = [str(p) for p in products]

    # 1–2. Historical backtest + selection (CV model instances are not retained).
    cv_models = None
    from pkg.ts_v2.models.registry import models_from_config

    cv_models = models_from_config(cfg.candidate_models)
    backtest = run_backtest(
        sales,
        product_list,
        models=cv_models,
        config=cfg,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    cv_model_ids = tuple(id(m) for m in cv_models)
    del cv_models

    selections = select_models(backtest, product_list, config=cfg)

    # 3–6. Fresh production refit per SKU.
    final_map: dict[str, ProductFinalForecast] = {}
    all_horizon: list[HorizonForecast] = []
    for product in product_list:
        final = refit_and_forecast_product(
            sales,
            product,
            final_origin,
            selections[product],
            backtest=backtest,
            config=cfg,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
        )
        reused = [mid for mid in final.metadata.get("refit_model_ids", ()) if mid in cv_model_ids]
        if reused:
            raise RuntimeError(
                f"{product!r}: production refit reused CV model instance(s)"
            )
        final_map[product] = final
        all_horizon.extend(final.horizon_forecasts)

    first_sel = selections[product_list[0]] if product_list else None
    legacy_selection = None
    if first_sel is not None:
        from pkg.ts_v2.types import SelectionResult

        legacy_selection = SelectionResult(
            product=first_sel.product,
            origin=final_origin,
            best_model_name=first_sel.selected_model,
            scores=dict(first_sel.candidate_scores),
            metric=first_sel.metric,
        )

    return EngineResult(
        config_name="default",
        selection=legacy_selection,
        selections=selections,
        final_forecasts=final_map,
        forecasts=tuple(all_horizon),
        backtest=backtest,
        extras={"cv_model_ids": cv_model_ids},
    )


def forecast_products(
    sales: pd.DataFrame,
    products: Iterable[str],
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> EngineResult:
    """Run backtest, select, and production-refit for one origin."""
    return forecast_with_backtest(
        sales,
        products,
        origin,
        config=config,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )


def default_engine_config() -> TSForecastConfig:
    """Return the frozen default V2 config (copy-safe via frozen dataclass)."""
    return DEFAULT_CONFIG
