"""Multi-origin / multi-horizon backtesting for V2.

Unlike V1's single 80/20 rolling 1-step RMSE on a scaled series, V2 evaluates
explicit origins and horizons against **raw** actuals using the same
:func:`~pkg.ts_v2.models.run_model` path for every candidate.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from pkg.ts_v2.backtest_origins import (
    OriginCoverage,
    discover_origins,
    eval_window_for_origin,
)
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series, product_monthly_sales
from pkg.ts_v2.dates import make_forecast_window, parse_origin, validate_shamsi_yyyymm
from pkg.ts_v2.metrics import metrics_summary_row
from pkg.ts_v2.models.base import ForecastModel, is_failure, is_success, run_model
from pkg.ts_v2.models.registry import models_from_config
from pkg.ts_v2.types import (
    BacktestFold,
    BacktestResult,
    ForecastOrigin,
    ForecastResult,
    ForecastWindow,
    HorizonForecast,
    ModelFailure,
    PreparedSeries,
)

PREDICTION_COLUMNS = (
    "product",
    "model",
    "origin",
    "target_date",
    "horizon",
    "actual",
    "prediction",
)


def make_folds(
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[BacktestFold]:
    """Build backtest folds from :func:`~pkg.ts_v2.dates.make_forecast_window`."""
    cfg = config or DEFAULT_CONFIG
    folds: list[BacktestFold] = []
    for origin in origins:
        window = make_forecast_window(origin, config=cfg)
        folds.append(
            BacktestFold(
                origin=parse_origin(window.forecast_origin),
                train_end_exclusive=window.forecast_origin,
                horizons=window.horizons,
                window=window,
            )
        )
    return folds


def windows_for_origins(
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[ForecastWindow]:
    """Explicit forecast windows for each evaluation origin."""
    cfg = config or DEFAULT_CONFIG
    return [make_forecast_window(origin, config=cfg) for origin in origins]


def forecast_fold(
    model: ForecastModel,
    train_series: pd.Series,
    fold: BacktestFold,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ForecastResult | ModelFailure:
    """Run one candidate on one fold via the shared :func:`run_model` interface."""
    cfg = config or DEFAULT_CONFIG
    if fold.evaluable_target_dates and fold.evaluable_horizons and fold.window is not None:
        window = ForecastWindow(
            forecast_origin=fold.window.forecast_origin,
            training_end=fold.window.training_end,
            target_dates=fold.evaluable_target_dates,
            horizons=tuple(int(h) for h in fold.evaluable_horizons),
        )
    else:
        window = fold.window or make_forecast_window(fold.origin, config=cfg)
    return run_model(model, train_series, window)


def _actual_at(full_sales: pd.Series, target_date: int) -> float:
    t = validate_shamsi_yyyymm(int(target_date))
    if t not in full_sales.index:
        return float("nan")
    return float(full_sales.loc[t])


def _fold_from_coverage(coverage: OriginCoverage) -> BacktestFold:
    eval_win = eval_window_for_origin(coverage)
    return BacktestFold(
        origin=coverage.origin,
        train_end_exclusive=eval_win.forecast_origin,
        horizons=eval_win.horizons,
        window=eval_win,
        evaluable_target_dates=eval_win.target_dates,
        evaluable_horizons=eval_win.horizons,
        max_evaluated_horizon=coverage.max_evaluated_horizon,
        full_horizon_coverage=coverage.full_horizon_coverage,
    )


def _coverage_row(
    predictions: pd.DataFrame,
    *,
    product: str,
    model: str,
    forecast_horizon: int,
) -> dict:
    sub = predictions.loc[
        (predictions["product"] == product) & (predictions["model"] == model)
    ]
    if sub.empty:
        return {
            "product": product,
            "model": model,
            "number_of_origins": 0,
            "number_of_predictions": 0,
            "evaluated_horizons": (),
            "max_evaluated_horizon": 0,
            "n_full_horizon_origins": 0,
        }
    horizons = tuple(sorted(int(h) for h in sub["horizon"].unique()))
    max_h = int(max(horizons)) if horizons else 0
    per_origin_max = sub.groupby("origin")["horizon"].max()
    n_full = int((per_origin_max >= forecast_horizon).sum())
    return {
        "product": product,
        "model": model,
        "number_of_origins": int(sub["origin"].nunique()),
        "number_of_predictions": int(len(sub)),
        "evaluated_horizons": horizons,
        "max_evaluated_horizon": max_h,
        "n_full_horizon_origins": n_full,
    }


def assert_backtest_no_leakage(
    predictions: pd.DataFrame,
    prepared_by_origin: Mapping[int, PreparedSeries],
) -> None:
    """For each origin: ``max(training_date) < origin <= min(target_date)``."""
    if predictions is None or predictions.empty:
        return
    for origin, group in predictions.groupby("origin"):
        origin_i = validate_shamsi_yyyymm(int(origin))
        prepared = prepared_by_origin.get(origin_i)
        if prepared is None or not prepared.dates:
            raise AssertionError(f"missing prepared series for origin {origin_i}")
        max_train = max(int(d) for d in prepared.dates)
        min_target = int(group["target_date"].min())
        if not (max_train < origin_i <= min_target):
            raise AssertionError(
                f"leakage/alignment at origin {origin_i}: "
                f"max_train={max_train}, min_target={min_target}; "
                f"require max_train < origin <= min_target"
            )
        if any(int(d) >= origin_i for d in prepared.dates):
            raise AssertionError(
                f"prepared training dates include origin or later at {origin_i}"
            )


def backtest_product(
    sales: pd.DataFrame,
    product: str,
    models: Sequence[ForecastModel],
    *,
    config: Optional[TSForecastConfig] = None,
    explicit_origins: Optional[Sequence[int]] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> BacktestResult:
    """Expanding-window backtest for one SKU across historical origins."""
    cfg = config or DEFAULT_CONFIG
    full_sales = product_monthly_sales(
        sales,
        product,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    origin_covers = discover_origins(
        full_sales,
        config=cfg,
        explicit_origins=explicit_origins,
    )

    pred_rows: list[dict] = []
    fail_rows: list[dict] = []
    prepared_by_origin: dict[int, PreparedSeries] = {}

    for cover in origin_covers:
        prepared = prepare_monthly_series(
            sales,
            product,
            cover.window.forecast_origin,
            config=cfg,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
        )
        if prepared.n_observations < cfg.min_train_months:
            continue
        prepared_by_origin[int(cover.window.forecast_origin)] = prepared
        fold = _fold_from_coverage(cover)
        eval_win = fold.window
        assert eval_win is not None

        for model in models:
            outcome = run_model(model, prepared.values, eval_win)
            if is_failure(outcome):
                assert isinstance(outcome, ModelFailure)
                fail_rows.append(
                    {
                        "product": product,
                        "model": outcome.model_name,
                        "origin": int(cover.window.forecast_origin),
                        "reason": outcome.reason,
                        "error_type": outcome.error_type,
                    }
                )
                continue
            assert is_success(outcome)
            assert isinstance(outcome, ForecastResult)
            for h, target, pred in zip(
                outcome.horizons,
                outcome.target_dates,
                outcome.predictions,
            ):
                pred_rows.append(
                    {
                        "product": product,
                        "model": outcome.model_name,
                        "origin": int(cover.window.forecast_origin),
                        "target_date": int(target),
                        "horizon": int(h),
                        "actual": _actual_at(full_sales, int(target)),
                        "prediction": float(pred),
                    }
                )

    predictions = (
        pd.DataFrame(pred_rows, columns=list(PREDICTION_COLUMNS))
        if pred_rows
        else pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    )
    failures = pd.DataFrame(fail_rows) if fail_rows else pd.DataFrame(
        columns=["product", "model", "origin", "reason", "error_type"]
    )

    if not predictions.empty:
        assert_backtest_no_leakage(predictions, prepared_by_origin)

    coverage_rows = []
    metrics_rows = []
    model_names = sorted({m.name for m in models})
    for model_name in model_names:
        cov = _coverage_row(
            predictions,
            product=product,
            model=model_name,
            forecast_horizon=cfg.forecast_horizon,
        )
        coverage_rows.append(cov)
        sub = predictions.loc[
            (predictions["product"] == product) & (predictions["model"] == model_name)
        ]
        metrics_rows.append(
            metrics_summary_row(
                product,
                model_name,
                sub,
                {
                    "number_of_origins": cov["number_of_origins"],
                    "number_of_predictions": cov["number_of_predictions"],
                    "evaluated_horizons": cov["evaluated_horizons"],
                    "max_evaluated_horizon": cov["max_evaluated_horizon"],
                    "n_full_horizon_origins": cov["n_full_horizon_origins"],
                },
                config=cfg,
            )
        )

    coverage = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
    metrics = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    return BacktestResult(
        predictions=predictions,
        coverage=coverage,
        metrics=metrics,
        failures=failures,
    )


def run_backtest(
    sales: pd.DataFrame,
    products: Iterable[str],
    *,
    models: Optional[Sequence[ForecastModel]] = None,
    model_names: Optional[Sequence[str]] = None,
    config: Optional[TSForecastConfig] = None,
    explicit_origins: Optional[Sequence[int]] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> BacktestResult:
    """Evaluate candidates across products, origins, and horizons (raw units).

    For each historical origin ``O``:

    - **Train** on all months with ``date < O`` (via :func:`prepare_monthly_series`).
    - **Forecast** evaluable horizons through ``O + H - 1`` without using
      post-origin actuals during fit.
    - **Score** only where warehouse actuals exist (no fabricated targets).

    Model selection score is horizon-equal: mean of per-horizon MAEs, not
    row-weighted MAE across all prediction cells.
    """
    cfg = config or DEFAULT_CONFIG
    if models is None:
        names = tuple(model_names) if model_names is not None else cfg.candidate_models
        models = models_from_config(names)
    if not models:
        raise ValueError("run_backtest requires at least one model")

    product_list = [str(p) for p in products]
    all_pred: list[pd.DataFrame] = []
    all_cov: list[pd.DataFrame] = []
    all_met: list[pd.DataFrame] = []
    all_fail: list[pd.DataFrame] = []

    for product in product_list:
        result = backtest_product(
            sales,
            product,
            models,
            config=cfg,
            explicit_origins=explicit_origins,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
        )
        if not result.predictions.empty:
            all_pred.append(result.predictions)
        if not result.coverage.empty:
            all_cov.append(result.coverage)
        if not result.metrics.empty:
            all_met.append(result.metrics)
        if not result.failures.empty:
            all_fail.append(result.failures)

    predictions = (
        pd.concat(all_pred, ignore_index=True)
        if all_pred
        else pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    )
    coverage = pd.concat(all_cov, ignore_index=True) if all_cov else pd.DataFrame()
    metrics = pd.concat(all_met, ignore_index=True) if all_met else pd.DataFrame()
    failures = pd.concat(all_fail, ignore_index=True) if all_fail else pd.DataFrame(
        columns=["product", "model", "origin", "reason", "error_type"]
    )
    return BacktestResult(
        predictions=predictions,
        coverage=coverage,
        metrics=metrics,
        failures=failures,
    )


def predictions_to_horizon_forecasts(frame: pd.DataFrame) -> list[HorizonForecast]:
    """Convert the out-of-fold prediction dataframe to legacy cell objects."""
    out: list[HorizonForecast] = []
    if frame is None or frame.empty:
        return out
    for row in frame.itertuples(index=False):
        out.append(
            HorizonForecast(
                product=str(row.product),
                origin=parse_origin(int(row.origin)),
                horizon=int(row.horizon),
                target_shamsi_yyyymm=int(row.target_date),
                yhat=float(row.prediction),
                model_name=str(row.model),
            )
        )
    return out
