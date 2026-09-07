"""V3A outer expanding-CV backtest engine.

Evaluates A0–A5 under the same historical forecasting contract as TS V2:
``ForecastWindow``, ``date < forecast_origin``, horizons 1..15, and
horizon-equal MAE. Does not modify V1/V2 behavior, select architectures,
refit on full history, or integrate with the server runner.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from pkg.ts_v2.backtest import assert_backtest_no_leakage
from pkg.ts_v2.backtest_origins import (
    OriginCoverage,
    discover_origins,
    eval_window_for_origin,
)
from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series, product_monthly_sales
from pkg.ts_v2.dates import validate_shamsi_yyyymm
from pkg.ts_v2.types import PreparedSeries
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.eligibility import IneligibleForTrainingError
from pkg.ts_v3a.metrics import metrics_summary_row
from pkg.ts_v3a.model_factory import coerce_architecture_list, create_neural_model
from pkg.ts_v3a.types import TrainMetadata
from pkg.ts_v3a.windows import InsufficientHistoryError

PREDICTION_COLUMNS = (
    "product_id",
    "product",
    "architecture",
    "seed",
    "origin",
    "target_date",
    "horizon",
    "actual",
    "prediction",
)

FOLD_METADATA_COLUMNS = (
    "product_id",
    "product",
    "architecture",
    "seed",
    "origin",
    "available_history_length",
    "lookback",
    "hidden_units",
    "second_hidden_units",
    "parameter_count",
    "train_window_count",
    "validation_window_count",
    "best_epoch",
    "epochs_ran",
    "best_val_loss",
    "training_start",
    "training_end",
    "validation_start",
    "validation_end",
    "runtime_seconds",
    "status",
    "unavailable_reason",
    "error_type",
    "error_message",
)

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"


@dataclass
class NeuralOuterBacktestResult:
    """Out-of-fold predictions, fold metadata, and architecture metrics."""

    predictions: pd.DataFrame
    fold_metadata: pd.DataFrame
    metrics: pd.DataFrame


def _actual_at(full_sales: pd.Series, target_date: int) -> float:
    t = validate_shamsi_yyyymm(int(target_date))
    if t not in full_sales.index:
        return float("nan")
    return float(full_sales.loc[t])


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PREDICTION_COLUMNS))


def _empty_fold_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FOLD_METADATA_COLUMNS))


def _product_id_lookup(
    sales: pd.DataFrame,
    product: str,
    *,
    product_col: str,
    product_id_col: Optional[str],
) -> Optional[Any]:
    if not product_id_col or product_id_col not in sales.columns:
        return None
    sub = sales.loc[sales[product_col].astype(str) == str(product), product_id_col]
    if sub.empty:
        return None
    return sub.iloc[0]


def _metadata_from_train(
    *,
    product: str,
    product_id: Optional[Any],
    architecture: str,
    seed: int,
    origin: int,
    available_history_length: int,
    meta: TrainMetadata,
    runtime_seconds: float,
) -> dict[str, Any]:
    params = dict(meta.parameters or {})
    lookback = params.get("lookback")
    hidden = params.get("hidden_units", params.get("l1"))
    second = params.get("second_hidden_units", params.get("l2"))
    return {
        "product_id": product_id,
        "product": product,
        "architecture": architecture,
        "seed": int(seed),
        "origin": int(origin),
        "available_history_length": int(available_history_length),
        "lookback": int(lookback) if lookback is not None else None,
        "hidden_units": int(hidden) if hidden is not None else None,
        "second_hidden_units": int(second) if second is not None else None,
        "parameter_count": meta.parameter_count,
        "train_window_count": meta.train_window_count,
        "validation_window_count": meta.validation_window_count,
        "best_epoch": meta.best_epoch,
        "epochs_ran": meta.epochs_ran,
        "best_val_loss": meta.best_val_loss,
        "training_start": meta.training_start,
        "training_end": meta.training_end,
        "validation_start": meta.validation_start,
        "validation_end": meta.validation_end,
        "runtime_seconds": float(runtime_seconds),
        "status": STATUS_OK,
        "unavailable_reason": None,
        "error_type": None,
        "error_message": None,
    }


def _unavailable_metadata(
    *,
    product: str,
    product_id: Optional[Any],
    architecture: str,
    seed: int,
    origin: int,
    available_history_length: int,
    runtime_seconds: float,
    status: str,
    unavailable_reason: Optional[str],
    error_type: Optional[str],
    error_message: Optional[str],
    lookback: Optional[int] = None,
    hidden_units: Optional[int] = None,
    second_hidden_units: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "product": product,
        "architecture": architecture,
        "seed": int(seed),
        "origin": int(origin),
        "available_history_length": int(available_history_length),
        "lookback": lookback,
        "hidden_units": hidden_units,
        "second_hidden_units": second_hidden_units,
        "parameter_count": None,
        "train_window_count": None,
        "validation_window_count": None,
        "best_epoch": None,
        "epochs_ran": None,
        "best_val_loss": None,
        "training_start": None,
        "training_end": None,
        "validation_start": None,
        "validation_end": None,
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "unavailable_reason": unavailable_reason,
        "error_type": error_type,
        "error_message": error_message,
    }


def _reason_from_exception(exc: BaseException) -> tuple[str, str, str]:
    """Return (status, unavailable_reason, error_type) plus message separately."""
    if isinstance(exc, InsufficientHistoryError):
        return STATUS_UNAVAILABLE, "insufficient_history", type(exc).__name__
    if isinstance(exc, IneligibleForTrainingError):
        reason = None
        if getattr(exc, "eligibility", None) is not None:
            reason = exc.eligibility.reason
        return (
            STATUS_UNAVAILABLE,
            reason or "ineligible_for_training",
            type(exc).__name__,
        )
    if isinstance(exc, ValueError) and "lookback" in str(exc).lower():
        return STATUS_UNAVAILABLE, "insufficient_history_for_lookback", type(exc).__name__
    return STATUS_ERROR, None, type(exc).__name__


def _coverage_for_slice(
    predictions: pd.DataFrame,
    *,
    product: str,
    architecture: str,
    forecast_horizon: int,
) -> dict[str, Any]:
    sub = predictions.loc[
        (predictions["product"] == product)
        & (predictions["architecture"] == architecture)
    ]
    if sub.empty:
        return {
            "number_of_origins": 0,
            "number_of_predictions": 0,
            "evaluated_horizons": (),
            "max_evaluated_horizon": 0,
        }
    horizons = tuple(sorted(int(h) for h in sub["horizon"].unique()))
    max_h = int(max(horizons)) if horizons else 0
    return {
        "number_of_origins": int(sub["origin"].nunique()),
        "number_of_predictions": int(len(sub)),
        "evaluated_horizons": horizons,
        "max_evaluated_horizon": max_h,
        "n_full_horizon_origins": int(
            (sub.groupby("origin")["horizon"].max() >= forecast_horizon).sum()
        ),
    }


def _run_one_fold(
    *,
    prepared: PreparedSeries,
    cover: OriginCoverage,
    architecture: ArchitectureName,
    seed: int,
    config: NeuralExperimentConfig,
    product: str,
    product_id: Optional[Any],
    full_sales: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit + predict one fold; return (prediction rows, fold metadata row)."""
    origin = int(cover.window.forecast_origin)
    window = eval_window_for_origin(cover)
    history = prepared.values
    available_history_length = int(prepared.n_observations)
    evaluable_h = {int(h) for h in cover.evaluable_horizons}
    arch_name = architecture.value
    t0 = time.perf_counter()

    model = None
    try:
        model = create_neural_model(architecture, config)
        model.fit(history, window, seed=int(seed))
        outcome = model.predict(window)
        runtime = time.perf_counter() - t0

        meta = model.metadata_
        if meta is None:
            raise RuntimeError("model.fit completed without TrainMetadata")

        fold_meta = _metadata_from_train(
            product=product,
            product_id=product_id,
            architecture=arch_name,
            seed=seed,
            origin=origin,
            available_history_length=available_history_length,
            meta=meta,
            runtime_seconds=runtime,
        )

        pred_rows: list[dict[str, Any]] = []
        for h, target, pred in zip(
            outcome.horizons,
            outcome.target_dates,
            outcome.predictions,
        ):
            if int(h) not in evaluable_h:
                continue
            pred_rows.append(
                {
                    "product_id": product_id,
                    "product": product,
                    "architecture": arch_name,
                    "seed": int(seed),
                    "origin": origin,
                    "target_date": int(target),
                    "horizon": int(h),
                    "actual": _actual_at(full_sales, int(target)),
                    "prediction": float(pred),
                }
            )
        return pred_rows, fold_meta
    except Exception as exc:  # noqa: BLE001 — fold isolation
        runtime = time.perf_counter() - t0
        status, unavailable_reason, error_type = _reason_from_exception(exc)
        # Prefer typed eligibility reason when present.
        if isinstance(exc, IneligibleForTrainingError) and exc.eligibility is not None:
            unavailable_reason = exc.eligibility.reason or unavailable_reason
        lookback = None
        hidden = None
        second = None
        if model is not None:
            lookback = getattr(model.config, "lookback", None)
            hidden = getattr(model.config, "hidden_units", None)
            second = getattr(model.config, "second_hidden_units", None)
            resolved = getattr(model, "_resolved_spec", None)
            if resolved is not None:
                lookback = resolved.lookback
                hidden = resolved.l1
                second = resolved.l2
        fold_meta = _unavailable_metadata(
            product=product,
            product_id=product_id,
            architecture=arch_name,
            seed=seed,
            origin=origin,
            available_history_length=available_history_length,
            runtime_seconds=runtime,
            status=status,
            unavailable_reason=unavailable_reason,
            error_type=error_type,
            error_message=str(exc),
            lookback=int(lookback) if lookback is not None else None,
            hidden_units=int(hidden) if hidden is not None else None,
            second_hidden_units=int(second) if second is not None else None,
        )
        return [], fold_meta
    finally:
        # Discard fitted model after the fold (no cross-fold weight reuse).
        model = None


def backtest_product_architectures(
    sales: pd.DataFrame,
    product: str,
    *,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    config: Optional[NeuralExperimentConfig] = None,
    v2_config: Optional[TSForecastConfig] = None,
    explicit_origins: Optional[Sequence[int]] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
    product_id_col: Optional[str] = "product_id",
) -> NeuralOuterBacktestResult:
    """Expanding outer CV for one SKU across architectures and seeds."""
    cfg = config or DEFAULT_CONFIG
    v2_cfg = v2_config or V2_DEFAULT_CONFIG
    arch_list = coerce_architecture_list(architectures)
    seed_list = tuple(int(s) for s in (seeds if seeds is not None else cfg.random_seeds))
    if not seed_list:
        raise ValueError("seeds must contain at least one seed")

    product_id = _product_id_lookup(
        sales, product, product_col=product_col, product_id_col=product_id_col
    )
    full_sales = product_monthly_sales(
        sales,
        product,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    origin_covers = discover_origins(
        full_sales,
        config=v2_cfg,
        explicit_origins=explicit_origins,
    )

    pred_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    prepared_by_origin: dict[int, PreparedSeries] = {}

    for cover in origin_covers:
        prepared = prepare_monthly_series(
            sales,
            product,
            cover.window.forecast_origin,
            config=v2_cfg,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
        )
        if prepared.n_observations < v2_cfg.min_train_months:
            # Same gate as V2: skip origin entirely (no fake folds).
            continue
        prepared_by_origin[int(cover.window.forecast_origin)] = prepared

        for architecture in arch_list:
            for seed in seed_list:
                fold_preds, fold_meta = _run_one_fold(
                    prepared=prepared,
                    cover=cover,
                    architecture=architecture,
                    seed=seed,
                    config=cfg,
                    product=product,
                    product_id=product_id,
                    full_sales=full_sales,
                )
                pred_rows.extend(fold_preds)
                meta_rows.append(fold_meta)

    predictions = (
        pd.DataFrame(pred_rows, columns=list(PREDICTION_COLUMNS))
        if pred_rows
        else _empty_predictions()
    )
    fold_metadata = (
        pd.DataFrame(meta_rows, columns=list(FOLD_METADATA_COLUMNS))
        if meta_rows
        else _empty_fold_metadata()
    )

    if not predictions.empty:
        assert_backtest_no_leakage(predictions, prepared_by_origin)

    metrics_rows: list[dict[str, Any]] = []
    for architecture in arch_list:
        arch_name = architecture.value
        cov = _coverage_for_slice(
            predictions,
            product=product,
            architecture=arch_name,
            forecast_horizon=v2_cfg.forecast_horizon,
        )
        unavailable_count = 0
        if not fold_metadata.empty:
            unavailable_count = int(
                (
                    (fold_metadata["product"] == product)
                    & (fold_metadata["architecture"] == arch_name)
                    & (fold_metadata["status"] != STATUS_OK)
                ).sum()
            )
        sub = predictions.loc[
            (predictions["product"] == product)
            & (predictions["architecture"] == arch_name)
        ]
        metrics_rows.append(
            metrics_summary_row(
                product,
                arch_name,
                sub,
                number_of_origins=cov["number_of_origins"],
                number_of_predictions=cov["number_of_predictions"],
                evaluated_horizons=cov["evaluated_horizons"],
                max_evaluated_horizon=cov["max_evaluated_horizon"],
                unavailable_fold_count=unavailable_count,
                forecast_horizon=v2_cfg.forecast_horizon,
                v2_config=v2_cfg,
            )
        )

    metrics = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    return NeuralOuterBacktestResult(
        predictions=predictions,
        fold_metadata=fold_metadata,
        metrics=metrics,
    )


def run_outer_backtest(
    sales: pd.DataFrame,
    products: Iterable[str],
    *,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    config: Optional[NeuralExperimentConfig] = None,
    v2_config: Optional[TSForecastConfig] = None,
    explicit_origins: Optional[Sequence[int]] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
    product_id_col: Optional[str] = "product_id",
) -> NeuralOuterBacktestResult:
    """Evaluate A0–A5 across products, outer origins, and seeds (raw units).

    For each historical origin ``O``:

    - select history with ``date < O`` via V2 ``prepare_monthly_series``
    - resolve architecture configuration from that history only (A0 tiers)
    - train a fresh model through ``NeuralTrainer``
    - forecast exactly h1..h15 and score evaluable outer actuals only
    - discard the fitted model after the fold

    Primary architecture metric is ``mean_horizon_MAE`` (equal-weight mean of
    horizon MAEs). Seeds are pooled for metrics; seed remains on OOF rows.
    """
    product_list = [str(p) for p in products]
    all_pred: list[pd.DataFrame] = []
    all_meta: list[pd.DataFrame] = []
    all_met: list[pd.DataFrame] = []

    for product in product_list:
        result = backtest_product_architectures(
            sales,
            product,
            architectures=architectures,
            seeds=seeds,
            config=config,
            v2_config=v2_config,
            explicit_origins=explicit_origins,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
            product_id_col=product_id_col,
        )
        if not result.predictions.empty:
            all_pred.append(result.predictions)
        if not result.fold_metadata.empty:
            all_meta.append(result.fold_metadata)
        if not result.metrics.empty:
            all_met.append(result.metrics)

    predictions = (
        pd.concat(all_pred, ignore_index=True) if all_pred else _empty_predictions()
    )
    fold_metadata = (
        pd.concat(all_meta, ignore_index=True) if all_meta else _empty_fold_metadata()
    )
    metrics = pd.concat(all_met, ignore_index=True) if all_met else pd.DataFrame()
    return NeuralOuterBacktestResult(
        predictions=predictions,
        fold_metadata=fold_metadata,
        metrics=metrics,
    )


__all__ = [
    "FOLD_METADATA_COLUMNS",
    "PREDICTION_COLUMNS",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "NeuralOuterBacktestResult",
    "backtest_product_architectures",
    "run_outer_backtest",
]
