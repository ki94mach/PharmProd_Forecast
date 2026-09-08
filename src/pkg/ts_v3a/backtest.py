"""V3A outer expanding-CV backtest engine.

Evaluates A0–A5 under the V2 historical forecasting contracts:
- A0/A1: production time contract (train through last_complete, 16→15 bridge)
- A2–A5: screening contract (train through origin−1, no bridge)

Horizon-equal MAE scoring is unchanged. Does not modify V1 behavior, select
architectures, refit on full history, or integrate with the server runner.

Each configured seed is an independent training run. Seed-level OOF rows are
always retained. A seed-ensemble forecast (mean across successful seeds) is
emitted only when at least ``min_successful_seeds`` succeed for that
architecture × origin (default 3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd

from pkg.ts_v2.backtest import assert_backtest_no_leakage
from pkg.ts_v2.backtest_origins import (
    OriginCoverage,
    discover_origins,
)
from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series, product_monthly_sales
from pkg.ts_v2.dates import (
    make_forecast_window,
    make_screening_forecast_window,
    validate_shamsi_yyyymm,
)
from pkg.ts_v2.types import ForecastWindow, PreparedSeries
from pkg.ts_v3a.architectures import (
    ArchitectureName,
    coerce_architecture_name,
    uses_production_time_contract,
)
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.eligibility import IneligibleForTrainingError
from pkg.ts_v3a.metrics import (
    PREDICTION_KIND_SEED,
    build_seed_ensemble_predictions,
    ensemble_metrics_table,
    seed_metrics_table,
    stability_table,
)
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
    "prediction_kind",
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
    "scaler_mean",
    "scaler_scale",
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
    """Out-of-fold seed predictions, seed-ensemble, and evaluation tables.

    ``predictions`` / ``seed_predictions`` hold every individual seed OOF row.
    ``ensemble_predictions`` holds mean forecasts across successful seeds only
    when ``min_successful_seeds`` is met. ``metrics`` aliases ``ensemble_metrics``
    (primary architecture screening score).
    """

    predictions: pd.DataFrame
    fold_metadata: pd.DataFrame
    metrics: pd.DataFrame
    ensemble_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    ensemble_origin_status: pd.DataFrame = field(default_factory=pd.DataFrame)
    seed_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    ensemble_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def seed_predictions(self) -> pd.DataFrame:
        """Alias for per-seed OOF predictions."""
        return self.predictions


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
    scaler = dict(meta.scaler_params or {})
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
        "scaler_mean": scaler.get("mean"),
        "scaler_scale": scaler.get("scale"),
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
        "scaler_mean": None,
        "scaler_scale": None,
        "status": status,
        "unavailable_reason": unavailable_reason,
        "error_type": error_type,
        "error_message": error_message,
    }


def _reason_from_exception(exc: BaseException) -> tuple[str, Optional[str], str]:
    """Return (status, unavailable_reason, error_type)."""
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


def _resolve_min_successful_seeds(
    seed_list: Sequence[int],
    *,
    config: NeuralExperimentConfig,
    min_successful_seeds: Optional[int],
) -> int:
    min_ok = int(
        min_successful_seeds
        if min_successful_seeds is not None
        else config.min_successful_seeds
    )
    if min_ok < 1:
        raise ValueError(f"min_successful_seeds must be >= 1, got {min_ok}")
    if min_ok > len(seed_list):
        raise ValueError(
            f"min_successful_seeds={min_ok} exceeds number of configured seeds "
            f"({len(seed_list)}); pass fewer min_successful_seeds or more seeds"
        )
    return min_ok


def _assemble_evaluation(
    *,
    predictions: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    seed_list: Sequence[int],
    arch_names: Sequence[str],
    products: Sequence[str],
    min_successful_seeds: int,
    forecast_horizon: int,
    v2_cfg: TSForecastConfig,
) -> NeuralOuterBacktestResult:
    if not predictions.empty and "prediction_kind" not in predictions.columns:
        predictions = predictions.copy()
        predictions["prediction_kind"] = PREDICTION_KIND_SEED

    ensemble_predictions, ensemble_origin_status = build_seed_ensemble_predictions(
        predictions,
        fold_metadata,
        seeds=seed_list,
        min_successful_seeds=min_successful_seeds,
        status_ok=STATUS_OK,
    )
    seed_metrics = seed_metrics_table(
        predictions,
        fold_metadata,
        forecast_horizon=forecast_horizon,
        v2_config=v2_cfg,
        status_ok=STATUS_OK,
    )
    ensemble_metrics = ensemble_metrics_table(
        ensemble_predictions,
        ensemble_origin_status,
        products=products,
        architectures=list(arch_names),
        forecast_horizon=forecast_horizon,
        v2_config=v2_cfg,
    )
    stability = stability_table(
        predictions,
        seed_metrics,
        forecast_horizon=forecast_horizon,
    )
    return NeuralOuterBacktestResult(
        predictions=predictions,
        fold_metadata=fold_metadata,
        metrics=ensemble_metrics,
        ensemble_predictions=ensemble_predictions,
        ensemble_origin_status=ensemble_origin_status,
        seed_metrics=seed_metrics,
        ensemble_metrics=ensemble_metrics,
        stability=stability,
    )


def assemble_neural_backtest_result(
    *,
    predictions: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    seeds: Sequence[int],
    architectures: Sequence[Union[str, ArchitectureName]],
    products: Sequence[str],
    min_successful_seeds: int,
    config: Optional[NeuralExperimentConfig] = None,
    v2_config: Optional[TSForecastConfig] = None,
) -> NeuralOuterBacktestResult:
    """Public helper to rebuild ensemble/metrics tables from cumulative OOF rows."""
    cfg = config or DEFAULT_CONFIG
    v2_cfg = v2_config or V2_DEFAULT_CONFIG
    arch_names = [coerce_architecture_name(a).value for a in architectures]
    return _assemble_evaluation(
        predictions=predictions if predictions is not None else _empty_predictions(),
        fold_metadata=(
            fold_metadata if fold_metadata is not None else _empty_fold_metadata()
        ),
        seed_list=tuple(int(s) for s in seeds),
        arch_names=arch_names,
        products=[str(p) for p in products],
        min_successful_seeds=int(min_successful_seeds),
        forecast_horizon=int(v2_cfg.forecast_horizon),
        v2_cfg=v2_cfg,
    )


def _window_for_architecture(
    origin_ym: int,
    architecture: ArchitectureName,
    *,
    v2_config: TSForecastConfig,
) -> ForecastWindow:
    """Production window for A0/A1; screening window for A2–A5."""
    if uses_production_time_contract(architecture):
        return make_forecast_window(origin_ym, config=v2_config)
    return make_screening_forecast_window(origin_ym, config=v2_config)


def _run_one_fold(
    *,
    prepared: PreparedSeries,
    cover: OriginCoverage,
    window: ForecastWindow,
    architecture: ArchitectureName,
    seed: int,
    config: NeuralExperimentConfig,
    product: str,
    product_id: Optional[Any],
    full_sales: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit + predict one fold; return (prediction rows, fold metadata row)."""
    origin = int(cover.window.forecast_origin)
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
                    "prediction_kind": PREDICTION_KIND_SEED,
                }
            )
        return pred_rows, fold_meta
    except Exception as exc:  # noqa: BLE001 — fold isolation
        runtime = time.perf_counter() - t0
        status, unavailable_reason, error_type = _reason_from_exception(exc)
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
    min_successful_seeds: Optional[int] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
    product_id_col: Optional[str] = "product_id",
    progress_prefix: str = "",
) -> NeuralOuterBacktestResult:
    """Expanding outer CV for one SKU across architectures and seeds.

    Seeds are never mixed across SKU / origin / architecture. Each seed trains
    a fresh model. Seed-ensemble means require at least ``min_successful_seeds``
    successful seeds (default 3 from config).
    """
    cfg = config or DEFAULT_CONFIG
    v2_cfg = v2_config or V2_DEFAULT_CONFIG
    arch_list = coerce_architecture_list(architectures)
    seed_list = tuple(int(s) for s in (seeds if seeds is not None else cfg.random_seeds))
    if not seed_list:
        raise ValueError("seeds must contain at least one seed")
    min_ok = _resolve_min_successful_seeds(
        seed_list, config=cfg, min_successful_seeds=min_successful_seeds
    )

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
    prepared_by_arch_origin: dict[tuple[str, int], PreparedSeries] = {}
    windows_by_arch_origin: dict[tuple[str, int], ForecastWindow] = {}

    for cover in origin_covers:
        origin_ym = int(cover.window.forecast_origin)
        for architecture in arch_list:
            window = _window_for_architecture(
                origin_ym, architecture, v2_config=v2_cfg
            )
            prepared = prepare_monthly_series(
                sales,
                product,
                window,
                config=v2_cfg,
                product_col=product_col,
                date_col=date_col,
                sales_col=sales_col,
            )
            if prepared.n_observations < v2_cfg.min_train_months:
                continue
            arch_key = (architecture.value, origin_ym)
            prepared_by_arch_origin[arch_key] = prepared
            windows_by_arch_origin[arch_key] = window

            for seed in seed_list:
                fold_preds, fold_meta = _run_one_fold(
                    prepared=prepared,
                    cover=cover,
                    window=window,
                    architecture=architecture,
                    seed=seed,
                    config=cfg,
                    product=product,
                    product_id=product_id,
                    full_sales=full_sales,
                )
                pred_rows.extend(fold_preds)
                meta_rows.append(fold_meta)
                if progress_prefix:
                    rt = fold_meta.get("runtime_seconds")
                    print(
                        f"{progress_prefix} origin={origin_ym} "
                        f"arch={architecture.value} seed={seed} "
                        f"status={fold_meta.get('status')} runtime_s={rt}",
                        flush=True,
                    )

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
        for architecture in arch_list:
            arch_name = architecture.value
            sub = predictions.loc[predictions["architecture"] == arch_name]
            if sub.empty:
                continue
            prepared_map = {
                origin: prepared_by_arch_origin[(arch_name, int(origin))]
                for origin in sub["origin"].unique()
                if (arch_name, int(origin)) in prepared_by_arch_origin
            }
            windows_map = {
                origin: windows_by_arch_origin[(arch_name, int(origin))]
                for origin in sub["origin"].unique()
                if (arch_name, int(origin)) in windows_by_arch_origin
            }
            assert_backtest_no_leakage(
                sub, prepared_map, windows_by_origin=windows_map
            )

    return _assemble_evaluation(
        predictions=predictions,
        fold_metadata=fold_metadata,
        seed_list=seed_list,
        arch_names=[a.value for a in arch_list],
        products=[product],
        min_successful_seeds=min_ok,
        forecast_horizon=v2_cfg.forecast_horizon,
        v2_cfg=v2_cfg,
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
    min_successful_seeds: Optional[int] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
    product_id_col: Optional[str] = "product_id",
) -> NeuralOuterBacktestResult:
    """Evaluate A0–A5 across products, outer origins, and seeds (raw units).

    For each historical origin ``O`` and each configured seed:

    - A0/A1: production contract (train through ``O-2``, bridge 16→15)
    - A2–A5: screening contract (train through ``O-1``, no bridge)
    - resolve architecture configuration from that history only (A0 tiers)
    - train a fresh model through ``NeuralTrainer`` (independent seed run)
    - forecast delivered h1..h15 and score evaluable outer actuals only
    - discard the fitted model after the fold

    Seed-ensemble ``prediction = mean(successful seeds)`` is built only when
    at least ``min_successful_seeds`` (default 3) succeed for that
    architecture × origin. Primary architecture metric is ensemble
    ``mean_horizon_MAE``.
    """
    cfg = config or DEFAULT_CONFIG
    v2_cfg = v2_config or V2_DEFAULT_CONFIG
    arch_list = coerce_architecture_list(architectures)
    seed_list = tuple(int(s) for s in (seeds if seeds is not None else cfg.random_seeds))
    min_ok = _resolve_min_successful_seeds(
        seed_list, config=cfg, min_successful_seeds=min_successful_seeds
    )

    product_list = [str(p) for p in products]
    all_pred: list[pd.DataFrame] = []
    all_meta: list[pd.DataFrame] = []
    n_products = len(product_list)
    n_arch = len(arch_list)
    n_seeds = len(seed_list)
    n_origins = len(tuple(explicit_origins)) if explicit_origins is not None else None
    planned = (
        n_products * n_arch * n_seeds * int(n_origins)
        if n_origins is not None
        else None
    )
    fold_i = 0
    print(
        f"backtest_start products={n_products} arch={n_arch} seeds={n_seeds} "
        f"origins={n_origins} planned_folds={planned}",
        flush=True,
    )

    for pi, product in enumerate(product_list, start=1):
        print(f"backtest_product_begin {pi}/{n_products} product={product!r}", flush=True)
        result = backtest_product_architectures(
            sales,
            product,
            architectures=architectures,
            seeds=seed_list,
            config=cfg,
            v2_config=v2_cfg,
            explicit_origins=explicit_origins,
            min_successful_seeds=min_ok,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
            product_id_col=product_id_col,
            progress_prefix=f"[{pi}/{n_products} {product}]",
        )
        n_folds = 0 if result.fold_metadata.empty else len(result.fold_metadata)
        fold_i += n_folds
        print(
            f"backtest_product_done {pi}/{n_products} product={product!r} "
            f"folds={n_folds} cumulative_folds={fold_i}",
            flush=True,
        )
        if not result.predictions.empty:
            all_pred.append(result.predictions)
        if not result.fold_metadata.empty:
            all_meta.append(result.fold_metadata)

    print(f"backtest_assemble folds={fold_i}", flush=True)

    predictions = (
        pd.concat(all_pred, ignore_index=True) if all_pred else _empty_predictions()
    )
    fold_metadata = (
        pd.concat(all_meta, ignore_index=True) if all_meta else _empty_fold_metadata()
    )
    return _assemble_evaluation(
        predictions=predictions,
        fold_metadata=fold_metadata,
        seed_list=seed_list,
        arch_names=[a.value for a in arch_list],
        products=product_list,
        min_successful_seeds=min_ok,
        forecast_horizon=v2_cfg.forecast_horizon,
        v2_cfg=v2_cfg,
    )


__all__ = [
    "FOLD_METADATA_COLUMNS",
    "PREDICTION_COLUMNS",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "NeuralOuterBacktestResult",
    "assemble_neural_backtest_result",
    "backtest_product_architectures",
    "run_outer_backtest",
]
