"""V3A outer-backtest metrics (raw units, equal-weight horizon MAE).

Wraps V2 metric helpers so the V3A report surface uses ``mean_horizon_MAE``
as the primary architecture score. Horizons receive equal weight; row counts
must not dominate shorter horizons.

Also builds seed-ensemble forecasts (mean across successful seeds) and
per-seed / ensemble / stability metric tables.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.metrics import (
    aggregate_metrics,
    horizon_bias,
    horizon_mae,
    horizon_rmse,
    horizon_wmape,
    selection_mae_from_horizons,
)

PREDICTION_KIND_SEED = "seed"
PREDICTION_KIND_ENSEMBLE = "seed_ensemble"
ENSEMBLE_UNAVAILABLE_REASON = "insufficient_successful_seeds"

ENSEMBLE_PREDICTION_COLUMNS = (
    "product_id",
    "product",
    "architecture",
    "origin",
    "target_date",
    "horizon",
    "actual",
    "prediction",
    "n_successful_seeds",
    "prediction_kind",
)

ENSEMBLE_ORIGIN_STATUS_COLUMNS = (
    "product_id",
    "product",
    "architecture",
    "origin",
    "n_successful_seeds",
    "status",
    "unavailable_reason",
)


def mean_horizon_mae(predictions: pd.DataFrame) -> float:
    """Equal-weight mean of available horizon-level MAEs (not row-weighted)."""
    return float(selection_mae_from_horizons(horizon_mae(predictions)))


def metrics_summary_row(
    product: str,
    architecture: str,
    predictions: pd.DataFrame,
    *,
    number_of_origins: int,
    number_of_predictions: int,
    evaluated_horizons: tuple[int, ...],
    max_evaluated_horizon: int,
    unavailable_fold_count: int,
    forecast_horizon: int = 15,
    v2_config: Optional[TSForecastConfig] = None,
    seed: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Flat metrics row for one (product, architecture[, seed]) slice."""
    cfg = v2_config or V2_DEFAULT_CONFIG
    m = aggregate_metrics(predictions, config=cfg)
    h_mae: pd.Series = m["horizon_mae"]  # type: ignore[assignment]
    row: dict[str, Any] = {
        "product": product,
        "architecture": architecture,
        "mean_horizon_MAE": m["selection_mae"],
        "overall_rmse": m["overall_rmse"],
        "overall_bias": m["overall_bias"],
        "overall_wmape": m["overall_wmape"],
        "number_of_origins": int(number_of_origins),
        "number_of_predictions": int(number_of_predictions),
        "evaluated_horizons": evaluated_horizons,
        "max_evaluated_horizon": int(max_evaluated_horizon),
        "unavailable_fold_count": int(unavailable_fold_count),
    }
    if seed is not None:
        row["seed"] = int(seed)
    for h in range(1, int(forecast_horizon) + 1):
        row[f"mae_h{h}"] = float(h_mae[h]) if h in h_mae.index else float("nan")
    if extra:
        row.update(extra)
    return row


def _coverage_fields(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions is None or predictions.empty:
        return {
            "number_of_origins": 0,
            "number_of_predictions": 0,
            "evaluated_horizons": (),
            "max_evaluated_horizon": 0,
        }
    horizons = tuple(sorted(int(h) for h in predictions["horizon"].unique()))
    max_h = int(max(horizons)) if horizons else 0
    return {
        "number_of_origins": int(predictions["origin"].nunique()),
        "number_of_predictions": int(len(predictions)),
        "evaluated_horizons": horizons,
        "max_evaluated_horizon": max_h,
    }


def _empty_ensemble_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ENSEMBLE_PREDICTION_COLUMNS))


def _empty_ensemble_origin_status() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ENSEMBLE_ORIGIN_STATUS_COLUMNS))


def build_seed_ensemble_predictions(
    seed_predictions: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    seeds: Sequence[int],
    min_successful_seeds: int,
    status_ok: str = "ok",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean successful-seed forecasts per cell; gate by ``min_successful_seeds``.

    A seed counts as successful for ``(product, architecture, origin)`` only when
    fold metadata status is ``ok`` and at least one OOF prediction row exists.

    If fewer than ``min_successful_seeds`` succeed, no ensemble prediction rows
    are emitted for that origin (status ``unavailable``).
    """
    seed_set = {int(s) for s in seeds}
    min_ok = int(min_successful_seeds)
    if min_ok < 1:
        raise ValueError(f"min_successful_seeds must be >= 1, got {min_ok}")

    if fold_metadata is None or fold_metadata.empty:
        return _empty_ensemble_predictions(), _empty_ensemble_origin_status()

    meta = fold_metadata.copy()
    meta["seed"] = meta["seed"].astype(int)
    meta["origin"] = meta["origin"].astype(int)

    pred = (
        seed_predictions.copy()
        if seed_predictions is not None and not seed_predictions.empty
        else pd.DataFrame(
            columns=[
                "product_id",
                "product",
                "architecture",
                "seed",
                "origin",
                "target_date",
                "horizon",
                "actual",
                "prediction",
            ]
        )
    )
    if not pred.empty:
        pred["seed"] = pred["seed"].astype(int)
        pred["origin"] = pred["origin"].astype(int)
        pred["horizon"] = pred["horizon"].astype(int)

    # Keys attempted from fold metadata (configured seeds only).
    keys = (
        meta.loc[meta["seed"].isin(seed_set), ["product_id", "product", "architecture", "origin"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    status_rows: list[dict[str, Any]] = []
    ens_rows: list[dict[str, Any]] = []

    for _, key in keys.iterrows():
        product = str(key["product"])
        architecture = str(key["architecture"])
        origin = int(key["origin"])
        product_id = key.get("product_id")

        fold_mask = (
            (meta["product"] == product)
            & (meta["architecture"] == architecture)
            & (meta["origin"] == origin)
            & (meta["seed"].isin(seed_set))
        )
        folds = meta.loc[fold_mask]
        ok_seeds = set(
            int(s)
            for s in folds.loc[folds["status"] == status_ok, "seed"].tolist()
        )

        if not pred.empty:
            pred_mask = (
                (pred["product"] == product)
                & (pred["architecture"] == architecture)
                & (pred["origin"] == origin)
                & (pred["seed"].isin(ok_seeds))
            )
            cell = pred.loc[pred_mask]
            seeds_with_preds = set(int(s) for s in cell["seed"].unique()) if not cell.empty else set()
        else:
            cell = pred
            seeds_with_preds = set()

        successful = sorted(ok_seeds & seeds_with_preds)
        n_ok = len(successful)

        if n_ok < min_ok:
            status_rows.append(
                {
                    "product_id": product_id,
                    "product": product,
                    "architecture": architecture,
                    "origin": origin,
                    "n_successful_seeds": n_ok,
                    "status": "unavailable",
                    "unavailable_reason": ENSEMBLE_UNAVAILABLE_REASON,
                }
            )
            continue

        status_rows.append(
            {
                "product_id": product_id,
                "product": product,
                "architecture": architecture,
                "origin": origin,
                "n_successful_seeds": n_ok,
                "status": "ok",
                "unavailable_reason": None,
            }
        )

        grouped = cell.loc[cell["seed"].isin(successful)].groupby(
            ["target_date", "horizon"], sort=True
        )
        for (target_date, horizon), g in grouped:
            ens_rows.append(
                {
                    "product_id": product_id,
                    "product": product,
                    "architecture": architecture,
                    "origin": origin,
                    "target_date": int(target_date),
                    "horizon": int(horizon),
                    "actual": float(g["actual"].iloc[0]),
                    "prediction": float(g["prediction"].astype(float).mean()),
                    "n_successful_seeds": n_ok,
                    "prediction_kind": PREDICTION_KIND_ENSEMBLE,
                }
            )

    ensemble_predictions = (
        pd.DataFrame(ens_rows, columns=list(ENSEMBLE_PREDICTION_COLUMNS))
        if ens_rows
        else _empty_ensemble_predictions()
    )
    ensemble_origin_status = (
        pd.DataFrame(status_rows, columns=list(ENSEMBLE_ORIGIN_STATUS_COLUMNS))
        if status_rows
        else _empty_ensemble_origin_status()
    )
    return ensemble_predictions, ensemble_origin_status


def seed_metrics_table(
    seed_predictions: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    forecast_horizon: int = 15,
    v2_config: Optional[TSForecastConfig] = None,
    status_ok: str = "ok",
) -> pd.DataFrame:
    """One metrics row per ``(product, architecture, seed)``."""
    if seed_predictions is None or seed_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    keys = (
        seed_predictions[["product", "architecture", "seed"]]
        .drop_duplicates()
        .sort_values(["product", "architecture", "seed"])
    )
    for _, key in keys.iterrows():
        product = str(key["product"])
        architecture = str(key["architecture"])
        seed = int(key["seed"])
        sub = seed_predictions.loc[
            (seed_predictions["product"] == product)
            & (seed_predictions["architecture"] == architecture)
            & (seed_predictions["seed"].astype(int) == seed)
        ]
        unavailable = 0
        if fold_metadata is not None and not fold_metadata.empty:
            unavailable = int(
                (
                    (fold_metadata["product"] == product)
                    & (fold_metadata["architecture"] == architecture)
                    & (fold_metadata["seed"].astype(int) == seed)
                    & (fold_metadata["status"] != status_ok)
                ).sum()
            )
        cov = _coverage_fields(sub)
        rows.append(
            metrics_summary_row(
                product,
                architecture,
                sub,
                seed=seed,
                number_of_origins=cov["number_of_origins"],
                number_of_predictions=cov["number_of_predictions"],
                evaluated_horizons=cov["evaluated_horizons"],
                max_evaluated_horizon=cov["max_evaluated_horizon"],
                unavailable_fold_count=unavailable,
                forecast_horizon=forecast_horizon,
                v2_config=v2_config,
            )
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def ensemble_metrics_table(
    ensemble_predictions: pd.DataFrame,
    ensemble_origin_status: pd.DataFrame,
    *,
    products: Optional[Sequence[str]] = None,
    architectures: Optional[Sequence[str]] = None,
    forecast_horizon: int = 15,
    v2_config: Optional[TSForecastConfig] = None,
) -> pd.DataFrame:
    """One metrics row per ``(product, architecture)`` from ensemble OOF."""
    rows: list[dict[str, Any]] = []

    key_pairs: list[tuple[str, str]] = []
    if products is not None and architectures is not None:
        for product in products:
            for architecture in architectures:
                key_pairs.append((str(product), str(architecture)))
    elif ensemble_origin_status is not None and not ensemble_origin_status.empty:
        key_pairs = [
            (str(r.product), str(r.architecture))
            for r in ensemble_origin_status[["product", "architecture"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]
    elif ensemble_predictions is not None and not ensemble_predictions.empty:
        key_pairs = [
            (str(r.product), str(r.architecture))
            for r in ensemble_predictions[["product", "architecture"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]

    for product, architecture in key_pairs:
        if ensemble_predictions is None or ensemble_predictions.empty:
            sub = pd.DataFrame()
        else:
            sub = ensemble_predictions.loc[
                (ensemble_predictions["product"] == product)
                & (ensemble_predictions["architecture"] == architecture)
            ]
        n_with = 0
        n_unavail = 0
        if ensemble_origin_status is not None and not ensemble_origin_status.empty:
            st = ensemble_origin_status.loc[
                (ensemble_origin_status["product"] == product)
                & (ensemble_origin_status["architecture"] == architecture)
            ]
            n_with = int((st["status"] == "ok").sum())
            n_unavail = int((st["status"] != "ok").sum())
        cov = _coverage_fields(sub)
        rows.append(
            metrics_summary_row(
                product,
                architecture,
                sub,
                number_of_origins=cov["number_of_origins"],
                number_of_predictions=cov["number_of_predictions"],
                evaluated_horizons=cov["evaluated_horizons"],
                max_evaluated_horizon=cov["max_evaluated_horizon"],
                unavailable_fold_count=n_unavail,
                forecast_horizon=forecast_horizon,
                v2_config=v2_config,
                extra={
                    "n_origins_with_ensemble": n_with,
                    "unavailable_ensemble_origin_count": n_unavail,
                },
            )
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def stability_table(
    seed_predictions: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    *,
    forecast_horizon: int = 15,
) -> pd.DataFrame:
    """Cross-seed stability diagnostics per ``(product, architecture)``."""
    if seed_metrics is None or seed_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    keys = (
        seed_metrics[["product", "architecture"]]
        .drop_duplicates()
        .sort_values(["product", "architecture"])
    )
    for _, key in keys.iterrows():
        product = str(key["product"])
        architecture = str(key["architecture"])
        sm = seed_metrics.loc[
            (seed_metrics["product"] == product)
            & (seed_metrics["architecture"] == architecture)
        ]
        mae_vals = pd.to_numeric(sm["mean_horizon_MAE"], errors="coerce").dropna()
        mae_mean = float(mae_vals.mean()) if not mae_vals.empty else float("nan")
        mae_std = float(mae_vals.std(ddof=0)) if len(mae_vals) else float("nan")
        mae_min = float(mae_vals.min()) if not mae_vals.empty else float("nan")
        mae_max = float(mae_vals.max()) if not mae_vals.empty else float("nan")
        mae_cv = (
            float(mae_std / mae_mean)
            if mae_vals.size and np.isfinite(mae_mean) and mae_mean > 0.0
            else float("nan")
        )
        row: dict[str, Any] = {
            "product": product,
            "architecture": architecture,
            "n_seeds_with_metrics": int(len(mae_vals)),
            "mae_std_across_seeds": mae_std,
            "mae_min_across_seeds": mae_min,
            "mae_max_across_seeds": mae_max,
            "mae_range": (
                float(mae_max - mae_min)
                if np.isfinite(mae_min) and np.isfinite(mae_max)
                else float("nan")
            ),
            "mae_cv": mae_cv,
        }

        if seed_predictions is not None and not seed_predictions.empty:
            sp = seed_predictions.loc[
                (seed_predictions["product"] == product)
                & (seed_predictions["architecture"] == architecture)
            ]
        else:
            sp = pd.DataFrame()

        for h in range(1, int(forecast_horizon) + 1):
            col = f"pred_std_h{h}"
            if sp.empty:
                row[col] = float("nan")
                continue
            h_sub = sp.loc[sp["horizon"].astype(int) == h]
            if h_sub.empty:
                row[col] = float("nan")
                continue
            cell_stds: list[float] = []
            for _, g in h_sub.groupby(["origin", "target_date"], sort=False):
                if len(g) < 2:
                    continue
                cell_stds.append(float(g["prediction"].astype(float).std(ddof=0)))
            row[col] = float(np.mean(cell_stds)) if cell_stds else float("nan")
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


__all__ = [
    "ENSEMBLE_ORIGIN_STATUS_COLUMNS",
    "ENSEMBLE_PREDICTION_COLUMNS",
    "ENSEMBLE_UNAVAILABLE_REASON",
    "PREDICTION_KIND_ENSEMBLE",
    "PREDICTION_KIND_SEED",
    "build_seed_ensemble_predictions",
    "ensemble_metrics_table",
    "horizon_bias",
    "horizon_mae",
    "horizon_rmse",
    "horizon_wmape",
    "mean_horizon_mae",
    "metrics_summary_row",
    "seed_metrics_table",
    "selection_mae_from_horizons",
    "stability_table",
]
