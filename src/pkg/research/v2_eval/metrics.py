"""Section C: paired accuracy metrics on identical matched rows.

Conventions (fixed in :mod:`pkg.research.v2_eval.config`):

- ``WMAPE`` is a percent and delegates to :func:`pkg.benchmark.evaluate.wmape`,
  i.e. ``100 * sum|actual - pred| / sum|actual|`` over the whole slice. It is
  never the mean of per-SKU WMAPEs.
- ``mean_horizon_MAE`` is the equally weighted mean of horizon-level MAEs, the
  same construction as :func:`pkg.ts_v2.metrics.selection_mae_from_horizons`.
- ``signed bias = mean(prediction - actual)``; positive means overforecasting.
- Relative improvement ``100 * (legacy - v2) / legacy`` is only defined for the
  non-negative error metrics and returns NaN when the denominator is zero or
  non-finite. Bias is compared through its distance from zero instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as benchmark_wmape
from pkg.research.v2_eval.config import CATEGORY_COLUMNS, HORIZON_GROUPS

ERROR_METRICS = ("mean_horizon_mae", "wmape", "rmse", "mae")


def wmape_pct(actual: pd.Series, prediction: pd.Series) -> float:
    """Portfolio WMAPE in percent; NaN when sum|actual| is zero."""
    return float(benchmark_wmape(np.asarray(actual, dtype=float),
                                 np.asarray(prediction, dtype=float)))


def rmse(actual: pd.Series, prediction: pd.Series) -> float:
    err = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2)))


def mae(actual: pd.Series, prediction: pd.Series) -> float:
    err = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    if err.size == 0:
        return float("nan")
    return float(np.mean(np.abs(err)))


def signed_bias(actual: pd.Series, prediction: pd.Series) -> float:
    err = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err))


def mean_horizon_mae(frame: pd.DataFrame, error_column: str) -> float:
    """Equally weighted mean of horizon-level MAEs."""
    if frame.empty:
        return float("nan")
    per_horizon = frame.groupby("horizon")[error_column].mean()
    values = per_horizon.dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def relative_improvement_pct(legacy: float, v2: float) -> float:
    """``100 * (legacy - v2) / legacy`` with explicit zero-denominator handling."""
    if legacy is None or not np.isfinite(legacy) or legacy == 0:
        return float("nan")
    return float((legacy - v2) / legacy * 100.0)


def bias_movement(legacy: float, v2: float) -> dict[str, float | str]:
    """Compare signed biases by distance from zero rather than by ratio."""
    if not np.isfinite(legacy) or not np.isfinite(v2):
        return {
            "bias_abs_legacy": float("nan"),
            "bias_abs_v2": float("nan"),
            "bias_abs_change": float("nan"),
            "bias_direction": "undefined",
        }
    abs_legacy = abs(legacy)
    abs_v2 = abs(v2)
    change = abs_v2 - abs_legacy
    if change < 0:
        direction = "toward_zero"
    elif change > 0:
        direction = "away_from_zero"
    else:
        direction = "unchanged"
    if np.sign(legacy) != np.sign(v2) and legacy != 0 and v2 != 0:
        direction += "_sign_flip"
    return {
        "bias_abs_legacy": float(abs_legacy),
        "bias_abs_v2": float(abs_v2),
        "bias_abs_change": float(change),
        "bias_direction": direction,
    }


def metric_pair_rows(frame: pd.DataFrame, scope: str, label: str) -> list[dict]:
    """One row per metric comparing legacy and V2 over the same rows."""
    if frame.empty:
        return []
    actual = frame["actual"]
    legacy = frame["legacy_prediction"]
    v2 = frame["v2_prediction"]

    values = {
        "mean_horizon_mae": (
            mean_horizon_mae(frame, "absolute_error_legacy"),
            mean_horizon_mae(frame, "absolute_error_v2"),
        ),
        "wmape": (wmape_pct(actual, legacy), wmape_pct(actual, v2)),
        "rmse": (rmse(actual, legacy), rmse(actual, v2)),
        "mae": (mae(actual, legacy), mae(actual, v2)),
    }

    rows = []
    for metric, (legacy_value, v2_value) in values.items():
        rows.append(
            {
                "scope": scope,
                "slice": label,
                "metric": metric,
                "legacy": legacy_value,
                "v2": v2_value,
                "absolute_change": v2_value - legacy_value,
                "relative_improvement_pct": relative_improvement_pct(
                    legacy_value, v2_value
                ),
                "n": int(len(frame)),
                "n_products": int(frame["product_id"].nunique()),
                "n_origins": int(frame["forecast_origin"].nunique()),
            }
        )

    bias_legacy = signed_bias(actual, legacy)
    bias_v2 = signed_bias(actual, v2)
    movement = bias_movement(bias_legacy, bias_v2)
    rows.append(
        {
            "scope": scope,
            "slice": label,
            "metric": "signed_bias",
            "legacy": bias_legacy,
            "v2": bias_v2,
            "absolute_change": bias_v2 - bias_legacy,
            # Deliberately blank: the relative-error formula is meaningless for
            # a signed quantity that can cross zero.
            "relative_improvement_pct": float("nan"),
            "n": int(len(frame)),
            "n_products": int(frame["product_id"].nunique()),
            "n_origins": int(frame["forecast_origin"].nunique()),
            **movement,
        }
    )
    return rows


def overall_metrics(
    matched: pd.DataFrame,
    sensitivities: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    rows = metric_pair_rows(matched, "overall", "all_matched_rows")
    for label, frame in (sensitivities or {}).items():
        rows += metric_pair_rows(frame, "sensitivity", label)
    return pd.DataFrame(rows)


def horizon_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in matched.groupby("horizon"):
        for row in metric_pair_rows(group, "horizon", f"h{int(horizon)}"):
            row["horizon"] = int(horizon)
            rows.append(row)
    for name, _low, _high in HORIZON_GROUPS:
        group = matched.loc[matched["horizon_group"] == name]
        for row in metric_pair_rows(group, "horizon_group", name):
            row["horizon"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def origin_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, group in matched.groupby("forecast_origin"):
        quarters = sorted(set(group["quarter"]))
        legacy_quarters = sorted(set(group["legacy_quarter"].dropna()))
        for row in metric_pair_rows(group, "origin", str(int(origin))):
            row["forecast_origin"] = int(origin)
            row["v2_quarter"] = ",".join(quarters)
            row["legacy_quarter"] = ",".join(legacy_quarters)
            rows.append(row)
    return pd.DataFrame(rows)


def product_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    """Per-product paired metrics plus contribution to total error change."""
    rows = []
    for product, group in matched.groupby("product_id"):
        wmape_legacy = wmape_pct(group["actual"], group["legacy_prediction"])
        wmape_v2 = wmape_pct(group["actual"], group["v2_prediction"])
        rows.append(
            {
                "product_id": product,
                "n": int(len(group)),
                "n_origins": int(group["forecast_origin"].nunique()),
                "actual_volume": float(group["actual"].abs().sum()),
                "wmape_legacy": wmape_legacy,
                "wmape_v2": wmape_v2,
                "relative_improvement_pct": relative_improvement_pct(
                    wmape_legacy, wmape_v2
                ),
                "mae_legacy": float(group["absolute_error_legacy"].mean()),
                "mae_v2": float(group["absolute_error_v2"].mean()),
                "bias_legacy": float(group["signed_error_legacy"].mean()),
                "bias_v2": float(group["signed_error_v2"].mean()),
                "total_absolute_error_legacy": float(
                    group["absolute_error_legacy"].sum()
                ),
                "total_absolute_error_v2": float(group["absolute_error_v2"].sum()),
                "total_absolute_error_reduction": float(
                    group["absolute_error_reduction"].sum()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    total_reduction = frame["total_absolute_error_reduction"].sum()
    frame["share_of_net_error_reduction"] = (
        frame["total_absolute_error_reduction"] / total_reduction
        if total_reduction != 0
        else np.nan
    )
    return frame.sort_values("actual_volume", ascending=False).reset_index(drop=True)


def category_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    """Breakdown over the frozen product categories, when present."""
    rows = []
    for column in CATEGORY_COLUMNS:
        if column not in matched.columns:
            continue
        for value, group in matched.groupby(matched[column].astype(str)):
            for row in metric_pair_rows(group, f"category:{column}", str(value)):
                row["category_column"] = column
                row["category_value"] = str(value)
                rows.append(row)
    return pd.DataFrame(rows)
