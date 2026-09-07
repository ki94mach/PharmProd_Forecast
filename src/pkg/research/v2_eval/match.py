"""Section B: the matched legacy-vs-V2 prediction panel.

One row per ``product_id + forecast_origin + target_date + horizon``. Rows are
kept only when both engines produced a forecast and an actual exists; missing
forecasts and missing actuals are reported as coverage, never imputed as zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pkg.research.v2_eval.config import (
    CATEGORY_COLUMNS,
    HORIZON_GROUPS,
    MATCH_KEYS,
    V2EvalConfig,
)
from pkg.research.v2_eval.load import BackfillInputs


@dataclass(frozen=True)
class MatchedPanel:
    """Matched rows plus the unmatched accounting that explains them."""

    matched: pd.DataFrame
    v2_scored: pd.DataFrame
    unmatched_v2: pd.DataFrame
    unmatched_legacy: pd.DataFrame
    coverage_summary: dict[str, int]


def horizon_group(horizon: int) -> str:
    for name, low, high in HORIZON_GROUPS:
        if low <= horizon <= high:
            return name
    return "other"


def attach_actuals(frame: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Left-join actuals without filling; unmatched target months stay NaN."""
    out = frame.merge(
        sales,
        on=["product_id", "target_date"],
        how="left",
        validate="many_to_one",
    )
    return out


def build_matched_panel(data: BackfillInputs) -> MatchedPanel:
    v2 = data.v2_forecasts.copy()
    legacy = data.legacy.copy()

    v2_scored = attach_actuals(v2, data.sales)

    legacy_cols = [
        *MATCH_KEYS,
        "legacy_prediction",
        "actual_legacy_panel",
        "legacy_quarter",
        "legacy_model",
        *[c for c in CATEGORY_COLUMNS if c in legacy.columns],
    ]
    legacy_slim = legacy[legacy_cols]

    matched = v2_scored.merge(
        legacy_slim,
        on=list(MATCH_KEYS),
        how="inner",
        validate="one_to_one",
    )

    # Consistency gate: the frozen legacy panel and the raw sales panel must
    # agree on the actual for every matched key.
    both = matched.dropna(subset=["actual"])
    disagreement = both.loc[
        (both["actual"] - both["actual_legacy_panel"]).abs() > 1e-6
    ]
    if not disagreement.empty:
        raise AssertionError(
            "actual mismatch between raw sales and the legacy panel on "
            f"{len(disagreement)} matched rows; first key="
            f"{disagreement.iloc[0][list(MATCH_KEYS)].to_dict()}"
        )
    # The legacy panel is complete by construction, so fall back to it only
    # where the raw panel has no row (never to zero).
    matched["actual"] = matched["actual"].fillna(matched["actual_legacy_panel"])

    matched = matched.rename(columns={"forecast": "v2_prediction"})
    matched["v2_raw_prediction"] = matched["raw_forecast"]
    matched["v2_model"] = matched["model"]

    unmatched_v2 = v2_scored.merge(
        legacy_slim[list(MATCH_KEYS)].assign(_legacy=1),
        on=list(MATCH_KEYS),
        how="left",
    )
    unmatched_v2 = unmatched_v2.loc[unmatched_v2["_legacy"].isna()].drop(
        columns=["_legacy"]
    )
    unmatched_v2["reason"] = np.where(
        unmatched_v2["actual"].isna(),
        "no_legacy_forecast_and_no_actual",
        "no_legacy_forecast",
    )

    unmatched_legacy = legacy_slim.merge(
        v2[list(MATCH_KEYS)].assign(_v2=1), on=list(MATCH_KEYS), how="left"
    )
    unmatched_legacy = unmatched_legacy.loc[unmatched_legacy["_v2"].isna()].drop(
        columns=["_v2"]
    )
    unmatched_legacy["reason"] = "no_v2_forecast"

    evaluated = matched.dropna(subset=["actual", "v2_prediction", "legacy_prediction"])
    if len(evaluated) != len(matched):
        matched = evaluated.copy()

    matched = _add_error_columns(matched)
    matched["horizon_group"] = matched["horizon"].map(horizon_group)

    ordered = [
        "product_id",
        "forecast_origin",
        "target_date",
        "horizon",
        "horizon_group",
        "quarter",
        "legacy_quarter",
        "actual",
        "legacy_prediction",
        "v2_prediction",
        "v2_raw_prediction",
        "absolute_error_legacy",
        "absolute_error_v2",
        "signed_error_legacy",
        "signed_error_v2",
        "absolute_error_reduction",
        "legacy_model",
        "v2_model",
        *[c for c in CATEGORY_COLUMNS if c in matched.columns],
        "job_slug",
    ]
    matched = matched[[c for c in ordered if c in matched.columns]]

    coverage_summary = {
        "v2_forecast_rows": int(len(v2)),
        "v2_rows_with_actual": int(v2_scored["actual"].notna().sum()),
        "legacy_rows": int(len(legacy)),
        "matched_rows": int(len(matched)),
        "unmatched_v2_rows": int(len(unmatched_v2)),
        "unmatched_legacy_rows": int(len(unmatched_legacy)),
        "matched_products": int(matched["product_id"].nunique()),
        "matched_origins": int(matched["forecast_origin"].nunique()),
    }
    return MatchedPanel(
        matched=matched.reset_index(drop=True),
        v2_scored=v2_scored,
        unmatched_v2=unmatched_v2.reset_index(drop=True),
        unmatched_legacy=unmatched_legacy.reset_index(drop=True),
        coverage_summary=coverage_summary,
    )


def _add_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Signed error is prediction - actual, so positive means overforecasting.
    out["signed_error_legacy"] = out["legacy_prediction"] - out["actual"]
    out["signed_error_v2"] = out["v2_prediction"] - out["actual"]
    out["absolute_error_legacy"] = out["signed_error_legacy"].abs()
    out["absolute_error_v2"] = out["signed_error_v2"].abs()
    out["absolute_error_reduction"] = (
        out["absolute_error_legacy"] - out["absolute_error_v2"]
    )
    return out


def validate_matched_panel(panel: MatchedPanel) -> list[str]:
    """Structural checks; returns a list of human-readable problems."""
    problems: list[str] = []
    matched = panel.matched

    dup = int(matched.duplicated(subset=list(MATCH_KEYS)).sum())
    if dup:
        problems.append(f"{dup} duplicate keys in the matched panel")

    for column in ("actual", "legacy_prediction", "v2_prediction"):
        n_missing = int(matched[column].isna().sum())
        if n_missing:
            problems.append(f"{n_missing} missing values in {column}")

    if not np.allclose(
        matched["absolute_error_reduction"],
        matched["absolute_error_legacy"] - matched["absolute_error_v2"],
        equal_nan=True,
    ):
        problems.append("absolute_error_reduction is not legacy - v2")

    bad_horizons = sorted(set(matched["horizon"]) - set(range(1, 16)))
    if bad_horizons:
        problems.append(f"unexpected horizons {bad_horizons}")

    return problems


def fully_actualised_pairs(matched: pd.DataFrame) -> pd.DataFrame:
    """Restrict to SKU/origin pairs with all 15 horizons scored on both sides."""
    sizes = matched.groupby(["product_id", "forecast_origin"]).size()
    keep = sizes.loc[sizes == 15].index
    if len(keep) == 0:
        return matched.iloc[0:0].copy()
    index = matched.set_index(["product_id", "forecast_origin"]).index
    return matched.loc[index.isin(keep)].copy()


def write_matched_panel(panel: MatchedPanel, config: V2EvalConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    panel.matched.to_parquet(
        config.out_dir / "matched_predictions.parquet", index=False
    )
