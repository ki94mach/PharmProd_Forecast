"""Point-in-time feature construction rules for residual-XGBoost v1.

Every CLEAN_QUANT / anchor feature must declare how ``feature_as_of_date``
is derived. Training for evaluation origin O requires:

  training_forecast_origin < O
  training_target_date <= O-2
  feature_as_of_date <= O-2
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import CLEAN_QUANT_FEATURES

# Anchor for frozen hybrid residual (not in CLEAN_QUANT_FEATURES).
HYBRID_ANCHOR = "f_a0_hybrid"
TS_RESID_FEATURE_LIST = [HYBRID_ANCHOR] + list(CLEAN_QUANT_FEATURES)


def _asof_forecast_origin(row: pd.Series) -> int:
    """Calendar / horizon / static attrs known at forecast issue time."""
    return int(row["forecast_origin"])


def _asof_sales_lag(lag: int) -> Callable[[pd.Series], int]:
    def _fn(row: pd.Series) -> int:
        return int(shamsi_add_months(int(row["forecast_origin"]), -lag))

    return _fn


def _asof_sales_roll3(row: pd.Series) -> int:
    """Roll3 uses lags 1..3; latest month touched is origin-1."""
    return int(shamsi_add_months(int(row["forecast_origin"]), -1))


def _asof_static(_row: pd.Series) -> int:
    """Product encodings are static attributes; treat as always available."""
    return 0


FEATURE_ASOF_RULES: dict[str, dict[str, Any]] = {
    HYBRID_ANCHOR: {
        "rule": "forecast_issued_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Frozen hybrid forecast for the training row's own origin.",
    },
    "horizon": {
        "rule": "known_at_forecast_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Horizon index known when forecast is issued.",
    },
    "month": {
        "rule": "target_calendar_known_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Target-month calendar attr known at origin (no sales look-ahead).",
    },
    "quarter": {
        "rule": "target_calendar_known_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Target-quarter calendar attr known at origin.",
    },
    "sales_lag_1": {
        "rule": "sales_at_origin_minus_1",
        "as_of_fn": _asof_sales_lag(1),
        "notes": "Sales at Fo-1 only.",
    },
    "sales_lag_2": {
        "rule": "sales_at_origin_minus_2",
        "as_of_fn": _asof_sales_lag(2),
        "notes": "Sales at Fo-2 only.",
    },
    "sales_lag_3": {
        "rule": "sales_at_origin_minus_3",
        "as_of_fn": _asof_sales_lag(3),
        "notes": "Sales at Fo-3 only.",
    },
    "sales_lag_12": {
        "rule": "sales_at_origin_minus_12",
        "as_of_fn": _asof_sales_lag(12),
        "notes": "Sales at Fo-12 only.",
    },
    "sales_roll3": {
        "rule": "mean_of_sales_lag_1_2_3",
        "as_of_fn": _asof_sales_roll3,
        "notes": "Latest sales month in roll is Fo-1.",
    },
    "model_enc": {
        "rule": "static_product_attribute",
        "as_of_fn": _asof_static,
        "notes": "Static encoding; as_of sentinel 0.",
    },
    "field_enc": {
        "rule": "static_product_attribute",
        "as_of_fn": _asof_static,
        "notes": "Static encoding; as_of sentinel 0.",
    },
    "form_enc": {
        "rule": "static_product_attribute",
        "as_of_fn": _asof_static,
        "notes": "Static encoding; as_of sentinel 0.",
    },
    "provider_enc": {
        "rule": "static_product_attribute",
        "as_of_fn": _asof_static,
        "notes": "Static encoding; as_of sentinel 0.",
    },
}


def assert_all_features_have_pit_rules(features: list[str] | None = None) -> None:
    feats = list(features) if features is not None else list(TS_RESID_FEATURE_LIST)
    missing = [f for f in feats if f not in FEATURE_ASOF_RULES]
    if missing:
        raise ValueError(f"Features missing PIT as_of rules: {missing}")


def feature_as_of_date(feature: str, row: pd.Series) -> int:
    assert_all_features_have_pit_rules([feature])
    return int(FEATURE_ASOF_RULES[feature]["as_of_fn"](row))


def row_max_feature_as_of(row: pd.Series, features: list[str] | None = None) -> int:
    feats = list(features) if features is not None else list(TS_RESID_FEATURE_LIST)
    assert_all_features_have_pit_rules(feats)
    return max(feature_as_of_date(f, row) for f in feats)


def pit_feature_schema_manifest() -> dict[str, Any]:
    assert_all_features_have_pit_rules()
    return {
        "feature_list": list(TS_RESID_FEATURE_LIST),
        "anchor": HYBRID_ANCHOR,
        "clean_quant_features": list(CLEAN_QUANT_FEATURES),
        "features": {
            name: {
                "rule": meta["rule"],
                "notes": meta["notes"],
            }
            for name, meta in FEATURE_ASOF_RULES.items()
            if name in TS_RESID_FEATURE_LIST
        },
    }


def filter_train_rows_pit(
    frame: pd.DataFrame,
    *,
    evaluation_origin: int,
    origin_minus_2: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Keep rows satisfying all three PIT predicates for evaluation origin O.

    For the registered residual feature set, every ``as_of`` depends only on
    ``forecast_origin``; the latest sales touch is Fo-1, so
    ``feature_as_of_date <= O-2`` is enforced as ``Fo-1 <= O-2``.
    """
    if frame.empty:
        return frame.copy()
    feats = list(features) if features is not None else list(TS_RESID_FEATURE_LIST)
    assert_all_features_have_pit_rules(feats)
    o = int(evaluation_origin)
    cut = int(origin_minus_2)
    fo = frame["forecast_origin"].astype(int)
    td = frame["target_date"].astype(int)
    # Vectorized Fo-1 (latest sales as_of among registered features).
    fo_minus_1 = fo.map(lambda x: shamsi_add_months(int(x), -1))
    mask = (fo < o) & (td <= cut) & (fo_minus_1 <= cut)
    return frame.loc[mask].copy()
