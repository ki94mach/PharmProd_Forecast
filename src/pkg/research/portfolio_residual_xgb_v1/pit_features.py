"""Point-in-time feature construction rules for residual-XGBoost.

Production semantics (v1.1): last_complete_month = forecast_origin − 2.

  sales_lag_1  = last_complete_month
  sales_lag_2  = last_complete_month − 1
  sales_lag_3  = last_complete_month − 2
  sales_lag_12 = last_complete_month − 11
  sales_roll3  = mean of the three months ending at last_complete_month

No sales-derived feature may use origin−1 or later.

Training for evaluation origin O requires:
  training_forecast_origin < O
  training_target_date <= O−2
  every sales feature date <= O−2
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import CLEAN_QUANT_FEATURES

HYBRID_ANCHOR = "f_a0_hybrid"
TS_RESID_FEATURE_LIST = [HYBRID_ANCHOR] + list(CLEAN_QUANT_FEATURES)

SALES_FEATURE_NAMES: tuple[str, ...] = (
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
)

# Offset from forecast_origin to the sales month used by each lag:
# last_complete = Fo-2; lag_k uses last_complete-(k-1) = Fo-(k+1)
_SALES_LAG_ORIGIN_OFFSET: dict[str, int] = {
    "sales_lag_1": -2,  # Fo-2
    "sales_lag_2": -3,  # Fo-3
    "sales_lag_3": -4,  # Fo-4
    "sales_lag_12": -13,  # Fo-13 = (Fo-2)-11
}


def last_complete_month(forecast_origin: int) -> int:
    """Production last complete sales month = origin − 2."""
    return int(shamsi_add_months(int(forecast_origin), -2))


def sales_feature_month(feature: str, forecast_origin: int) -> int:
    """Absolute Shamsi month touched by a sales-derived feature."""
    fo = int(forecast_origin)
    if feature == "sales_roll3":
        return last_complete_month(fo)  # latest month in the 3-month window
    if feature not in _SALES_LAG_ORIGIN_OFFSET:
        raise KeyError(feature)
    return int(shamsi_add_months(fo, _SALES_LAG_ORIGIN_OFFSET[feature]))


def _asof_forecast_origin(row: pd.Series) -> int:
    return int(row["forecast_origin"])


def _asof_sales_feature(feature: str) -> Callable[[pd.Series], int]:
    def _fn(row: pd.Series) -> int:
        return sales_feature_month(feature, int(row["forecast_origin"]))

    return _fn


def _asof_static(_row: pd.Series) -> int:
    return 0


FEATURE_ASOF_RULES: dict[str, dict[str, Any]] = {
    HYBRID_ANCHOR: {
        "rule": "forecast_issued_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Frozen hybrid forecast for the training row's own origin.",
        "sales_derived": False,
    },
    "horizon": {
        "rule": "known_at_forecast_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Horizon index known when forecast is issued.",
        "sales_derived": False,
    },
    "month": {
        "rule": "target_calendar_known_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Target-month calendar attr known at origin (no sales look-ahead).",
        "sales_derived": False,
    },
    "quarter": {
        "rule": "target_calendar_known_at_origin",
        "as_of_fn": _asof_forecast_origin,
        "notes": "Target-quarter calendar attr known at origin.",
        "sales_derived": False,
    },
    "sales_lag_1": {
        "rule": "sales_at_last_complete_month",
        "as_of_fn": _asof_sales_feature("sales_lag_1"),
        "notes": "last_complete_month = Fo-2. Never uses Fo-1.",
        "sales_derived": True,
    },
    "sales_lag_2": {
        "rule": "sales_at_last_complete_minus_1",
        "as_of_fn": _asof_sales_feature("sales_lag_2"),
        "notes": "last_complete_month-1 = Fo-3.",
        "sales_derived": True,
    },
    "sales_lag_3": {
        "rule": "sales_at_last_complete_minus_2",
        "as_of_fn": _asof_sales_feature("sales_lag_3"),
        "notes": "last_complete_month-2 = Fo-4.",
        "sales_derived": True,
    },
    "sales_lag_12": {
        "rule": "sales_at_last_complete_minus_11",
        "as_of_fn": _asof_sales_feature("sales_lag_12"),
        "notes": "last_complete_month-11 = Fo-13.",
        "sales_derived": True,
    },
    "sales_roll3": {
        "rule": "mean_three_months_ending_last_complete",
        "as_of_fn": _asof_sales_feature("sales_roll3"),
        "notes": "Mean of Fo-4, Fo-3, Fo-2; latest = last_complete_month.",
        "sales_derived": True,
    },
    "model_enc": {
        "rule": "categorical_fit_on_train_only",
        "as_of_fn": _asof_static,
        "notes": "Fit LabelEncoder on training rows only; unknown → dedicated code.",
        "sales_derived": False,
    },
    "field_enc": {
        "rule": "categorical_fit_on_train_only",
        "as_of_fn": _asof_static,
        "notes": "Fit LabelEncoder on training rows only; unknown → dedicated code.",
        "sales_derived": False,
    },
    "form_enc": {
        "rule": "categorical_fit_on_train_only",
        "as_of_fn": _asof_static,
        "notes": "Fit LabelEncoder on training rows only; unknown → dedicated code.",
        "sales_derived": False,
    },
    "provider_enc": {
        "rule": "categorical_fit_on_train_only",
        "as_of_fn": _asof_static,
        "notes": "Fit LabelEncoder on training rows only; unknown → dedicated code.",
        "sales_derived": False,
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


def row_max_sales_feature_date(row: pd.Series) -> int:
    """Latest Shamsi month among sales-derived features for this row."""
    fo = int(row["forecast_origin"])
    return max(sales_feature_month(f, fo) for f in SALES_FEATURE_NAMES)


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
        "production_semantics": {
            "last_complete_month": "forecast_origin-2",
            "sales_lag_1": "last_complete_month",
            "sales_lag_2": "last_complete_month-1",
            "sales_lag_3": "last_complete_month-2",
            "sales_lag_12": "last_complete_month-11",
            "sales_roll3": "mean of three months ending at last_complete_month",
            "forbidden": "No sales-derived feature may use origin-1 or later.",
        },
        "features": {
            name: {
                "rule": meta["rule"],
                "notes": meta["notes"],
                "sales_derived": bool(meta.get("sales_derived", False)),
            }
            for name, meta in FEATURE_ASOF_RULES.items()
            if name in TS_RESID_FEATURE_LIST
        },
        "categorical_encoding": {
            "fit_on": "training_rows_only",
            "unknown_category": "__UNKNOWN__",
            "deterministic": True,
        },
    }


def filter_train_rows_pit(
    frame: pd.DataFrame,
    *,
    evaluation_origin: int,
    origin_minus_2: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Keep rows satisfying all PIT predicates for evaluation origin O.

    Sales as-of uses last_complete_month = Fo-2 (never Fo-1).
    """
    if frame.empty:
        return frame.copy()
    feats = list(features) if features is not None else list(TS_RESID_FEATURE_LIST)
    assert_all_features_have_pit_rules(feats)
    o = int(evaluation_origin)
    cut = int(origin_minus_2)
    fo = frame["forecast_origin"].astype(int)
    td = frame["target_date"].astype(int)
    # Latest sales month touched = Fo-2
    fo_last_complete = fo.map(lambda x: last_complete_month(int(x)))
    mask = (fo < o) & (td <= cut) & (fo_last_complete <= cut)
    return frame.loc[mask].copy()


def assert_no_sales_uses_origin_minus_1(forecast_origin: int) -> None:
    """Hard check: no sales feature month is Fo-1 or later."""
    fo = int(forecast_origin)
    forbidden = shamsi_add_months(fo, -1)
    for name in SALES_FEATURE_NAMES:
        m = sales_feature_month(name, fo)
        if m >= forbidden:
            raise AssertionError(
                f"{name} month {m} uses origin-1 or later relative to Fo={fo}"
            )
