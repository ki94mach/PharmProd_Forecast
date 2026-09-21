"""Expanding-window origin classification for residual-XGBoost v1.

Origins are the independent temporal units. Never use a random row split.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.research.portfolio_residual_xgb_v1.constants import (
    MIN_PRIOR_MATURED_ORIGINS,
    MIN_TRAIN_ROWS,
    MIN_TRAIN_SKU_ORIGINS,
    ORIGIN_CLASS_EVALUABLE,
    ORIGIN_CLASS_INSUFFICIENT,
    ORIGIN_CLASS_WARMUP,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import filter_train_rows_pit


def sorted_residual_origins(
    origins: Sequence[int] | None = None,
) -> list[int]:
    base = (
        sorted(int(o) for o in origins)
        if origins is not None
        else sorted(RESIDUAL_DEVELOPMENT_ORIGINS)
    )
    return base


def classify_origin(
    *,
    evaluation_origin: int,
    residual_origins_sorted: Sequence[int],
    train_rows: int,
    train_sku_origins: int,
    min_prior_matured_origins: int = MIN_PRIOR_MATURED_ORIGINS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_train_sku_origins: int = MIN_TRAIN_SKU_ORIGINS,
) -> str:
    origins = list(residual_origins_sorted)
    o = int(evaluation_origin)
    if o not in origins:
        raise ValueError(f"evaluation_origin {o} not in residual_development set")
    idx = origins.index(o)
    n_prior = idx  # strictly earlier residual origins
    if n_prior < int(min_prior_matured_origins):
        return ORIGIN_CLASS_WARMUP
    if int(train_rows) < int(min_train_rows) or int(train_sku_origins) < int(
        min_train_sku_origins
    ):
        return ORIGIN_CLASS_INSUFFICIENT
    return ORIGIN_CLASS_EVALUABLE


def build_oof_split_manifest(
    residual_panel: pd.DataFrame,
    *,
    residual_origins: Sequence[int] | None = None,
    min_prior_matured_origins: int = MIN_PRIOR_MATURED_ORIGINS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_train_sku_origins: int = MIN_TRAIN_SKU_ORIGINS,
) -> dict[str, Any]:
    """Classify each residual_development origin under expanding-window PIT rules."""
    origins = sorted_residual_origins(residual_origins)
    rows_out: list[dict[str, Any]] = []
    for o in origins:
        o2 = shamsi_add_months(int(o), -2)
        train = filter_train_rows_pit(
            residual_panel,
            evaluation_origin=int(o),
            origin_minus_2=o2,
        )
        test = residual_panel.loc[
            residual_panel["forecast_origin"].astype(int) == int(o)
        ]
        n_train = int(len(train))
        n_sku = (
            int(train.groupby(["product", "forecast_origin"]).ngroups)
            if n_train
            else 0
        )
        klass = classify_origin(
            evaluation_origin=int(o),
            residual_origins_sorted=origins,
            train_rows=n_train,
            train_sku_origins=n_sku,
            min_prior_matured_origins=min_prior_matured_origins,
            min_train_rows=min_train_rows,
            min_train_sku_origins=min_train_sku_origins,
        )
        rows_out.append(
            {
                "evaluation_origin": int(o),
                "origin_minus_2": int(o2),
                "class": klass,
                "n_prior_residual_origins": int(origins.index(o)),
                "n_train_rows": n_train,
                "n_train_sku_origins": n_sku,
                "n_test_rows": int(len(test)),
                "pit_rules": {
                    "training_forecast_origin_lt_O": True,
                    "training_target_date_le_O_minus_2": True,
                    "feature_as_of_date_le_O_minus_2": True,
                },
            }
        )

    by_class: dict[str, list[int]] = {
        ORIGIN_CLASS_WARMUP: [],
        ORIGIN_CLASS_INSUFFICIENT: [],
        ORIGIN_CLASS_EVALUABLE: [],
    }
    for r in rows_out:
        by_class[r["class"]].append(int(r["evaluation_origin"]))

    return {
        "split_method": "expanding_window_by_origin",
        "random_row_split": False,
        "minima": {
            "min_prior_matured_origins": int(min_prior_matured_origins),
            "min_train_rows": int(min_train_rows),
            "min_train_sku_origins": int(min_train_sku_origins),
        },
        "origins": rows_out,
        "by_class": by_class,
        "eligible_evaluation_origins": list(by_class[ORIGIN_CLASS_EVALUABLE]),
    }
