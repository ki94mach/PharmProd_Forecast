"""Deployable frozen-policy classification for reliability portfolio activation.

Identifier / availability only: does not open HB forecast qty columns.
Naive availability uses sales through origin−2 (production naive_last_value rule).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    DEPLOYABLE_BRANCH_A0,
    DEPLOYABLE_BRANCH_BOTH,
    DEPLOYABLE_BRANCH_NEITHER_NAIVE,
    DEPLOYABLE_BRANCH_V21,
    SALES_FULL_PORTFOLIO_REL,
)


def classify_deployable_branch(
    product: str,
    origin: int,
    *,
    a0_complete: set[tuple[str, int]],
    v21_complete: set[tuple[str, int]],
) -> str:
    key = (str(product), int(origin))
    a0_ok = key in a0_complete
    v21_ok = key in v21_complete
    if a0_ok and v21_ok:
        return DEPLOYABLE_BRANCH_BOTH
    if v21_ok:
        return DEPLOYABLE_BRANCH_V21
    if a0_ok:
        return DEPLOYABLE_BRANCH_A0
    return DEPLOYABLE_BRANCH_NEITHER_NAIVE


def naive_available_through_origin_minus_2(
    sales: pd.DataFrame,
    product: str,
    origin: int,
) -> dict[str, Any]:
    """True iff ≥1 sales row with date ≤ origin−2 (production naive_last_value rule).

    Does not return the naive forecast value (no outcome scoring).
    """
    cutoff = int(shamsi_add_months(int(origin), -2))
    g = sales.loc[
        (sales["product"].astype(str) == str(product))
        & (pd.to_numeric(sales["date"], errors="coerce") <= cutoff)
    ]
    # Require a finite sales observation to mirror last_train_value
    if g.empty:
        return {
            "naive_available": False,
            "origin_minus_2_cutoff": cutoff,
            "reason": "no_sales_through_origin_minus_2",
        }
    vals = pd.to_numeric(g["sales"], errors="coerce")
    if not bool(vals.notna().any()):
        return {
            "naive_available": False,
            "origin_minus_2_cutoff": cutoff,
            "reason": "sales_rows_present_but_all_non_finite",
        }
    return {
        "naive_available": True,
        "origin_minus_2_cutoff": cutoff,
        "reason": None,
    }


def reconcile_hb_complete_deployable(
    *,
    human_complete: set[tuple[str, int]],
    a0_complete: set[tuple[str, int]],
    v21_complete: set[tuple[str, int]],
    sales: pd.DataFrame,
) -> dict[str, Any]:
    """Reconcile every HB-complete product×origin to a deployable forecast.

    Portfolio activation includes production naive_last_value for neither-native.
    Returns explicit failures when naive cannot be built from sales through Fo−2.
    """
    by_branch: dict[str, int] = {
        DEPLOYABLE_BRANCH_BOTH: 0,
        DEPLOYABLE_BRANCH_V21: 0,
        DEPLOYABLE_BRANCH_A0: 0,
        DEPLOYABLE_BRANCH_NEITHER_NAIVE: 0,
    }
    deployable_pairs: set[tuple[str, int]] = set()
    neither_with_naive: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for product, origin in sorted(human_complete):
        branch = classify_deployable_branch(
            product,
            origin,
            a0_complete=a0_complete,
            v21_complete=v21_complete,
        )
        by_branch[branch] = by_branch.get(branch, 0) + 1
        row: dict[str, Any] = {
            "product": product,
            "forecast_origin": int(origin),
            "deployable_branch": branch,
        }
        if branch == DEPLOYABLE_BRANCH_NEITHER_NAIVE:
            nav = naive_available_through_origin_minus_2(sales, product, origin)
            row.update(nav)
            if nav["naive_available"]:
                deployable_pairs.add((product, int(origin)))
                neither_with_naive.append(row)
            else:
                failures.append(
                    {
                        **row,
                        "fail_reason": (
                            "HB-complete product×origin has no native V2.1/A0 and "
                            "no production naive_last_value from sales through "
                            f"origin−2 ({nav['origin_minus_2_cutoff']})"
                        ),
                    }
                )
        else:
            row["naive_available"] = None
            row["origin_minus_2_cutoff"] = int(shamsi_add_months(int(origin), -2))
            deployable_pairs.add((product, int(origin)))
        classifications.append(row)

    n_human = len(human_complete)
    n_deployable = len(deployable_pairs)
    n_fail = len(failures)
    status = "PASSED" if n_fail == 0 and n_deployable == n_human else "FAILED"
    return {
        "rule": (
            "Portfolio activation uses deployable frozen policy: native V2.1/A0 "
            "when present; else production naive_last_value from sales through "
            "origin−2. Every HB-complete product×origin must have a deployable "
            "forecast."
        ),
        "naive_construction": {
            "method": "production naive_last_value",
            "sales_through": "origin_minus_2",
            "module_reference": "pkg.portfolio_ts_lock.dense_actuals.last_train_value",
            "sales_source_relative": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "hb_forecast_columns_read": [],
        },
        "n_hb_complete_product_origins_required": n_human,
        "n_deployable_with_forecast": n_deployable,
        "n_failed_no_deployable_forecast": n_fail,
        "reconciliation_status": status,
        "by_branch_among_hb_complete": by_branch,
        "n_neither_naive_with_naive_available": len(neither_with_naive),
        "n_neither_naive_failed": sum(
            1
            for f in failures
            if f.get("deployable_branch") == DEPLOYABLE_BRANCH_NEITHER_NAIVE
        ),
        "failures": failures,
        "sku_x_group_mnih_policy": {
            "native_only_for_MNIH": True,
            "any_evaluated_neither_naive_blocks_MNIH": True,
            "note": (
                "Portfolio activation may include naive_fallback rows; "
                "SKU×validity-group MACHINE_NONINFERIOR_TO_HUMAN remains "
                "blocked if any evaluated origin is neither_naive."
            ),
        },
        "portfolio_activation_population": {
            "definition": (
                "HB-complete product×origins with a deployable frozen-policy "
                "forecast (native or naive_last_value)"
            ),
            "n_product_origin_pairs": n_deployable,
            "includes_neither_naive_when_naive_available": True,
        },
        # classifications omitted from JSON by default (large); sample kept
        "classifications_sample": classifications[:30],
        "n_classifications": len(classifications),
    }
