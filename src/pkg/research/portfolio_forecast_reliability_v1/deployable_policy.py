"""Deployable frozen-policy classification for reliability portfolio activation.

Identifier / availability only: does not open HB forecast qty columns.
Never zero-fills or modifies the frozen TS policy.
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
    DEPLOYABLE_PORTFOLIO_COHORT,
    NO_MACHINE_FORECAST,
    NO_MACHINE_FORECAST_COHORT,
    SALES_FULL_PORTFOLIO_REL,
)


def classify_native_branch(
    product: str,
    origin: int,
    *,
    a0_complete: set[tuple[str, int]],
    v21_complete: set[tuple[str, int]],
) -> str | None:
    """Return native branch, or None if neither V2.1 nor A0 is complete."""
    key = (str(product), int(origin))
    a0_ok = key in a0_complete
    v21_ok = key in v21_complete
    if a0_ok and v21_ok:
        return DEPLOYABLE_BRANCH_BOTH
    if v21_ok:
        return DEPLOYABLE_BRANCH_V21
    if a0_ok:
        return DEPLOYABLE_BRANCH_A0
    return None


def naive_available_through_origin_minus_2(
    sales: pd.DataFrame,
    product: str,
    origin: int,
) -> dict[str, Any]:
    """True iff ≥1 finite sales row with date ≤ origin−2 (production naive rule).

    Does not return the naive forecast value (no outcome scoring / no zero-fill).
    """
    cutoff = int(shamsi_add_months(int(origin), -2))
    g = sales.loc[
        (sales["product"].astype(str) == str(product))
        & (pd.to_numeric(sales["date"], errors="coerce") <= cutoff)
    ]
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
    """Partition all HB-complete product×origins into deployable vs NO_MACHINE_FORECAST.

    Identity (must hold):
      n_native + n_valid_naive + n_no_machine_forecast = n_hb_complete
    """
    by_disposition: dict[str, int] = {
        DEPLOYABLE_BRANCH_BOTH: 0,
        DEPLOYABLE_BRANCH_V21: 0,
        DEPLOYABLE_BRANCH_A0: 0,
        DEPLOYABLE_BRANCH_NEITHER_NAIVE: 0,  # valid naive_last_value only
        NO_MACHINE_FORECAST: 0,
    }
    deployable_pairs: set[tuple[str, int]] = set()
    no_machine_pairs: set[tuple[str, int]] = set()
    no_machine_rows: list[dict[str, Any]] = []
    valid_naive_rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for product, origin in sorted(human_complete):
        native = classify_native_branch(
            product,
            origin,
            a0_complete=a0_complete,
            v21_complete=v21_complete,
        )
        row: dict[str, Any] = {
            "product": product,
            "forecast_origin": int(origin),
        }
        if native is not None:
            row["disposition"] = native
            row["deployable"] = True
            row["enters_accuracy_denominator"] = True
            row["naive_available"] = None
            row["origin_minus_2_cutoff"] = int(shamsi_add_months(int(origin), -2))
            by_disposition[native] += 1
            deployable_pairs.add((product, int(origin)))
        else:
            nav = naive_available_through_origin_minus_2(sales, product, origin)
            row.update(nav)
            if nav["naive_available"]:
                row["disposition"] = DEPLOYABLE_BRANCH_NEITHER_NAIVE
                row["deployable"] = True
                row["enters_accuracy_denominator"] = True
                row["note"] = (
                    "Production naive_last_value fallback from sales through "
                    "origin−2; frozen TS policy unchanged (no zero-fill)."
                )
                by_disposition[DEPLOYABLE_BRANCH_NEITHER_NAIVE] += 1
                deployable_pairs.add((product, int(origin)))
                valid_naive_rows.append(row)
            else:
                row["disposition"] = NO_MACHINE_FORECAST
                row["deployable"] = False
                row["enters_accuracy_denominator"] = False
                row["coverage_failure"] = True
                row["fail_reason"] = (
                    "HB-complete product×origin has no native V2.1/A0 and no "
                    "sales through origin−2 for naive_last_value; classified "
                    f"{NO_MACHINE_FORECAST} (not a deployable naive forecast; "
                    "never zero-fill)."
                )
                by_disposition[NO_MACHINE_FORECAST] += 1
                no_machine_pairs.add((product, int(origin)))
                no_machine_rows.append(row)
        classifications.append(row)

    n_human = len(human_complete)
    n_native = (
        by_disposition[DEPLOYABLE_BRANCH_BOTH]
        + by_disposition[DEPLOYABLE_BRANCH_V21]
        + by_disposition[DEPLOYABLE_BRANCH_A0]
    )
    n_valid_naive = by_disposition[DEPLOYABLE_BRANCH_NEITHER_NAIVE]
    n_no_machine = by_disposition[NO_MACHINE_FORECAST]
    n_deployable = len(deployable_pairs)
    identity_ok = (n_native + n_valid_naive + n_no_machine) == n_human
    identity_ok = identity_ok and (n_deployable + n_no_machine) == n_human
    identity_ok = identity_ok and n_deployable == (n_native + n_valid_naive)

    no_machine_skus = sorted({p for p, _o in no_machine_pairs})
    coverage_pct = (
        round(100.0 * n_deployable / n_human, 2) if n_human else None
    )

    return {
        "rule": (
            "DEPLOYABLE_PORTFOLIO_COHORT = HB-complete product×origins with an "
            "actual frozen-policy forecast (native V2.1/A0 or valid "
            "naive_last_value from sales through origin−2). "
            "Cases without either are NO_MACHINE_FORECAST — coverage failures, "
            "excluded from accuracy denominators. Never zero-fill or modify the "
            "frozen TS policy."
        ),
        "naive_construction": {
            "method": "production naive_last_value",
            "sales_through": "origin_minus_2",
            "module_reference": "pkg.portfolio_ts_lock.dense_actuals.last_train_value",
            "sales_source_relative": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "hb_forecast_columns_read": [],
            "zero_fill_forbidden": True,
            "frozen_ts_policy_modified": False,
        },
        "identity": {
            "n_native": n_native,
            "n_valid_naive": n_valid_naive,
            "n_no_machine_forecast": n_no_machine,
            "n_hb_complete": n_human,
            "equation": "n_native + n_valid_naive + n_no_machine_forecast = n_hb_complete",
            "holds": identity_ok,
            "expected_example": "615 + 224 + 41 = 880",
        },
        "n_hb_complete_product_origins_required": n_human,
        "n_deployable_with_forecast": n_deployable,
        "n_no_machine_forecast": n_no_machine,
        "deployable_coverage_fraction": (
            n_deployable / n_human if n_human else None
        ),
        "deployable_coverage_pct": coverage_pct,
        "reconciliation_status": "RECONCILED" if identity_ok else "IDENTITY_BROKEN",
        "by_disposition_among_hb_complete": by_disposition,
        # back-compat alias (neither_naive count = valid naive only)
        "by_branch_among_hb_complete": {
            DEPLOYABLE_BRANCH_BOTH: by_disposition[DEPLOYABLE_BRANCH_BOTH],
            DEPLOYABLE_BRANCH_V21: by_disposition[DEPLOYABLE_BRANCH_V21],
            DEPLOYABLE_BRANCH_A0: by_disposition[DEPLOYABLE_BRANCH_A0],
            DEPLOYABLE_BRANCH_NEITHER_NAIVE: n_valid_naive,
            NO_MACHINE_FORECAST: n_no_machine,
        },
        "n_neither_naive_with_naive_available": n_valid_naive,
        "n_failed_no_deployable_forecast": n_no_machine,  # legacy key → NO_MACHINE
        DEPLOYABLE_PORTFOLIO_COHORT: {
            "definition": (
                "HB-complete product×origins with an actual frozen-policy "
                "forecast, including valid naive_last_value fallbacks"
            ),
            "n_product_origin_pairs": n_deployable,
            "n_native": n_native,
            "n_valid_naive": n_valid_naive,
            "enters_accuracy_denominator": True,
            "is_portfolio_activation_gate_population": True,
        },
        NO_MACHINE_FORECAST_COHORT: {
            "definition": (
                "HB-complete product×origins with neither native V2.1/A0 nor "
                "sales through origin−2; not deployable; never zero-filled"
            ),
            "n_product_origin_pairs": n_no_machine,
            "n_skus": len(no_machine_skus),
            "skus": no_machine_skus,
            "enters_accuracy_denominator": False,
            "reported_as": "coverage_failures",
            "rows": no_machine_rows,
        },
        "sku_x_group_mnih_policy": {
            "native_comparisons_only_for_MNIH": True,
            "any_evaluated_neither_naive_blocks_MNIH": True,
            "any_evaluated_NO_MACHINE_FORECAST_blocks_MNIH": True,
            "blocks_yield": "HUMAN_REVIEW_REQUIRED",
            "note": (
                "MACHINE_NONINFERIOR_TO_HUMAN requires native group-complete "
                "origins (≥3). neither_naive or NO_MACHINE_FORECAST on any "
                "evaluated origin → HUMAN_REVIEW_REQUIRED. Missing/incomplete "
                "HB evidence → INSUFFICIENT_HUMAN_EVIDENCE."
            ),
        },
        "portfolio_activation_population": {
            "name": DEPLOYABLE_PORTFOLIO_COHORT,
            "n_product_origin_pairs": n_deployable,
            "not": "HUMAN_COMPARABLE_COHORT with ≥3 origins",
        },
        "classifications_sample": classifications[:30],
        "n_classifications": len(classifications),
        "valid_naive_sample": valid_naive_rows[:20],
    }
