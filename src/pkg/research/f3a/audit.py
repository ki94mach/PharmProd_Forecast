"""Pre-model F3A lifecycle audit (no XGB, no threshold tuning)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.features.lifecycle import (
    SCORED_FEATURE,
    add_lifecycle_features,
    product_lifecycle_catalog,
)


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return float("nan")
    return float(n) / float(d) * 100.0


def _age_percentiles(ages: np.ndarray) -> dict[str, float]:
    finite = ages[np.isfinite(ages)]
    if len(finite) == 0:
        keys = ("min_age", "p10_age", "p25_age", "median_age", "p75_age", "p90_age", "max_age")
        return {k: float("nan") for k in keys}
    qs = np.quantile(finite, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        "min_age": float(qs[0]),
        "p10_age": float(qs[1]),
        "p25_age": float(qs[2]),
        "median_age": float(qs[3]),
        "p75_age": float(qs[4]),
        "p90_age": float(qs[5]),
        "max_age": float(qs[6]),
    }


def age_quartile_edges(ages: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Quartile edges from available ages. None if too few finite values."""
    finite = np.asarray(ages, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        return None
    q1, q2, q3 = np.quantile(finite, [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def assign_age_group(age: float, edges: Optional[tuple[float, float, float]]) -> str:
    if not np.isfinite(age):
        return "age_missing"
    if edges is None:
        return "age_available"
    q1, q2, q3 = edges
    if age <= q1:
        return "Q1_youngest"
    if age <= q2:
        return "Q2"
    if age <= q3:
        return "Q3"
    return "Q4_oldest"


def audit_lifecycle(
    panel: pd.DataFrame,
    sales_hist: pd.DataFrame,
    *,
    origin_col: str = "origin",
) -> dict[str, pd.DataFrame]:
    """Audit scored age + diagnostics on PRIMARY test rows before F3A fits."""
    enriched = add_lifecycle_features(panel, sales_hist, origin_col=origin_col)
    catalog = product_lifecycle_catalog(sales_hist)
    products = sorted(enriched["product"].astype(str).unique())
    cat_mvp = catalog.loc[catalog["product"].isin(products)].copy()
    missing_prod = [p for p in products if p not in set(catalog["product"])]
    if missing_prod:
        extra = pd.DataFrame(
            {
                "product": missing_prod,
                "first_positive_sale_month": np.nan,
                "first_nonzero_sale_month": np.nan,
                "earliest_available_sales_month": (
                    catalog["earliest_available_sales_month"].iloc[0]
                    if len(catalog)
                    else np.nan
                ),
                "first_sale_left_censored": 0,
            }
        )
        cat_mvp = pd.concat([cat_mvp, extra], ignore_index=True)
    cat_mvp = cat_mvp.sort_values("product").reset_index(drop=True)

    age = enriched[SCORED_FEATURE].to_numpy(dtype=float)
    n_rows = len(enriched)
    n_products = int(enriched["product"].nunique())
    age_available = int(np.isfinite(age).sum())
    age_missing = n_rows - age_available
    left_prod = int((cat_mvp["first_sale_left_censored"] == 1).sum())
    left_rows = int((enriched["first_sale_left_censored"] == 1).sum())
    has_prior = enriched["has_prior_positive_sale"].to_numpy(dtype=float)
    n_has_prior = int((has_prior == 1).sum())
    prior_const = bool(np.unique(has_prior).size <= 1)

    pos = enriched["first_positive_sale_month"].to_numpy(dtype=float)
    nz = enriched["first_nonzero_sale_month"].to_numpy(dtype=float)
    both = np.isfinite(pos) & np.isfinite(nz)
    disagree_rows = int((both & (pos != nz)).sum())
    only_nz = int((~np.isfinite(pos) & np.isfinite(nz)).sum())
    cat_dis = cat_mvp.copy()
    cat_dis["positive_ne_nonzero"] = (
        cat_dis["first_positive_sale_month"].notna()
        & cat_dis["first_nonzero_sale_month"].notna()
        & (
            cat_dis["first_positive_sale_month"]
            != cat_dis["first_nonzero_sale_month"]
        )
    )
    disagree_products = int(cat_dis["positive_ne_nonzero"].sum())

    pooled = pd.DataFrame(
        [
            {
                "n_rows": n_rows,
                "n_products": n_products,
                "age_available_rows": age_available,
                "age_missing_rows": age_missing,
                "age_coverage_pct": _pct(age_available, n_rows),
                "left_censored_products": left_prod,
                "left_censored_product_pct": _pct(left_prod, n_products),
                "left_censored_rows": left_rows,
                "has_prior_positive_sale_rows": n_has_prior,
                "has_prior_positive_sale_constant": prior_const,
                "first_nonzero_disagree_products": disagree_products,
                "first_nonzero_disagree_rows": disagree_rows,
                "first_nonzero_only_rows": only_nz,
                **_age_percentiles(age),
            }
        ]
    )

    origin_rows = []
    for origin, g in enriched.groupby(origin_col):
        a = g[SCORED_FEATURE].to_numpy(dtype=float)
        finite = a[np.isfinite(a)]
        origin_rows.append(
            {
                "origin": int(origin),
                "n_rows": len(g),
                "age_coverage_pct": _pct(int(np.isfinite(a).sum()), len(g)),
                "min_age": float(np.min(finite)) if len(finite) else float("nan"),
                "median_age": float(np.median(finite)) if len(finite) else float("nan"),
                "max_age": float(np.max(finite)) if len(finite) else float("nan"),
            }
        )
    by_origin = pd.DataFrame(origin_rows).sort_values("origin").reset_index(drop=True)

    nz_audit = pd.DataFrame(
        [
            {
                "definition": "first_positive_sale = earliest month with sales > 0",
                "alternative_not_scored": "first_nonzero_sale = earliest month with sales != 0",
                "disagree_products": disagree_products,
                "disagree_rows": disagree_rows,
                "alternative_only_rows": only_nz,
                "note": (
                    "Zeros are not commercial launch. Negatives may be "
                    "returns/adjustments. Alternative is audited, not scored."
                ),
            }
        ]
    )

    return {
        "product_audit": cat_mvp[
            [
                "product",
                "first_positive_sale_month",
                "earliest_available_sales_month",
                "first_sale_left_censored",
            ]
        ],
        "coverage": pooled,
        "by_origin": by_origin,
        "first_nonzero_audit": nz_audit,
        "enriched_panel": enriched,
        "catalog": cat_mvp,
        "age_edges": age_quartile_edges(age),
    }
