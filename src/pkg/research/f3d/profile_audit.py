"""Pre-model F3D patient-consumption-profile feature audit (no XGB, no WMAPE)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark import load_benchmark
from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.research.f3d.config import f3d_profile_audit_dir
from pkg.research.features.patient_consumption import (
    FEATURE_NAMES,
    add_patient_consumption_features,
    load_frozen_profile,
)

QUANTILE_VALS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
QUANTILE_NAMES = ("min", "p10", "p25", "median", "p75", "p90", "max")


def _pct(n: int, d: int) -> float:
    return float(n) / float(d) * 100.0 if d > 0 else float("nan")


def _finite_mask(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").map(lambda v: bool(np.isfinite(v)))


def coverage_overall(enriched: pd.DataFrame) -> pd.DataFrame:
    n_rows = int(len(enriched))
    n_products = int(enriched["product"].nunique())
    type_avail = int(enriched["is_continuous_consumption"].notna().sum())
    log_avail = int(_finite_mask(enriched["log_patient_annual_consumption"]).sum())
    both_avail = int(
        (
            enriched["is_continuous_consumption"].notna()
            & _finite_mask(enriched["log_patient_annual_consumption"])
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "n_rows": n_rows,
                "n_products": n_products,
                "type_indicator_available_rows": type_avail,
                "type_indicator_coverage_pct": _pct(type_avail, n_rows),
                "log_annual_available_rows": log_avail,
                "log_annual_coverage_pct": _pct(log_avail, n_rows),
                "both_available_rows": both_avail,
                "both_coverage_pct": _pct(both_avail, n_rows),
            }
        ]
    )


def coverage_by_origin(
    enriched: pd.DataFrame, origin_col: str = "origin"
) -> pd.DataFrame:
    rows = []
    for o in sorted(PRIMARY_ORIGINS):
        g = enriched.loc[enriched[origin_col].astype(int) == int(o)]
        n = len(g)
        np_ = int(g["product"].nunique())
        t = int(g["is_continuous_consumption"].notna().sum())
        la = int(_finite_mask(g["log_patient_annual_consumption"]).sum())
        rows.append(
            {
                "origin": int(o),
                "n_rows": n,
                "n_products": np_,
                "type_available": t,
                "type_coverage_pct": _pct(t, n),
                "log_annual_available": la,
                "log_annual_coverage_pct": _pct(la, n),
            }
        )
    return pd.DataFrame(rows)


def coverage_by_product(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in enriched.groupby("product"):
        n = len(g)
        t = int(g["is_continuous_consumption"].notna().sum())
        la = int(_finite_mask(g["log_patient_annual_consumption"]).sum())
        rows.append(
            {
                "product": str(product),
                "n_rows": n,
                "type_available": t,
                "type_coverage_pct": _pct(t, n),
                "log_annual_available": la,
                "log_annual_coverage_pct": _pct(la, n),
            }
        )
    return pd.DataFrame(rows)


def distributions(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in (
        "patient_annual_consumption",
        "log_patient_annual_consumption",
        "is_continuous_consumption",
    ):
        if feat not in enriched.columns:
            continue
        vals = pd.to_numeric(enriched[feat], errors="coerce").dropna()
        row: dict = {"feature": feat, "count": len(vals), "missing": len(enriched) - len(vals)}
        if len(vals) > 0:
            qs = np.quantile(vals, QUANTILE_VALS)
            for lbl, q in zip(QUANTILE_NAMES, qs):
                row[lbl] = float(q)
        else:
            for lbl in QUANTILE_NAMES:
                row[lbl] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_profile_features(
    *,
    out_dir: Optional[Path] = None,
    verify_freeze: bool = True,
) -> dict:
    """Full F3D Step 2 feature audit (no XGBoost)."""
    aud_dir = out_dir or f3d_profile_audit_dir()
    aud_dir.mkdir(parents=True, exist_ok=True)

    profile = load_frozen_profile()
    ds = load_benchmark(verify_checksums=verify_freeze)

    # Attach features to matched universe (primary origins)
    matched = ds.matched_universe.copy()
    neg_report: list = []
    enriched = add_patient_consumption_features(matched, profile, negative_report=neg_report)

    # Use origin column present in the dataset
    origin_col = "origin" if "origin" in enriched.columns else "ts_origin"

    overall = coverage_overall(enriched)
    by_origin = coverage_by_origin(enriched, origin_col)
    by_product = coverage_by_product(enriched)
    dists = distributions(enriched)

    overall.to_csv(aud_dir / "coverage_overall.csv", index=False)
    by_origin.to_csv(aud_dir / "coverage_by_origin.csv", index=False)
    by_product.to_csv(aud_dir / "coverage_by_product.csv", index=False)
    dists.to_csv(aud_dir / "distributions.csv", index=False)

    neg_df = pd.DataFrame(neg_report)
    if not neg_df.empty:
        neg_df.to_csv(aud_dir / "negative_period_rows.csv", index=False)

    return {
        "overall": overall,
        "by_origin": by_origin,
        "by_product": by_product,
        "distributions": dists,
        "negative_report": neg_df,
        "enriched": enriched,
        "out_dir": aud_dir,
    }
