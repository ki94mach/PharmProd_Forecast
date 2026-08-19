"""Pre-model F3C inventory-feature audit (no XGB, no threshold tuning)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark import load_benchmark
from pkg.benchmark.config import PRIMARY_ORIGINS, default_benchmark_root
from pkg.research.f3c.config import f3c_feature_audit_dir, f3c_source_dir
from pkg.research.features.inventory import (
    FEATURE_NAMES,
    RAW_QTY_NAMES,
    add_inventory_features,
    load_frozen_distributor_inventory,
    load_frozen_factory_inventory,
)

DIST_COLS = (
    "distributor_inventory_qty",
    "log_distributor_inventory_qty",
    "factory_inventory_qty",
    "log_factory_inventory_qty",
)
QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p1", "p10", "p25", "median", "p75", "p90", "p99", "max")


def _pct(n: int, d: int) -> float:
    return float(n) / float(d) * 100.0 if d > 0 else float("nan")


def _finite_mask(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").map(lambda v: bool(np.isfinite(v)))


def coverage_overall(enriched: pd.DataFrame) -> pd.DataFrame:
    n_rows = int(len(enriched))
    n_products = int(enriched["product"].nunique())
    dist_avail = int(_finite_mask(enriched["log_distributor_inventory_qty"]).sum())
    fact_avail = int(_finite_mask(enriched["log_factory_inventory_qty"]).sum())
    both_avail = int(
        (_finite_mask(enriched["log_distributor_inventory_qty"])
         & _finite_mask(enriched["log_factory_inventory_qty"])).sum()
    )
    return pd.DataFrame([{
        "n_rows": n_rows,
        "n_products": n_products,
        "distributor_available_rows": dist_avail,
        "distributor_coverage_pct": _pct(dist_avail, n_rows),
        "factory_available_rows": fact_avail,
        "factory_coverage_pct": _pct(fact_avail, n_rows),
        "both_available_rows": both_avail,
        "both_coverage_pct": _pct(both_avail, n_rows),
    }])


def coverage_by_origin(enriched: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    rows = []
    for o in sorted(PRIMARY_ORIGINS):
        g = enriched.loc[enriched[origin_col].astype(int) == int(o)]
        n = len(g)
        np_ = int(g["product"].nunique())
        d = int(_finite_mask(g["log_distributor_inventory_qty"]).sum())
        f = int(_finite_mask(g["log_factory_inventory_qty"]).sum())
        rows.append({
            "origin": int(o), "n_rows": n, "n_products": np_,
            "distributor_available": d,
            "distributor_coverage_pct": _pct(d, n),
            "factory_available": f,
            "factory_coverage_pct": _pct(f, n),
        })
    return pd.DataFrame(rows)


def coverage_by_product(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in enriched.groupby("product"):
        n = len(g)
        d = int(_finite_mask(g["log_distributor_inventory_qty"]).sum())
        f = int(_finite_mask(g["log_factory_inventory_qty"]).sum())
        rows.append({
            "product": str(product), "n_rows": n,
            "distributor_available": d,
            "distributor_coverage_pct": _pct(d, n),
            "factory_available": f,
            "factory_coverage_pct": _pct(f, n),
        })
    return pd.DataFrame(rows)


def missingness(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, reason_col in [
        ("log_distributor_inventory_qty", "distributor_missing_reason"),
        ("log_factory_inventory_qty", "factory_missing_reason"),
    ]:
        if reason_col not in enriched.columns:
            continue
        vc = enriched[reason_col].value_counts()
        n = len(enriched)
        for reason, count in vc.items():
            rows.append({
                "feature": feat,
                "reason": str(reason),
                "n_rows": int(count),
                "pct": _pct(int(count), n),
            })
    return pd.DataFrame(rows)


def distributions(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DIST_COLS:
        if col not in enriched.columns:
            continue
        vals = pd.to_numeric(enriched[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        rec = {"feature": col, "n_finite": int(len(finite)), "n": int(len(vals))}
        if len(finite) == 0:
            for name in QUANTILE_NAMES:
                rec[name] = float("nan")
        else:
            qs = np.quantile(finite, QUANTILES)
            for name, q in zip(QUANTILE_NAMES, qs):
                rec[name] = float(q)
        rows.append(rec)
    return pd.DataFrame(rows)


def temporal_variation(enriched: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    rows = []
    for product, g in enriched.groupby("product"):
        dist_vals = pd.to_numeric(g["distributor_inventory_qty"], errors="coerce")
        fact_vals = pd.to_numeric(g["factory_inventory_qty"], errors="coerce")
        d_states = int(dist_vals.dropna().nunique())
        f_states = int(fact_vals.dropna().nunique())
        rows.append({
            "product": str(product),
            "n_distinct_distributor_states": d_states,
            "n_distinct_factory_states": f_states,
        })
    df = pd.DataFrame(rows)
    return df


def audit_inventory_features(
    *,
    out_dir: Optional[Path] = None,
    verify_freeze: bool = False,
) -> dict:
    """Attach PIT inventory features and audit. No XGB."""
    out_dir = Path(out_dir) if out_dir is not None else f3c_feature_audit_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_benchmark(verify_checksums=verify_freeze)
    matched = ds.matched_universe.copy()
    matched["product"] = matched["product"].astype(str)

    dist_hist = load_frozen_distributor_inventory()
    fact_hist = load_frozen_factory_inventory()

    origin_col = "origin"
    enriched = add_inventory_features(matched, dist_hist, fact_hist, origin_col=origin_col)

    overall = coverage_overall(enriched)
    by_origin = coverage_by_origin(enriched, origin_col)
    by_product = coverage_by_product(enriched)
    miss = missingness(enriched)
    dist = distributions(enriched)
    temp_var = temporal_variation(enriched, origin_col)

    n_dist_gt1 = int((temp_var["n_distinct_distributor_states"] > 1).sum())
    n_fact_gt1 = int((temp_var["n_distinct_factory_states"] > 1).sum())

    overall.to_csv(out_dir / "coverage_overall.csv", index=False)
    by_origin.to_csv(out_dir / "coverage_by_origin.csv", index=False)
    by_product.to_csv(out_dir / "coverage_by_product.csv", index=False)
    miss.to_csv(out_dir / "missingness.csv", index=False)
    dist.to_csv(out_dir / "distributions.csv", index=False)
    temp_var.to_csv(out_dir / "temporal_variation.csv", index=False)

    return {
        "enriched": enriched,
        "overall": overall,
        "by_origin": by_origin,
        "by_product": by_product,
        "missingness": miss,
        "distributions": dist,
        "temporal_variation": temp_var,
        "n_products_dist_gt1_state": n_dist_gt1,
        "n_products_fact_gt1_state": n_fact_gt1,
        "out_dir": out_dir,
    }
