"""Pre-model F3B price-feature audit (no XGB, no threshold tuning)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark import load_benchmark
from pkg.benchmark.config import PRIMARY_ORIGINS, default_benchmark_root
from pkg.research.f3b.config import f3b_feature_audit_dir, f3b_source_dir
from pkg.research.f3b.prepare import _file_fingerprint, assert_freeze_untouched
from pkg.research.features.price import (
    DIAGNOSTIC_NAMES,
    FEATURE_NAMES,
    add_price_features,
    load_frozen_price_history,
)

DIST_COLS = (
    "consumer_price_asof_origin",
    "log_consumer_price_asof_origin",
    "last_consumer_price_change_pct",
    "months_since_last_consumer_price_change",
)
QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p1", "p10", "p25", "median", "p75", "p90", "p99", "max")
EXTREME_ABS_PCT = 1.0


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return float("nan")
    return float(n) / float(d) * 100.0


def _finite_mask(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").map(lambda v: bool(np.isfinite(v)))


def coverage_overall(enriched: pd.DataFrame) -> pd.DataFrame:
    n_rows = int(len(enriched))
    n_products = int(enriched["product"].nunique())
    row = {
        "n_rows": n_rows,
        "n_products": n_products,
        "current_price_available_rows": int(_finite_mask(enriched["consumer_price_asof_origin"]).sum()),
        "last_change_available_rows": int(
            _finite_mask(enriched["last_consumer_price_change_pct"]).sum()
        ),
        "months_since_change_available_rows": int(
            _finite_mask(enriched["months_since_last_consumer_price_change"]).sum()
        ),
    }
    row["current_price_coverage_pct"] = _pct(row["current_price_available_rows"], n_rows)
    row["last_change_coverage_pct"] = _pct(row["last_change_available_rows"], n_rows)
    row["months_since_change_coverage_pct"] = _pct(
        row["months_since_change_available_rows"], n_rows
    )
    return pd.DataFrame([row])


def distributions(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DIST_COLS:
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


def coverage_by_origin(enriched: pd.DataFrame, origin_col: str = "origin") -> pd.DataFrame:
    rows = []
    for origin, g in enriched.groupby(enriched[origin_col].astype(int)):
        n = int(len(g))
        rows.append(
            {
                "origin": int(origin),
                "n_rows": n,
                "current_price_coverage_pct": _pct(
                    int(_finite_mask(g["consumer_price_asof_origin"]).sum()), n
                ),
                "change_pct_coverage_pct": _pct(
                    int(_finite_mask(g["last_consumer_price_change_pct"]).sum()), n
                ),
                "months_since_change_coverage_pct": _pct(
                    int(_finite_mask(g["months_since_last_consumer_price_change"]).sum()), n
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)


def coverage_by_product(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in enriched.groupby(enriched["product"].astype(str)):
        n = int(len(g))
        rows.append(
            {
                "product": product,
                "n_rows": n,
                "current_price_coverage_pct": _pct(
                    int(_finite_mask(g["consumer_price_asof_origin"]).sum()), n
                ),
                "change_pct_coverage_pct": _pct(
                    int(_finite_mask(g["last_consumer_price_change_pct"]).sum()), n
                ),
                "months_since_change_coverage_pct": _pct(
                    int(_finite_mask(g["months_since_last_consumer_price_change"]).sum()), n
                ),
                "n_distinct_asof_prices": int(
                    pd.to_numeric(g["consumer_price_asof_origin"], errors="coerce")
                    .dropna()
                    .nunique()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_distinct_asof_prices", "product"], ascending=[False, True])
        .reset_index(drop=True)
    )


def temporal_variation(enriched: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    products = sorted(enriched["product"].astype(str).unique())
    hist = history.copy()
    hist["product"] = hist["product"].astype(str)
    hist["consumer_price"] = pd.to_numeric(hist["consumer_price"], errors="coerce")
    rows = []
    for sku in products:
        h = hist.loc[hist["product"] == sku]
        n_hist_states = int(h.loc[h["consumer_price"] > 0, "consumer_price"].nunique())
        g = enriched.loc[enriched["product"].astype(str) == sku]
        asof = pd.to_numeric(g["consumer_price_asof_origin"], errors="coerce")
        n_origin_states = int(asof.dropna().nunique())
        rows.append(
            {
                "product": sku,
                "n_distinct_price_states": n_hist_states,
                "n_distinct_price_states_across_primary_origins": n_origin_states,
                "price_state_changes_across_primary_origins": n_origin_states > 1,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["price_state_changes_across_primary_origins", "product"],
        ascending=[False, True],
    ).reset_index(drop=True)


def extreme_changes(enriched: pd.DataFrame) -> pd.DataFrame:
    sub = enriched.copy()
    sub["last_consumer_price_change_pct"] = pd.to_numeric(
        sub["last_consumer_price_change_pct"], errors="coerce"
    )
    empty_cols = [
        "product",
        "origin",
        "consumer_price_asof_origin",
        "previous_consumer_price",
        "last_consumer_price_change_pct",
        "last_change_month",
        "flag_abs_pct_ge_1",
        "flag_abs_ge_p99",
    ]
    grain = ["product", "origin"]
    keep = grain + [
        "consumer_price_asof_origin",
        "previous_consumer_price",
        "last_consumer_price_change_pct",
        "last_change_month",
    ]
    keep = [c for c in keep if c in sub.columns]
    finite = sub.loc[np.isfinite(sub["last_consumer_price_change_pct"]), keep]
    if finite.empty:
        return pd.DataFrame(columns=empty_cols)
    finite = finite.drop_duplicates(subset=[c for c in grain if c in finite.columns])
    p99 = float(np.quantile(np.abs(finite["last_consumer_price_change_pct"]), 0.99))
    flagged = finite.loc[
        (finite["last_consumer_price_change_pct"].abs() >= EXTREME_ABS_PCT)
        | (finite["last_consumer_price_change_pct"].abs() >= p99)
    ].copy()
    flagged["flag_abs_pct_ge_1"] = flagged["last_consumer_price_change_pct"].abs() >= EXTREME_ABS_PCT
    flagged["flag_abs_ge_p99"] = flagged["last_consumer_price_change_pct"].abs() >= p99
    cols = [c for c in empty_cols if c in flagged.columns]
    out = flagged[cols].copy()
    out["_abs"] = out["last_consumer_price_change_pct"].abs()
    return out.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)


def audit_price_features(
    *,
    dataset=None,
    price_history: Optional[pd.DataFrame] = None,
    out_dir: Optional[Path] = None,
    verify_freeze: bool = True,
) -> dict:
    """Attach PIT price features to PRIMARY rows and write audit CSVs. No XGB."""
    out_dir = Path(out_dir) if out_dir is not None else f3b_feature_audit_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = default_benchmark_root()
    freeze_before = _file_fingerprint(bench) if verify_freeze and bench.exists() else {}

    ds = dataset or load_benchmark(verify_checksums=False)
    origins = set(int(o) for o in PRIMARY_ORIGINS)
    panel = ds.matched_universe.copy()
    panel = panel.loc[panel["origin"].astype(int).isin(origins)].copy()
    if price_history is None:
        source_parquet = f3b_source_dir() / "price_history.parquet"
        if not source_parquet.exists():
            raise FileNotFoundError(source_parquet)
        hist = load_frozen_price_history(source_parquet)
    else:
        hist = price_history

    enriched = add_price_features(panel, hist, origin_col="origin")
    overall = coverage_overall(enriched)
    dist = distributions(enriched)
    by_o = coverage_by_origin(enriched)
    by_p = coverage_by_product(enriched)
    temporal = temporal_variation(enriched, hist)
    extremes = extreme_changes(enriched)
    n_vary = int(temporal["price_state_changes_across_primary_origins"].sum())

    overall.to_csv(out_dir / "coverage_overall.csv", index=False)
    dist.to_csv(out_dir / "distributions.csv", index=False)
    by_o.to_csv(out_dir / "coverage_by_origin.csv", index=False)
    by_p.to_csv(out_dir / "coverage_by_product.csv", index=False)
    temporal.to_csv(out_dir / "temporal_variation.csv", index=False)
    extremes.to_csv(out_dir / "extreme_changes.csv", index=False)

    if verify_freeze and freeze_before:
        assert_freeze_untouched(bench, freeze_before)

    return {
        "enriched": enriched,
        "overall": overall,
        "distributions": dist,
        "by_origin": by_o,
        "by_product": by_p,
        "temporal": temporal,
        "extremes": extremes,
        "n_products_varying_across_origins": n_vary,
        "scored_features": FEATURE_NAMES,
        "diagnostic_features": DIAGNOSTIC_NAMES,
        "out_dir": out_dir,
        "n_primary_origins": len(PRIMARY_ORIGINS),
    }
