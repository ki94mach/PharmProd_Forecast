"""Build V3A architecture-validation product panel from frozen MVP sales."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import DEFAULT_CONFIG
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.intermittency import intermittency_stats
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import resolve_legacy_architecture

REPO_ROOT = Path(__file__).resolve().parents[4]
FROZEN_SALES = REPO_ROOT / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
MVP_CSV = REPO_ROOT / "src" / "pkg" / "benchmark" / "universes" / "mvp_products.csv"
PANEL_CSV = (
    REPO_ROOT
    / "src"
    / "pkg"
    / "benchmark"
    / "universes"
    / "v3a_architecture_validation_products.csv"
)
PANEL_META = PANEL_CSV.with_suffix(".meta.json")

# Full H15 on freeze (last≈140504 ⇒ origin+14 ≤ last). Earlier origins exercise A0 mid/short tiers.
DEFAULT_VALIDATION_ORIGINS: tuple[int, ...] = (140104, 140201, 140301, 140402)


def _longest_zero_run(values: np.ndarray) -> int:
    best = cur = 0
    for v in values:
        if float(v) == 0.0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _series_diagnostics(series: pd.Series) -> dict:
    vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    n = len(vals)
    nonzero = int(np.sum(vals != 0))
    zero = int(n - nonzero)
    mean = float(np.mean(vals)) if n else float("nan")
    median = float(np.median(vals)) if n else float("nan")
    std = float(np.std(vals, ddof=0)) if n else float("nan")
    cv = float(std / mean) if mean not in (0.0, -0.0) and np.isfinite(mean) else float("nan")
    inter = intermittency_stats(series)
    # Growth: last 12 vs prior 12 when possible
    growth = float("nan")
    if n >= 24:
        recent = float(np.mean(vals[-12:]))
        prior = float(np.mean(vals[-24:-12]))
        if prior != 0:
            growth = (recent - prior) / abs(prior)
    ac12 = float("nan")
    if n >= 24:
        x = vals[12:]
        y = vals[:-12]
        if np.std(x) > 0 and np.std(y) > 0:
            ac12 = float(np.corrcoef(x, y)[0, 1])
    return {
        "history_length": n,
        "nonzero_months": nonzero,
        "zero_months": zero,
        "zero_fraction": float(zero / n) if n else float("nan"),
        "longest_zero_run": _longest_zero_run(vals),
        "average_inter_demand_interval": inter.average_inter_demand_interval,
        "mean_sales": mean,
        "median_sales": median,
        "cv": cv,
        "growth_12m": growth,
        "acf_lag12": ac12,
    }


def _history_bucket(n: int) -> str:
    if n > 36:
        return ">36"
    if n > 24:
        return "25-36"
    if n > 12:
        return "13-24"
    return "<=12"


def build_validation_panel(
    *,
    sales_parquet: Path = FROZEN_SALES,
    mvp_csv: Path = MVP_CSV,
    origins: Sequence[int] = DEFAULT_VALIDATION_ORIGINS,
    target_n: int = 20,
    always_include: Sequence[str] = ("Recigen",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select ~target_n MVP products with diagnostic diversity.

    Returns ``(panel_df, origin_coverage_df)``.
    """
    mvp = pd.read_csv(mvp_csv)
    mvp["product"] = mvp["product"].astype(str).str.strip()
    sales = pd.read_parquet(sales_parquet)
    sales["product"] = sales["product"].astype(str).str.strip()
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce").fillna(0.0)

    last = int(sales["date"].max())
    origins = tuple(int(o) for o in origins)
    for o in origins:
        if shamsi_add_months(o, 14) > last:
            raise ValueError(
                f"origin {o} lacks full H15 actuals (last={last})"
            )

    diag_rows: list[dict] = []
    cov_rows: list[dict] = []
    for product in mvp["product"].tolist():
        sub = sales.loc[sales["product"] == product]
        if sub.empty:
            continue
        # Diagnostics as-of latest origin (primary selection features)
        latest = max(origins)
        prep = prepare_monthly_series(sales, product, latest, config=DEFAULT_CONFIG)
        d = _series_diagnostics(prep.values)
        meta = mvp.loc[mvp["product"] == product].iloc[0]
        row = {
            "product": product,
            "product_title": meta.get("product_title", product),
            "product_id": meta.get("product_id", ""),
            "generic": meta.get("generic", ""),
            "field": meta.get("field", ""),
            "product_form": meta.get("product_form", ""),
            "provider": meta.get("provider", ""),
            **d,
            "history_bucket_latest": _history_bucket(d["history_length"]),
        }
        # Per-origin history / A0 tier
        buckets = set()
        tiers = set()
        for origin in origins:
            p = prepare_monthly_series(sales, product, origin, config=DEFAULT_CONFIG)
            n = int(p.n_observations)
            spec = resolve_legacy_architecture(n)
            buckets.add(_history_bucket(n))
            tiers.add(f"{spec.l1}/{spec.l2}/{spec.lookback}")
            cov_rows.append(
                {
                    "product": product,
                    "origin": origin,
                    "history_length": n,
                    "history_bucket": _history_bucket(n),
                    "a0_l1": spec.l1,
                    "a0_l2": spec.l2,
                    "a0_lookback": spec.lookback,
                    "a0_max_epochs": spec.max_epochs,
                }
            )
        row["history_buckets_across_origins"] = "|".join(sorted(buckets))
        row["a0_tiers_across_origins"] = "|".join(sorted(tiers))
        diag_rows.append(row)

    diag = pd.DataFrame(diag_rows)
    cov = pd.DataFrame(cov_rows)
    if diag.empty:
        raise RuntimeError("no MVP products found in frozen sales")

    picked: list[str] = []
    reasons: dict[str, str] = {}

    def _add(product: str, reason: str) -> None:
        if product in picked or product not in set(diag["product"]):
            return
        picked.append(product)
        reasons[product] = reason

    for p in always_include:
        _add(str(p), "always_include_smoke_continuity")

    # Quota fills
    def _pick_from(mask: pd.Series, reason: str, n: int = 1) -> None:
        cand = diag.loc[mask & ~diag["product"].isin(picked)].sort_values(
            "mean_sales", ascending=False
        )
        for product in cand["product"].head(n):
            _add(str(product), reason)

    # History diversity using across-origin buckets
    _pick_from(diag["history_buckets_across_origins"].str.contains(r">36", regex=True), "long_history_>36", 4)
    _pick_from(diag["history_buckets_across_origins"].str.contains("25-36", regex=False), "medium_history_25-36", 3)
    _pick_from(diag["history_buckets_across_origins"].str.contains("13-24", regex=False), "short_history_13-24", 3)

    # Intermittent / stable
    _pick_from(diag["zero_fraction"] >= 0.25, "intermittent_high_zero_fraction", 3)
    _pick_from(diag["zero_fraction"] <= 0.05, "stable_low_zero_fraction", 3)

    # Seasonal / trend
    _pick_from(diag["acf_lag12"].fillna(0) >= 0.4, "seasonal_acf12", 2)
    _pick_from(diag["growth_12m"].fillna(0) >= 0.15, "growing_12m", 2)
    _pick_from(diag["growth_12m"].fillna(0) <= -0.15, "declining_12m", 2)

    # Volume extremes among remaining
    rem = diag.loc[~diag["product"].isin(picked)].sort_values("mean_sales", ascending=False)
    if not rem.empty:
        _add(str(rem.iloc[0]["product"]), "high_volume")
    rem = diag.loc[~diag["product"].isin(picked)].sort_values("mean_sales", ascending=True)
    if not rem.empty:
        _add(str(rem.iloc[0]["product"]), "low_volume")

    # Provider / field diversity fill to target_n
    used_providers = set(diag.loc[diag["product"].isin(picked), "provider"].astype(str))
    used_fields = set(diag.loc[diag["product"].isin(picked), "field"].astype(str))
    rem = diag.loc[~diag["product"].isin(picked)].copy()
    rem["_div"] = rem.apply(
        lambda r: int(str(r["provider"]) not in used_providers)
        + int(str(r["field"]) not in used_fields),
        axis=1,
    )
    rem = rem.sort_values(["_div", "mean_sales"], ascending=[False, False])
    for _, r in rem.iterrows():
        if len(picked) >= target_n:
            break
        _add(str(r["product"]), "diversity_fill")
        used_providers.add(str(r["provider"]))
        used_fields.add(str(r["field"]))

    # Ensure A0 mid/short tiers appear somewhere in coverage for picked set
    picked_cov = cov.loc[cov["product"].isin(picked)]
    need_mid = not ((picked_cov["a0_l1"] == 128) & (picked_cov["a0_lookback"] == 6)).any()
    need_short = not ((picked_cov["a0_l1"] == 128) & (picked_cov["a0_lookback"] == 3)).any()
    if need_mid or need_short:
        for _, r in cov.sort_values("history_length").iterrows():
            if len(picked) >= target_n + 3:
                break
            if need_mid and r["a0_l1"] == 128 and r["a0_lookback"] == 6:
                _add(str(r["product"]), "a0_mid_tier_coverage")
                need_mid = False
            if need_short and r["a0_l1"] == 128 and r["a0_lookback"] == 3:
                _add(str(r["product"]), "a0_short_tier_coverage")
                need_short = False

    panel = diag.loc[diag["product"].isin(picked)].copy()
    panel["selection_reason"] = panel["product"].map(reasons)
    # Stable order: always_include first, then by mean_sales desc
    panel["_ord"] = panel["product"].map({p: i for i, p in enumerate(picked)})
    panel = panel.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    return panel, cov.loc[cov["product"].isin(picked)].reset_index(drop=True)


def write_validation_panel(
    panel: pd.DataFrame,
    *,
    path: Path = PANEL_CSV,
    origins: Sequence[int] = DEFAULT_VALIDATION_ORIGINS,
    sales_source: str = "src/data/benchmarks/v1/raw/sales.parquet",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    meta = {
        "name": "v3a_architecture_validation_products",
        "n_products": int(len(panel)),
        "sales_source": sales_source,
        "origins": list(int(o) for o in origins),
        "parent_universe": "mvp_products",
        "logical_product_key": "ProductTitleEN (column product)",
    }
    PANEL_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path
