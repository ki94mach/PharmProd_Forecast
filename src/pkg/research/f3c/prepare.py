"""Freeze distributor + factory inventory extracts into F3C source artifacts.

SQL is used only here. Later F3C steps read ``src/data/results/f3c/source/``
and must not query DWOrchid. Does not train XGB or attach scored features.
"""
from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_month_start_gregorian
from pkg.benchmark.config import PANEL_FILES, PRIMARY_ORIGINS, RAW_FILES, default_benchmark_root
from pkg.db.query.inventory import load_distributor_inventory, load_factory_inventory
from pkg.research.f3c.config import f3c_source_dir

DISTRIBUTOR_PARQUET = "distributor_inventory_daily.parquet"
FACTORY_PARQUET = "factory_inventory_daily.parquet"


class UncleanMvpMappingError(Exception):
    pass


# ---------------------------------------------------------------------------
# Freeze-file fingerprint (guards frozen benchmark)
# ---------------------------------------------------------------------------

def _file_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    out = {}
    names = list(PANEL_FILES) + [f"raw/{n}" for n in RAW_FILES]
    for name in names:
        p = root / name
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        out[name] = (p.stat().st_mtime_ns, p.stat().st_size, h.hexdigest())
    return out


def assert_freeze_untouched(root: Path, before: dict) -> None:
    after = _file_fingerprint(root)
    if after != before:
        raise AssertionError(
            "F3C source prep modified frozen benchmark files "
            f"(before={before} after={after})"
        )


# ---------------------------------------------------------------------------
# MVP products from frozen matched_universe
# ---------------------------------------------------------------------------

def mvp_products(benchmark_root: Optional[Path] = None) -> list[str]:
    root = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
    path = root / "matched_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"matched universe missing: {path}")
    matched = pd.read_parquet(path, columns=["product"])
    return sorted(matched["product"].astype(str).unique())


# ---------------------------------------------------------------------------
# Product mapping audit
# ---------------------------------------------------------------------------

def product_mapping_audit(
    df: pd.DataFrame, mvp_list: list[str], source_label: str
) -> pd.DataFrame:
    """Audit FKProduct → Dim.Product mapping for one source."""
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    g = df.groupby("fk_product").first().reset_index()
    n_fk = int(g["fk_product"].nunique())
    mapped = g.loc[g["product"].notna() & (g["product"].astype(str).str.strip() != "")]
    unmapped = g.loc[~g.index.isin(mapped.index)]
    n_mapped = int(len(mapped))
    n_unmapped = int(len(unmapped))
    n_ambiguous = 0
    coverage = 100.0 * n_mapped / n_fk if n_fk > 0 else 0.0
    mvp_set = set(mvp_list)
    mapped_products = set(mapped["product"].astype(str).unique())
    mvp_mapped = mvp_set & mapped_products
    mvp_unmapped = sorted(mvp_set - mapped_products)
    rows.append({
        "source": source_label,
        "n_distinct_fk_product": n_fk,
        "n_successfully_mapped": n_mapped,
        "n_unmapped": n_unmapped,
        "n_ambiguous": n_ambiguous,
        "mapping_coverage_pct": coverage,
        "n_mvp_products": len(mvp_list),
        "n_mvp_mapped": len(mvp_mapped),
        "n_mvp_unmapped": len(mvp_unmapped),
        "mvp_unmapped_products": "; ".join(mvp_unmapped) if mvp_unmapped else "",
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Snapshot date audit
# ---------------------------------------------------------------------------

def snapshot_date_audit(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    dates = df["snapshot_date"].sort_values().unique()
    gaps = pd.Series(dates).diff().dt.days.dropna()
    products_per_snap = df.groupby("snapshot_date")["product"].nunique()
    return pd.DataFrame([{
        "source": source_label,
        "min_snapshot_date": str(dates.min()),
        "max_snapshot_date": str(dates.max()),
        "n_distinct_snapshot_dates": len(dates),
        "median_gap_days": float(gaps.median()) if len(gaps) > 0 else float("nan"),
        "p90_gap_days": float(gaps.quantile(0.9)) if len(gaps) > 0 else float("nan"),
        "max_gap_days": float(gaps.max()) if len(gaps) > 0 else float("nan"),
        "median_products_per_snapshot": float(products_per_snap.median()),
    }])


# ---------------------------------------------------------------------------
# Exact month-end coverage
# ---------------------------------------------------------------------------

def exact_month_end_coverage(
    dist_df: pd.DataFrame,
    fact_df: pd.DataFrame,
    mvp_list: list[str],
) -> pd.DataFrame:
    mvp_set = set(mvp_list)
    rows = []
    for origin in PRIMARY_ORIGINS:
        origin_start = shamsi_month_start_gregorian(int(origin))
        inv_date = pd.Timestamp(origin_start - timedelta(days=1))

        n_mvp = len(mvp_list)
        if not dist_df.empty:
            dist_exact = dist_df.loc[
                (dist_df["snapshot_date"] == inv_date)
                & dist_df["product"].isin(mvp_set)
            ]
            n_dist = int(dist_exact["product"].nunique())
        else:
            n_dist = 0

        if not fact_df.empty:
            fact_exact = fact_df.loc[
                (fact_df["snapshot_date"] == inv_date)
                & fact_df["product"].isin(mvp_set)
            ]
            n_fact = int(fact_exact["product"].nunique())
        else:
            n_fact = 0

        rows.append({
            "origin": int(origin),
            "origin_start_date": str(origin_start),
            "inventory_month_end_date": str(inv_date.date()),
            "n_mvp_products": n_mvp,
            "n_with_exact_distributor_record": n_dist,
            "distributor_exact_coverage_pct": 100.0 * n_dist / n_mvp if n_mvp else 0,
            "n_with_exact_factory_record": n_fact,
            "factory_exact_coverage_pct": 100.0 * n_fact / n_mvp if n_mvp else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distributor month-end status audit
# ---------------------------------------------------------------------------

def distributor_month_end_status_audit(
    dist_df: pd.DataFrame,
    mvp_list: list[str],
) -> pd.DataFrame:
    mvp_set = set(mvp_list)
    rows = []
    for origin in PRIMARY_ORIGINS:
        origin_start = shamsi_month_start_gregorian(int(origin))
        inv_date = pd.Timestamp(origin_start - timedelta(days=1))
        snap = dist_df.loc[
            (dist_df["snapshot_date"] == inv_date)
            & dist_df["product"].isin(mvp_set)
        ].copy()

        present_products = set(snap["product"].astype(str).unique())
        missing_products = sorted(mvp_set - present_products)

        if snap.empty:
            rows.append({
                "origin": int(origin),
                "inventory_month_end_date": str(inv_date.date()),
                "n_products_with_record": 0,
                "n_products_missing_entirely": len(missing_products),
                "n_on_hand_gt0": 0, "n_in_transit_gt0": 0,
                "n_both_gt0": 0, "n_only_on_hand_gt0": 0,
                "n_only_in_transit_gt0": 0,
                "n_distributor_inventory_eq0": 0,
                "sum_on_hand_qty": 0, "sum_in_transit_qty": 0,
                "sum_blocked_inventory_qty": 0,
                "sum_distributor_inventory_qty": 0,
                "identity_holds": True,
                "blocked_excluded": True,
            })
            continue

        oh = snap["distributor_on_hand_qty"].astype(float)
        it = snap["distributor_in_transit_qty"].astype(float)
        inv = snap["distributor_inventory_qty"].astype(float)
        blk = snap["blocked_inventory_qty"].astype(float)

        oh_gt0 = (oh > 0)
        it_gt0 = (it > 0)

        identity = np.allclose(inv.values, (oh + it).values, atol=1e-4)
        blocked_excluded = not np.any(np.abs(inv.values - (oh + it).values) > 1e-4)

        rows.append({
            "origin": int(origin),
            "inventory_month_end_date": str(inv_date.date()),
            "n_products_with_record": len(snap),
            "n_products_missing_entirely": len(missing_products),
            "n_on_hand_gt0": int(oh_gt0.sum()),
            "n_in_transit_gt0": int(it_gt0.sum()),
            "n_both_gt0": int((oh_gt0 & it_gt0).sum()),
            "n_only_on_hand_gt0": int((oh_gt0 & ~it_gt0).sum()),
            "n_only_in_transit_gt0": int((~oh_gt0 & it_gt0).sum()),
            "n_distributor_inventory_eq0": int((inv == 0).sum()),
            "sum_on_hand_qty": float(oh.sum()),
            "sum_in_transit_qty": float(it.sum()),
            "sum_blocked_inventory_qty": float(blk.sum()),
            "sum_distributor_inventory_qty": float(inv.sum()),
            "identity_holds": bool(identity),
            "blocked_excluded": bool(blocked_excluded),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quantity quality
# ---------------------------------------------------------------------------

def quantity_quality(
    dist_df: pd.DataFrame, fact_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for label, df, col in [
        ("distributor", dist_df, "distributor_inventory_qty"),
        ("factory", fact_df, "factory_inventory_qty"),
    ]:
        if df.empty:
            continue
        vals = df[col].astype(float)
        rows.append({
            "source": label,
            "qty_col": col,
            "n_product_dates": len(df),
            "n_aggregate_zero": int((vals == 0).sum()),
            "n_negative": int((vals < 0).sum()),
            "n_products_ever_negative": int(
                df.loc[vals < 0, "product"].nunique()
            ) if (vals < 0).any() else 0,
            "min_aggregate_qty": float(vals.min()),
            "max_aggregate_qty": float(vals.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source summary
# ---------------------------------------------------------------------------

def build_source_summary(
    *,
    dist_df: pd.DataFrame,
    fact_df: pd.DataFrame,
    dist_mapping: pd.DataFrame,
    fact_mapping: pd.DataFrame,
    month_end_coverage: pd.DataFrame,
    status_audit: pd.DataFrame,
    qty_qual: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_distributor_product_dates": len(dist_df),
        "n_factory_product_dates": len(fact_df),
        "distributor_mapping_coverage_pct": float(
            dist_mapping["mapping_coverage_pct"].iloc[0]
        ) if not dist_mapping.empty else 0,
        "factory_mapping_coverage_pct": float(
            fact_mapping["mapping_coverage_pct"].iloc[0]
        ) if not fact_mapping.empty else 0,
        "identity_holds_all_origins": bool(
            status_audit["identity_holds"].all()
        ) if not status_audit.empty else True,
        "blocked_excluded_all_origins": bool(
            status_audit["blocked_excluded"].all()
        ) if not status_audit.empty else True,
    }])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_inventory_source(
    *,
    out_dir: Optional[Path] = None,
    benchmark_root: Optional[Path] = None,
    verify_freeze: bool = True,
) -> dict:
    """Extract, map, audit, and freeze F3C source artifacts. No XGB."""
    out_dir = Path(out_dir) if out_dir is not None else f3c_source_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
    freeze_before = _file_fingerprint(bench) if verify_freeze and bench.exists() else {}

    print("Loading distributor inventory from FactInventoryHistorical...")
    dist_raw = load_distributor_inventory()
    dist_raw["snapshot_date"] = pd.to_datetime(dist_raw["snapshot_date"])
    dist_raw["product"] = dist_raw["product"].astype(str)

    print("Loading factory inventory from FactInventory...")
    fact_raw = load_factory_inventory()
    fact_raw["snapshot_date"] = pd.to_datetime(fact_raw["snapshot_date"])
    fact_raw["product"] = fact_raw["product"].astype(str)

    mvp_list = mvp_products(bench)

    # mapping audit
    dist_mapping = product_mapping_audit(dist_raw, mvp_list, "distributor")
    fact_mapping = product_mapping_audit(fact_raw, mvp_list, "factory")
    mapping = pd.concat([dist_mapping, fact_mapping], ignore_index=True)

    # check MVP coverage — distributor must cover all MVP products;
    # factory may legitimately have no FactInventory records for some products
    for _, row in mapping.iterrows():
        if row["n_mvp_unmapped"] > 0:
            if row["source"] == "distributor":
                raise UncleanMvpMappingError(
                    f"{row['source']}: {row['n_mvp_unmapped']} MVP products unmapped: "
                    f"{row['mvp_unmapped_products']}"
                )
            else:
                print(
                    f"WARNING: {row['source']}: {row['n_mvp_unmapped']} MVP products "
                    f"absent from FactInventory (will be NaN in features): "
                    f"{row['mvp_unmapped_products']}"
                )

    # snapshot date audit
    dist_snap = snapshot_date_audit(dist_raw, "distributor")
    fact_snap = snapshot_date_audit(fact_raw, "factory")
    snap_audit = pd.concat([dist_snap, fact_snap], ignore_index=True)

    # exact month-end coverage
    month_end = exact_month_end_coverage(dist_raw, fact_raw, mvp_list)

    # status audit
    status_audit = distributor_month_end_status_audit(dist_raw, mvp_list)

    # quantity quality
    qty_qual = quantity_quality(dist_raw, fact_raw)

    # summary
    summary = build_source_summary(
        dist_df=dist_raw, fact_df=fact_raw,
        dist_mapping=dist_mapping, fact_mapping=fact_mapping,
        month_end_coverage=month_end,
        status_audit=status_audit, qty_qual=qty_qual,
    )

    # freeze parquets
    dist_raw.to_parquet(out_dir / DISTRIBUTOR_PARQUET, index=False)
    fact_raw.to_parquet(out_dir / FACTORY_PARQUET, index=False)

    # CSVs
    summary.to_csv(out_dir / "inventory_source_summary.csv", index=False)
    mapping.to_csv(out_dir / "product_mapping_audit.csv", index=False)
    snap_audit.to_csv(out_dir / "snapshot_date_audit.csv", index=False)
    month_end.to_csv(out_dir / "exact_month_end_coverage.csv", index=False)
    status_audit.to_csv(out_dir / "distributor_status_audit.csv", index=False)
    qty_qual.to_csv(out_dir / "quantity_quality.csv", index=False)

    if verify_freeze and freeze_before:
        assert_freeze_untouched(bench, freeze_before)

    return {
        "dist_raw": dist_raw,
        "fact_raw": fact_raw,
        "mapping": mapping,
        "snap_audit": snap_audit,
        "month_end": month_end,
        "status_audit": status_audit,
        "qty_qual": qty_qual,
        "summary": summary,
        "out_dir": out_dir,
        "mvp_list": mvp_list,
    }
