"""One-shot builder: live DB + TS CSVs -> src/data/benchmarks/v1/.

Run::

    python -m pkg.benchmark.freeze
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pkg.benchmark.calendar import shamsi_add_months, shamsi_month_diff
from pkg.benchmark.config import (
    ALLOWED_FORECAST_QRTS,
    BENCHMARK_VERSION,
    EXCLUDED_EMPTY_FORECAST_QRTS,
    EXCLUDED_ODD_COVERAGE_PRODUCTS,
    EXPECTED_ANALYSIS_A_PRIMARY,
    EXPECTED_ANALYSIS_B_PRIMARY,
    FORECAST_HORIZON_MONTHS,
    INCOMPLETE_SHAMSI_MONTHS,
    PANEL_FILES,
    PRIMARY_ORIGINS,
    RAW_FILES,
    default_benchmark_root,
    manifest_path,
)
from pkg.benchmark.dataset import file_sha256, horizon_bucket, prep_lags
from pkg.utils import DATA_DIR


def extract_product_forecast_window(
    df: pd.DataFrame, n_months: int = FORECAST_HORIZON_MONTHS
) -> pd.DataFrame:
    meta_cols = [
        c for c in ("product_fa", "provider", "model", "dep", "status", "qrt") if c in df.columns
    ]
    parts = []
    for product, g in df.groupby("product", sort=False):
        agg_kw = {"forecast": ("forecast", "mean")}
        agg_kw.update({c: (c, "first") for c in meta_cols})
        row = g.groupby("date", as_index=False).agg(**agg_kw)
        row["product"] = product
        dates = sorted(row["date"].astype(int).tolist())
        if not dates:
            continue
        window = dates[-n_months:]
        origin = int(window[0])
        gg = row[row["date"].isin(window)].copy()
        gg["origin"] = origin
        gg["horizon"] = [shamsi_month_diff(int(d), origin) + 1 for d in gg["date"]]
        parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def modal_origin_for_qrt(g: pd.DataFrame, n_months: int = FORECAST_HORIZON_MONTHS) -> int:
    all_origins = []
    exact_origins = []
    for _, pg in g.groupby("product", sort=False):
        origin = int(pg["origin"].iloc[0])
        dates = sorted(int(d) for d in pg["date"].unique())
        all_origins.append(origin)
        expected = [shamsi_add_months(origin, i) for i in range(n_months)]
        if dates == expected:
            exact_origins.append(origin)
    n_prod = max(len(all_origins), 1)
    if exact_origins:
        exact_vc = pd.Series(exact_origins).value_counts()
        mode_exact = int(exact_vc.index[0])
        if int(exact_vc.iloc[0]) >= max(5, int(0.2 * n_prod)):
            return mode_exact
    return int(pd.Series(all_origins).value_counts().index[0])


def align_forecast_to_modal_origin(
    df: pd.DataFrame, n_months: int = FORECAST_HORIZON_MONTHS
) -> pd.DataFrame:
    if df.empty:
        return df
    parts = []
    for qrt, g in df.groupby("qrt", sort=False):
        modal = modal_origin_for_qrt(g, n_months=n_months)
        gg = g[g["date"].astype(int) >= modal].copy()
        gg["origin"] = int(modal)
        gg["horizon"] = [shamsi_month_diff(int(d), int(modal)) + 1 for d in gg["date"]]
        gg = gg[(gg["horizon"] >= 1) & (gg["horizon"] <= n_months)].copy()
        parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def load_forecast_csvs(
    results_dir: Path, allowed_qrts=ALLOWED_FORECAST_QRTS
) -> pd.DataFrame:
    frames = []
    for qrt in allowed_qrts:
        csv_path = results_dir / qrt / f"{qrt}_total_forecast.csv"
        if not csv_path.exists():
            print(f"WARNING: missing allowed qrt={qrt} path={csv_path}")
            continue
        raw = pd.read_csv(csv_path)
        if raw.empty:
            print(f"WARNING: empty allowed qrt={qrt} path={csv_path}")
            continue
        raw["date"] = raw["date"].astype(int)
        raw["forecast"] = pd.to_numeric(raw["forecast"], errors="coerce")
        raw["qrt"] = qrt
        raw = raw[~raw["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
        windowed = extract_product_forecast_window(raw)
        frames.append(windowed)
        print(f"loaded {qrt}: rows={len(windowed)} products={windowed['product'].nunique()}")
    for qrt in EXCLUDED_EMPTY_FORECAST_QRTS:
        csv_path = results_dir / qrt / f"{qrt}_total_forecast.csv"
        n = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
        print(
            f"WARNING: excluded empty/placeholder qrt={qrt} path={csv_path} "
            f"rows={n} (not in ALLOWED_FORECAST_QRTS)"
        )
    if not frames:
        raise FileNotFoundError(f"No allowed forecast CSVs under {results_dir}")
    forecast_all = pd.concat(frames, ignore_index=True)
    return align_forecast_to_modal_origin(forecast_all, FORECAST_HORIZON_MONTHS)


def _sales_at(sales_pivot: pd.Series, product: str, ym: int) -> float:
    try:
        return float(sales_pivot.loc[(product, ym)])
    except KeyError:
        return np.nan


def _add_baseline_features(
    panel: pd.DataFrame,
    sales_agg: pd.DataFrame,
    product_attrs: pd.DataFrame,
) -> pd.DataFrame:
    sales_pivot = sales_agg.set_index(["product", "date"])["sales"]
    feat = panel.merge(product_attrs, on="product", how="left")
    feat["month"] = feat["date"] % 100
    feat["quarter"] = ((feat["month"] - 1) // 3) + 1
    for lag in (1, 2, 3, 12):
        col = f"sales_lag_{lag}"
        feat[col] = [
            _sales_at(sales_pivot, p, shamsi_add_months(o, -lag))
            for p, o in zip(feat["product"], feat["origin"])
        ]
    feat["sales_roll3"] = feat[["sales_lag_1", "sales_lag_2", "sales_lag_3"]].mean(axis=1)
    for col, src in [
        ("model_enc", "model"),
        ("field_enc", "Field"),
        ("form_enc", "ProductForm"),
        ("provider_enc", "Provider"),
    ]:
        enc = LabelEncoder()
        values = feat[src].fillna("__missing__").astype(str)
        feat[col] = enc.fit_transform(values)
    return feat


def build_panels(
    results_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build ts / budget / matched universes from live sources.

    Returns panels plus a dict of raw frames for archival.
    """
    from pkg.db.query.constants import TARGET_GENERIC_EN
    from pkg.db.query.dim_product import load_dim_product
    from pkg.db.query.product_budget import load_line_budget_forecasts
    from pkg.db.query.sales import load_sales_data

    results_dir = Path(results_dir) if results_dir else Path(DATA_DIR) / "results"

    sales_df = load_sales_data()
    sales_df["date"] = sales_df["date"].astype(int)
    sales_df["sales"] = pd.to_numeric(sales_df["sales"], errors="coerce")
    sales_df = sales_df[~sales_df["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()

    dim_product_df = load_dim_product()
    target_products = dim_product_df[
        dim_product_df["GenericEN"].isin(TARGET_GENERIC_EN)
    ].copy()
    target_product_names = set(target_products["ProductTitleEN"].dropna().astype(str))
    target_product_names -= EXCLUDED_ODD_COVERAGE_PRODUCTS
    target_products = target_products[
        ~target_products["ProductTitleEN"].astype(str).isin(EXCLUDED_ODD_COVERAGE_PRODUCTS)
    ].copy()

    product_attrs = (
        target_products[["ProductTitleEN", "GenericEN", "Field", "ProductForm", "Provider"]]
        .drop_duplicates(subset=["ProductTitleEN"])
        .rename(columns={"ProductTitleEN": "product", "GenericEN": "generic"})
    )

    sales_agg = sales_df.groupby(["product", "date"], as_index=False)["sales"].sum()
    sales_agg = sales_agg[sales_agg["product"].isin(target_product_names)].copy()

    # --- TS panel ---
    forecast_df = load_forecast_csvs(results_dir)
    forecast_df = forecast_df[forecast_df["product"].isin(target_product_names)].copy()
    panel = forecast_df.merge(sales_agg, on=["product", "date"], how="inner")
    panel = panel[~panel["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
    panel["residual"] = panel["sales"] - panel["forecast"]
    panel = panel[
        (panel["horizon"] >= 1) & (panel["horizon"] <= FORECAST_HORIZON_MONTHS)
    ].copy()
    feat = _add_baseline_features(panel, sales_agg, product_attrs)
    feat_model = feat.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["residual", "forecast", "sales"]
    ).copy()
    for c in ["sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_12", "sales_roll3"]:
        feat_model[c] = feat_model[c].fillna(0)

    # --- Budget panel ---
    budget_raw = load_line_budget_forecasts(earliest_edition_only=True)
    budget_raw = budget_raw[budget_raw["product"].isin(target_product_names)].copy()
    budget_raw = budget_raw[~budget_raw["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
    budget_raw["forecast"] = pd.to_numeric(budget_raw["forecast"], errors="coerce")

    budget_forecast_df = (
        budget_raw.groupby(["product", "qrt", "version", "date"], as_index=False)
        .agg(forecast=("forecast", "mean"), generic=("generic", "first"))
    )
    parts = []
    for (product, qrt), g in budget_forecast_df.groupby(["product", "qrt"], sort=False):
        gg = g.sort_values("date").copy()
        origin = int(gg["date"].min())
        gg["origin"] = origin
        gg["horizon"] = [shamsi_month_diff(int(d), origin) + 1 for d in gg["date"]]
        parts.append(gg)
    budget_forecast_df = (
        pd.concat(parts, ignore_index=True) if parts else budget_forecast_df.iloc[0:0].copy()
    )
    budget_forecast_df = budget_forecast_df[
        (budget_forecast_df["horizon"] >= 1)
        & (budget_forecast_df["horizon"] <= FORECAST_HORIZON_MONTHS)
    ].copy()
    budget_forecast_df["model"] = "LineBudget"

    budget_panel = budget_forecast_df.merge(sales_agg, on=["product", "date"], how="inner")
    budget_panel = budget_panel[~budget_panel["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
    budget_panel["residual"] = budget_panel["sales"] - budget_panel["forecast"]
    budget_panel = budget_panel.dropna(subset=["residual", "forecast", "sales"]).copy()
    budget_feat = _add_baseline_features(budget_panel, sales_agg, product_attrs)

    # --- Universes ---
    budget_universe = budget_feat.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["residual", "forecast", "sales", "horizon", "month", "quarter"]
    ).copy()
    budget_universe = prep_lags(budget_universe)
    budget_universe = budget_universe.rename(
        columns={
            "forecast": "budget_forecast",
            "origin": "budget_origin",
            "horizon": "budget_horizon",
            "version": "budget_version",
            "residual": "budget_residual",
        }
    )
    budget_universe["target_date"] = budget_universe["date"].astype(int)
    budget_universe["budget_origin"] = budget_universe["budget_origin"].astype(int)
    budget_universe["budget_horizon"] = budget_universe["budget_horizon"].astype(int)
    budget_universe["horizon"] = budget_universe["budget_horizon"]
    budget_universe["horizon_bucket"] = budget_universe["horizon"].map(horizon_bucket)

    ts_universe = feat_model.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["residual", "forecast", "sales", "horizon", "month", "quarter"]
    ).copy()
    ts_universe = prep_lags(ts_universe)
    ts_universe = ts_universe.rename(
        columns={
            "forecast": "ts_forecast",
            "origin": "ts_origin",
            "horizon": "ts_horizon",
            "residual": "ts_residual",
        }
    )
    ts_universe["target_date"] = ts_universe["date"].astype(int)
    ts_universe["ts_origin"] = ts_universe["ts_origin"].astype(int)
    ts_universe["ts_horizon"] = ts_universe["ts_horizon"].astype(int)
    ts_universe["horizon"] = ts_universe["ts_horizon"]
    ts_universe["horizon_bucket"] = ts_universe["horizon"].map(horizon_bucket)

    ts_side = ts_universe.copy()
    budget_side = budget_universe[
        [
            "product",
            "qrt",
            "date",
            "budget_forecast",
            "budget_origin",
            "budget_horizon",
            "budget_version",
            "sales",
        ]
    ].rename(columns={"sales": "budget_sales"})

    matched_all = ts_side.merge(
        budget_side, on=["product", "qrt", "date"], how="inner", suffixes=("", "_bud")
    )
    matched_all["sales"] = matched_all["sales"].astype(float)
    matched_all["human_adjustment"] = (
        matched_all["budget_forecast"] - matched_all["ts_forecast"]
    )
    matched_all["same_origin"] = matched_all["ts_origin"] == matched_all["budget_origin"]
    matched_all["same_horizon"] = matched_all["ts_horizon"] == matched_all["budget_horizon"]
    matched_all["horizon"] = matched_all["ts_horizon"]
    matched_all["horizon_bucket"] = matched_all["horizon"].map(horizon_bucket)
    matched_all["origin"] = matched_all["ts_origin"]
    matched_universe = matched_all.loc[matched_all["same_horizon"]].copy()
    matched_universe = prep_lags(matched_universe)

    raw = {
        "sales": sales_agg,
        "line_budget": budget_raw,
        "product_attrs": product_attrs,
        "results_dir": results_dir,
        "forecast_df": forecast_df,
    }
    print(
        f"built panels: ts={len(ts_universe)} budget={len(budget_universe)} "
        f"matched={len(matched_universe)}"
    )
    return ts_universe, budget_universe, matched_universe, raw


def write_freeze(
    out_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    *,
    update_tracked_manifest: bool = True,
) -> Path:
    """Build panels and write parquet + manifest under ``out_dir``."""
    out = Path(out_dir) if out_dir else default_benchmark_root()
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "ts_csvs").mkdir(exist_ok=True)

    ts_u, bud_u, matched_u, raw = build_panels(results_dir=results_dir)

    ts_path = out / "ts_universe.parquet"
    bud_path = out / "budget_universe.parquet"
    matched_path = out / "matched_universe.parquet"
    ts_u.to_parquet(ts_path, index=False)
    bud_u.to_parquet(bud_path, index=False)
    matched_u.to_parquet(matched_path, index=False)

    raw["sales"].to_parquet(out / "raw" / "sales.parquet", index=False)
    raw["line_budget"].to_parquet(out / "raw" / "line_budget.parquet", index=False)
    raw["product_attrs"].to_parquet(out / "raw" / "product_attrs.parquet", index=False)

    results_dir = Path(raw["results_dir"])
    for qrt in ALLOWED_FORECAST_QRTS:
        src = results_dir / qrt / f"{qrt}_total_forecast.csv"
        if src.exists():
            dest_dir = out / "ts_csvs" / qrt
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / src.name)

    checksums = {}
    for name in PANEL_FILES:
        checksums[name] = file_sha256(out / name)
    for name in RAW_FILES:
        rel = f"raw/{name}"
        checksums[rel] = file_sha256(out / rel)

    for csv_path in sorted((out / "ts_csvs").rglob("*.csv")):
        rel = csv_path.relative_to(out).as_posix()
        checksums[rel] = file_sha256(csv_path)

    man = {
        "version": BENCHMARK_VERSION,
        "description": (
            "Frozen matched Human/TS rolling-origin benchmark from "
            "notebooks/residual_prediction.ipynb Analysis B PRIMARY."
        ),
        "primary_origins": list(PRIMARY_ORIGINS),
        "row_counts": {
            "ts_universe": int(len(ts_u)),
            "budget_universe": int(len(bud_u)),
            "matched_universe": int(len(matched_u)),
            "ts_products": int(ts_u["product"].nunique()),
            "budget_products": int(bud_u["product"].nunique()),
            "matched_products": int(matched_u["product"].nunique()),
        },
        "incomplete_shamsi_months": sorted(INCOMPLETE_SHAMSI_MONTHS),
        "excluded_odd_coverage_products": sorted(EXCLUDED_ODD_COVERAGE_PRODUCTS),
        "allowed_forecast_qrts": list(ALLOWED_FORECAST_QRTS),
        "expected_analysis_b_primary_wmape": EXPECTED_ANALYSIS_B_PRIMARY,
        "expected_analysis_a_primary_wmape": EXPECTED_ANALYSIS_A_PRIMARY,
        "checksums": checksums,
        "schema": {
            "matched_universe_required": [
                "product",
                "qrt",
                "date",
                "target_date",
                "origin",
                "horizon",
                "sales",
                "ts_forecast",
                "budget_forecast",
                "human_adjustment",
                "sales_lag_1",
                "sales_lag_2",
                "sales_lag_3",
                "sales_lag_12",
                "sales_roll3",
                "month",
                "quarter",
                "model_enc",
                "field_enc",
                "form_enc",
                "provider_enc",
            ],
        },
    }

    local_manifest = out / "manifest.json"
    with local_manifest.open("w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
        f.write("\n")

    if update_tracked_manifest:
        tracked = manifest_path()
        with tracked.open("w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
            f.write("\n")
        print(f"updated tracked manifest: {tracked}")

    print(f"wrote freeze to {out}")
    return out


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Freeze benchmark v1 panels")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: src/data/benchmarks/v1)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="TS vintage CSV root (default: src/data/results)",
    )
    parser.add_argument(
        "--no-tracked-manifest",
        action="store_true",
        help="Do not overwrite pkg/benchmark/v1_manifest.json",
    )
    args = parser.parse_args(argv)
    write_freeze(
        out_dir=args.out,
        results_dir=args.results_dir,
        update_tracked_manifest=not args.no_tracked_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
