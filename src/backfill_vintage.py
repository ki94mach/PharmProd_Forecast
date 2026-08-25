"""
Reconstruct historical TS forecast vintages with the same modules as production.

Uses SalesForecasting / SalesForecast, but truncates sales to date < origin so the
forecast window is as-of that historical start date. Default product scope is
TARGET_GENERIC_EN (aligned with residual_prediction.ipynb).

Examples (run from src/):

  python backfill_vintage.py --qrt 1404Q1 --start-date 140312 --force
  python backfill_vintage.py --qrt 1405Q1 --start-date 140501 --force
  python backfill_vintage.py --qrt 1402Q4 --start-date 140210 --force --skip-excel

After a successful run, add the qrt to ALLOWED_FORECAST_QRTS in
notebooks/residual_prediction.ipynb and re-run the residual experiment.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pkg.env import load_project_env

load_project_env()

from pkg.db.query.constants import TARGET_GENERIC_EN
from pkg.db.query.dim_product import load_dim_product
from pkg.db.query.sales import load_sales_data
from pkg.sales_forecasting import SalesForecasting
from pkg.utils import DATA_DIR, manage_excel, pivot_and_format_data, update_department_info

# Keep reconstructed vintages aligned with residual notebook exclusions
EXCLUDED_ODD_COVERAGE_PRODUCTS = {
    "Nanojade 90 Old",
    "Nanojade 180 Old",
    "Nanojade 360 Old",
    "Kidi Mab",
    "Alvocade 3.5",
}

FORECAST_HORIZON_MONTHS = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill a historical TS forecast vintage (same models as main.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--qrt", required=True, help="Quarter label, e.g. 1404Q1")
    parser.add_argument(
        "--start-date",
        required=True,
        type=int,
        help="Forecast origin Shamsi YYYYMM (sales kept strictly before this)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required to overwrite: backup then reset {qrt}_total_forecast.csv",
    )
    parser.add_argument(
        "--products",
        choices=["target"],
        default="target",
        help="Product scope (default: TARGET_GENERIC_EN only)",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Write CSV only (skip department Excel packaging)",
    )
    return parser.parse_args()


def target_product_names() -> set[str]:
    dim = load_dim_product()
    names = set(
        dim.loc[dim["GenericEN"].isin(TARGET_GENERIC_EN), "ProductTitleEN"]
        .dropna()
        .astype(str)
    )
    return names - EXCLUDED_ODD_COVERAGE_PRODUCTS


def reset_forecast_csv(csv_path: str, headers: list[str], force: bool) -> Path | None:
    """Backup existing CSV (if any) and rewrite headers-only. Returns backup path."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists() and path.stat().st_size > 0:
        if not force:
            raise SystemExit(
                f"Refusing to overwrite existing {path}. Re-run with --force "
                "(placeholder/resume CSVs must be cleared before backfill)."
            )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        print(f"backed up existing CSV -> {backup_path}")

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"reset forecast CSV (headers only): {path}")
    return backup_path


def modal_origin_from_csv(df: pd.DataFrame, n_months: int = FORECAST_HORIZON_MONTHS) -> tuple[int | None, dict]:
    """Infer modal origin as first month of each product's last n dates."""
    if df.empty:
        return None, {}
    work = df.copy()
    work["date"] = work["date"].astype(int)
    origins = []
    for _, g in work.groupby("product", sort=False):
        dates = sorted(g["date"].unique().tolist())
        if not dates:
            continue
        window = dates[-n_months:]
        origins.append(int(window[0]))
    if not origins:
        return None, {}
    vc = pd.Series(origins).value_counts()
    return int(vc.index[0]), vc.to_dict()


def validate_forecast_csv(csv_path: str, expected_origin: int) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("VALIDATION FAIL: forecast CSV is empty")

    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    n_rows = len(df)
    n_products = df["product"].nunique()
    nonzero = int((df["forecast"].fillna(0) != 0).sum())
    nonzero_frac = nonzero / n_rows if n_rows else 0.0
    model_nonnull = int(df["model"].notna().sum()) if "model" in df.columns else 0
    model_frac = model_nonnull / n_rows if n_rows else 0.0

    modal_origin, origin_counts = modal_origin_from_csv(df)
    print("\n=== backfill validation ===")
    print(f"csv: {csv_path}")
    print(f"rows={n_rows} products={n_products}")
    print(f"nonzero_forecast_rows={nonzero} ({nonzero_frac:.1%})")
    print(f"nonnull_model_rows={model_nonnull} ({model_frac:.1%})")
    print(f"expected_origin={expected_origin} modal_origin={modal_origin}")
    print(f"origin_counts (top): {dict(list(sorted(origin_counts.items(), key=lambda x: -x[1])[:8]))}")

    if nonzero == 0:
        raise SystemExit(
            "VALIDATION FAIL: all forecasts are zero (still a placeholder). "
            "Check sales truncation, stale gate, and --force reset."
        )
    if modal_origin is None:
        raise SystemExit("VALIDATION FAIL: could not infer modal origin")
    if modal_origin != int(expected_origin):
        # Soft warning: sparse products can pull max(sales)+1 off the requested origin
        print(
            f"WARNING: modal origin {modal_origin} != expected {expected_origin}. "
            "Majority products should still align; inspect origin_counts."
        )
    else:
        print("VALIDATION OK: modal origin matches --start-date")


def prepare_sales(start_date: int, product_names: set[str]) -> pd.DataFrame:
    sale_df = load_sales_data()
    sale_df["date"] = sale_df["date"].astype(int)
    before = len(sale_df)
    sale_df = sale_df[sale_df["product"].isin(product_names)].copy()
    sale_df = sale_df[sale_df["date"] < int(start_date)].copy()
    print(
        f"sales filtered: {before} -> {len(sale_df)} rows "
        f"({sale_df['product'].nunique()} products) with date < {start_date}"
    )
    if sale_df.empty:
        raise SystemExit("No sales rows after product + date filters")
    print(
        f"sales date range after cutoff: {int(sale_df['date'].min())} .. {int(sale_df['date'].max())}"
    )
    return sale_df


def run_backfill(qrt: str, start_date: int, force: bool, skip_excel: bool) -> None:
    product_names = target_product_names()
    print(f"target products in scope: {len(product_names)}")

    sf = SalesForecasting(qrt)
    reset_forecast_csv(sf.forecasts, sf.headers, force=force)

    sale_df = prepare_sales(start_date, product_names)
    # Empty forecast frame => process all products in sale_df (no resume skip)
    forecast_df = pd.DataFrame(columns=sf.headers)

    print(f"\nRunning TS models for qrt={qrt}, origin={start_date} ...")
    forecast_total_df = sf.process_sales_data(
        sale_df, forecast_df, start_date, skip_forecast=False
    )
    print(f"wrote {len(forecast_total_df)} forecast rows -> {sf.forecasts}")

    if not skip_excel:
        print("Building department Excel outputs ...")
        updated_dep_dict = update_department_info(qrt)
        excel_df = forecast_total_df.copy()
        excel_df["sales"] = excel_df["forecast"]
        excel_df["type"] = "forecast"
        pivot = pivot_and_format_data(excel_df, updated_dep_dict, start_date)
        manage_excel(pivot, os.path.join(DATA_DIR, "results", qrt), qrt)
        print(f"excel outputs under {os.path.join(DATA_DIR, 'results', qrt)}")
    else:
        print("skipped Excel packaging (--skip-excel)")

    validate_forecast_csv(sf.forecasts, expected_origin=start_date)


def main():
    args = parse_args()
    try:
        start_date = int(args.start_date)
    except (TypeError, ValueError):
        print("Error: --start-date must be an integer YYYYMM (e.g. 140312).", file=sys.stderr)
        sys.exit(1)

    if not args.force:
        print(
            "Note: --force is required to clear an existing CSV before backfill.",
            file=sys.stderr,
        )

    run_backfill(
        qrt=args.qrt,
        start_date=start_date,
        force=args.force,
        skip_excel=args.skip_excel,
    )
    print("Done.")


if __name__ == "__main__":
    main()
