"""Extract a versioned full-portfolio sales Parquet (read-only for jobs).

Uses ``pkg.db.query.sales.load_sales_data`` and ``dim_product`` basket universe.
Does **not** overwrite ``src/data/benchmarks/v1/raw/sales.parquet``.
Does **not** run forecasting.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import INCOMPLETE_SHAMSI_MONTHS
from pkg.db.query.dim_product import load_dim_product, select_basket_products
from pkg.db.query.sales import load_sales_data

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = (
    REPO / "src" / "data" / "results" / "product_universe" / "sales_full_portfolio_v1"
)
SNAPSHOT_VERSION = "sales_full_portfolio_v1"
ARCHVAL55_SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
PINNED_ARCHVAL55_SHA = (
    "1d146d49261357f6b35d743a4c4064b22abe82047f6f5c36bbe4496d03f1ffc7"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def products_list_sha256(products: Sequence[str]) -> str:
    canon = sorted({str(p).strip() for p in products if str(p).strip()}, key=str.casefold)
    return hashlib.sha256(("\n".join(canon) + "\n").encode("utf-8")).hexdigest()


def series_history_sha256(dates: Iterable[int], values: Iterable[float]) -> str:
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        if v != v:  # NaN
            token = f"{int(d)}|\n"
        else:
            token = f"{int(d)}|{float(v):.10g}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def aggregate_full_portfolio_sales(
    raw: pd.DataFrame,
    universe_products: Sequence[str],
) -> pd.DataFrame:
    """Freeze aggregation contract without TARGET_GENERIC_EN / odd-coverage filters."""
    work = raw.copy()
    work["product"] = work["product"].astype(str).str.strip()
    work["date"] = pd.to_numeric(work["date"], errors="coerce").astype("Int64")
    work["sales"] = pd.to_numeric(work["sales"], errors="coerce")
    work = work.loc[work["date"].notna()].copy()
    work["date"] = work["date"].astype(int)
    work = work.loc[~work["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
    allowed = {str(p).strip() for p in universe_products}
    work = work.loc[work["product"].isin(allowed)].copy()
    agg = (
        work.groupby(["product", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["product", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    return agg.loc[:, ["product", "date", "sales"]]


def build_history_hashes(
    sales: pd.DataFrame,
    origins: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for product, g in sales.groupby("product", sort=False):
        g = g.sort_values("date")
        full_hash = series_history_sha256(
            [int(x) for x in g["date"].tolist()],
            [float(x) if pd.notna(x) else float("nan") for x in g["sales"].tolist()],
        )
        rows.append(
            {
                "product": str(product),
                "forecast_origin": "",
                "last_complete": "",
                "scope": "full_series",
                "n_history_rows": int(len(g)),
                "history_sha256": full_hash,
            }
        )
        for origin in origins:
            last_complete = shamsi_add_months(int(origin), -2)
            hist = g.loc[g["date"] <= last_complete].sort_values("date")
            h = series_history_sha256(
                [int(x) for x in hist["date"].tolist()],
                [
                    float(x) if pd.notna(x) else float("nan")
                    for x in hist["sales"].tolist()
                ],
            )
            rows.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "last_complete": int(last_complete),
                    "scope": "through_last_complete",
                    "n_history_rows": int(len(hist)),
                    "history_sha256": h,
                }
            )
    return pd.DataFrame(rows)


def extract(
    *,
    out_dir: Optional[Path] = None,
    candidate_origins: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Run extract; return meta dict. Refuses overwrite if parquet already exists."""
    directory = Path(out_dir) if out_dir is not None else OUT_DIR
    parquet_path = directory / "sales.parquet"
    if parquet_path.is_file():
        raise FileExistsError(
            f"Refusing to overwrite immutable snapshot at {parquet_path}. "
            "Use a new version directory if a refresh is required."
        )

    extracted_at = _utc_now()
    dim = load_dim_product()
    basket = select_basket_products(dim)
    universe = sorted(
        basket["ProductTitleEN"].astype(str).str.strip().tolist(),
        key=str.casefold,
    )
    universe_sha = products_list_sha256(universe)

    raw = load_sales_data()
    sales = aggregate_full_portfolio_sales(raw, universe)

    # Default candidate origins for history hashes: broad quarterly + known HB/archval.
    if candidate_origins is None:
        vintages = REPO / "src" / "pkg" / "benchmark" / "vintages" / "ts_backfill_1401Q1_1405Q2.csv"
        origins: list[int] = []
        if vintages.is_file():
            vdf = pd.read_csv(vintages)
            origins.extend(int(x) for x in vdf["forecast_origin"].tolist())
        origins.extend([140104, 140201, 140301, 140402])
        budget_path = (
            REPO / "src" / "data" / "benchmarks" / "v1" / "budget_universe.parquet"
        )
        if budget_path.is_file():
            bud = pd.read_parquet(budget_path, columns=["budget_origin"])
            origins.extend(int(x) for x in bud["budget_origin"].dropna().unique())
        candidate_origins = sorted(set(origins))

    hashes = build_history_hashes(sales, list(candidate_origins))

    directory.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(parquet_path, index=False)
    hashes.to_csv(
        directory / "product_history_hashes.csv",
        index=False,
        lineterminator="\n",
        encoding="utf-8",
    )

    size_bytes = int(parquet_path.stat().st_size)
    content_sha = file_sha256(parquet_path)
    old_sha = file_sha256(ARCHVAL55_SALES) if ARCHVAL55_SALES.is_file() else None

    meta: dict[str, Any] = {
        "snapshot_version": SNAPSHOT_VERSION,
        "extracted_at_utc": extracted_at,
        "immutable": True,
        "read_only_for_jobs": True,
        "source": {
            "interface": "pkg.db.query.sales.load_sales_data",
            "table": "[DWOrchid].[dbo].[Flat_Fact_Sale]",
            "universe_interface": (
                "pkg.db.query.dim_product.load_dim_product / select_basket_products"
            ),
            "dim_table": "[Iris_DW].[Dim].[Product]",
            "aggregation_contract": (
                "SUM(DQTY) by ProductTitleEN × ShamsiYearMonth; "
                f"drop incomplete months {sorted(INCOMPLETE_SHAMSI_MONTHS)}; "
                "groupby(product, date).sum(); filter to dim basket universe; "
                "no TARGET_GENERIC_EN filter; no EXCLUDED_ODD_COVERAGE filter"
            ),
            "schema": {"columns": ["product", "date", "sales"]},
        },
        "product_universe": {
            "n_products": len(universe),
            "products_sha256": universe_sha,
        },
        "sales": {
            "path": parquet_path.as_posix(),
            "n_rows": int(len(sales)),
            "n_products_with_sales": int(sales["product"].nunique()) if len(sales) else 0,
            "min_date": int(sales["date"].min()) if len(sales) else None,
            "max_date": int(sales["date"].max()) if len(sales) else None,
            "size_bytes": size_bytes,
            "content_sha256": content_sha,
        },
        "archval55_reference": {
            "path": ARCHVAL55_SALES.as_posix(),
            "pinned_sha256": PINNED_ARCHVAL55_SHA,
            "live_sha256": old_sha,
            "matches_pinned": (
                old_sha.lower() == PINNED_ARCHVAL55_SHA.lower() if old_sha else False
            ),
            "note": "Not overwritten; retained as 55-SKU experiment snapshot.",
        },
        "history_hash_origins": list(candidate_origins),
    }
    atomic_write_json(directory / "meta.json", meta)
    return meta


def main() -> int:
    meta = extract()
    print(json.dumps(meta["sales"], indent=2))
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
