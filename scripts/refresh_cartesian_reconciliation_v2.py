"""Refresh cartesian reconciliation after origin-relative NO_SALES fix."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.forecast_jobs.plan import decode_product_key
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import DEFAULT_CONFIG

OUT = REPO / "src/data/results/product_universe/portfolio_backfill_prep_v2"
INV = REPO / "src/data/results/product_universe/backfill_inventory_v2"
MIN_TRAIN = 12
A0_SEEDS = [41, 42, 43]


def main() -> int:
    locked = [
        int(r["forecast_origin"])
        for r in json.loads((INV / "proposed_origins_locked.json").read_text(encoding="utf-8"))[
            "locked_origins"
        ]
    ]
    dense = pd.read_parquet(OUT / "sales_dense_for_eligibility.parquet")
    old = pd.read_csv(OUT / "cartesian_reconciliation_v2.csv")
    products = sorted(old["product"].astype(str).unique())
    dense_by = {str(p): g for p, g in dense.groupby("product")}

    def train_months(g: pd.DataFrame, origin: int) -> int:
        lc = shamsi_add_months(int(origin), -2)
        return int((pd.to_numeric(g["date"], errors="coerce") <= lc).sum())

    def full_at_origin(g: pd.DataFrame, origin: int) -> bool:
        series = pd.Series(
            pd.to_numeric(g["sales"], errors="coerce").to_numpy(),
            index=pd.Index([int(x) for x in g["date"].tolist()], dtype="int64"),
        )
        covers = discover_origins(
            series, config=DEFAULT_CONFIG, explicit_origins=(int(origin),)
        )
        return bool(covers) and bool(covers[0].full_horizon_coverage)

    def po_classify(g, origin: int) -> str:
        if g is None or g.empty:
            return "NO_SALES"
        lc = shamsi_add_months(int(origin), -2)
        before = g.loc[pd.to_numeric(g["date"], errors="coerce") <= lc]
        if before.empty or int(before["sales"].fillna(0).gt(0).sum()) == 0:
            return "NO_SALES"
        if train_months(g, origin) < MIN_TRAIN:
            return "EXPECTED_MODEL_INELIGIBLE"
        if full_at_origin(g, origin):
            return "ELIGIBLE"
        return "INSUFFICIENT_HISTORY"

    success: set[tuple] = set()
    for run_id, _eng in (
        ("portfolio_v21_backfill_v2", "v2.1"),
        ("portfolio_a0_backfill_v2", "a0"),
    ):
        conn = sqlite3.connect(str(REPO / "src/data/forecast_jobs" / run_id / "state.sqlite"))
        for row in conn.execute(
            "SELECT product_id, forecast_origin, engine_version FROM jobs WHERE status='SUCCESS'"
        ):
            product, seed = decode_product_key(row[0])
            success.add(
                (
                    str(product),
                    int(row[1]),
                    str(row[2]).lower(),
                    None if seed is None else int(seed),
                )
            )
        conn.close()

    rows = []
    for product in products:
        g = dense_by.get(product)
        for origin in locked:
            st = po_classify(g, origin)
            for engine, seeds in (("v2.1", [None]), ("a0", list(A0_SEEDS))):
                for seed in seeds:
                    if st == "ELIGIBLE":
                        key = (product, origin, engine, seed)
                        bucket = "SUCCESS_IMPORTED" if key in success else "PENDING_MISSING"
                    elif st == "EXPECTED_MODEL_INELIGIBLE":
                        bucket = "SKIPPED_EXPECTED_MODEL_INELIGIBLE"
                    elif st == "INSUFFICIENT_HISTORY":
                        bucket = "SKIPPED_INSUFFICIENT_HISTORY"
                    elif st == "NO_SALES":
                        bucket = "SKIPPED_NO_SALES"
                    else:
                        bucket = "SKIPPED_OTHER"
                    rows.append(
                        {
                            "product": product,
                            "forecast_origin": origin,
                            "engine": engine,
                            "seed": "" if seed is None else seed,
                            "product_origin_status": st,
                            "bucket": bucket,
                        }
                    )
    slots = pd.DataFrame(rows)
    slots.to_csv(OUT / "cartesian_reconciliation_v2.csv", index=False, lineterminator="\n")

    def bc(eng: str) -> dict[str, int]:
        return slots.loc[slots.engine == eng, "bucket"].value_counts().astype(int).to_dict()

    bv, ba = bc("v2.1"), bc("a0")
    sv, sa = int(sum(bv.values())), int(sum(ba.values()))
    recon = {
        "buckets_v21": bv,
        "buckets_a0": ba,
        "grid_v21": 5445,
        "grid_a0": 16335,
        "sum_v21_buckets": sv,
        "sum_a0_buckets": sa,
        "grid_v21_ok": sv == 5445,
        "grid_a0_ok": sa == 16335,
        "SKIPPED_NO_SALES_note": (
            "No positive sales available through origin-2 for that product-origin "
            "(not necessarily no sales in the full source)."
        ),
        "success_after_effective_hash": {
            "v2.1": int(bv.get("SUCCESS_IMPORTED", 0)),
            "a0": int(ba.get("SUCCESS_IMPORTED", 0)),
        },
    }
    (OUT / "cartesian_reconciliation_v2.json").write_text(
        json.dumps(recon, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(recon, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
