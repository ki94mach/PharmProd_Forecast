"""Post-execute verifier for portfolio_backfill_prep_v2 smoke runs.

Checks training contract, forecast shape, A0 seed completeness, status
partition, and resume semantics. Does not execute forecasting.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(REPO / "scripts"))

from pkg.benchmark.backfill_runner.state import (  # noqa: E402
    JOB_FAILED,
    JOB_INELIGIBLE,
    JOB_PENDING,
    JOB_SKIPPED,
    JOB_SUCCESS,
    JobStateStore,
    should_run_job,
)
from pkg.benchmark.calendar import shamsi_add_months  # noqa: E402
from pkg.forecast_jobs.plan import decode_product_key  # noqa: E402
from pkg.ts_v2.config import DEFAULT_CONFIG  # noqa: E402
from pkg.ts_v2.data import prepare_monthly_series  # noqa: E402
from pkg.ts_v2.dates import make_forecast_window  # noqa: E402

from export_portfolio_backfill_panel import (  # noqa: E402
    REQUIRED_A0_SEEDS,
    assemble_success_panel,
    seed_mean_panel,
)

FORECAST_JOBS = REPO / "src" / "data" / "forecast_jobs"
SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)

SMOKE_CASES = [
    ("Recigen AutoInjector", 140402),
    ("Cinnapoietin 4000", 140402),
    ("OvuGen", 140402),
    ("Canvert Fade", 140402),
    ("Altebrel 25", 140401),
]


def _status_hist(run_dir: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(run_dir / "state.sqlite"))
    rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
    conn.close()
    return {str(s): int(n) for s, n in rows}


def _check_training_contract(sales: pd.DataFrame, product: str, origin: int) -> list[str]:
    errs: list[str] = []
    window = make_forecast_window(int(origin), config=DEFAULT_CONFIG)
    prepared = prepare_monthly_series(sales, product, int(origin), config=DEFAULT_CONFIG)
    if prepared.values.empty:
        errs.append(f"{product}@{origin}: empty prepared series")
        return errs
    if int(prepared.values.index.max()) != int(window.training_end):
        errs.append(
            f"{product}@{origin}: training_end mismatch "
            f"{prepared.values.index.max()} != {window.training_end}"
        )
    bridge = shamsi_add_months(int(origin), -1)
    if bridge in set(int(x) for x in prepared.values.index):
        errs.append(f"{product}@{origin}: bridge month {bridge} present in train")
    if int(window.training_end) != shamsi_add_months(int(origin), -2):
        errs.append(f"{product}@{origin}: window.training_end != origin-2")
    return errs


def _check_forecast_frame(frame: pd.DataFrame, product: str, origin: int) -> list[str]:
    errs: list[str] = []
    if frame.empty:
        errs.append(f"{product}@{origin}: empty forecast")
        return errs
    work = frame.copy()
    if "horizon" in work.columns:
        work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce")
    if "target_date" in work.columns:
        work["target_date"] = pd.to_numeric(work["target_date"], errors="coerce")
    if "forecast" in work.columns:
        work["forecast"] = pd.to_numeric(work["forecast"], errors="coerce")
    if len(work) != 15:
        errs.append(f"{product}@{origin}: expected 15 rows, got {len(work)}")
    h1 = work.loc[work["horizon"] == 1] if "horizon" in work.columns else work.iloc[0:0]
    if h1.empty:
        errs.append(f"{product}@{origin}: missing horizon 1")
    else:
        td = int(h1.iloc[0]["target_date"])
        if td != int(origin):
            errs.append(f"{product}@{origin}: h1 target_date {td} != origin")
    preds = work["forecast"] if "forecast" in work.columns else pd.Series(dtype=float)
    if not preds.notna().all():
        errs.append(f"{product}@{origin}: non-finite forecast present")
    if (preds < 0).any():
        errs.append(f"{product}@{origin}: negative forecast present")
    return errs


def verify_run(*, run_id: str, engine: str, expect_pending_ok: bool = False) -> dict[str, Any]:
    run_dir = FORECAST_JOBS / run_id
    errors: list[str] = []
    hist = _status_hist(run_dir)

    # Status partition
    for st in (JOB_SKIPPED, JOB_INELIGIBLE, JOB_FAILED, JOB_SUCCESS, JOB_PENDING):
        _ = hist.get(st, 0)
    if hist.get(JOB_SUCCESS, 0) and hist.get("allowlist_skip", 0):
        errors.append("allowlist_skip must not appear as SUCCESS")

    # Empty forecast cannot be SUCCESS
    conn = sqlite3.connect(str(run_dir / "state.sqlite"))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(
        "SELECT product_id, forecast_origin, status, output_path FROM jobs WHERE status=?",
        (JOB_SUCCESS,),
    ):
        out = Path(str(row["output_path"] or ""))
        fc = out / "forecast.csv"
        if not fc.is_file() or pd.read_csv(fc).empty:
            errors.append(
                f"SUCCESS without non-empty forecast: {row['product_id']}@{row['forecast_origin']}"
            )
    conn.close()

    # Resume does not re-claim SUCCESS/SKIPPED
    store = JobStateStore(run_dir)
    conn = sqlite3.connect(str(run_dir / "state.sqlite"))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(
        "SELECT * FROM jobs WHERE status IN (?, ?)", (JOB_SUCCESS, JOB_SKIPPED)
    ):
        from pkg.benchmark.backfill_runner.state import JobIdentity, JobRecord

        identity = JobIdentity(
            experiment_id=str(row["experiment_id"]),
            engine_version=str(row["engine_version"]),
            config_hash=str(row["config_hash"]),
            quarter=str(row["quarter"]),
            forecast_origin=int(row["forecast_origin"]),
            product_id=str(row["product_id"]),
        )
        rec = store.get_job(identity.job_id)
        assert rec is not None
        if should_run_job(rec, resume=True, retry_failed=False, force_job=False):
            errors.append(f"resume would re-run terminal job {identity.job_id}")
    conn.close()

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str).str.strip()
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")

    for product, origin in SMOKE_CASES:
        errors.extend(_check_training_contract(sales, product, origin))

    panel = assemble_success_panel(run_dir, engine=engine)
    n_success = int(hist.get(JOB_SUCCESS, 0))
    if n_success == 0 and expect_pending_ok:
        return {
            "run_id": run_id,
            "engine": engine,
            "status_counts": hist,
            "n_success": 0,
            "ok": len(errors) == 0,
            "errors": errors,
            "note": "no SUCCESS yet (pre-execute / pending smoke)",
        }

    if not panel.empty:
        for (product, origin), g in panel.groupby(["product", "forecast_origin"]):
            errors.extend(_check_forecast_frame(g, str(product), int(origin)))

    if engine.lower() == "a0" and n_success:
        # All three seeds for each case that has any SUCCESS
        for product, origin in SMOKE_CASES:
            sub = panel.loc[
                (panel["product"] == product) & (panel["forecast_origin"] == origin)
            ]
            if sub.empty:
                continue
            seeds = set(int(x) for x in sub["seed"].dropna())
            if seeds and seeds != set(REQUIRED_A0_SEEDS):
                if not (set(REQUIRED_A0_SEEDS) <= seeds):
                    errors.append(
                        f"A0 {product}@{origin}: seeds {sorted(seeds)} missing required "
                        f"{sorted(REQUIRED_A0_SEEDS)}"
                    )
        mean = seed_mean_panel(panel)
        # Groups in mean must have had all seeds
        if not mean.empty:
            for (product, origin), g in panel.groupby(["product", "forecast_origin"]):
                seeds = set(int(x) for x in g["seed"].dropna())
                in_mean = not mean.loc[
                    (mean["product"] == product) & (mean["forecast_origin"] == origin)
                ].empty
                if in_mean and not (set(REQUIRED_A0_SEEDS) <= seeds):
                    errors.append(
                        f"seed_mean emitted without all seeds for {product}@{origin}"
                    )
                if (not in_mean) and set(REQUIRED_A0_SEEDS) <= seeds:
                    errors.append(
                        f"seed_mean missing despite complete seeds for {product}@{origin}"
                    )

    return {
        "run_id": run_id,
        "engine": engine,
        "status_counts": hist,
        "n_success": n_success,
        "ok": len(errors) == 0,
        "errors": errors,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--engine", required=True, choices=["v2.1", "a0"])
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="Pass when smoke has not been executed yet (0 SUCCESS ok)",
    )
    args = p.parse_args()
    result = verify_run(
        run_id=args.run_id,
        engine=args.engine,
        expect_pending_ok=bool(args.allow_empty),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
