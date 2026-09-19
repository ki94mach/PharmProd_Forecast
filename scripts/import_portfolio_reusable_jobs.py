"""Import reusable V2.1/A0 jobs into a new portfolio run (copy-only; no execute).

Validates per-product history hashes against sales_full_portfolio_v1.
Does not mutate source run trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.backfill_runner.state import resolve_git_commit  # noqa: E402
from pkg.forecast_jobs.config import load_job_config  # noqa: E402
from pkg.forecast_jobs.plan import PlannedJob, encode_product_key  # noqa: E402
from pkg.forecast_jobs.runner import prepare_run  # noqa: E402
from pkg.forecast_jobs.store import atomic_write_json, job_artifact_dir  # noqa: E402
from pkg.ts_v2.data import prepare_monthly_series  # noqa: E402

INV = REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v2"
SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)
FORECAST_JOBS = REPO / "src" / "data" / "forecast_jobs"
ARCHVAL25 = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval25_export"
)
A0_MODEL = "legacy_adaptive_recursive_lstm"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def series_history_sha256(dates, values) -> str:
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        if v != v:
            token = f"{int(d)}|\n"
        else:
            token = f"{int(d)}|{float(v):.10g}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def history_hash(sales: pd.DataFrame, product: str, origin: int) -> str:
    """Effective prepared training series through origin-2 (zero-filled gaps)."""
    prepared = prepare_monthly_series(sales, product, int(origin))
    series = prepared.values
    if series.empty:
        return ""
    return series_history_sha256(
        [int(x) for x in series.index.tolist()],
        [float(x) if pd.notna(x) else float("nan") for x in series.tolist()],
    )


def _source_root(run_id: str) -> Path:
    if run_id == "archval25_export" or run_id == ARCHVAL25.name:
        return ARCHVAL25
    p = FORECAST_JOBS / run_id
    if p.is_dir():
        return p
    # archval25 export folder name
    if (ARCHVAL25 / "state.sqlite").is_file() and "archval25" in run_id:
        return ARCHVAL25
    return p


def _lookup_source_output(
    run_root: Path, *, engine: str, product: str, origin: int, seed: Optional[int]
) -> Optional[Path]:
    db = run_root / "state.sqlite"
    if not db.is_file():
        return None
    product_id = encode_product_key(product, seed)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT output_path FROM jobs WHERE lower(engine_version)=? "
            "AND product_id=? AND forecast_origin=? AND status='SUCCESS' LIMIT 1",
            (engine.lower(), product_id, int(origin)),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    path = Path(str(row[0]))
    if path.is_dir() and (path / "forecast.csv").is_file():
        return path
    # Remap absolute paths from other machines
    rel_jobs = run_root / "jobs" / engine.lower()
    if rel_jobs.is_dir():
        # try slug match
        from pkg.benchmark.calendar import quarter_from_origin

        q = quarter_from_origin(int(origin))
        safe = product_id.replace("/", "_").replace("\\", "_")
        for name in (f"{q}__{int(origin)}__{safe}", f"{q}__{safe}"):
            cand = rel_jobs / name
            if (cand / "forecast.csv").is_file():
                return cand
    return path if path.exists() else None


def _planned_match(plan_jobs: list[PlannedJob], row: dict[str, Any]) -> Optional[PlannedJob]:
    seed_raw = row.get("seed")
    seed = (
        None
        if seed_raw == "" or seed_raw is None or (isinstance(seed_raw, float) and pd.isna(seed_raw))
        else int(seed_raw)
    )
    for job in plan_jobs:
        if (
            str(job.product) == str(row["product"])
            and int(job.origin) == int(row["forecast_origin"])
            and str(job.architecture).lower() == str(row["engine"]).lower()
            and job.seed == seed
        ):
            return job
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", choices=("v2.1", "a0"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-hash-gate",
        action="store_true",
        help=(
            "Import when forecast.csv exists even if inventory product_history_hash "
            "does not match live prepare_monthly_series hash (inventory still uses "
            "sparse hashes; archval artifacts were trained with prepare)."
        ),
    )
    args = ap.parse_args()

    config = load_job_config(args.config)
    root, cfg_hash, plan, _sales_df, _snap = prepare_run(config, dry_run=False)
    # Rebuild plan with ensure_jobs for full Cartesian
    from pkg.forecast_jobs.plan import build_job_plan
    from pkg.benchmark.backfill_runner.state import JobStateStore

    state = JobStateStore(root)
    git = resolve_git_commit()
    plan = build_job_plan(
        config, config_hash=cfg_hash, state=state, git_commit=git, ensure_jobs=True
    )

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str).str.strip()
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")
    # Exclude partial month from hash validation basis
    sales = sales.loc[sales["date"] <= 140504].copy()

    reusable = pd.read_csv(INV / "reusable_completed_jobs.csv")
    reusable = reusable.loc[reusable["engine"].astype(str).str.lower() == args.engine].copy()

    imported = 0
    skipped_hash = 0
    skipped_missing = 0
    skipped_no_plan = 0
    report_rows = []

    for _, row in reusable.iterrows():
        rec = row.to_dict()
        product = str(rec["product"])
        origin = int(rec["forecast_origin"])
        seed_raw = rec.get("seed")
        seed = (
            None
            if seed_raw == "" or (isinstance(seed_raw, float) and pd.isna(seed_raw))
            else int(seed_raw)
        )
        live_hash = history_hash(sales, product, origin)
        inv_hash = str(rec.get("product_history_hash") or "")
        if (
            inv_hash
            and inv_hash != live_hash
            and not args.skip_hash_gate
        ):
            skipped_hash += 1
            report_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "seed": "" if seed is None else seed,
                    "status": "skipped_hash_mismatch",
                }
            )
            continue

        planned = _planned_match(plan.jobs if hasattr(plan, "jobs") else [], rec)
        # plan.jobs may only include PENDING; search all products via ensure
        if planned is None:
            # synthesize identity from config for marking
            from pkg.benchmark.calendar import quarter_from_origin
            from pkg.benchmark.backfill_runner.state import JobIdentity

            q = quarter_from_origin(origin)
            identity = JobIdentity(
                experiment_id=config.run_id,
                engine_version=args.engine,
                config_hash=cfg_hash,
                quarter=q,
                forecast_origin=origin,
                product_id=encode_product_key(product, seed),
            )
            planned = PlannedJob(
                architecture=args.engine,
                origin=origin,
                quarter=q,
                product=product,
                seed=seed,
                identity=identity,
            )
            state.ensure_job(identity, git_commit=git)

        run_id = str(rec.get("run_id") or "")
        src_root = _source_root(run_id)
        src_dir = _lookup_source_output(
            src_root, engine=args.engine, product=product, origin=origin, seed=seed
        )
        if src_dir is None or not (Path(src_dir) / "forecast.csv").is_file():
            # try archval25 if not found
            if src_root != ARCHVAL25:
                src_dir = _lookup_source_output(
                    ARCHVAL25, engine=args.engine, product=product, origin=origin, seed=seed
                )
        if src_dir is None or not (Path(src_dir) / "forecast.csv").is_file():
            skipped_missing += 1
            report_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "seed": "" if seed is None else seed,
                    "status": "skipped_missing_artifact",
                    "run_id": run_id,
                }
            )
            continue

        src_dir = Path(src_dir)
        out_dir = job_artifact_dir(root, planned.identity)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / "forecast.csv", out_dir / "forecast.csv")
            if (src_dir / "job.json").is_file():
                shutil.copy2(src_dir / "job.json", out_dir / "job.json")
            else:
                atomic_write_json(
                    out_dir / "job.json",
                    {
                        "job_id": planned.identity.job_id,
                        "run_id": config.run_id,
                        "architecture": args.engine,
                        "product": product,
                        "forecast_origin": origin,
                        "seed": seed,
                        "selected_model": rec.get("model_version"),
                        "imported_from": run_id,
                        "product_history_hash": live_hash,
                        "success": True,
                    },
                )
            model = str(rec.get("model_version") or "")
            if args.engine == "a0" and model.startswith(A0_MODEL):
                model = A0_MODEL if ":" in model or model.startswith(A0_MODEL) else model
            now = _utc()
            state.mark_success(
                planned.identity,
                output_path=str(out_dir),
                selected_model=str(rec.get("model_version") or model),
                started_at=now,
                finished_at=now,
                runtime_seconds=0.0,
                git_commit=git,
            )
        imported += 1
        report_rows.append(
            {
                "product": product,
                "forecast_origin": origin,
                "seed": "" if seed is None else seed,
                "status": "imported",
                "run_id": run_id,
                "product_history_hash": live_hash,
            }
        )

    summary = {
        "config": str(args.config),
        "engine": args.engine,
        "run_id": config.run_id,
        "imported": imported,
        "skipped_hash": skipped_hash,
        "skipped_missing": skipped_missing,
        "skipped_no_plan": skipped_no_plan,
        "dry_run": bool(args.dry_run),
    }
    out_dir = (
        REPO
        / "src"
        / "data"
        / "results"
        / "product_universe"
        / "portfolio_backfill_prep_v1"
        / f"import_{args.engine.replace('.', '')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(report_rows).to_csv(
        out_dir / "import_log.csv", index=False, lineterminator="\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
