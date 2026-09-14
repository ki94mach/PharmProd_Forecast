#!/usr/bin/env python3
"""A0-only refit of the 90 empty-prediction jobs into a new run version.

Does not mutate archval55_v21_a0_a1_fallback_v1. Uses the forecast_jobs
runner primitives with an A0-only config and an exact job allow-list.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from pkg.benchmark.backfill_runner.threads import apply_inner_thread_limits
from pkg.benchmark.backfill_runner.state import JobIdentity, JobStateStore
from pkg.benchmark.calendar import quarter_from_origin
from pkg.forecast_jobs.config import load_job_config
from pkg.forecast_jobs.engines import execute_architecture_job
from pkg.forecast_jobs.plan import PlannedJob, encode_product_key
from pkg.forecast_jobs.runner import (
    _mp_run_one_job,
    _run_one_planned_job,
    prepare_run,
)
from pkg.forecast_jobs.signals import ShutdownCoordinator
import threading

REPO = Path(__file__).resolve().parents[1]
BASE_YAML = REPO / "configs" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1.win.yaml"
INELIG = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval55_decision_v1"
    / "ineligible_jobs.csv"
)
ALLOWLIST = (
    REPO / "configs" / "forecast_jobs" / "archval55_a0_empty_fix_v1.allowlist.json"
)
NEW_RUN_ID = "archval55_a0_empty_fix_v1"


def _empty_jobs_from_csv(path: Path) -> pd.DataFrame:
    jobs = pd.read_csv(path)
    return jobs.loc[
        (jobs["engine"].astype(str).str.lower() == "a0")
        & jobs["reason_code"].astype(str).str.contains("empty", case=False, na=False)
    ].copy()


def _empty_jobs_from_allowlist(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = payload.get("jobs") or []
    return pd.DataFrame(jobs)


def _load_empty_jobs(*, ineligible_csv: Path, allowlist: Path) -> pd.DataFrame:
    if Path(allowlist).is_file():
        return _empty_jobs_from_allowlist(allowlist)
    if Path(ineligible_csv).is_file():
        return _empty_jobs_from_csv(ineligible_csv)
    raise FileNotFoundError(
        f"need allow-list JSON ({allowlist}) or ineligible CSV ({ineligible_csv})"
    )

def _write_refit_yaml(
    *,
    out_path: Path,
    products: list[str],
    origins: list[int],
    seeds: list[int],
    workers: int,
) -> Path:
    raw = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    raw["run"] = {
        "run_id": NEW_RUN_ID,
        "output_dir": "src/data/forecast_jobs",
    }
    raw["architectures"] = ["a0"]
    raw["cohort"] = {"products": products}
    raw["origins"] = origins
    raw["seeds"] = seeds
    raw["execution"]["workers"] = int(workers)
    raw["execution"]["pool"] = "processes"
    raw["execution"]["resume"] = True
    raw["execution"]["force_job"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ineligible-csv", type=Path, default=INELIG)
    p.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--config-out",
        type=Path,
        default=REPO / "configs" / "forecast_jobs" / f"{NEW_RUN_ID}.win.yaml",
    )
    args = p.parse_args()

    empty = _load_empty_jobs(
        ineligible_csv=args.ineligible_csv, allowlist=args.allowlist
    )
    if empty.empty:
        raise SystemExit("no empty A0 jobs found")
    allow = {
        (str(r.product), int(r.forecast_origin), int(r.seed))
        for r in empty.itertuples(index=False)
    }
    products = sorted({p for p, _, _ in allow})
    origins = sorted({o for _, o, _ in allow})
    seeds = sorted({s for _, _, s in allow})
    cfg_path = _write_refit_yaml(
        out_path=args.config_out,
        products=products,
        origins=origins,
        seeds=seeds,
        workers=int(args.workers),
    )
    print(f"wrote config {cfg_path} products={len(products)} origins={origins} seeds={seeds}")
    print(f"allow-list size={len(allow)}")

    config = load_job_config(cfg_path)
    root, cfg_hash, plan, sales_df, snap = prepare_run(config, dry_run=args.dry_run)
    # Filter Cartesian expansion down to the exact 90 empties.
    jobs = [
        j
        for j in plan.jobs
        if (str(j.product), int(j.origin), int(j.seed)) in allow
    ]
    # Also register any missing allow-list identities (plan may already have them).
    state = JobStateStore(root)
    planned: list[PlannedJob] = []
    for product, origin, seed in sorted(allow):
        quarter = quarter_from_origin(int(origin))
        identity = JobIdentity(
            experiment_id=config.run_id,
            engine_version="a0",
            config_hash=cfg_hash,
            quarter=quarter,
            forecast_origin=int(origin),
            product_id=encode_product_key(product, seed),
        )
        state.ensure_job(identity, git_commit="unknown")
        planned.append(
            PlannedJob(
                architecture="a0",
                origin=int(origin),
                quarter=quarter,
                product=str(product),
                seed=int(seed),
                identity=identity,
            )
        )
    print(f"planned refit jobs={len(planned)} run_root={root}")
    if args.dry_run:
        return 0

    workers_n = max(1, int(args.workers))
    apply_inner_thread_limits(workers=workers_n)
    shutdown = ShutdownCoordinator()
    shutdown.install()
    print_lock = threading.Lock()
    outcomes: list[str] = []

    try:
        if workers_n == 1:
            for job in planned:
                if shutdown.should_stop():
                    break
                outcomes.append(
                    _run_one_planned_job(
                        config=config,
                        job=job,
                        sales_df=sales_df,
                        state=state,
                        cfg_hash=cfg_hash,
                        git_commit="unknown",
                        executor=execute_architecture_job,
                        shutdown=shutdown,
                        print_lock=print_lock,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=workers_n) as pool:
                futs = [
                    pool.submit(
                        _mp_run_one_job,
                        config,
                        job,
                        sales_df,
                        cfg_hash,
                        "unknown",
                    )
                    for job in planned
                ]
                for fut in as_completed(futs):
                    outcomes.append(fut.result())
                    if shutdown.should_stop():
                        break
    finally:
        shutdown.uninstall()

    counts = {
        "success": sum(1 for o in outcomes if o == "success"),
        "ineligible": sum(1 for o in outcomes if o == "ineligible"),
        "failed": sum(1 for o in outcomes if o == "failed"),
        "skipped": sum(1 for o in outcomes if o == "skipped"),
        "stopped": sum(1 for o in outcomes if o == "stopped"),
        "n_outcomes": len(outcomes),
    }
    (root / "refit_summary.json").write_text(
        json.dumps(
            {
                "run_id": NEW_RUN_ID,
                "allow_list": len(allow),
                "counts": counts,
                "status_counts": state.status_counts(config.run_id),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))
    print("status", state.status_counts(config.run_id))
    return 0 if counts["failed"] == 0 and counts["success"] + counts["ineligible"] >= len(allow) else 1


if __name__ == "__main__":
    raise SystemExit(main())
