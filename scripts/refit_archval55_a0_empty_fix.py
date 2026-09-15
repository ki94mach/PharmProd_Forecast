#!/usr/bin/env python3
"""A0-only refit of the 90 empty-prediction jobs into a new run version.

Does not mutate archval55_v21_a0_a1_fallback_v1. Uses forecast_jobs runner
primitives with an A0-only config and an exact job allow-list.

On Windows, jobs are submitted in a rolling window of ``--workers`` so we do
not pickle the full sales frame 90 times up-front (that looked like 1 worker).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path

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


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _run_pool_rolling(
    *,
    config,
    planned: list[PlannedJob],
    sales_df: pd.DataFrame,
    cfg_hash: str,
    workers_n: int,
    shutdown: ShutdownCoordinator,
) -> list[str]:
    """Keep at most ``workers_n`` futures in flight; submit the rest as slots free."""
    outcomes: list[str] = []
    pending: dict[Future, PlannedJob] = {}
    it = iter(planned)
    submitted = 0
    total = len(planned)

    with ProcessPoolExecutor(max_workers=workers_n) as pool:
        def _submit_one(job: PlannedJob) -> None:
            nonlocal submitted
            fut = pool.submit(
                _mp_run_one_job,
                config,
                job,
                sales_df,
                cfg_hash,
                "unknown",
            )
            pending[fut] = job
            submitted += 1
            _log(
                f"submit {submitted}/{total} "
                f"{job.product} @ {job.origin} seed={job.seed} "
                f"in_flight={len(pending)}"
            )

        # Prime the pool.
        for _ in range(workers_n):
            if shutdown.should_stop():
                break
            try:
                job = next(it)
            except StopIteration:
                break
            _submit_one(job)

        while pending:
            if shutdown.should_stop():
                break
            # Wait for any one to finish, then top up.
            done_set = set()
            for fut in as_completed(list(pending.keys()), timeout=None):
                done_set.add(fut)
                break
            for fut in done_set:
                job = pending.pop(fut)
                try:
                    outcome = fut.result()
                except Exception as exc:  # noqa: BLE001
                    _log(
                        f"FAILED future {job.product} @ {job.origin} "
                        f"seed={job.seed}: {type(exc).__name__}: {exc}"
                    )
                    outcome = "failed"
                outcomes.append(outcome)
                _log(
                    f"done {len(outcomes)}/{total} "
                    f"{job.product} @ {job.origin} seed={job.seed} "
                    f"outcome={outcome} in_flight={len(pending)}"
                )
                if shutdown.should_stop():
                    break
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                _submit_one(nxt)

    return outcomes


def main() -> int:
    # Unbuffered-ish when redirected from the Task Scheduler launcher.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

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
    _log(f"wrote config {cfg_path} products={len(products)} origins={origins} seeds={seeds}")
    _log(f"allow-list size={len(allow)}")

    config = load_job_config(cfg_path)
    root, cfg_hash, plan, sales_df, snap = prepare_run(config, dry_run=args.dry_run)
    _log(f"prepare_run done root={root} cartesian_remaining={plan.remaining}")

    state = JobStateStore(root)
    # Resume-safe: only schedule allow-list jobs that are not already SUCCESS.
    planned: list[PlannedJob] = []
    skipped_done = 0
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
        record = state.ensure_job(identity, git_commit="unknown")
        if record.status == "SUCCESS":
            skipped_done += 1
            continue
        if record.status == "RUNNING":
            # Reclaim stale RUNNING from a killed prior attempt.
            state.reclaim_stale_running(config.run_id)
            record = state.ensure_job(identity, git_commit="unknown")
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
    _log(
        f"planned refit jobs={len(planned)} already_success={skipped_done} "
        f"workers={int(args.workers)}"
    )
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
            outcomes = _run_pool_rolling(
                config=config,
                planned=planned,
                sales_df=sales_df,
                cfg_hash=cfg_hash,
                workers_n=workers_n,
                shutdown=shutdown,
            )
    finally:
        shutdown.uninstall()

    counts = {
        "success": sum(1 for o in outcomes if o == "success"),
        "ineligible": sum(1 for o in outcomes if o == "ineligible"),
        "failed": sum(1 for o in outcomes if o == "failed"),
        "skipped": sum(1 for o in outcomes if o == "skipped"),
        "stopped": sum(1 for o in outcomes if o == "stopped"),
        "n_outcomes": len(outcomes),
        "already_success_skipped": skipped_done,
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
    _log(json.dumps(counts, indent=2))
    _log(f"status {state.status_counts(config.run_id)}")
    ok = counts["failed"] == 0 and (
        counts["success"] + counts["ineligible"] + skipped_done >= len(allow)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
