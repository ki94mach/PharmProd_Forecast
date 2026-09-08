"""Orchestrate resumable forecast_jobs runs (V2.1 / A0 / A1)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from pkg.benchmark.backfill_runner.state import (
    JobStateStore,
    RunLock,
    compute_config_hash,
    resolve_git_commit,
)
from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult
from pkg.forecast_jobs.config import JobConfigError, JobRunConfig
from pkg.forecast_jobs.cutoff import apply_cutoff_policy
from pkg.forecast_jobs.engines import execute_architecture_job, target_dates_for_origin
from pkg.forecast_jobs.plan import JobPlan, PlannedJob, build_job_plan
from pkg.forecast_jobs.signals import ShutdownCoordinator
from pkg.forecast_jobs.snapshot import (
    SalesSnapshotError,
    assert_snapshot_matches_recorded,
    read_input_snapshot,
    validate_sales_config,
    write_input_snapshot,
)
from pkg.forecast_jobs.store import (
    persist_job_result,
    read_manifest,
    run_dir,
    write_manifest,
    write_resolved_config,
)

JobExecutor = Callable[..., EngineJobResult]


@dataclass
class RunSummary:
    run_id: str
    config_hash: str
    run_path: Path
    planned: int
    succeeded: int
    failed: int
    skipped: int
    stopped_early: bool
    stop_reason: Optional[str]
    dry_run: bool


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scientific_config_hash(config: JobRunConfig, snapshot_sha: str) -> str:
    """Hash scientific config + frozen sales content hash."""
    payload = dict(config.scientific_payload())
    payload["sales_content_sha256"] = str(snapshot_sha).lower()
    return compute_config_hash(payload)


def prepare_run(
    config: JobRunConfig,
    *,
    dry_run: bool = False,
) -> tuple[Path, str, JobPlan, pd.DataFrame, dict[str, Any]]:
    """Validate snapshot, acquire layout, build plan (no model fits)."""
    snapshot = validate_sales_config(config.sales)
    cfg_hash = scientific_config_hash(config, snapshot.content_sha256)
    root = run_dir(config.output_dir, config.run_id)
    root.mkdir(parents=True, exist_ok=True)

    recorded = read_input_snapshot(root)
    if recorded is not None:
        assert_snapshot_matches_recorded(snapshot, recorded)
        man = read_manifest(root)
        if man is not None and man.get("config_hash") not in (None, cfg_hash):
            raise JobConfigError(
                f"config_hash mismatch for run {config.run_id!r}: "
                f"manifest={man.get('config_hash')!r} current={cfg_hash!r}"
            )
    else:
        write_input_snapshot(root, snapshot)

    write_resolved_config(root, config)
    git_commit = resolve_git_commit()
    write_manifest(
        root,
        config=config,
        config_hash=cfg_hash,
        git_commit=git_commit,
        input_snapshot=snapshot.to_dict(),
        status="planned" if dry_run else "running",
    )

    state = JobStateStore(root)
    if config.execution.resume:
        state.reclaim_stale_running(config.run_id)

    plan = build_job_plan(
        config,
        config_hash=cfg_hash,
        state=state,
        git_commit=git_commit,
        ensure_jobs=True,
    )
    sales_df = None
    if not dry_run:
        from pkg.forecast_jobs.snapshot import load_sales_frame

        sales_df = load_sales_frame(config.sales, snapshot=snapshot)
    return root, cfg_hash, plan, sales_df if sales_df is not None else pd.DataFrame(), snapshot.to_dict()


def _build_request(
    config: JobRunConfig,
    job: PlannedJob,
    sales_df: pd.DataFrame,
) -> tuple[EngineJobRequest, pd.DataFrame]:
    truncated = apply_cutoff_policy(
        sales_df,
        job.origin,
        policy_name=config.cutoff_policy.name,
    )
    product_sales = truncated.loc[
        truncated["product"].astype(str) == str(job.product)
    ].copy()
    targets = target_dates_for_origin(job.origin, config.horizon)
    request = EngineJobRequest(
        engine=job.architecture,
        product=job.product,
        quarter=job.quarter,
        forecast_origin=job.origin,
        horizon=int(config.horizon),
        target_dates=targets,
        training_sales=product_sales,
        meta={"seed": job.seed, "cutoff_policy": config.cutoff_policy.name},
    )
    return request, truncated


def run_forecast_jobs(
    config: JobRunConfig,
    *,
    dry_run: bool = False,
    execute: bool = False,
    job_executor: Optional[JobExecutor] = None,
    install_signals: bool = True,
) -> RunSummary:
    """Plan and optionally execute jobs for one run_id.

    ``dry_run`` validates snapshot + writes plan artifacts without fitting.
    ``execute`` runs pending jobs. Passing neither is treated as dry_run.
    """
    if not dry_run and not execute:
        dry_run = True

    root, cfg_hash, plan, sales_df, snap_dict = prepare_run(config, dry_run=dry_run)
    git_commit = resolve_git_commit()

    if dry_run and not execute:
        write_manifest(
            root,
            config=config,
            config_hash=cfg_hash,
            git_commit=git_commit,
            input_snapshot=snap_dict,
            status="dry_run",
        )
        return RunSummary(
            run_id=config.run_id,
            config_hash=cfg_hash,
            run_path=root,
            planned=plan.remaining,
            succeeded=0,
            failed=0,
            skipped=plan.already_completed,
            stopped_early=False,
            stop_reason=None,
            dry_run=True,
        )

    lock = RunLock(root)
    shutdown = ShutdownCoordinator()
    succeeded = 0
    failed = 0
    stopped_early = False
    stop_reason: Optional[str] = None
    executor = job_executor or (
        lambda **kwargs: execute_architecture_job(**kwargs)
    )

    try:
        lock.acquire()
        if install_signals:
            shutdown.install()
        state = JobStateStore(root)
        if config.execution.resume:
            state.reclaim_stale_running(config.run_id)
        # Rebuild plan after reclaim so newly pending jobs are included.
        plan = build_job_plan(
            config,
            config_hash=cfg_hash,
            state=state,
            git_commit=git_commit,
            ensure_jobs=True,
        )

        for job in plan.jobs:
            if shutdown.should_stop():
                stopped_early = True
                stop_reason = shutdown.reason
                break

            claimed = state.try_claim_job(job.identity, git_commit=git_commit)
            if claimed is None:
                continue

            start_s = _utc_stamp()
            try:
                request, sales_cut = _build_request(config, job, sales_df)
                result = executor(
                    architecture=job.architecture,
                    request=request,
                    seed=job.seed,
                    sales_for_neural=sales_cut,
                )
            except Exception as exc:  # noqa: BLE001
                result = EngineJobResult(
                    success=False,
                    product=job.product,
                    quarter=job.quarter,
                    forecast_origin=job.origin,
                    error_message=str(exc),
                    error_type=type(exc).__name__,
                )

            end_s = _utc_stamp()
            # Approximate duration from ISO stamps is unnecessary; use 0.0 when unknown.
            runtime_seconds = 0.0
            try:
                t0 = datetime.strptime(start_s, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                t1 = datetime.strptime(end_s, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                runtime_seconds = max(0.0, (t1 - t0).total_seconds())
            except ValueError:
                pass
            log_payload = {
                "job_id": job.identity.job_id,
                "run_id": config.run_id,
                "architecture": job.architecture,
                "config_hash": cfg_hash,
                "quarter": job.quarter,
                "forecast_origin": job.origin,
                "product": job.product,
                "seed": job.seed,
                "started_at": start_s,
                "finished_at": end_s,
                "runtime_seconds": runtime_seconds,
                "success": bool(result.success),
                "selected_model": result.selected_model,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "git_commit": git_commit,
            }
            out_dir = persist_job_result(
                root, job.identity, result, log_payload=log_payload
            )
            if result.success:
                state.mark_success(
                    job.identity,
                    output_path=str(out_dir),
                    selected_model=result.selected_model,
                    started_at=start_s,
                    finished_at=end_s,
                    runtime_seconds=runtime_seconds,
                    git_commit=git_commit,
                )
                succeeded += 1
                print(
                    f"[SUCCESS] arch={job.architecture} origin={job.origin} "
                    f"product={job.product} seed={job.seed} model={result.selected_model}"
                )
            else:
                state.mark_failed(
                    job.identity,
                    error_type=result.error_type or "Error",
                    error_message=result.error_message or "unknown",
                    started_at=start_s,
                    finished_at=end_s,
                    runtime_seconds=runtime_seconds,
                    git_commit=git_commit,
                    output_path=str(out_dir),
                )
                failed += 1
                print(
                    f"[FAILED] arch={job.architecture} origin={job.origin} "
                    f"product={job.product} seed={job.seed} "
                    f"err={result.error_message}"
                )

        status = "stopped" if stopped_early else "complete"
        write_manifest(
            root,
            config=config,
            config_hash=cfg_hash,
            git_commit=git_commit,
            input_snapshot=snap_dict,
            status=status,
        )
    finally:
        if install_signals:
            shutdown.uninstall()
        try:
            lock.release()
        except Exception:
            pass

    return RunSummary(
        run_id=config.run_id,
        config_hash=cfg_hash,
        run_path=root,
        planned=plan.total_jobs,
        succeeded=succeeded,
        failed=failed,
        skipped=plan.already_completed,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        dry_run=False,
    )


def print_status(config: JobRunConfig) -> int:
    root = run_dir(config.output_dir, config.run_id)
    db = root / "state.sqlite"
    if not db.exists():
        print(f"No checkpoint DB at {db}")
        return 1
    state = JobStateStore(root)
    counts = state.status_counts(config.run_id)
    man = read_manifest(root) or {}
    print(f"run_id={config.run_id}")
    print(f"run_dir={root}")
    print(f"config_hash={man.get('config_hash')}")
    print(f"status={man.get('status')}")
    print(
        " ".join(f"{k}={counts.get(k, 0)}" for k in ("PENDING", "RUNNING", "SUCCESS", "FAILED"))
    )
    return 0
