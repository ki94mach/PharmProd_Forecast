"""Orchestrate resumable forecast_jobs runs (V2.1 / A0 / A1)."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from pkg.benchmark.backfill_runner.threads import apply_inner_thread_limits
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

# Per-job wall timeout for one A0/A1/V2.1 fit (seconds).
_DEFAULT_JOB_TIMEOUT_S = 60 * 45


class _NeverStop:
    """Stand-in shutdown flag for process-pool workers (signals stay in parent)."""

    def should_stop(self) -> bool:
        return False


def _mp_run_one_job(
    config: JobRunConfig,
    job: PlannedJob,
    sales_df: pd.DataFrame,
    cfg_hash: str,
    git_commit: Optional[str],
) -> str:
    """Picklable process-pool entrypoint (module-level for Windows spawn)."""
    # Cap nested BLAS/OMP inside each worker process.
    apply_inner_thread_limits(workers=2)
    state = JobStateStore(run_dir(config.output_dir, config.run_id))
    return _run_one_planned_job(
        config=config,
        job=job,
        sales_df=sales_df,
        state=state,
        cfg_hash=cfg_hash,
        git_commit=git_commit,
        executor=execute_architecture_job,
        shutdown=_NeverStop(),  # type: ignore[arg-type]
        print_lock=threading.Lock(),
    )


def _mp_run_one_job_exit(
    result_queue: "mp.Queue[tuple[str, str]]",
    config: JobRunConfig,
    job: PlannedJob,
    sales_df: pd.DataFrame,
    cfg_hash: str,
    git_commit: Optional[str],
) -> None:
    """Child process target: run one job, push outcome, hard-exit.

    Keras/TF atexit handlers often hang on Windows after a successful fit.
    ``os._exit`` skips them so the parent can start the next wave.
    """
    outcome = "failed"
    try:
        outcome = _mp_run_one_job(config, job, sales_df, cfg_hash, git_commit)
        result_queue.put(("ok", outcome))
    except Exception as exc:  # noqa: BLE001
        try:
            result_queue.put(("err", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        outcome = "failed"
    finally:
        # noqa: S110 — intentional hard exit to avoid TF teardown deadlock
        os._exit(0 if outcome in {"success", "ineligible", "skipped"} else 1)


def _run_jobs_in_process_waves(
    *,
    jobs: list[PlannedJob],
    config: JobRunConfig,
    sales_df: pd.DataFrame,
    cfg_hash: str,
    git_commit: Optional[str],
    workers_n: int,
    shutdown: ShutdownCoordinator,
    job_timeout_s: float = _DEFAULT_JOB_TIMEOUT_S,
) -> list[str]:
    """Run jobs in waves of fresh spawn processes (one TF fit per process)."""
    ctx = mp.get_context("spawn")
    remaining = list(jobs)
    outcomes: list[str] = []
    print(
        f"process waves: jobs={len(remaining)} workers={workers_n} "
        f"(spawn+os._exit per job, timeout_s={int(job_timeout_s)})",
        flush=True,
    )
    while remaining and not shutdown.should_stop():
        batch = remaining[:workers_n]
        remaining = remaining[workers_n:]
        print(
            f"wave size={len(batch)} remaining_after={len(remaining)}",
            flush=True,
        )
        procs: list[tuple[Any, Any, PlannedJob]] = []
        for job in batch:
            q: mp.Queue = ctx.Queue()
            proc = ctx.Process(
                target=_mp_run_one_job_exit,
                args=(q, config, job, sales_df, cfg_hash, git_commit),
            )
            proc.start()
            procs.append((proc, q, job))

        for proc, q, job in procs:
            proc.join(timeout=float(job_timeout_s))
            if proc.is_alive():
                print(
                    f"timeout killing {job.product} @ {job.origin} "
                    f"seed={job.seed} pid={proc.pid}",
                    flush=True,
                )
                proc.terminate()
                proc.join(timeout=30)
                if proc.is_alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.join(timeout=10)
                outcomes.append("failed")
                continue
            try:
                status, payload = q.get_nowait()
            except Exception:
                outcomes.append("failed")
                print(
                    f"no result queue for {job.product} @ {job.origin} "
                    f"seed={job.seed} exitcode={proc.exitcode}",
                    flush=True,
                )
                continue
            if status == "ok":
                outcomes.append(str(payload))
            else:
                outcomes.append("failed")
                print(
                    f"worker error {job.product} @ {job.origin} "
                    f"seed={job.seed}: {payload}",
                    flush=True,
                )
            if shutdown.should_stop():
                break
        if shutdown.should_stop():
            for leftover, _q, _job in procs:
                if leftover.is_alive():
                    leftover.terminate()
                    leftover.join(timeout=10)
            break
    return outcomes


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
    ineligible: int = 0


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
        recorded_hash = None if man is None else man.get("config_hash")
        if recorded_hash not in (None, cfg_hash):
            # Absolute sales.path is machine-local and participates in the
            # scientific hash. Across laptop→VPS copies the bytes match but
            # the path string (and thus config_hash) differs. Keep the
            # recorded hash so existing SUCCESS/INELIGIBLE job_ids stay valid.
            same_bytes = str(recorded.get("content_sha256", "")).lower() == str(
                snapshot.content_sha256
            ).lower()
            if same_bytes:
                cfg_hash = str(recorded_hash)
            else:
                raise JobConfigError(
                    f"config_hash mismatch for run {config.run_id!r}: "
                    f"manifest={recorded_hash!r} current={cfg_hash!r}"
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
        meta={
            "seed": job.seed,
            "cutoff_policy": config.cutoff_policy.name,
            "fallback_policy": config.fallback.policy,
            "fallback_version": config.fallback.version,
        },
    )
    return request, truncated


def _run_one_planned_job(
    *,
    config: JobRunConfig,
    job: PlannedJob,
    sales_df: pd.DataFrame,
    state: JobStateStore,
    cfg_hash: str,
    git_commit: Optional[str],
    executor: JobExecutor,
    shutdown: ShutdownCoordinator,
    print_lock: threading.Lock,
) -> str:
    """Claim + fit one job. Returns success|failed|skipped|stopped."""
    if shutdown.should_stop():
        return "stopped"

    claimed = state.try_claim_job(job.identity, git_commit=git_commit)
    if claimed is None:
        return "skipped"

    start_s = _utc_stamp()
    try:
        request, sales_cut = _build_request(config, job, sales_df)
        # V2.1 trains on production-cut rows (request.training_sales).
        # A0/A1 still enforce train-through-last_complete inside
        # prepare_monthly_series / ForecastWindow, but need the fuller
        # panel so discover_origins can see matured target actuals
        # (production cutoff alone ends at O-2 and drops the origin).
        neural_sales = (
            sales_df
            if str(job.architecture).lower() in {"a0", "a1"}
            else sales_cut
        )
        result = executor(
            architecture=job.architecture,
            request=request,
            seed=job.seed,
            sales_for_neural=neural_sales,
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
        "ineligible": bool(getattr(result, "ineligible", False)),
        "selected_model": result.selected_model,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "git_commit": git_commit,
    }
    out_dir = persist_job_result(
        run_dir(config.output_dir, config.run_id),
        job.identity,
        result,
        log_payload=log_payload,
    )
    if getattr(result, "ineligible", False) and result.success:
        state.mark_ineligible(
            job.identity,
            output_path=str(out_dir),
            selected_model=result.selected_model,
            error_type=result.error_type,
            error_message=result.error_message,
            started_at=start_s,
            finished_at=end_s,
            runtime_seconds=runtime_seconds,
            git_commit=git_commit,
        )
        reason = ""
        if result.extras:
            reason = str(result.extras.get("fallback_reason") or "")
        with print_lock:
            print(
                f"[INELIGIBLE] arch={job.architecture} origin={job.origin} "
                f"product={job.product} seed={job.seed} "
                f"fallback={result.selected_model} reason={reason or result.error_message}"
            )
        return "ineligible"

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
        with print_lock:
            print(
                f"[SUCCESS] arch={job.architecture} origin={job.origin} "
                f"product={job.product} seed={job.seed} model={result.selected_model}"
            )
        return "success"

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
    with print_lock:
        print(
            f"[FAILED] arch={job.architecture} origin={job.origin} "
            f"product={job.product} seed={job.seed} "
            f"err={result.error_message}"
        )
    return "failed"


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

    Parallelism is job-level only (``execution.workers``). Use
    ``execution.pool: processes`` (default) for true multi-core on large hosts;
    ``threads`` keeps a single process (GIL-limited, lower RAM).
    Nested BLAS/OMP threads stay capped when workers > 1.
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
            ineligible=0,
        )

    workers_n = max(1, int(config.execution.workers))
    pool_mode = str(getattr(config.execution, "pool", "processes") or "processes").lower()
    # Custom executors (unit tests) cannot cross process boundaries.
    if job_executor is not None and pool_mode == "processes" and workers_n > 1:
        pool_mode = "threads"
    thread_config = apply_inner_thread_limits(workers=workers_n)
    print(f"workers={workers_n} pool={pool_mode} thread_config={thread_config}")

    lock = RunLock(root)
    shutdown = ShutdownCoordinator()
    succeeded = 0
    failed = 0
    ineligible = 0
    stopped_early = False
    stop_reason: Optional[str] = None
    executor = job_executor or (
        lambda **kwargs: execute_architecture_job(**kwargs)
    )
    print_lock = threading.Lock()

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

        def _one_thread(job: PlannedJob) -> str:
            return _run_one_planned_job(
                config=config,
                job=job,
                sales_df=sales_df,
                state=state,
                cfg_hash=cfg_hash,
                git_commit=git_commit,
                executor=executor,
                shutdown=shutdown,
                print_lock=print_lock,
            )

        outcomes: list[str] = []
        if workers_n == 1:
            for job in plan.jobs:
                if shutdown.should_stop():
                    stopped_early = True
                    stop_reason = shutdown.reason
                    break
                outcomes.append(_one_thread(job))
        elif pool_mode == "processes":
            # Keras/TF hangs process-pool workers on Windows after a fit
            # (atexit / session teardown). Use spawn waves + os._exit.
            outcomes.extend(
                _run_jobs_in_process_waves(
                    jobs=list(plan.jobs),
                    config=config,
                    sales_df=sales_df,
                    cfg_hash=cfg_hash,
                    git_commit=git_commit,
                    workers_n=workers_n,
                    shutdown=shutdown,
                )
            )
            if shutdown.should_stop():
                stopped_early = True
                stop_reason = shutdown.reason or "shutdown"
        else:
            with ThreadPoolExecutor(max_workers=workers_n) as pool:
                futures = []
                for job in plan.jobs:
                    if shutdown.should_stop():
                        stopped_early = True
                        stop_reason = shutdown.reason
                        break
                    futures.append(pool.submit(_one_thread, job))
                for fut in as_completed(futures):
                    outcomes.append(fut.result())
                    if shutdown.should_stop():
                        stopped_early = True
                        stop_reason = shutdown.reason

        succeeded = sum(1 for o in outcomes if o == "success")
        ineligible = sum(1 for o in outcomes if o == "ineligible")
        failed = sum(1 for o in outcomes if o == "failed")
        if any(o == "stopped" for o in outcomes):
            stopped_early = True
            stop_reason = stop_reason or shutdown.reason or "shutdown"

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
        ineligible=ineligible,
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
        " ".join(
            f"{k}={counts.get(k, 0)}"
            for k in ("PENDING", "RUNNING", "SUCCESS", "INELIGIBLE", "FAILED")
        )
    )
    return 0
