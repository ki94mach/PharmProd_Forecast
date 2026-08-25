"""Historical forecast backfill orchestrator (no model logic)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from pkg.benchmark.backfill_runner.state import (
    JOB_FAILED,
    JOB_PENDING,
    JOB_RUNNING,
    JOB_SUCCESS,
    JobIdentity,
    JobStateStore,
    RunLock,
    RunLockError,
    compute_config_hash,
    make_experiment_id,
    resolve_git_commit,
    should_run_job,
)
from pkg.benchmark.backfill_runner.store import (
    BackfillStore,
    atomic_write_json,
    default_backfill_root,
)
from pkg.benchmark.backfill_runner.threads import apply_inner_thread_limits
from pkg.benchmark.backfill_runner.types import (
    ForecastEngine,
    request_from_vintage,
    target_dates_for_origin,
)
from pkg.benchmark.calendar import iter_shamsi_quarters
from pkg.benchmark.universes import load_universe_product_names
from pkg.benchmark.vintages import VintageSpec, load_vintage_manifest_by_name


@dataclass
class BackfillPlan:
    experiment_id: str
    engine: str
    config_hash: str
    vintages_requested: list[str]
    vintages_eligible: list[str]
    products: list[str]
    jobs: list[tuple[VintageSpec, str, JobIdentity]]
    already_completed: int
    remaining: int
    total_jobs: int
    status_counts: dict[str, int] = field(default_factory=dict)
    workers: int = 1

    def report_lines(self) -> list[str]:
        counts = self.status_counts
        return [
            f"experiment_id={self.experiment_id}",
            f"engine={self.engine}",
            f"config_hash={self.config_hash}",
            f"workers={self.workers}",
            f"vintages_requested={len(self.vintages_requested)}: "
            f"{', '.join(self.vintages_requested)}",
            f"vintages_eligible_now={len(self.vintages_eligible)}: "
            f"{', '.join(self.vintages_eligible)}",
            f"products={len(self.products)}",
            f"total_sku_vintage_jobs={self.total_jobs}",
            f"expected_job_count={self.remaining}",
            f"already_completed={self.already_completed}",
            f"remaining={self.remaining}",
            (
                "status_counts: "
                f"PENDING={counts.get(JOB_PENDING, 0)} "
                f"RUNNING={counts.get(JOB_RUNNING, 0)} "
                f"SUCCESS={counts.get(JOB_SUCCESS, 0)} "
                f"FAILED={counts.get(JOB_FAILED, 0)}"
            ),
        ]


@dataclass
class BackfillRunSummary:
    plan: BackfillPlan
    n_success: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    n_reclaimed_stale: int = 0
    dry_run: bool = False
    workers: int = 1
    thread_config: dict[str, str] = field(default_factory=dict)
    runtime_seconds: Optional[float] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class _JobOutcome:
    identity: JobIdentity
    kind: str  # success | failed | skipped | not_claimed
    message: str = ""
    selected_model: Optional[str] = None
    duration: Optional[float] = None
    order_key: tuple[str, str] = ("", "")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enforce_historical_cutoff(
    sales: pd.DataFrame,
    forecast_origin: int,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Return only rows with ``date < forecast_origin`` (Shamsi YYYYMM)."""
    if sales is None or sales.empty:
        return sales.iloc[0:0].copy() if sales is not None else pd.DataFrame()
    work = sales.copy()
    work[date_col] = work[date_col].map(lambda x: int(x))
    origin = int(forecast_origin)
    out = work.loc[work[date_col] < origin].copy()
    if (out[date_col] >= origin).any():
        raise RuntimeError("internal error: cutoff leaked origin-or-later rows")
    return out


def filter_vintages(
    specs: Sequence[VintageSpec],
    *,
    quarter: Optional[str] = None,
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    include_future: bool = False,
) -> list[VintageSpec]:
    out = list(specs)
    if quarter:
        out = [s for s in out if s.quarter == quarter]
    elif quarter_from or quarter_to:
        start = quarter_from or specs[0].quarter
        end = quarter_to or specs[-1].quarter
        allowed = set(iter_shamsi_quarters(start, end))
        out = [s for s in out if s.quarter in allowed]
    if not include_future:
        out = [s for s in out if s.status != "future_origin"]
    return out


def experiment_dir(output_root: Path, experiment_id: str) -> Path:
    return Path(output_root) / experiment_id


def build_config_payload(
    *,
    engine: str,
    vintage_name: str,
    universe_name: str,
    horizon: int = 15,
) -> dict[str, Any]:
    # Worker count is intentionally excluded — reproducibility of forecasts
    # must not depend on parallelism settings.
    return {
        "engine_version": str(engine),
        "vintage_manifest": str(vintage_name),
        "universe_manifest": str(universe_name),
        "horizon": int(horizon),
        "training_cutoff_rule": "date < forecast_origin",
    }


def build_backfill_plan(
    *,
    engine: str,
    vintage_name: str,
    universe_name: str,
    state: JobStateStore,
    experiment_id: str,
    config_hash: str,
    git_commit: str,
    quarter: Optional[str] = None,
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    products: Optional[Sequence[str]] = None,
    product: Optional[str] = None,
    include_future: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    force_job: bool = False,
    ensure_jobs: bool = True,
    workers: int = 1,
) -> BackfillPlan:
    all_specs = load_vintage_manifest_by_name(vintage_name, validate=True)
    requested = [s.quarter for s in all_specs]
    eligible_specs = filter_vintages(
        all_specs,
        quarter=quarter,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
        include_future=include_future,
    )
    product_list = load_universe_product_names(universe_name, validate=True)
    if product:
        products = [product]
    if products:
        wanted = {str(p) for p in products}
        product_list = [p for p in product_list if p in wanted]
        missing = wanted - set(product_list)
        if missing:
            raise ValueError(
                f"products not in universe {universe_name!r}: {sorted(missing)[:10]}"
            )

    jobs: list[tuple[VintageSpec, str, JobIdentity]] = []
    n_done = 0
    for vintage in eligible_specs:
        for product_id in product_list:
            identity = JobIdentity(
                experiment_id=experiment_id,
                engine_version=str(engine),
                config_hash=config_hash,
                quarter=vintage.quarter,
                forecast_origin=int(vintage.forecast_origin),
                product_id=str(product_id),
            )
            if ensure_jobs:
                record = state.ensure_job(identity, git_commit=git_commit)
                if force_job and record.status == JOB_SUCCESS:
                    record = state.reset_for_force(identity, git_commit=git_commit)
                if should_run_job(
                    record,
                    resume=resume,
                    retry_failed=retry_failed,
                    force_job=force_job,
                ):
                    jobs.append((vintage, product_id, identity))
                elif record.status == JOB_SUCCESS:
                    n_done += 1
            else:
                jobs.append((vintage, product_id, identity))

    # Deterministic queue order: vintage order × product universe order.
    total = len(eligible_specs) * len(product_list)
    counts = state.status_counts(experiment_id) if ensure_jobs else {}
    return BackfillPlan(
        experiment_id=experiment_id,
        engine=str(engine),
        config_hash=config_hash,
        vintages_requested=requested,
        vintages_eligible=[s.quarter for s in eligible_specs],
        products=list(product_list),
        jobs=jobs,
        already_completed=n_done,
        remaining=len(jobs),
        total_jobs=total,
        status_counts=counts,
        workers=max(1, int(workers)),
    )


def load_default_sales() -> pd.DataFrame:
    """Prefer frozen benchmark sales; fall back to warehouse query."""
    from pkg.benchmark.config import default_benchmark_root

    frozen = default_benchmark_root() / "raw" / "sales.parquet"
    if frozen.exists():
        df = pd.read_parquet(frozen)
        df["date"] = pd.to_numeric(df["date"], errors="coerce").astype(int)
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        df["product"] = df["product"].astype(str)
        return df
    from pkg.db.query.sales import load_sales_data

    df = load_sales_data()
    df["date"] = pd.to_numeric(df["date"], errors="coerce").astype(int)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["product"] = df["product"].astype(str)
    return df


def print_status(
    *,
    output_root: Path,
    experiment_id: str,
    quarter: Optional[str] = None,
    product: Optional[str] = None,
) -> int:
    exp_dir = experiment_dir(output_root, experiment_id)
    if not (exp_dir / "backfill.sqlite").exists():
        print(f"No checkpoint DB at {exp_dir / 'backfill.sqlite'}")
        return 1
    state = JobStateStore(exp_dir)
    counts = state.status_counts(experiment_id)
    print(f"experiment_id={experiment_id}")
    print(
        f"PENDING={counts.get(JOB_PENDING, 0)} "
        f"RUNNING={counts.get(JOB_RUNNING, 0)} "
        f"SUCCESS={counts.get(JOB_SUCCESS, 0)} "
        f"FAILED={counts.get(JOB_FAILED, 0)}"
    )
    jobs = state.list_jobs(
        experiment_id,
        quarter=quarter,
        product_id=product,
    )
    for job in jobs[:100]:
        print(
            f"  [{job.status}] {job.identity.quarter} "
            f"origin={job.identity.forecast_origin} "
            f"product={job.identity.product_id} "
            f"attempts={job.attempt_count} "
            f"err={job.error_message}"
        )
    if len(jobs) > 100:
        print(f"  ... ({len(jobs) - 100} more)")
    return 0


def _execute_claimed_job(
    *,
    engine: ForecastEngine,
    store: BackfillStore,
    state: JobStateStore,
    identity: JobIdentity,
    vintage: VintageSpec,
    product_id: str,
    sales_df: pd.DataFrame,
    meta_by_product: dict[str, dict[str, Any]],
    git_commit: str,
    print_lock: threading.Lock,
) -> _JobOutcome:
    """Run one already-claimed job. Failures stay isolated to this SKU-vintage."""
    start = _utc_now()
    start_s = _utc_stamp(start)
    order_key = (identity.quarter, identity.product_id)
    try:
        train = enforce_historical_cutoff(sales_df, vintage.forecast_origin)
        train = train.loc[train["product"].astype(str) == str(product_id)].copy()
        request = request_from_vintage(
            engine=engine.name,
            product=product_id,
            vintage=vintage,
            training_sales=train,
            meta=meta_by_product.get(product_id, {}),
        )
        if not request.training_sales.empty:
            assert (
                request.training_sales["date"].astype(int).max()
                < int(request.forecast_origin)
            )
        result = engine.forecast_product(request)
        end = _utc_now()
        end_s = _utc_stamp(end)
        duration = (end - start).total_seconds()
        log_payload = {
            "job_id": identity.job_id,
            "experiment_id": identity.experiment_id,
            "engine_version": identity.engine_version,
            "config_hash": identity.config_hash,
            "quarter": identity.quarter,
            "forecast_origin": identity.forecast_origin,
            "product_id": identity.product_id,
            "started_at": start_s,
            "finished_at": end_s,
            "runtime_seconds": duration,
            "success": bool(result.success),
            "selected_model": result.selected_model,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "git_commit": git_commit,
        }
        line = (
            f"[{'SUCCESS' if result.success else 'FAILED'}] "
            f"engine={identity.engine_version} qrt={identity.quarter} "
            f"origin={identity.forecast_origin} product={identity.product_id} "
            f"duration={duration:.2f}s model={result.selected_model} "
            f"err={result.error_message}"
        )
        with print_lock:
            print(line)
        if result.success:
            out_dir = store.persist_success(identity, result, log_payload)
            state.mark_success(
                identity,
                output_path=str(out_dir),
                selected_model=result.selected_model,
                started_at=start_s,
                finished_at=end_s,
                runtime_seconds=duration,
                git_commit=git_commit,
            )
            return _JobOutcome(
                identity=identity,
                kind="success",
                selected_model=result.selected_model,
                duration=duration,
                order_key=order_key,
            )
        out_dir = store.persist_failure(identity, log_payload)
        state.mark_failed(
            identity,
            error_type=result.error_type,
            error_message=result.error_message,
            started_at=start_s,
            finished_at=end_s,
            runtime_seconds=duration,
            git_commit=git_commit,
            output_path=str(out_dir),
        )
        return _JobOutcome(
            identity=identity,
            kind="failed",
            message=str(result.error_message or ""),
            duration=duration,
            order_key=order_key,
        )
    except Exception as exc:  # noqa: BLE001 — never kill the whole backfill
        end = _utc_now()
        end_s = _utc_stamp(end)
        duration = (end - start).total_seconds()
        log_payload = {
            "job_id": identity.job_id,
            "experiment_id": identity.experiment_id,
            "engine_version": identity.engine_version,
            "config_hash": identity.config_hash,
            "quarter": identity.quarter,
            "forecast_origin": identity.forecast_origin,
            "product_id": identity.product_id,
            "started_at": start_s,
            "finished_at": end_s,
            "runtime_seconds": duration,
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "git_commit": git_commit,
        }
        with print_lock:
            print(
                f"[FAILED] engine={identity.engine_version} "
                f"qrt={identity.quarter} product={identity.product_id} "
                f"err={exc}"
            )
        try:
            out_dir = store.persist_failure(identity, log_payload)
            state.mark_failed(
                identity,
                error_type=type(exc).__name__,
                error_message=str(exc),
                started_at=start_s,
                finished_at=end_s,
                runtime_seconds=duration,
                git_commit=git_commit,
                output_path=str(out_dir),
            )
        except Exception as persist_exc:  # noqa: BLE001
            return _JobOutcome(
                identity=identity,
                kind="failed",
                message=f"persist failed: {persist_exc}; original: {exc}",
                duration=duration,
                order_key=order_key,
            )
        return _JobOutcome(
            identity=identity,
            kind="failed",
            message=str(exc),
            duration=duration,
            order_key=order_key,
        )


def _process_plan_job(
    *,
    vintage: VintageSpec,
    product_id: str,
    identity: JobIdentity,
    engine: ForecastEngine,
    store: BackfillStore,
    state: JobStateStore,
    sales_df: pd.DataFrame,
    meta_by_product: dict[str, dict[str, Any]],
    git_commit: str,
    force_job: bool,
    print_lock: threading.Lock,
) -> _JobOutcome:
    order_key = (identity.quarter, identity.product_id)
    record = state.get_job(identity.job_id)
    if record is None:
        record = state.ensure_job(identity, git_commit=git_commit)

    if record.status == JOB_SUCCESS and not force_job:
        return _JobOutcome(identity=identity, kind="skipped", order_key=order_key)

    if force_job:
        if store.has_complete_artifacts(identity):
            store.clear_artifacts(identity)
        if record.status != JOB_PENDING:
            state.reset_for_force(identity, git_commit=git_commit)

    claimed = state.try_claim_job(identity, git_commit=git_commit)
    if claimed is None:
        latest = state.get_job(identity.job_id)
        if latest is not None and latest.status == JOB_SUCCESS and not force_job:
            return _JobOutcome(identity=identity, kind="skipped", order_key=order_key)
        return _JobOutcome(
            identity=identity,
            kind="not_claimed",
            message="job already claimed or not runnable",
            order_key=order_key,
        )

    return _execute_claimed_job(
        engine=engine,
        store=store,
        state=state,
        identity=identity,
        vintage=vintage,
        product_id=product_id,
        sales_df=sales_df,
        meta_by_product=meta_by_product,
        git_commit=git_commit,
        print_lock=print_lock,
    )


def run_backfill(
    *,
    engine: ForecastEngine,
    vintage_name: str,
    universe_name: str = "mvp_products",
    output_root: Optional[Path] = None,
    sales: Optional[pd.DataFrame] = None,
    experiment_id: Optional[str] = None,
    quarter: Optional[str] = None,
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    products: Optional[Sequence[str]] = None,
    product: Optional[str] = None,
    resume: bool = True,
    retry_failed: bool = False,
    force_job: bool = False,
    dry_run: bool = False,
    include_future: bool = False,
    product_meta: Optional[dict[str, dict[str, Any]]] = None,
    acquire_lock: bool = True,
    workers: int = 1,
) -> BackfillRunSummary:
    """Orchestrate historical backfill with durable SQLite checkpoints.

    Parallelism is only at the SKU-vintage job level (``workers``). Model
    selection inside an engine must not spawn additional job pools.
    """
    workers_n = max(1, int(workers))
    thread_config = apply_inner_thread_limits(workers=workers_n)

    root = Path(output_root) if output_root is not None else default_backfill_root()
    exp_id = experiment_id or make_experiment_id(vintage_name, universe_name, engine.name)
    exp_dir = experiment_dir(root, exp_id)
    config = build_config_payload(
        engine=engine.name,
        vintage_name=vintage_name,
        universe_name=universe_name,
    )
    config_hash = compute_config_hash(config)
    git_commit = resolve_git_commit()

    lock: Optional[RunLock] = None
    if acquire_lock and not dry_run:
        lock = RunLock(exp_dir)
        lock.acquire()

    run_started = _utc_now()
    try:
        state = JobStateStore(exp_dir)
        state.upsert_experiment(
            experiment_id=exp_id,
            vintage_manifest=vintage_name,
            universe_manifest=universe_name,
            engine_version=engine.name,
            config=config,
            config_hash=config_hash,
            git_commit=git_commit,
        )

        n_reclaimed = 0
        if resume or retry_failed:
            n_reclaimed = state.reclaim_stale_running(
                exp_id, quarter=quarter, product_id=product
            )

        store = BackfillStore(exp_dir, engine.name)
        plan = build_backfill_plan(
            engine=engine.name,
            vintage_name=vintage_name,
            universe_name=universe_name,
            state=state,
            experiment_id=exp_id,
            config_hash=config_hash,
            git_commit=git_commit,
            quarter=quarter,
            quarter_from=quarter_from,
            quarter_to=quarter_to,
            products=products,
            product=product,
            include_future=include_future,
            resume=resume,
            retry_failed=retry_failed,
            force_job=force_job,
            ensure_jobs=not dry_run,
            workers=workers_n,
        )
        plan.status_counts = state.status_counts(exp_id)

        for line in plan.report_lines():
            print(line)
        print(f"thread_config={thread_config}")
        if n_reclaimed:
            print(f"reclaimed_stale_running={n_reclaimed}")

        if dry_run:
            print(
                f"dry-run: expected_job_count={plan.remaining} "
                f"(no model fitting; workers would be {workers_n})"
            )
            for vintage, product_id, identity in plan.jobs[:50]:
                targets = target_dates_for_origin(
                    vintage.forecast_origin, vintage.horizon
                )
                print(
                    f"  {vintage.quarter} origin={vintage.forecast_origin} "
                    f"product={product_id} job_id={identity.job_id[:8]}… "
                    f"targets={targets[0]}..{targets[-1]}"
                )
            if len(plan.jobs) > 50:
                print(f"  ... ({len(plan.jobs) - 50} more jobs)")
            return BackfillRunSummary(
                plan=plan,
                dry_run=True,
                n_reclaimed_stale=n_reclaimed,
                workers=workers_n,
                thread_config=thread_config,
                runtime_seconds=0.0,
            )

        sales_df = sales if sales is not None else load_default_sales()
        summary = BackfillRunSummary(
            plan=plan,
            dry_run=False,
            n_reclaimed_stale=n_reclaimed,
            workers=workers_n,
            thread_config=thread_config,
        )
        meta_by_product = product_meta or {}
        print_lock = threading.Lock()

        def _one(item: tuple[VintageSpec, str, JobIdentity]) -> _JobOutcome:
            vintage, product_id, identity = item
            return _process_plan_job(
                vintage=vintage,
                product_id=product_id,
                identity=identity,
                engine=engine,
                store=store,
                state=state,
                sales_df=sales_df,
                meta_by_product=meta_by_product,
                git_commit=git_commit,
                force_job=force_job,
                print_lock=print_lock,
            )

        outcomes: list[_JobOutcome] = []
        if workers_n == 1:
            for item in plan.jobs:
                outcomes.append(_one(item))
        else:
            with ThreadPoolExecutor(max_workers=workers_n) as pool:
                futures = [pool.submit(_one, item) for item in plan.jobs]
                for fut in as_completed(futures):
                    outcomes.append(fut.result())

        # Deterministic reporting order: (quarter, product_id).
        outcomes.sort(key=lambda o: o.order_key)
        for outcome in outcomes:
            if outcome.kind == "success":
                summary.n_success += 1
            elif outcome.kind == "failed":
                summary.n_failed += 1
                summary.errors.append(
                    f"{outcome.identity.quarter}/{outcome.identity.product_id}: "
                    f"{outcome.message}"
                )
            elif outcome.kind == "skipped":
                summary.n_skipped += 1
            # not_claimed: another worker won the race; not an error

        runtime = (_utc_now() - run_started).total_seconds()
        summary.runtime_seconds = runtime
        run_meta = {
            "experiment_id": exp_id,
            "engine_version": engine.name,
            "config_hash": config_hash,
            "requested_workers": workers_n,
            "thread_config": thread_config,
            "runtime_seconds": runtime,
            "n_success": summary.n_success,
            "n_failed": summary.n_failed,
            "n_skipped": summary.n_skipped,
            "finished_at": _utc_stamp(),
            "git_commit": git_commit,
        }
        atomic_write_json(exp_dir / "run_meta.json", run_meta)
        print(
            f"done: success={summary.n_success} failed={summary.n_failed} "
            f"skipped={summary.n_skipped} reclaimed_stale={summary.n_reclaimed_stale} "
            f"workers={workers_n} runtime_seconds={runtime:.2f}"
        )
        return summary
    finally:
        if lock is not None:
            lock.release()
