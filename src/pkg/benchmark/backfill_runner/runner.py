"""Historical forecast backfill orchestrator (no model logic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from pkg.benchmark.backfill_runner.store import BackfillStore, default_backfill_root
from pkg.benchmark.backfill_runner.types import (
    EngineJobRequest,
    ForecastEngine,
    JobKey,
    JobLogRecord,
    request_from_vintage,
    target_dates_for_origin,
)
from pkg.benchmark.calendar import iter_shamsi_quarters
from pkg.benchmark.universes import load_universe_product_names
from pkg.benchmark.vintages import VintageSpec, load_vintage_manifest_by_name


@dataclass
class BackfillPlan:
    engine: str
    vintages_requested: list[str]
    vintages_eligible: list[str]
    products: list[str]
    jobs: list[tuple[VintageSpec, str]]
    already_completed: int
    remaining: int
    total_jobs: int

    def report_lines(self) -> list[str]:
        return [
            f"engine={self.engine}",
            f"vintages_requested={len(self.vintages_requested)}: "
            f"{', '.join(self.vintages_requested)}",
            f"vintages_eligible_now={len(self.vintages_eligible)}: "
            f"{', '.join(self.vintages_eligible)}",
            f"products={len(self.products)}",
            f"total_sku_vintage_jobs={self.total_jobs}",
            f"already_completed={self.already_completed}",
            f"remaining={self.remaining}",
        ]


@dataclass
class BackfillRunSummary:
    plan: BackfillPlan
    n_success: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enforce_historical_cutoff(
    sales: pd.DataFrame,
    forecast_origin: int,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Return only rows with ``date < forecast_origin`` (Shamsi YYYYMM).

    This cutoff is enforced by the runner, outside forecasting engines.
    """
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
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    include_future: bool = False,
) -> list[VintageSpec]:
    out = list(specs)
    if quarter_from or quarter_to:
        start = quarter_from or specs[0].quarter
        end = quarter_to or specs[-1].quarter
        allowed = set(iter_shamsi_quarters(start, end))
        out = [s for s in out if s.quarter in allowed]
    if not include_future:
        out = [s for s in out if s.status != "future_origin"]
    return out


def build_backfill_plan(
    *,
    engine: str,
    vintage_name: str,
    universe_name: str,
    store: BackfillStore,
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    products: Optional[Sequence[str]] = None,
    include_future: bool = False,
    resume: bool = True,
) -> BackfillPlan:
    all_specs = load_vintage_manifest_by_name(vintage_name, validate=True)
    requested = [s.quarter for s in all_specs]
    eligible_specs = filter_vintages(
        all_specs,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
        include_future=include_future,
    )
    product_list = load_universe_product_names(universe_name, validate=True)
    if products:
        wanted = {str(p) for p in products}
        product_list = [p for p in product_list if p in wanted]
        missing = wanted - set(product_list)
        if missing:
            raise ValueError(
                f"products not in universe {universe_name!r}: {sorted(missing)[:10]}"
            )

    jobs: list[tuple[VintageSpec, str]] = []
    completed = store.completed_keys() if resume else set()
    n_done = 0
    for vintage in eligible_specs:
        for product in product_list:
            key = (vintage.quarter, product)
            if resume and key in completed:
                n_done += 1
                continue
            jobs.append((vintage, product))

    total = len(eligible_specs) * len(product_list)
    return BackfillPlan(
        engine=str(engine),
        vintages_requested=requested,
        vintages_eligible=[s.quarter for s in eligible_specs],
        products=list(product_list),
        jobs=jobs,
        already_completed=n_done,
        remaining=len(jobs),
        total_jobs=total,
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


def run_backfill(
    *,
    engine: ForecastEngine,
    vintage_name: str,
    universe_name: str = "mvp_products",
    output_root: Optional[Path] = None,
    sales: Optional[pd.DataFrame] = None,
    quarter_from: Optional[str] = None,
    quarter_to: Optional[str] = None,
    products: Optional[Sequence[str]] = None,
    resume: bool = True,
    dry_run: bool = False,
    include_future: bool = False,
    product_meta: Optional[dict[str, dict[str, Any]]] = None,
) -> BackfillRunSummary:
    """Orchestrate historical backfill for one engine/version."""
    root = Path(output_root) if output_root is not None else default_backfill_root()
    store = BackfillStore(root, engine.name)
    plan = build_backfill_plan(
        engine=engine.name,
        vintage_name=vintage_name,
        universe_name=universe_name,
        store=store,
        quarter_from=quarter_from,
        quarter_to=quarter_to,
        products=products,
        include_future=include_future,
        resume=resume,
    )

    for line in plan.report_lines():
        print(line)

    if dry_run:
        print("dry-run: work plan (no model fitting)")
        for vintage, product in plan.jobs[:50]:
            targets = target_dates_for_origin(vintage.forecast_origin, vintage.horizon)
            print(
                f"  {vintage.quarter} origin={vintage.forecast_origin} "
                f"product={product} targets={targets[0]}..{targets[-1]}"
            )
        if len(plan.jobs) > 50:
            print(f"  ... ({len(plan.jobs) - 50} more jobs)")
        return BackfillRunSummary(plan=plan, dry_run=True)

    store.write_run_metadata(
        {
            "engine": engine.name,
            "vintage_manifest": vintage_name,
            "universe": universe_name,
            "started_at_utc": _utc_stamp(_utc_now()),
            "resume": resume,
            "plan": {
                "vintages_eligible": plan.vintages_eligible,
                "n_products": len(plan.products),
                "total_jobs": plan.total_jobs,
                "already_completed": plan.already_completed,
                "remaining": plan.remaining,
            },
        }
    )

    sales_df = sales if sales is not None else load_default_sales()
    summary = BackfillRunSummary(plan=plan, dry_run=False)
    meta_by_product = product_meta or {}

    for vintage, product in plan.jobs:
        key = JobKey(
            engine=engine.name,
            quarter=vintage.quarter,
            product=product,
            forecast_origin=int(vintage.forecast_origin),
        )
        if resume and store.is_complete(key):
            summary.n_skipped += 1
            continue

        start = _utc_now()
        try:
            store.assert_writable(key, resume=resume)
            train = enforce_historical_cutoff(sales_df, vintage.forecast_origin)
            train = train.loc[train["product"].astype(str) == str(product)].copy()
            request = request_from_vintage(
                engine=engine.name,
                product=product,
                vintage=vintage,
                training_sales=train,
                meta=meta_by_product.get(product, {}),
            )
            # Re-assert cutoff on the request payload.
            if not request.training_sales.empty:
                assert (
                    request.training_sales["date"].astype(int).max()
                    < int(request.forecast_origin)
                )
            result = engine.forecast_product(request)
            end = _utc_now()
            duration = (end - start).total_seconds()
            log = JobLogRecord(
                engine=engine.name,
                quarter=vintage.quarter,
                forecast_origin=int(vintage.forecast_origin),
                product=product,
                start_time_utc=_utc_stamp(start),
                end_time_utc=_utc_stamp(end),
                duration_seconds=float(duration),
                success=bool(result.success),
                selected_model=result.selected_model,
                error_message=result.error_message,
                status="complete" if result.success else "failed",
            )
            print(
                f"[{log.status}] engine={log.engine} qrt={log.quarter} "
                f"origin={log.forecast_origin} product={log.product} "
                f"duration={log.duration_seconds:.2f}s "
                f"model={log.selected_model} err={log.error_message}"
            )
            if result.success:
                store.persist_success(key, result, log)
                summary.n_success += 1
            else:
                store.persist_failure(key, log)
                summary.n_failed += 1
                summary.errors.append(
                    f"{vintage.quarter}/{product}: {result.error_message}"
                )
        except Exception as exc:  # noqa: BLE001 — never kill the whole backfill
            end = _utc_now()
            log = JobLogRecord(
                engine=engine.name,
                quarter=vintage.quarter,
                forecast_origin=int(vintage.forecast_origin),
                product=product,
                start_time_utc=_utc_stamp(start),
                end_time_utc=_utc_stamp(end),
                duration_seconds=float((end - start).total_seconds()),
                success=False,
                error_message=str(exc),
                status="failed",
            )
            print(
                f"[failed] engine={log.engine} qrt={log.quarter} "
                f"product={log.product} err={log.error_message}"
            )
            try:
                store.persist_failure(key, log)
            except Exception as persist_exc:  # noqa: BLE001
                summary.errors.append(
                    f"{vintage.quarter}/{product}: persist failed: {persist_exc}"
                )
            summary.n_failed += 1
            summary.errors.append(f"{vintage.quarter}/{product}: {exc}")

    print(
        f"done: success={summary.n_success} failed={summary.n_failed} "
        f"skipped={summary.n_skipped}"
    )
    return summary
