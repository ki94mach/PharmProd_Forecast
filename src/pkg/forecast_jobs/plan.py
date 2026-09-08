"""Job planning: expand cohort × origins × architectures × seeds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from pkg.benchmark.calendar import quarter_from_origin
from pkg.benchmark.backfill_runner.state import JobIdentity, JobStateStore, should_run_job
from pkg.benchmark.universes import load_universe_product_names
from pkg.forecast_jobs.config import JobRunConfig


def encode_product_key(product: str, seed: Optional[int]) -> str:
    """Encode product (+ optional neural seed) into the SQLite product_id column."""
    if seed is None:
        return str(product)
    return f"{product}__seed{int(seed)}"


def decode_product_key(product_key: str) -> tuple[str, Optional[int]]:
    marker = "__seed"
    if marker not in product_key:
        return str(product_key), None
    product, _, seed_s = product_key.rpartition(marker)
    try:
        return product, int(seed_s)
    except ValueError:
        return str(product_key), None


def resolve_products(config: JobRunConfig) -> list[str]:
    """Resolve cohort products from universe and/or explicit list."""
    products: list[str] = []
    if config.cohort.universe:
        products = list(
            load_universe_product_names(config.cohort.universe, validate=True)
        )
    if config.cohort.products:
        wanted = [str(p) for p in config.cohort.products]
        if products:
            allowed = set(products)
            missing = [p for p in wanted if p not in allowed]
            if missing:
                raise ValueError(
                    f"cohort.products not in universe {config.cohort.universe!r}: "
                    f"{missing[:10]}"
                )
            products = [p for p in products if p in set(wanted)]
        else:
            products = wanted
    if not products:
        raise ValueError("resolved product cohort is empty")
    return products


@dataclass(frozen=True)
class PlannedJob:
    architecture: str
    origin: int
    quarter: str
    product: str
    seed: Optional[int]
    identity: JobIdentity


@dataclass
class JobPlan:
    run_id: str
    config_hash: str
    products: list[str]
    jobs: list[PlannedJob]
    already_completed: int
    remaining: int
    total_jobs: int
    status_counts: dict[str, int]


def build_job_plan(
    config: JobRunConfig,
    *,
    config_hash: str,
    state: Optional[JobStateStore] = None,
    git_commit: str = "unknown",
    ensure_jobs: bool = True,
) -> JobPlan:
    """Expand the Cartesian job grid and optionally register SQLite rows."""
    products = resolve_products(config)
    jobs: list[PlannedJob] = []
    n_done = 0

    for architecture in config.architectures:
        seed_values: Sequence[Optional[int]]
        if architecture in {"a0", "a1"}:
            seed_values = list(config.seeds)
        else:
            seed_values = [None]

        for origin in config.origins:
            quarter = quarter_from_origin(int(origin))
            for product in products:
                for seed in seed_values:
                    product_key = encode_product_key(product, seed)
                    identity = JobIdentity(
                        experiment_id=config.run_id,
                        engine_version=str(architecture),
                        config_hash=config_hash,
                        quarter=quarter,
                        forecast_origin=int(origin),
                        product_id=product_key,
                    )
                    planned = PlannedJob(
                        architecture=str(architecture),
                        origin=int(origin),
                        quarter=quarter,
                        product=str(product),
                        seed=seed if seed is None else int(seed),
                        identity=identity,
                    )
                    if state is not None and ensure_jobs:
                        record = state.ensure_job(identity, git_commit=git_commit)
                        if config.execution.force_job and record.status == "SUCCESS":
                            record = state.reset_for_force(
                                identity, git_commit=git_commit
                            )
                        if should_run_job(
                            record,
                            resume=config.execution.resume,
                            retry_failed=config.execution.retry_failed,
                            force_job=config.execution.force_job,
                        ):
                            jobs.append(planned)
                        elif record.status == "SUCCESS":
                            n_done += 1
                    else:
                        jobs.append(planned)

    total = 0
    for architecture in config.architectures:
        n_seeds = len(config.seeds) if architecture in {"a0", "a1"} else 1
        total += len(config.origins) * len(products) * n_seeds

    counts = (
        state.status_counts(config.run_id)
        if state is not None and ensure_jobs
        else {}
    )
    return JobPlan(
        run_id=config.run_id,
        config_hash=config_hash,
        products=products,
        jobs=jobs,
        already_completed=n_done,
        remaining=len(jobs),
        total_jobs=total,
        status_counts=counts,
    )
