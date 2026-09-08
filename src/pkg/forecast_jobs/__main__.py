"""``python -m pkg.forecast_jobs`` — resumable V2.1 / A0 / A1 server runner."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from pkg.env import load_project_env

load_project_env()

from pkg.forecast_jobs.config import JobConfigError, load_job_config
from pkg.forecast_jobs.runner import print_status, run_forecast_jobs
from pkg.forecast_jobs.snapshot import SalesSnapshotError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Resumable server runner for V2.1, A0, and A1. "
            "Reads a versioned YAML/JSON config, validates a frozen sales Parquet "
            "snapshot, and writes per-run_id checkpoints under "
            "src/data/forecast_jobs/{run_id}/."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to versioned YAML or JSON job configuration",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override run.run_id from the config (separate output directory)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate config + sales snapshot and exit (no job execution)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, write plan/manifest/snapshot artifacts, do not fit models",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Execute pending jobs (server mode). Implies resume semantics from config",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print SQLite job status for the run and exit",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Force execution.resume=true",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Force execution.retry_failed=true",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_job_config(args.config, run_id_override=args.run_id)
    except JobConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # CLI flags override execution block.
    if args.resume or args.retry_failed:
        from dataclasses import replace
        from pkg.forecast_jobs.config import ExecutionConfig

        config = replace(
            config,
            execution=ExecutionConfig(
                resume=True if args.resume else config.execution.resume,
                retry_failed=(
                    True if args.retry_failed else config.execution.retry_failed
                ),
                force_job=config.execution.force_job,
                workers=config.execution.workers,
            ),
        )

    if args.status:
        return print_status(config)

    try:
        if args.validate and not args.dry_run and not args.execute:
            from pkg.forecast_jobs.snapshot import validate_sales_config
            from pkg.forecast_jobs.plan import build_job_plan, resolve_products
            from pkg.forecast_jobs.runner import scientific_config_hash

            snap = validate_sales_config(config.sales)
            products = resolve_products(config)
            cfg_hash = scientific_config_hash(config, snap.content_sha256)
            plan = build_job_plan(
                config, config_hash=cfg_hash, state=None, ensure_jobs=False
            )
            print(f"config_ok run_id={config.run_id}")
            print(f"config_hash={cfg_hash}")
            print(f"sales_sha256={snap.content_sha256}")
            print(f"sales_n_rows={snap.n_rows}")
            print(f"products={len(products)} origins={len(config.origins)}")
            print(f"architectures={list(config.architectures)} seeds={list(config.seeds)}")
            print(f"total_jobs={plan.total_jobs}")
            return 0

        summary = run_forecast_jobs(
            config,
            dry_run=bool(args.dry_run) or (not args.execute),
            execute=bool(args.execute),
        )
    except (JobConfigError, SalesSnapshotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {exc}", file=sys.stderr)
        return 1

    print(
        f"run_id={summary.run_id} config_hash={summary.config_hash} "
        f"path={summary.run_path} dry_run={summary.dry_run} "
        f"planned={summary.planned} success={summary.succeeded} "
        f"failed={summary.failed} skipped={summary.skipped} "
        f"stopped_early={summary.stopped_early} stop_reason={summary.stop_reason}"
    )
    if not summary.dry_run and summary.failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
