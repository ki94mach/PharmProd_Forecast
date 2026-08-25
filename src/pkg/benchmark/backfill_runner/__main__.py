"""``python -m pkg.benchmark.backfill_runner``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


from pkg.env import load_project_env

load_project_env()

from pkg.benchmark.backfill_runner.engines import available_engines, get_engine
from pkg.benchmark.backfill_runner.manifest import (
    ExperimentManifestError,
    make_experiment_id,
)
from pkg.benchmark.backfill_runner.runner import print_status, run_backfill
from pkg.benchmark.backfill_runner.state import RunLockError
from pkg.benchmark.backfill_runner.store import default_backfill_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Historical forecast backfill orchestrator with durable SQLite "
            "checkpoints, immutable experiment manifests, and an exclusive run lock. "
            "Writes under data/backfills/{experiment_id}/{engine}/ — not production CSVs."
        )
    )
    p.add_argument(
        "--engine",
        required=True,
        choices=sorted(set(available_engines()) | {"v3"}),
        help="Forecasting engine/version (v3 not implemented yet)",
    )
    p.add_argument(
        "--vintages",
        required=True,
        help="Vintage manifest stem, e.g. ts_backfill_1401Q1_1405Q2",
    )
    p.add_argument(
        "--universe",
        default="mvp_products",
        help="Universe manifest stem (default: mvp_products)",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Override experiment id (default: ts_mvp_backfill_{start}_{end}). "
            "Engine is always a subdirectory under the experiment id."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Output root (default: {default_backfill_root()})",
    )
    p.add_argument("--quarter", default=None, help="Single quarter filter (e.g. 1405Q1)")
    p.add_argument("--quarter-from", default=None, help="Inclusive start quarter filter")
    p.add_argument("--quarter-to", default=None, help="Inclusive end quarter filter")
    p.add_argument("--product", default=None, help="Single product filter")
    p.add_argument(
        "--products",
        default=None,
        help="Comma-separated product filter (subset of universe)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip SUCCESS jobs; reclaim stale RUNNING and continue PENDING",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run FAILED jobs (implies resume semantics for SUCCESS skip)",
    )
    p.add_argument(
        "--force-job",
        action="store_true",
        help="Force recompute of matching jobs even if SUCCESS (clears artifacts)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print SQLite job status and exit (no forecasting)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print work plan and expected job count without fitting models",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent SKU-vintage jobs (default: 1). "
            "Does not auto-select CPU count."
        ),
    )
    p.add_argument(
        "--include-future",
        action="store_true",
        help="Include vintages with status=future_origin",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.engine == "v3":
        print("V3 engine is not implemented yet.", file=sys.stderr)
        return 2

    root = args.output_root or default_backfill_root()
    exp_id = args.experiment_id or make_experiment_id(args.vintages, args.universe)

    if args.status:
        return print_status(
            output_root=root,
            experiment_id=exp_id,
            engine=args.engine,
            quarter=args.quarter,
            product=args.product,
        )

    try:
        engine = get_engine(args.engine)
    except Exception as exc:  # noqa: BLE001
        print(f"engine error: {exc}", file=sys.stderr)
        return 2

    product_filter = None
    if args.products:
        product_filter = [p.strip() for p in args.products.split(",") if p.strip()]

    if int(args.workers) < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2

    resume = bool(args.resume or args.retry_failed)
    try:
        summary = run_backfill(
            engine=engine,
            vintage_name=args.vintages,
            universe_name=args.universe,
            output_root=root,
            experiment_id=exp_id,
            quarter=args.quarter,
            quarter_from=args.quarter_from,
            quarter_to=args.quarter_to,
            products=product_filter,
            product=args.product,
            resume=resume,
            retry_failed=bool(args.retry_failed),
            force_job=bool(args.force_job),
            dry_run=bool(args.dry_run),
            include_future=bool(args.include_future),
            workers=int(args.workers),
        )
    except RunLockError as exc:
        print(f"run lock: {exc}", file=sys.stderr)
        return 3
    except ExperimentManifestError as exc:
        print(f"experiment manifest: {exc}", file=sys.stderr)
        return 4

    if summary.dry_run:
        return 0
    return 0 if summary.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
