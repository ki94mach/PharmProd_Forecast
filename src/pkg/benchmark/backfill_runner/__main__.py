"""``python -m pkg.benchmark.backfill_runner``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from pkg.benchmark.backfill_runner.engines import available_engines, get_engine
from pkg.benchmark.backfill_runner.runner import run_backfill
from pkg.benchmark.backfill_runner.store import default_backfill_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Historical forecast backfill orchestrator (no model logic). "
            "Enforces sales.date < forecast_origin outside engines."
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
        "--output-root",
        type=Path,
        default=None,
        help=f"Output root (default: {default_backfill_root()})",
    )
    p.add_argument("--quarter-from", default=None, help="Inclusive start quarter filter")
    p.add_argument("--quarter-to", default=None, help="Inclusive end quarter filter")
    p.add_argument(
        "--products",
        default=None,
        help="Comma-separated product filter (subset of universe)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs with a valid .complete marker",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print work plan without fitting models",
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
    try:
        engine = get_engine(args.engine)
    except Exception as exc:  # noqa: BLE001
        print(f"engine error: {exc}", file=sys.stderr)
        return 2

    product_filter = None
    if args.products:
        product_filter = [p.strip() for p in args.products.split(",") if p.strip()]

    summary = run_backfill(
        engine=engine,
        vintage_name=args.vintages,
        universe_name=args.universe,
        output_root=args.output_root,
        quarter_from=args.quarter_from,
        quarter_to=args.quarter_to,
        products=product_filter,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        include_future=bool(args.include_future),
    )
    if summary.dry_run:
        return 0
    return 0 if summary.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
