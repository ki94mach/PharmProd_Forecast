"""TS V2 historical backfill evaluation CLI.

Run::

    python -m pkg.research.evaluate_v2_backfill

Read-only with respect to the backfill experiment and the frozen benchmark; the
only writes are the analysis artifacts under
``src/data/results/ts_v2_backfill_eval/`` and ``docs/ts_v2_backfill_eval.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.v2_eval.config import default_config
from pkg.research.v2_eval.run import evaluate_v2_backfill


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a completed TS V2 historical backfill against the frozen "
            "legacy TS forecasts on identical matched rows"
        )
    )
    parser.add_argument("--experiment-id", default="ts_mvp_backfill_1401Q1_1405Q2")
    parser.add_argument("--engine", default="v2")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-charts", action="store_true", help="skip figure generation"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = default_config(
            experiment_id=args.experiment_id,
            engine=args.engine,
            out_dir=args.out_dir,
            experiment_dir=args.experiment_dir,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        report = evaluate_v2_backfill(config=config, make_charts=not args.no_charts)
    except FileNotFoundError as exc:
        print(f"ERROR: missing input: {exc}", file=sys.stderr)
        print(
            "The frozen benchmark is required. Build it with: "
            "python -m pkg.benchmark.freeze",
            file=sys.stderr,
        )
        return 1
    except AssertionError as exc:
        print(f"ERROR: validation failed: {exc}", file=sys.stderr)
        return 2

    overall = report["overall"]
    primary = overall.loc[overall["scope"] == "overall"]
    verdict = report["verdict"]
    coverage = report["panel"].coverage_summary

    print("=== Coverage ===")
    for key, value in coverage.items():
        print(f"{key}: {value:,}")

    print("\n=== Overall metrics (matched rows) ===")
    print(
        primary[
            ["metric", "legacy", "v2", "absolute_change", "relative_improvement_pct", "n"]
        ].to_string(index=False)
    )

    print("\n=== Sensitivities (WMAPE) ===")
    sens = overall.loc[
        (overall["scope"] == "sensitivity") & (overall["metric"] == "wmape")
    ]
    print(
        sens[["slice", "n", "legacy", "v2", "relative_improvement_pct"]].to_string(
            index=False
        )
    )

    print("\n=== Verdict ===")
    print(verdict["label"])
    print(
        f"origins better: {verdict['origins_better']}/{verdict['origins_total']}; "
        f"products improved {verdict['pct_products_improved']:.1f}% vs regressed "
        f"{verdict['pct_products_regressed']:.1f}%"
    )

    audit = report["audit"]
    failed_checks = audit.loc[audit["status"] == "fail"]
    if len(failed_checks):
        print("\n=== Failing audit checks ===")
        print(failed_checks[["check", "finding"]].to_string(index=False))

    print(f"\nArtifacts: {config.out_dir}")
    print(f"Report: {config.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
