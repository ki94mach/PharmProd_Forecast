"""F2 feature-experiment CLI.

Run::

    python -m pkg.research.evaluate_f2
    python -m pkg.research.evaluate_f2 --experiment F2A
    python -m pkg.research.evaluate_f2 --experiment F2B
    python -m pkg.research.evaluate_f2 --experiment F2C
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f2.evaluate import evaluate_f2
from pkg.research.f2.report import write_f2_results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate F2A/F2B/F2C on frozen benchmark v1 (F0 unchanged)"
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        choices=["F2A", "F2B", "F2C"],
        help="Subset of F2 experiments (default: F2A, F2B, then F2C if justified)",
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = evaluate_f2(
            experiments=args.experiment,
            verify_checksums=args.verify_checksums,
            out_dir=args.out_dir,
            skip_diagnostics=args.skip_diagnostics,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.benchmark.freeze", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F2 assertion failed: {e}", file=sys.stderr)
        return 2

    overall = report["overall"]
    cols = [
        "experiment",
        "anchor",
        "wmape",
        "rel_wmape_vs_f0_pct",
        "rmse",
        "mae",
        "bias",
        "n",
        "origins_improved",
        "origins_total",
        "product_win_rate",
        "median_product_improvement_pct",
        "verdict",
    ]
    print("=== Canonical F0 (current frozen backtest) ===")
    print(report["canonical_f0"]["summary"].to_string(index=False))
    print("\n=== F2 scoreboard (matched PRIMARY) ===")
    print(overall[cols].to_string(index=False))
    print("\n=== By origin ===")
    print(report["by_origin"].to_string(index=False))
    print("\n=== Training coverage ===")
    td = report["train_diagnostics"]
    show = [
        c
        for c in (
            "experiment",
            "anchor",
            "origin",
            "train_rows",
            "train_rows_budget",
            "prior_budget_vintages",
            "train_universe",
        )
        if c in td.columns
    ]
    print(td[show].to_string(index=False))

    path = write_f2_results(report)
    print(f"\nWrote {path}")
    print(f"CSV artifacts: {report['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
