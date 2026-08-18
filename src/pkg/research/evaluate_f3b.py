"""F3B price-experiment CLI.

Run::

    python -m pkg.research.evaluate_f3b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3b.evaluate import evaluate_f3b
from pkg.research.f3b.experiment_report import write_f3b_results, write_gate_failure


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate F3B point-in-time consumer-price features on frozen benchmark v1"
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = evaluate_f3b(
            verify_checksums=args.verify_checksums,
            out_dir=args.out_dir,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.research.prepare_f3b", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F3B assertion failed: {e}", file=sys.stderr)
        write_gate_failure(str(e), out_dir=args.out_dir)
        return 2

    overall = report["overall"]
    cols = [
        "experiment",
        "anchor",
        "control",
        "wmape",
        "rel_wmape_vs_control_pct",
        "rmse",
        "mae",
        "bias",
        "n",
        "origins_improved",
        "origins_total",
        "product_win_rate",
        "median_product_improvement_pct",
    ]
    print("=== Canonical F0 (current frozen backtest) ===")
    print(report["canonical_f0"]["summary"].to_string(index=False))
    print("\n=== F3B scoreboard (matched PRIMARY) ===")
    print(overall[cols].to_string(index=False))
    print("\n=== Verdict ===")
    print(report["verdict"])

    path = write_f3b_results(report)
    print(f"\nWrote {path}")
    print(f"CSV artifacts: {report['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
