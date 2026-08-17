"""Feature-family ablation CLI.

Run::

    python -m pkg.research.evaluate_feature_ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.ablation.evaluate import evaluate_feature_ablation
from pkg.research.ablation.report import write_ablation_report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CORE vs F0_DEMAND ablation: F1/F2 as replacement vs addition "
            "(frozen F0 unchanged)"
        )
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Skip H6/H7 (CORE + F1/F2 demand + Human, no F0 demand)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = evaluate_feature_ablation(
            verify_checksums=args.verify_checksums,
            out_dir=args.out_dir,
            skip_secondary=args.skip_secondary,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.benchmark.freeze", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"Ablation assertion failed: {e}", file=sys.stderr)
        return 2

    overall = report["overall"]
    cols = [
        c
        for c in (
            "experiment",
            "anchor",
            "n_features",
            "wmape",
            "rel_wmape_vs_f0_pct",
            "origins_improved",
            "product_win_rate",
        )
        if c in overall.columns
    ]
    print("=== Canonical F0 (current frozen backtest) ===")
    print(report["canonical_f0"]["summary"].to_string(index=False))
    print("\n=== Reproduction gates ===")
    print(report["gates"].to_string(index=False))
    print("\n=== Scoreboard ===")
    print(overall[cols].to_string(index=False))
    print("\n=== Replacement effects ===")
    print(report["effects"].to_string(index=False))
    print("\n=== Cases A–E ===")
    print(report["classifications"].to_string(index=False))

    path = write_ablation_report(report)
    print(f"\nWrote {path}")
    print(f"CSV artifacts: {report['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
