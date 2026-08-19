"""F3D Step 2 CLI: controlled XGBoost experiment (patient-consumption profile).

Run::

    python -m pkg.research.evaluate_f3d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3d.evaluate import evaluate_f3d
from pkg.research.f3d.experiment_report import write_f3d_results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F3D Step 2: controlled XGB patient-consumption experiment"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = evaluate_f3d(
            out_dir=args.out_dir,
            verify_checksums=not args.skip_checksums,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F3D gate/assertion failed: {e}", file=sys.stderr)
        return 2

    print("=== F3D overall ===")
    print(result["overall"].to_string(index=False))
    print(f"\n=== Verdict: {result['verdict']} ===")

    path = write_f3d_results(result)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
