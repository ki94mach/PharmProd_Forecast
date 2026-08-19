"""F3C Step 3 CLI: controlled XGBoost experiment (inventory features).

Run::

    python -m pkg.research.evaluate_f3c
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3c.evaluate import evaluate_f3c
from pkg.research.f3c.experiment_report import write_f3c_results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F3C Step 3: controlled XGB inventory experiment"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = evaluate_f3c(
            out_dir=args.out_dir,
            verify_checksums=not args.skip_checksums,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F3C gate/assertion failed: {e}", file=sys.stderr)
        return 2

    print("=== F3C overall ===")
    print(result["overall"].to_string(index=False))
    print(f"\n=== Verdict: {result['verdict']} ===")

    path = write_f3c_results(result)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
