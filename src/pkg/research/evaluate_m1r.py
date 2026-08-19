"""CLI entrypoint for M1R deterministic rerun."""
from __future__ import annotations

import sys
import traceback

from pkg.research.m1r.run import run_m1r


def main() -> None:
    print("=== M1R deterministic reproducibility run ===")
    try:
        out = run_m1r()
    except Exception as exc:  # noqa: BLE001
        print(f"M1R FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    print("=== M1R COMPLETE ===")
    print(out)


if __name__ == "__main__":
    main()

