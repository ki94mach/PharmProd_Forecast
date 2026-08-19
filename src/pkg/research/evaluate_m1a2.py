"""CLI entrypoint for M1A2 fixed-200 structural tuning diagnostic."""
from __future__ import annotations

import sys
import traceback

from pkg.research.m1a2.run import run_m1a2


def main() -> None:
    print("=== M1A2 fixed-200 structural tuning ===")
    try:
        out = run_m1a2()
    except Exception as exc:  # noqa: BLE001
        print(f"M1A2 FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    print("=== M1A2 COMPLETE ===")
    print(out)


if __name__ == "__main__":
    main()
