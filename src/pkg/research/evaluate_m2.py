"""CLI entrypoint for M2 residual learner model-class benchmark."""
from __future__ import annotations

import sys
import traceback

from pkg.research.model_benchmark.run import run_m2


def main() -> None:
    print("=== M2 residual learner benchmark ===")
    try:
        out = run_m2()
    except Exception as exc:  # noqa: BLE001
        print(f"M2 FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    print("=== M2 COMPLETE ===")
    print(out)


if __name__ == "__main__":
    main()
