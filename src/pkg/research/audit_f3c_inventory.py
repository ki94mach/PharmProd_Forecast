"""F3C Step 2 CLI: point-in-time inventory feature audit (no XGBoost).

Run::

    python -m pkg.research.audit_f3c_inventory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3c.feature_audit import audit_inventory_features
from pkg.research.f3c.feature_report import write_feature_audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F3C Step 2: PIT inventory feature audit (no XGBoost)"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-freeze-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = audit_inventory_features(
            out_dir=args.out_dir,
            verify_freeze=not args.skip_freeze_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.research.prepare_f3c", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F3C assertion failed: {e}", file=sys.stderr)
        return 2

    print("=== F3C Step 2 PRIMARY coverage ===")
    print(result["overall"].to_string(index=False))
    print("\n=== Distributions ===")
    print(result["distributions"].to_string(index=False))
    print("\n=== By origin ===")
    print(result["by_origin"].to_string(index=False))

    path = write_feature_audit(result)
    print(f"\nWrote {path}")
    print(f"CSV artifacts: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
