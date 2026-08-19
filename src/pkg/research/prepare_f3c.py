"""F3C Step 1 CLI: freeze distributor + factory inventory source.

Run::

    python -m pkg.research.prepare_f3c
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3c.prepare import UncleanMvpMappingError, prepare_inventory_source
from pkg.research.f3c.report import write_source_audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F3C Step 1: freeze inventory source (no XGBoost)"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-freeze-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = prepare_inventory_source(
            out_dir=args.out_dir,
            verify_freeze=not args.skip_freeze_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except UncleanMvpMappingError as e:
        print(f"MVP MAPPING FAILED: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"F3C source prep failed: {e}", file=sys.stderr)
        raise

    print("=== F3C Step 1 source summary ===")
    print(result["summary"].to_string(index=False))
    print("\n=== Exact month-end coverage ===")
    print(result["month_end"].to_string(index=False))
    print("\n=== Distributor status audit ===")
    print(result["status_audit"].to_string(index=False))

    path = write_source_audit(result)
    print(f"\nWrote {path}")
    print(f"Frozen source: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
