"""F3D Step 1 CLI: freeze patient-consumption profile source from Dim.Product.

Run::

    python -m pkg.research.prepare_f3d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3d.prepare import (
    DuplicateConflictError,
    ZeroOverlapError,
    prepare_profile_source,
)
from pkg.research.f3d.report import write_profile_audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F3D Step 1: freeze patient-consumption profile (no XGBoost)"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-freeze-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = prepare_profile_source(
            out_dir=args.out_dir,
            verify_freeze=not args.skip_freeze_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except (DuplicateConflictError, ZeroOverlapError) as e:
        print(f"MAPPING FAILED: {e}", file=sys.stderr)
        return 2
    except AssertionError as e:
        print(f"SEMANTIC ASSERTION FAILED: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"F3D source prep failed: {e}", file=sys.stderr)
        raise

    print("=== F3D Step 1 coverage ===")
    print(result["audit"].to_string(index=False))
    print("\n=== PatientConsumeType counts ===")
    print(result["type_counts"].to_string(index=False))
    print("\n=== PatientConsumePerPeriod distributions ===")
    print(result["period_distributions"].to_string(index=False))

    if not result["negative_report"].empty:
        print("\n=== WARNING: negative PatientConsumePerPeriod ===")
        print(result["negative_report"].to_string(index=False))

    path = write_profile_audit(result)
    print(f"\nWrote {path}")
    print(f"Frozen source: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
