"""F3B Step 1 CLI: freeze Triple Price history + Dim.Product.

Run::

    python -m pkg.research.prepare_f3b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3b.prepare import prepare_price_source
from pkg.research.f3b.report import write_price_source_audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "F3B Step 1: freeze product-level price history from the Triple "
            "Price workbook + Dim.Product (no XGBoost)"
        )
    )
    parser.add_argument("--triple-xlsx", type=Path, default=None)
    parser.add_argument("--map-xlsx", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-freeze-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = prepare_price_source(
            triple_xlsx=args.triple_xlsx,
            map_xlsx=args.map_xlsx,
            out_dir=args.out_dir,
            verify_freeze=not args.skip_freeze_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"F3B source prep failed: {e}", file=sys.stderr)
        raise

    summary = result["summary"].iloc[0]
    print("=== F3B Step 1 price source ===")
    print(result["summary"].to_string(index=False))
    print("\n=== PRIMARY origin coverage (MVP, no features) ===")
    print(result["origins"].to_string(index=False))
    print(
        f"\nValid observations: {int(summary['n_valid_dated_observations'])}  "
        f"unmatched names: {int(summary['n_source_names_unmatched'])}  "
        f"ambiguous names: {int(summary['n_source_names_ambiguous'])}"
    )

    path = write_price_source_audit(result)
    print(f"\nWrote {path}")
    print(f"Frozen source: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
