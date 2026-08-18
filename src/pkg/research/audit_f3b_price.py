"""F3B Step 2 CLI: point-in-time price feature audit (no XGBoost).

Run::

    python -m pkg.research.audit_f3b_price
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.research.f3b.feature_audit import audit_price_features
from pkg.research.f3b.feature_report import write_price_feature_audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "F3B Step 2: attach leakage-safe consumer-price features from the "
            "frozen price history and audit PRIMARY coverage (no XGBoost)"
        )
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-freeze-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = audit_price_features(
            out_dir=args.out_dir,
            verify_freeze=not args.skip_freeze_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.research.prepare_f3b", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"F3B assertion failed: {e}", file=sys.stderr)
        return 2
    except AssertionError as e:
        print(f"F3B assertion failed: {e}", file=sys.stderr)
        return 2

    print("=== F3B Step 2 PRIMARY coverage ===")
    print(result["overall"].to_string(index=False))
    print("\n=== Distributions ===")
    print(result["distributions"].to_string(index=False))
    print("\n=== By origin ===")
    print(result["by_origin"].to_string(index=False))
    print(
        "\nMVP products with as-of price change across PRIMARY origins: "
        f"{result['n_products_varying_across_origins']}"
    )
    path = write_price_feature_audit(result)
    print(f"\nWrote {path}")
    print(f"CSV artifacts: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
