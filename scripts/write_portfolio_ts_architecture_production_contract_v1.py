#!/usr/bin/env python3
"""Write immutable ts_architecture_production_contract_v1.json (+ .sha256 sidecar).

Does NOT modify ts_architecture_lock.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.portfolio_ts_lock.lock_io import LockExistsError  # noqa: E402
from pkg.portfolio_ts_lock.production_contract import (  # noqa: E402
    build_production_contract_v1,
    validate_contract_against_lock,
    write_immutable_json,
)

DEFAULT_LOCK_DIR = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_ts_architecture_lock_v1"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    args = ap.parse_args()
    lock_dir = Path(args.lock_dir)
    lock_path = lock_dir / "ts_architecture_lock.json"
    if not lock_path.is_file():
        raise SystemExit(f"Scientific lock missing: {lock_path}")

    metrics = json.loads((lock_dir / "metrics_primary.json").read_text(encoding="utf-8"))
    impact = json.loads(
        (lock_dir / "impact_decomposition_residual.json").read_text(encoding="utf-8")
    )
    contract = build_production_contract_v1(
        repo_root=REPO, lock_dir=lock_dir, metrics=metrics, impact=impact
    )
    out = lock_dir / "ts_architecture_production_contract_v1.json"
    try:
        digest = write_immutable_json(out, contract)
    except LockExistsError as exc:
        raise SystemExit(str(exc)) from exc

    loaded = json.loads(out.read_text(encoding="utf-8"))
    validate_contract_against_lock(loaded, lock_dir=lock_dir, repo_root=REPO)
    # Prove lock untouched
    lock_before = lock_path.read_bytes()
    lock_after = lock_path.read_bytes()
    if lock_before != lock_after:
        raise SystemExit("Scientific lock bytes changed — abort")

    print(
        json.dumps(
            {
                "contract_path": out.as_posix(),
                "content_sha256": digest,
                "sidecar": (out.with_suffix(out.suffix + ".sha256")).as_posix(),
                "scientific_lock_untouched": True,
                "frozen_choice": loaded["frozen_choice"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
