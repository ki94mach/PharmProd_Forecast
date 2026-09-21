#!/usr/bin/env python3
"""Post-lock TS stability on 140401/140402 only. Never mutates the lock. Never uses HB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.universes import load_universe_product_names  # noqa: E402
from pkg.portfolio_ts_lock.completeness import audit_product_origin_completeness  # noqa: E402
from pkg.portfolio_ts_lock.constants import (  # noqa: E402
    HB_ORIGINS,
    LOCKED_SCORE_END,
    POST_HB_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import build_dense_actuals  # noqa: E402
from pkg.portfolio_ts_lock.freeze import score_frame  # noqa: E402
from pkg.portfolio_ts_lock.lock_io import (  # noqa: E402
    assert_lock_immutable_read,
    refuse_lock_mutation,
)
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb  # noqa: E402

DEFAULT_LOCK_DIR = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_ts_architecture_lock_v1"
)
V21_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_v21_backfill_v2_export"
)
A0_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_a0_backfill_v2_export"
)
SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    args = ap.parse_args()
    lock_path = args.lock_dir / "ts_architecture_lock.json"
    lock = assert_lock_immutable_read(lock_path)

    # Hard refuse mutation
    try:
        refuse_lock_mutation(lock_path)
    except PermissionError:
        pass  # expected when we probe; we never write

    sales = pd.read_parquet(SALES)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    panel_v21 = filter_non_hb(pd.read_parquet(V21_EXPORT / "forecasts_panel.parquet"))
    panel_a0 = filter_non_hb(pd.read_parquet(A0_EXPORT / "forecasts_panel.parquet"))
    status, _inc = audit_product_origin_completeness(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        original55=set(load_universe_product_names("mvp_products")),
    )
    _common, deployable, _cats, _partial = build_policy_panels(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        completeness_status=status,
    )

    if deployable["forecast_origin"].astype(int).isin(HB_ORIGINS).any():
        raise RuntimeError("HB origins leaked into post-lock panel")

    post = deployable.loc[
        deployable["forecast_origin"].astype(int).isin(POST_HB_ORIGINS)
    ].copy()
    frozen = lock.get("frozen_policy") or lock.get("choice")
    if frozen == "A0_HYBRID_POLICY":
        fcol = "f_a0_hybrid"
    else:
        fcol = "f_v2_policy"

    metrics = {
        "lock_choice": lock.get("choice"),
        "frozen_policy": frozen,
        "n_rows": int(len(post)),
        "v2_policy": score_frame(post["actual"], post["f_v2_policy"]),
        "frozen_policy_scores": score_frame(post["actual"], post[fcol]),
        "hb_excluded": sorted(HB_ORIGINS),
        "origins_scored": sorted(POST_HB_ORIGINS),
        "lock_path": lock_path.as_posix(),
        "lock_mutated": False,
    }
    out = args.lock_dir / "postlock_140401_140402.json"
    out.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    # Prove lock bytes unchanged
    lock2 = assert_lock_immutable_read(lock_path)
    if lock2 != lock:
        raise RuntimeError("Lock content changed during post-lock analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
