#!/usr/bin/env python3
"""Post-lock TS stability on 140401/140402 only.

Reads and validates scientific lock + production contract.
Never rewrites either immutable file. Never uses Human Budget origins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.universes import load_universe_product_names  # noqa: E402
from pkg.portfolio_ts_lock.completeness import audit_product_origin_completeness  # noqa: E402
from pkg.portfolio_ts_lock.constants import (  # noqa: E402
    HB_ORIGINS,
    HORIZON_GROUPS,
    LOCKED_SCORE_END,
    POST_HB_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import build_dense_actuals  # noqa: E402
from pkg.portfolio_ts_lock.decomposition import resolve_naive_deployable_partials  # noqa: E402
from pkg.portfolio_ts_lock.freeze import (  # noqa: E402
    horizon_group_wmape_lifts,
    score_frame,
)
from pkg.portfolio_ts_lock.lock_io import (  # noqa: E402
    assert_lock_immutable_read,
    refuse_lock_mutation,
)
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb  # noqa: E402
from pkg.portfolio_ts_lock.production_contract import (  # noqa: E402
    file_sha256,
    validate_contract_against_lock,
)

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


def deployment_stability_status(
    *,
    lift_deployable: float,
    lift_common: float,
    coverage_both_native: float,
    rmse_hybrid: float,
    rmse_v2: float,
) -> str:
    """SUPPORTIVE / CAUTION / BLOCK_DEPLOYMENT for post-HB stability only."""
    if lift_deployable <= -2.0 or (
        np.isfinite(rmse_hybrid)
        and np.isfinite(rmse_v2)
        and rmse_v2 > 0
        and rmse_hybrid > rmse_v2 * 1.10
    ):
        return "BLOCK_DEPLOYMENT"
    if lift_deployable < 0 or lift_common < 0 or coverage_both_native < 0.70:
        return "CAUTION"
    return "SUPPORTIVE"


def influential_skus(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = []
    for product, g in df.groupby("product", sort=False):
        ae_v2 = float(np.nansum(np.abs(g["f_v2_policy"] - g["actual"])))
        ae_hy = float(np.nansum(np.abs(g["f_a0_hybrid"] - g["actual"])))
        rows.append(
            {
                "product": str(product),
                "ae_reduction_v2_minus_hybrid": ae_v2 - ae_hy,
                "sum_ae_v2": ae_v2,
                "sum_ae_hybrid": ae_hy,
            }
        )
    out = pd.DataFrame(rows).sort_values("ae_reduction_v2_minus_hybrid")
    # most negative = hybrid worse; most positive = hybrid better
    return pd.concat(
        [
            out.head(k).assign(rank_type="top_hybrid_regressors"),
            out.tail(k).iloc[::-1].assign(rank_type="top_hybrid_gainers"),
        ],
        ignore_index=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    args = ap.parse_args()
    lock_dir = Path(args.lock_dir)
    lock_path = lock_dir / "ts_architecture_lock.json"
    contract_path = lock_dir / "ts_architecture_production_contract_v1.json"
    sidecar = contract_path.with_suffix(contract_path.suffix + ".sha256")

    lock_bytes = lock_path.read_bytes()
    contract_bytes = contract_path.read_bytes()
    lock = assert_lock_immutable_read(lock_path)
    contract = json.loads(contract_bytes.decode("utf-8"))
    validate_contract_against_lock(contract, lock_dir=lock_dir, repo_root=REPO)
    expected_side = sidecar.read_text(encoding="utf-8").strip()
    if file_sha256(contract_path) != expected_side:
        raise SystemExit("Production contract sidecar SHA-256 mismatch")

    for p in (lock_path, contract_path):
        try:
            refuse_lock_mutation(p)
        except PermissionError:
            pass

    sales = pd.read_parquet(SALES)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    panel_v21 = filter_non_hb(pd.read_parquet(V21_EXPORT / "forecasts_panel.parquet"))
    panel_a0 = filter_non_hb(pd.read_parquet(A0_EXPORT / "forecasts_panel.parquet"))
    if panel_v21["forecast_origin"].astype(int).isin(HB_ORIGINS).any():
        raise RuntimeError("HB in V2 panel")
    if panel_a0["forecast_origin"].astype(int).isin(HB_ORIGINS).any():
        raise RuntimeError("HB in A0 panel")

    status, _inc = audit_product_origin_completeness(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        original55=set(load_universe_product_names("mvp_products")),
    )
    common, deployable, _cats, partial = build_policy_panels(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        completeness_status=status,
    )
    naive_extra, _nofc = resolve_naive_deployable_partials(partial=partial, dense=dense)
    if not naive_extra.empty:
        deployable = pd.concat([deployable, naive_extra], ignore_index=True)

    post_dep = deployable.loc[
        deployable["forecast_origin"].astype(int).isin(POST_HB_ORIGINS)
    ].copy()
    post_nat = common.loc[
        common["forecast_origin"].astype(int).isin(POST_HB_ORIGINS)
    ].copy()
    if post_dep["forecast_origin"].astype(int).isin(HB_ORIGINS).any():
        raise RuntimeError("HB leaked into post-lock deployable")

    s_v2_d = score_frame(post_dep["actual"], post_dep["f_v2_policy"])
    s_hy_d = score_frame(post_dep["actual"], post_dep["f_a0_hybrid"])
    s_v2_n = score_frame(post_nat["actual"], post_nat["f_v2_policy"])
    s_hy_n = score_frame(post_nat["actual"], post_nat["f_a0_hybrid"])
    lift_d = float(s_v2_d["portfolio_wmape_pct"] - s_hy_d["portfolio_wmape_pct"])
    lift_n = float(s_v2_n["portfolio_wmape_pct"] - s_hy_n["portfolio_wmape_pct"])
    cov = float((post_dep["fallback_category"] == "both_native").mean()) if len(post_dep) else 0.0
    hz = horizon_group_wmape_lifts(post_dep, HORIZON_GROUPS)
    infl = influential_skus(post_dep, k=10)
    infl.to_csv(lock_dir / "postlock_influential_skus.csv", index=False)

    status_label = deployment_stability_status(
        lift_deployable=lift_d,
        lift_common=lift_n,
        coverage_both_native=cov,
        rmse_hybrid=float(s_hy_d["rmse"]),
        rmse_v2=float(s_v2_d["rmse"]),
    )

    report = {
        "origins_scored": sorted(POST_HB_ORIGINS),
        "hb_origins_excluded": sorted(HB_ORIGINS),
        "hb_outcomes_inspected": False,
        "scientific_lock_validated": True,
        "production_contract_validated": True,
        "frozen_choice": contract.get("frozen_choice"),
        "deployable_post_hb": {
            "n_rows": int(len(post_dep)),
            "n_product_origins": int(
                post_dep.groupby(["product", "forecast_origin"]).ngroups
            ),
            "v2": s_v2_d,
            "hybrid": s_hy_d,
            "lift_wmape_pp": lift_d,
            "both_native_coverage": cov,
            "fallback_counts": post_dep.groupby("fallback_category")
            .size()
            .astype(int)
            .to_dict(),
            "horizon_group_lifts": hz,
        },
        "common_native_post_hb": {
            "n_rows": int(len(post_nat)),
            "n_product_origins": int(
                post_nat.groupby(["product", "forecast_origin"]).ngroups
            ),
            "v2": s_v2_n,
            "hybrid": s_hy_n,
            "lift_wmape_pp": lift_n,
        },
        "deployment_stability_status": status_label,
        "lock_mutated": False,
        "contract_mutated": False,
    }
    out = lock_dir / "postlock_140401_140402.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if lock_path.read_bytes() != lock_bytes:
        raise SystemExit("Scientific lock mutated during post-lock — FAIL")
    if contract_path.read_bytes() != contract_bytes:
        raise SystemExit("Production contract mutated during post-lock — FAIL")

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
