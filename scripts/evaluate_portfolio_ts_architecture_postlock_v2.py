#!/usr/bin/env python3
"""Post-lock TS stability v2 on 140401/140402 only.

Writes postlock_140401_140402_v2.json. Never rewrites scientific lock,
production contract, or postlock v1. Never uses Human Budget origins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.universes import load_universe_product_names  # noqa: E402
from pkg.portfolio_ts_lock.bootstrap import (  # noqa: E402
    drop_top_k_hybrid_gainers,
    leave_one_origin,
)
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
    sku_origin_wins,
)
from pkg.portfolio_ts_lock.lock_io import (  # noqa: E402
    assert_lock_immutable_read,
    refuse_lock_mutation,
)
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb  # noqa: E402
from pkg.portfolio_ts_lock.postlock_stability import (  # noqa: E402
    deployment_stability_status_v2,
)
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


def _rel_improvement(lift_pp: float, v2_wmape: float) -> float:
    if not (v2_wmape == v2_wmape) or v2_wmape == 0:
        return float("nan")
    return float(lift_pp / v2_wmape)


def _pct_rmse_change(rmse_hy: float, rmse_v2: float) -> float:
    if not (rmse_v2 == rmse_v2) or rmse_v2 <= 0:
        return float("nan")
    if not (rmse_hy == rmse_hy):
        return float("nan")
    return float((rmse_hy / rmse_v2 - 1.0) * 100.0)


def _panel_block(df: pd.DataFrame, *, include_coverage: bool) -> dict[str, Any]:
    s_v2 = score_frame(df["actual"], df["f_v2_policy"])
    s_hy = score_frame(df["actual"], df["f_a0_hybrid"])
    lift = float(s_v2["portfolio_wmape_pct"] - s_hy["portfolio_wmape_pct"])
    hz = horizon_group_wmape_lifts(df, HORIZON_GROUPS)
    wins = sku_origin_wins(df)
    drop3 = drop_top_k_hybrid_gainers(df, k=3)
    loo = leave_one_origin(df)
    loo_records = loo.to_dict(orient="records") if not loo.empty else []
    block: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_product_origins": int(df.groupby(["product", "forecast_origin"]).ngroups),
        "v2": s_v2,
        "hybrid": s_hy,
        "lift_wmape_pp": lift,
        "relative_wmape_improvement": _rel_improvement(
            lift, float(s_v2["portfolio_wmape_pct"])
        ),
        "pct_rmse_change": _pct_rmse_change(float(s_hy["rmse"]), float(s_v2["rmse"])),
        "horizon_group_lifts": hz,
        "sku_origin_wins": wins,
        "drop_top3": drop3,
        "leave_one_origin": loo_records,
    }
    if include_coverage:
        cov = (
            float((df["fallback_category"] == "both_native").mean()) if len(df) else 0.0
        )
        block["both_native_coverage"] = cov
        block["fallback_counts"] = (
            df.groupby("fallback_category").size().astype(int).to_dict()
        )
    return block


def influential_skus(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = []
    for product, g in df.groupby("product", sort=False):
        ae_v2 = float((g["f_v2_policy"] - g["actual"]).abs().sum())
        ae_hy = float((g["f_a0_hybrid"] - g["actual"]).abs().sum())
        rows.append(
            {
                "product": str(product),
                "ae_reduction_v2_minus_hybrid": ae_v2 - ae_hy,
                "sum_ae_v2": ae_v2,
                "sum_ae_hybrid": ae_hy,
            }
        )
    out = pd.DataFrame(rows).sort_values("ae_reduction_v2_minus_hybrid")
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

    dep_block = _panel_block(post_dep, include_coverage=True)
    nat_block = _panel_block(post_nat, include_coverage=False)

    loo_lifts = [float(r["lift_wmape_pp"]) for r in dep_block["leave_one_origin"]]
    clf = deployment_stability_status_v2(
        deployable_wmape_lift=float(dep_block["lift_wmape_pp"]),
        rmse_hybrid=float(dep_block["hybrid"]["rmse"]),
        rmse_v2=float(dep_block["v2"]["rmse"]),
        p95_hybrid=float(dep_block["hybrid"]["p95_ae"]),
        p95_v2=float(dep_block["v2"]["p95_ae"]),
        horizon_group_lifts=dep_block["horizon_group_lifts"],
        leave_one_origin_lifts=loo_lifts,
        drop_top3_lift_wmape_pp=float(dep_block["drop_top3"]["lift_wmape_pp"]),
    )

    infl = influential_skus(post_dep, k=10)
    infl.to_csv(lock_dir / "postlock_influential_skus_v2.csv", index=False)
    loo_df = pd.DataFrame(dep_block["leave_one_origin"])
    if not loo_df.empty:
        loo_df.to_csv(lock_dir / "postlock_leave_one_origin_v2.csv", index=False)

    report = {
        "artifact_version": "postlock_140401_140402_v2",
        "origins_scored": sorted(POST_HB_ORIGINS),
        "hb_origins_excluded": sorted(HB_ORIGINS),
        "hb_outcomes_inspected": False,
        "scientific_lock_validated": True,
        "production_contract_validated": True,
        "frozen_choice": contract.get("frozen_choice"),
        "frozen_policy_unchanged": "A0_HYBRID_POLICY",
        "production_promotion": "BLOCKED_PENDING_FINAL_PIPELINE_VALIDATION",
        "deployable_post_hb": dep_block,
        "common_native_post_hb": nat_block,
        "deployment_stability_status": clf["deployment_stability_status"],
        "classifier_gates": clf["classifier_gates"],
        "n_horizon_groups_lift_ge_0": clf["n_horizon_groups_lift_ge_0"],
        "lock_sha256_unchanged": file_sha256(lock_path),
        "contract_sha256_unchanged": file_sha256(contract_path),
        "lock_mutated": False,
        "contract_mutated": False,
        "scientific_lock_choice": lock.get("choice"),
    }
    out = lock_dir / "postlock_140401_140402_v2.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if lock_path.read_bytes() != lock_bytes:
        raise SystemExit("Scientific lock mutated during post-lock v2 — FAIL")
    if contract_path.read_bytes() != contract_bytes:
        raise SystemExit("Production contract mutated during post-lock v2 — FAIL")

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
