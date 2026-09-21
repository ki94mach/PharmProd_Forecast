"""Post-run OOF evaluation against locked residual promotion gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkg.portfolio_ts_lock.bootstrap import drop_top_k_hybrid_gainers, sku_cluster_bootstrap
from pkg.portfolio_ts_lock.constants import HORIZON_GROUPS
from pkg.portfolio_ts_lock.freeze import horizon_group_wmape_lifts, score_frame, sku_origin_wins
from pkg.research.portfolio_residual_xgb_v1.methods import (
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
)
from pkg.research.portfolio_residual_xgb_v1.promotion_gates import PROMOTION_GATES


def _score_method(df: pd.DataFrame, col: str) -> dict[str, float]:
    m = score_frame(df["actual"], df[col])
    err = df[col].astype(float) - df["actual"].astype(float)
    ae = np.abs(err.to_numpy(dtype=float))
    m["p99_ae"] = float(np.quantile(ae, 0.99)) if len(ae) else float("nan")
    m["max_ae"] = float(np.nanmax(ae)) if len(ae) else float("nan")
    return m


def evaluate_oof_predictions(
    pred_paths: list[Path],
    *,
    out_path: Path,
    gates_path: Path | None = None,
) -> dict[str, Any]:
    frames = [pd.read_parquet(p) for p in pred_paths]
    pooled = pd.concat(frames, ignore_index=True)
    # Bootstrap/horizon helpers expect f_v2_policy / f_a0_hybrid naming.
    # OOF already has anchor column f_a0_hybrid - assign explicitly so we do not
    # create duplicate column names via rename (which yields 2-D selections).
    boot_df = pooled.copy()
    boot_df["f_v2_policy"] = pooled[METHOD_FROZEN_TS].astype(float)
    boot_df["f_a0_hybrid"] = pooled[METHOD_XGB].astype(float)

    scores = {
        METHOD_FROZEN_TS: _score_method(pooled, METHOD_FROZEN_TS),
        METHOD_MEDIAN: _score_method(pooled, METHOD_MEDIAN),
        METHOD_XGB: _score_method(pooled, METHOD_XGB),
    }
    lift_xgb = float(
        scores[METHOD_FROZEN_TS]["portfolio_wmape_pct"]
        - scores[METHOD_XGB]["portfolio_wmape_pct"]
    )

    by_origin = []
    for o, g in pooled.groupby(pooled["forecast_origin"].astype(int)):
        s_ts = _score_method(g, METHOD_FROZEN_TS)
        s_x = _score_method(g, METHOD_XGB)
        by_origin.append(
            {
                "forecast_origin": int(o),
                "lift_wmape_pp": float(
                    s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
                ),
                "frozen_ts_wmape": s_ts["portfolio_wmape_pct"],
                "xgb_wmape": s_x["portfolio_wmape_pct"],
            }
        )
    n_improved = sum(1 for r in by_origin if r["lift_wmape_pp"] > 0)

    hz = horizon_group_wmape_lifts(
        boot_df,
        HORIZON_GROUPS,
        v2_col="f_v2_policy",
        hy_col="f_a0_hybrid",
    )
    wins = sku_origin_wins(boot_df, v2_col="f_v2_policy", hy_col="f_a0_hybrid")
    drop3 = drop_top_k_hybrid_gainers(boot_df, k=3)
    boot = sku_cluster_bootstrap(boot_df, B=int(PROMOTION_GATES["bootstrap_B"]), seed=42)

    ts = scores[METHOD_FROZEN_TS]
    xg = scores[METHOD_XGB]
    gates_result = {
        "pooled_oof_wmape_improvement_over_frozen_ts": bool(lift_xgb > 0),
        "origins_improved_ge_4": bool(n_improved >= 4),
        "drop_top3_lift_positive": bool(drop3.get("lift_positive")),
        "rmse_not_more_than_2pct_worse": bool(
            xg["rmse"] <= ts["rmse"] * 1.02
        ),
        "p95_not_more_than_2pct_worse": bool(
            xg["p95_ae"] <= ts["p95_ae"] * 1.02
        ),
        "p99_not_more_than_2pct_worse": bool(
            xg["p99_ae"] <= ts["p99_ae"] * 1.02
        ),
        "no_horizon_group_regression_gt_2pp": bool(
            all(v == v and v >= -2.0 for v in hz.values())
        ),
        "sku_cluster_bootstrap_frac_positive_ge_90": bool(
            boot["frac_positive"] >= 0.90
        ),
    }
    all_pass = all(gates_result.values())

    worst = (
        pooled.loc[pooled["xgb_magnifies_ts_error"]]
        .sort_values("ae_delta_xgb_minus_ts", ascending=False)
        .head(20)
    )

    report = {
        "hb_outcomes_inspected": False,
        "n_rows": int(len(pooled)),
        "scores": scores,
        "lift_wmape_pp_xgb_vs_ts": lift_xgb,
        "by_origin": by_origin,
        "n_origins_improved": int(n_improved),
        "horizon_group_lifts": hz,
        "sku_origin_wins": wins,
        "drop_top3": drop3,
        "bootstrap": boot,
        "gates_result": gates_result,
        "promotion_gates_pass": bool(all_pass),
        "worst_xgb_magnification": worst.to_dict(orient="records"),
        "gates_path": str(gates_path) if gates_path else None,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(out_path)
    return report
