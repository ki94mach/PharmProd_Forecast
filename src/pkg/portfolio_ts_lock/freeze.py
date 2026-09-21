"""Strengthened freeze rule for portfolio TS architecture."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct
from pkg.portfolio_ts_lock.constants import (
    CHOICE_HYBRID,
    CHOICE_INSUFFICIENT,
    CHOICE_V2,
    POLICY_HYBRID,
    POLICY_V2,
    TIE_AE_EPS,
)


def score_frame(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    m = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    if m.empty:
        return {
            "n_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "portfolio_wmape_pct": np.nan,
            "bias": np.nan,
            "p90_ae": np.nan,
            "p95_ae": np.nan,
        }
    err = m["forecast"].to_numpy(dtype=float) - m["actual"].to_numpy(dtype=float)
    ae = np.abs(err)
    return {
        "n_rows": int(len(m)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "p95_ae": float(np.quantile(ae, 0.95)),
    }


def sku_origin_wins(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual",
    v2_col: str = "f_v2_policy",
    hy_col: str = "f_a0_hybrid",
) -> dict[str, Any]:
    """Count SKU×origin wins; ties when |AE_h - AE_v2| < TIE_AE_EPS on mean AE."""
    rows = []
    for (product, origin), g in df.groupby(["product", "forecast_origin"], sort=False):
        m = g[[actual_col, v2_col, hy_col]].dropna()
        if m.empty:
            continue
        ae_v2 = float(np.mean(np.abs(m[v2_col] - m[actual_col])))
        ae_hy = float(np.mean(np.abs(m[hy_col] - m[actual_col])))
        delta = ae_hy - ae_v2
        if abs(delta) < TIE_AE_EPS:
            winner = "tie"
        elif delta < 0:
            winner = "hybrid"
        else:
            winner = "v2"
        rows.append(winner)
    hybrid_wins = sum(1 for w in rows if w == "hybrid")
    v2_wins = sum(1 for w in rows if w == "v2")
    ties = sum(1 for w in rows if w == "tie")
    return {
        "hybrid_wins": int(hybrid_wins),
        "v2_wins": int(v2_wins),
        "ties": int(ties),
        "n_sku_origins": int(len(rows)),
    }


def make_recommendation(
    *,
    lift_deployable_wmape: float,
    lift_common_native_wmape: float,
    wins: dict[str, Any],
    bootstrap_frac_positive: float,
    drop_top3_lift_positive: bool,
    rmse_hybrid: float,
    rmse_v2: float,
    p95_hybrid: float,
    p95_v2: float,
    horizon_group_lifts: dict[str, float],
    unexplained_a0_defect: bool,
    pooled_deployable_lift_wmape: float | None = None,
    original55_volume_share: float | None = None,
    original55_ae_reduction: float | None = None,
    unseen_ae_reduction: float | None = None,
) -> dict[str, Any]:
    """Freeze hybrid only if unseen primary gates pass and pooled deployable not worse.

    If primary (unseen) wins but pooled is worse because original_55 dominate volume,
    return INSUFFICIENT_EVIDENCE for a single universal policy (no cohort-named rule).
    """
    bullets: list[str] = []

    if unexplained_a0_defect:
        return {
            "choice": CHOICE_INSUFFICIENT,
            "frozen_policy": None,
            "rationale_bullets": [
                "Unresolved A0 coverage or empty-prediction defect → INSUFFICIENT_EVIDENCE."
            ],
            "gates": {"unexplained_a0_defect": True},
        }

    gates = {
        "lift_deployable_positive": bool(lift_deployable_wmape > 0),
        "lift_common_native_positive": bool(lift_common_native_wmape > 0),
        "hybrid_wins_exceed_v2": int(wins.get("hybrid_wins", 0))
        > int(wins.get("v2_wins", 0)),
        "bootstrap_frac_ge_90": bool(bootstrap_frac_positive >= 0.90),
        "drop_top3_lift_positive": bool(drop_top3_lift_positive),
        "rmse_not_2pct_worse": bool(
            np.isnan(rmse_hybrid)
            or np.isnan(rmse_v2)
            or rmse_hybrid <= rmse_v2 * 1.02
        ),
        "p95_not_2pct_worse": bool(
            np.isnan(p95_hybrid)
            or np.isnan(p95_v2)
            or p95_hybrid <= p95_v2 * 1.02
        ),
        "no_horizon_group_regresses_gt_2": all(
            (np.isnan(v) or v >= -2.0) for v in horizon_group_lifts.values()
        ),
    }
    unseen_gates_pass = all(gates.values())
    pooled = (
        float(pooled_deployable_lift_wmape)
        if pooled_deployable_lift_wmape is not None
        else float("nan")
    )
    gates["pooled_deployable_not_worse"] = bool(np.isnan(pooled) or pooled >= 0)

    bullets.append(
        f"Unseen primary deployable WMAPE lift (V2−hybrid) = {lift_deployable_wmape:+.4g} pp."
    )
    bullets.append(
        f"Unseen primary common-native WMAPE lift (V2−hybrid) = {lift_common_native_wmape:+.4g} pp."
    )
    bullets.append(f"Pooled residual deployable WMAPE lift = {pooled:+.4g} pp.")
    bullets.append(
        f"SKU×origin wins hybrid/v2/ties = {wins.get('hybrid_wins')}/"
        f"{wins.get('v2_wins')}/{wins.get('ties')}."
    )
    bullets.append(f"Bootstrap P(lift>0) = {bootstrap_frac_positive:.2%}.")
    bullets.append(f"Drop-top-3 lift positive = {drop_top3_lift_positive}.")
    if original55_volume_share is not None:
        bullets.append(
            f"Original-55 share of residual deployable |actual| volume = "
            f"{float(original55_volume_share):.2%}."
        )
    bullets.append(f"Unseen gates: {gates}")

    conflict_55 = (
        unseen_gates_pass
        and pooled < 0
        and original55_volume_share is not None
        and float(original55_volume_share) >= 0.50
        and original55_ae_reduction is not None
        and unseen_ae_reduction is not None
        and float(original55_ae_reduction) < 0
        and float(unseen_ae_reduction) > 0
    )

    if unseen_gates_pass and gates["pooled_deployable_not_worse"]:
        choice = CHOICE_HYBRID
        frozen = POLICY_HYBRID
        bullets.append(
            "Unseen primary gates passed and pooled deployable WMAPE not worse "
            "→ freeze A0_HYBRID_POLICY."
        )
    elif conflict_55:
        choice = CHOICE_INSUFFICIENT
        frozen = None
        bullets.append(
            "Unseen primary favors hybrid but pooled residual deployable is worse, "
            "driven by original_55 volume dominance with negative AE reduction while "
            "unseen gains — no cohort-named production rule; "
            "INSUFFICIENT_EVIDENCE for a single universal TS policy."
        )
    elif unseen_gates_pass and pooled < 0:
        choice = CHOICE_V2
        frozen = POLICY_V2
        bullets.append(
            "Unseen primary gates passed but pooled residual deployable WMAPE is worse "
            "→ conservative global freeze V2_POLICY."
        )
    else:
        choice = CHOICE_V2
        frozen = POLICY_V2
        failed = [k for k, v in gates.items() if not v and k != "pooled_deployable_not_worse"]
        bullets.append(
            "Unseen hybrid gates failed ("
            + ", ".join(failed)
            + ") → freeze V2_POLICY."
        )

    return {
        "choice": choice,
        "frozen_policy": frozen,
        "rationale_bullets": bullets,
        "gates": gates,
        "unseen_gates_pass": unseen_gates_pass,
        "pooled_deployable_lift_wmape_pp": pooled,
        "lift_deployable_wmape_pp": float(lift_deployable_wmape),
        "lift_common_native_wmape_pp": float(lift_common_native_wmape),
        "conflict_original55_volume": bool(conflict_55),
    }


def horizon_group_wmape_lifts(
    df: pd.DataFrame,
    horizon_groups: tuple[tuple[str, frozenset[int]], ...],
    *,
    actual_col: str = "actual",
    v2_col: str = "f_v2_policy",
    hy_col: str = "f_a0_hybrid",
) -> dict[str, float]:
    """Lift = WMAPE(V2) - WMAPE(hybrid); negative means hybrid worse."""
    out: dict[str, float] = {}
    for name, hs in horizon_groups:
        g = df.loc[df["horizon"].astype(int).isin(hs)]
        if g.empty:
            out[name] = float("nan")
            continue
        s_v2 = score_frame(g[actual_col], g[v2_col])
        s_hy = score_frame(g[actual_col], g[hy_col])
        out[name] = float(s_v2["portfolio_wmape_pct"] - s_hy["portfolio_wmape_pct"])
    return out
