"""Strict product×origin completeness for architecture panels."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import (
    A0_SEEDS,
    HB_ORIGINS,
    POST_HB_ORIGINS,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)


def _filter_non_hb(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "forecast_origin" not in df.columns:
        return df.copy()
    return df.loc[
        ~pd.to_numeric(df["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))
    ].copy()


REQUIRED_HORIZONS = frozenset(range(1, 16))


def origin_role(origin: int) -> str:
    o = int(origin)
    if o in HB_ORIGINS:
        return "human_budget_final_evaluation"
    if o in RESIDUAL_DEVELOPMENT_ORIGINS:
        return "residual_development"
    if o in POST_HB_ORIGINS:
        return "post_hb_ts_only_validation"
    return "other"


def expected_targets(origin: int) -> dict[int, int]:
    """horizon -> target_date for production H=15."""
    o = int(origin)
    return {h: shamsi_add_months(o, h - 1) for h in range(1, 16)}


def _native_mask(df: pd.DataFrame) -> pd.Series:
    model = df["model"].astype(str) if "model" in df.columns else pd.Series("", index=df.index)
    mask = model != "naive_last_value"
    if "job_status" in df.columns:
        mask = mask & (df["job_status"].astype(str) == "SUCCESS")
    return mask


def _horizons_present(g: pd.DataFrame) -> set[int]:
    return {int(h) for h in pd.to_numeric(g["horizon"], errors="coerce").dropna().tolist()}


def audit_product_origin_completeness(
    *,
    panel_v21: pd.DataFrame,
    panel_a0: pd.DataFrame,
    dense: pd.DataFrame,
    original55: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (status_df, incomplete_csv_df).

    status_df has one row per product×origin with completeness flags.
    incomplete_csv_df lists only incomplete / excluded groups with reasons.
    """
    v21 = _filter_non_hb(panel_v21)
    a0 = _filter_non_hb(panel_a0)
    v21 = v21.loc[_native_mask(v21)].copy()
    a0 = a0.loc[_native_mask(a0)].copy()
    a0["seed"] = pd.to_numeric(a0["seed"], errors="coerce")

    pairs = set()
    if not v21.empty:
        pairs |= {
            (str(p), int(o))
            for p, o in zip(v21["product"], v21["forecast_origin"])
        }
    if not a0.empty:
        pairs |= {
            (str(p), int(o))
            for p, o in zip(a0["product"], a0["forecast_origin"])
        }

    dense_keys = {
        (str(r.product), int(r.date)) for r in dense.itertuples(index=False)
    }
    status_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []

    for product, origin in sorted(pairs):
        role = origin_role(origin)
        cohort = "original_55" if product in original55 else "previously_unseen"
        exp = expected_targets(origin)
        exp_h = set(exp.keys())
        exp_td = set(exp.values())

        # Dense actuals for all 15 targets
        missing_actual_tds = []
        for td in sorted(exp_td):
            if (product, int(td)) not in dense_keys:
                missing_actual_tds.append(int(td))
        dense_complete = len(missing_actual_tds) == 0

        # V2 native completeness
        g_v = v21.loc[
            (v21["product"].astype(str) == product)
            & (v21["forecast_origin"].astype(int) == origin)
        ]
        v_hs = _horizons_present(g_v)
        missing_v_h = sorted(exp_h - v_hs)
        # target dates present for those horizons
        v_td = set()
        h1_ok = False
        if not g_v.empty:
            v_td = {int(t) for t in pd.to_numeric(g_v["target_date"], errors="coerce").dropna()}
            h1 = g_v.loc[g_v["horizon"].astype(int) == 1]
            if not h1.empty:
                h1_ok = int(h1["target_date"].iloc[0]) == int(origin)
        missing_v_td = sorted(exp_td - v_td)
        # finite forecasts for all 15
        v_finite_15 = False
        if len(v_hs) == 15 and not missing_v_h:
            per_h = g_v.groupby(g_v["horizon"].astype(int))["forecast"].first()
            v_finite_15 = len(per_h) == 15 and bool(per_h.notna().all())
        v21_complete = (
            missing_v_h == []
            and missing_v_td == []
            and h1_ok
            and v_finite_15
            and v_hs == exp_h
        )

        # A0 per-seed completeness then triple
        a0_seed_complete: dict[int, bool] = {}
        a0_missing: dict[int, dict[str, Any]] = {}
        g_a = a0.loc[
            (a0["product"].astype(str) == product)
            & (a0["forecast_origin"].astype(int) == origin)
            & (a0["seed"].isin(list(A0_SEEDS)))
        ]
        for seed in sorted(A0_SEEDS):
            gs = g_a.loc[g_a["seed"].astype(int) == seed]
            hs = _horizons_present(gs)
            miss_h = sorted(exp_h - hs)
            tds = (
                {int(t) for t in pd.to_numeric(gs["target_date"], errors="coerce").dropna()}
                if not gs.empty
                else set()
            )
            miss_td = sorted(exp_td - tds)
            h1_ok_s = False
            if not gs.empty:
                h1s = gs.loc[gs["horizon"].astype(int) == 1]
                if not h1s.empty:
                    h1_ok_s = int(h1s["target_date"].iloc[0]) == int(origin)
            finite = False
            if hs == exp_h:
                per = gs.groupby(gs["horizon"].astype(int))["forecast"].first()
                finite = len(per) == 15 and bool(per.notna().all())
            ok = miss_h == [] and miss_td == [] and h1_ok_s and finite
            a0_seed_complete[seed] = ok
            if not ok:
                a0_missing[seed] = {
                    "missing_horizons": miss_h,
                    "missing_target_dates": miss_td,
                    "h1_eq_origin": h1_ok_s,
                }
        a0_complete = all(a0_seed_complete.values()) and len(a0_seed_complete) == 3

        # Joint availability (both natives complete + dense)
        joint_complete = bool(v21_complete and a0_complete and dense_complete)

        # Deployable-complete: at least one native complete + dense + full 15 policy
        # (policy can fill naive for missing engine)
        any_native = v21_complete or a0_complete
        deployable_complete = bool(any_native and dense_complete)

        status_rows.append(
            {
                "product": product,
                "forecast_origin": origin,
                "cohort": cohort,
                "origin_role": role,
                "v21_complete": v21_complete,
                "a0_complete": a0_complete,
                "dense_actuals_complete": dense_complete,
                "joint_both_native_complete": joint_complete,
                "deployable_complete": deployable_complete,
                "fallback_if_deployable": (
                    "both_native"
                    if v21_complete and a0_complete
                    else "v21_only"
                    if v21_complete and not a0_complete
                    else "a0_only"
                    if a0_complete and not v21_complete
                    else "naive_fallback"
                ),
            }
        )

        # Incomplete reporting (any engine-side or dense failure for pairs we saw)
        if not deployable_complete or (not joint_complete and (v21_complete or a0_complete or not dense_complete)):
            # Always record incomplete aspects
            reasons = []
            if not dense_complete:
                reasons.append("missing_dense_actuals")
            if not v21_complete and not a0_complete:
                reasons.append("no_complete_native_engine")
            if not v21_complete and len(g_v) > 0:
                reasons.append("v21_partial")
            if not a0_complete and len(g_a) > 0:
                reasons.append("a0_partial")
            if not v21_complete:
                incomplete_rows.append(
                    {
                        "product": product,
                        "forecast_origin": origin,
                        "cohort": cohort,
                        "origin_role": role,
                        "engine": "v2.1",
                        "missing_horizons": ",".join(str(x) for x in missing_v_h),
                        "missing_target_dates": ",".join(str(x) for x in missing_v_td),
                        "exclusion_reason": (
                            "incomplete_v21;"
                            + ("h1_ne_origin;" if not h1_ok and len(g_v) else "")
                            + ";".join(reasons)
                        ),
                    }
                )
            if not a0_complete:
                for seed, meta in a0_missing.items():
                    incomplete_rows.append(
                        {
                            "product": product,
                            "forecast_origin": origin,
                            "cohort": cohort,
                            "origin_role": role,
                            "engine": f"a0_seed_{seed}",
                            "missing_horizons": ",".join(
                                str(x) for x in meta["missing_horizons"]
                            ),
                            "missing_target_dates": ",".join(
                                str(x) for x in meta["missing_target_dates"]
                            ),
                            "exclusion_reason": "incomplete_a0_seed;"
                            + ("h1_ne_origin;" if not meta["h1_eq_origin"] else "")
                            + ";".join(reasons),
                        }
                    )
            if not dense_complete and v21_complete and a0_complete:
                incomplete_rows.append(
                    {
                        "product": product,
                        "forecast_origin": origin,
                        "cohort": cohort,
                        "origin_role": role,
                        "engine": "dense_actuals",
                        "missing_horizons": "",
                        "missing_target_dates": ",".join(str(x) for x in missing_actual_tds),
                        "exclusion_reason": "missing_dense_actuals",
                    }
                )

    status = pd.DataFrame(status_rows)
    incomplete = pd.DataFrame(incomplete_rows)
    return status, incomplete


def assert_rows_multiple_of_15(df: pd.DataFrame, label: str) -> dict[str, Any]:
    n = int(len(df))
    n_po = (
        int(df.groupby(["product", "forecast_origin"]).ngroups) if n else 0
    )
    ok = (n == n_po * 15) and (n % 15 == 0)
    if not ok:
        raise AssertionError(
            f"{label}: n_rows={n} n_product_origins={n_po} "
            f"expected n_rows == n_product_origins*15"
        )
    return {
        "label": label,
        "n_rows": n,
        "n_product_origins": n_po,
        "n_product_origins_x_15": n_po * 15,
        "reconciled": True,
    }
