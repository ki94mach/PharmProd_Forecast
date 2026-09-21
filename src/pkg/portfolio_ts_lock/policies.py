"""Policy panel construction for V2 / A0 / hybrid (complete product×origin only)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.portfolio_ts_lock.completeness import expected_targets
from pkg.portfolio_ts_lock.constants import (
    A0_SEEDS,
    FALLBACK_A0_ONLY,
    FALLBACK_BOTH_NATIVE,
    FALLBACK_NAIVE,
    FALLBACK_V21_ONLY,
    HB_ORIGINS,
    KEYS,
)
from pkg.portfolio_ts_lock.dense_actuals import last_train_value


def _assert_no_hb(df: pd.DataFrame, where: str) -> None:
    if "forecast_origin" not in df.columns or df.empty:
        return
    origins = pd.to_numeric(df["forecast_origin"], errors="coerce")
    bad = origins.isin(list(HB_ORIGINS))
    if bool(bad.any()):
        raise ValueError(
            f"HB origins leaked into {where}: "
            f"{sorted(origins.loc[bad].dropna().astype(int).unique().tolist())}"
        )


def filter_non_hb(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "forecast_origin" not in df.columns:
        return df.copy()
    return df.loc[
        ~pd.to_numeric(df["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))
    ].copy()


def _native_mask(df: pd.DataFrame) -> pd.Series:
    model = df["model"].astype(str) if "model" in df.columns else pd.Series("", index=df.index)
    mask = model != "naive_last_value"
    if "job_status" in df.columns:
        mask = mask & (df["job_status"].astype(str) == "SUCCESS")
    return mask


def _v21_forecast_map(panel_v21: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    p = filter_non_hb(panel_v21)
    p = p.loc[_native_mask(p)]
    out: dict[tuple[str, int, int], float] = {}
    for _, r in p.iterrows():
        key = (str(r["product"]), int(r["forecast_origin"]), int(r["horizon"]))
        if key not in out and pd.notna(r["forecast"]):
            out[key] = float(r["forecast"])
    return out


def _a0_seed_mean_map(panel_a0: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    """Only emit means for horizons where all three seeds present (caller gates PO)."""
    p = filter_non_hb(panel_a0)
    p = p.loc[_native_mask(p)].copy()
    p["seed"] = pd.to_numeric(p["seed"], errors="coerce")
    p = p.loc[p["seed"].isin(list(A0_SEEDS))]
    out: dict[tuple[str, int, int], float] = {}
    for (product, origin, horizon), g in p.groupby(
        ["product", "forecast_origin", "horizon"], sort=False
    ):
        seeds = {int(s) for s in g["seed"].dropna().tolist()}
        if seeds != set(A0_SEEDS):
            continue
        per = g.groupby("seed")["forecast"].first()
        if set(int(s) for s in per.index) != set(A0_SEEDS):
            continue
        if per.isna().any():
            continue
        out[(str(product), int(origin), int(horizon))] = float(per.mean())
    return out


def build_policy_panels(
    *,
    panel_v21: pd.DataFrame,
    panel_a0: pd.DataFrame,
    dense: pd.DataFrame,
    completeness_status: pd.DataFrame,
    eligible_keys: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (common_native, deployable, fallback_category_counts, partial_excluded).

    Decision panels include only complete product×origins (15 horizons + actuals).
    Partial groups are excluded from primary decision panels.
    """
    _ = eligible_keys  # reserved
    st = completeness_status.copy()
    if st.empty:
        empty = pd.DataFrame(columns=list(KEYS))
        return empty, empty, empty, empty

    deployable_pos = st.loc[st["deployable_complete"]].copy()
    joint_pos = st.loc[st["joint_both_native_complete"]].copy()
    partial = st.loc[~st["deployable_complete"]].copy()

    v21_map = _v21_forecast_map(panel_v21)
    a0_map = _a0_seed_mean_map(panel_a0)
    dense_map = {
        (str(r.product), int(r.date)): float(r.actual)
        for r in dense.itertuples(index=False)
    }

    def _expand(pos: pd.DataFrame, *, require_joint: bool) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in pos.itertuples(index=False):
            product = str(r.product)
            origin = int(r.forecast_origin)
            v21_ok = bool(r.v21_complete)
            a0_ok = bool(r.a0_complete)
            if require_joint and not (v21_ok and a0_ok):
                continue
            if not require_joint and not (v21_ok or a0_ok):
                continue
            exp = expected_targets(origin)
            naive = last_train_value(dense, product, origin)
            for h, td in exp.items():
                actual = dense_map.get((product, int(td)))
                if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                    raise RuntimeError(
                        f"deployable_complete but missing dense actual "
                        f"{product}@{origin} td={td}"
                    )
                f_v21 = v21_map.get((product, origin, h)) if v21_ok else None
                f_a0 = a0_map.get((product, origin, h)) if a0_ok else None
                if v21_ok and f_v21 is None:
                    raise RuntimeError(f"v21_complete missing forecast {product}@{origin} h={h}")
                if a0_ok and f_a0 is None:
                    raise RuntimeError(f"a0_complete missing seed-mean {product}@{origin} h={h}")

                if v21_ok and a0_ok:
                    cat = FALLBACK_BOTH_NATIVE
                    f_v2_policy = float(f_v21)
                    f_hybrid = 0.5 * float(f_v21) + 0.5 * float(f_a0)
                    f_a0_only = float(f_a0)
                elif v21_ok:
                    cat = FALLBACK_V21_ONLY
                    f_v2_policy = float(f_v21)
                    f_hybrid = float(f_v21)
                    f_a0_only = float("nan")
                elif a0_ok:
                    cat = FALLBACK_A0_ONLY
                    # V2_POLICY uses production naive; hybrid = A0
                    f_v2_policy = float(naive)
                    f_hybrid = float(f_a0)
                    f_a0_only = float(f_a0)
                else:
                    cat = FALLBACK_NAIVE
                    f_v2_policy = float(naive)
                    f_hybrid = float(naive)
                    f_a0_only = float("nan")

                rows.append(
                    {
                        "product": product,
                        "forecast_origin": origin,
                        "horizon": int(h),
                        "target_date": int(td),
                        "actual": float(actual),
                        "cohort": r.cohort,
                        "origin_role": r.origin_role,
                        "v21_ok": v21_ok,
                        "a0_ok": a0_ok,
                        "f_v21": float(f_v21) if f_v21 is not None else float("nan"),
                        "f_a0": float(f_a0) if f_a0 is not None else float("nan"),
                        "f_naive": float(naive),
                        "f_v2_policy": float(f_v2_policy),
                        "f_a0_only": f_a0_only,
                        "f_a0_hybrid": float(f_hybrid),
                        "fallback_category": cat,
                        "joint_availability": bool(v21_ok and a0_ok),
                    }
                )
        return pd.DataFrame(rows)

    deployable = _expand(deployable_pos, require_joint=False)
    common = _expand(joint_pos, require_joint=True)
    _assert_no_hb(deployable, "deployable")
    _assert_no_hb(common, "common_native")

    # Semantic asserts for a0_only
    if not deployable.empty:
        a0_only = deployable.loc[deployable["fallback_category"] == FALLBACK_A0_ONLY]
        if not a0_only.empty:
            if not np.allclose(
                a0_only["f_a0_hybrid"].to_numpy(dtype=float),
                a0_only["f_a0"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise AssertionError("A0-only: hybrid must equal A0")
            if not np.allclose(
                a0_only["f_v2_policy"].to_numpy(dtype=float),
                a0_only["f_naive"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise AssertionError("A0-only: V2_POLICY must equal production naive")

    if deployable.empty:
        cat_counts = pd.DataFrame(columns=["fallback_category", "n_rows", "n_product_origins"])
    else:
        cat_counts = (
            deployable.groupby("fallback_category", as_index=False)
            .agg(
                n_rows=("horizon", "size"),
                n_product_origins=("product", lambda s: deployable.loc[s.index]
                .groupby(["product", "forecast_origin"])
                .ngroups),
            )
        )
        # Fix n_product_origins properly
        rows = []
        for cat, g in deployable.groupby("fallback_category"):
            n_po = int(g.groupby(["product", "forecast_origin"]).ngroups)
            rows.append(
                {
                    "fallback_category": cat,
                    "n_rows": int(len(g)),
                    "n_product_origins": n_po,
                    "n_product_origins_x_15": n_po * 15,
                    "reconciled": int(len(g)) == n_po * 15,
                }
            )
        cat_counts = pd.DataFrame(rows)

    return common, deployable, cat_counts, partial
