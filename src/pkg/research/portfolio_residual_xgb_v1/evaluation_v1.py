"""Post-hoc residual-XGBoost v1.1 evaluation analysis (no refit, no residual lock).

Produces versioned artifacts under evaluation_v1/. Median correction is benchmark-only.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkg.portfolio_ts_lock.bootstrap import (
    drop_top_k_hybrid_gainers,
    hierarchical_origin_then_sku_bootstrap,
    leave_one_origin,
    leave_one_sku,
    sku_cluster_bootstrap,
)
from pkg.portfolio_ts_lock.constants import (
    HB_ORIGINS,
    HORIZON_GROUPS,
    POST_HB_ORIGINS,
)
from pkg.portfolio_ts_lock.freeze import (
    horizon_group_wmape_lifts,
    score_frame,
    sku_origin_wins,
)
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_residual_xgb_v1.methods import (
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    last_complete_month,
)
from pkg.research.portfolio_residual_xgb_v1.promotion_gates import PROMOTION_GATES
from pkg.research.portfolio_residual_xgb_v1.smoke_acceptance import PRED_KEYS

FAMILY_DIR_REL = Path("src/data/results/portfolio_residual_xgb_v1")
LOCK_DIR_REL = Path(
    "src/data/results/product_universe/portfolio_ts_architecture_lock_v1"
)
FULL_RUN_ID = "portfolio_residual_xgb_v1_1_full"
EVAL_DIR_NAME = "evaluation_v1"

EXPECTED_ORIGINS: tuple[int, ...] = (
    140107,
    140110,
    140201,
    140204,
    140207,
    140210,
    140301,
)
REQUIRED_HORIZONS: frozenset[int] = frozenset(range(1, 16))

ROUTER_WHITELIST: tuple[str, ...] = (
    "product",
    "forecast_origin",
    "train_n_months",
    "train_mean_sales",
    "train_zero_share",
)

METHOD_COLS: tuple[str, ...] = (METHOD_FROZEN_TS, METHOD_MEDIAN, METHOD_XGB)

REC_PROMOTE = "PROMOTE_TS_PLUS_RESIDUAL_XGB"
REC_KEEP = "KEEP_FROZEN_TS"
REC_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

CORR_EPS = 1e-9
# Prior VPS evaluate lift — reconciliation context only (never drives recommendation).
VPS_RECONCILE_LIFT_PP = -0.9394942800937258


class EvaluationOverwriteError(RuntimeError):
    """Raised when evaluation_v1/ exists and --force was not supplied."""


def decide_recommendation(
    *,
    integrity_ok: bool,
    non_computable_gates: list[str],
    promotion_gates_pass: bool,
) -> str | None:
    """Locked recommendation branch rule (no hard-coded outcomes)."""
    if not integrity_ok:
        return None
    if non_computable_gates:
        return REC_INSUFFICIENT
    if promotion_gates_pass:
        return REC_PROMOTE
    return REC_KEEP


def check_complete_horizons(df: pd.DataFrame) -> dict[str, Any]:
    """Every (product, forecast_origin) must have exactly horizons {1..15}."""
    incomplete: list[dict[str, Any]] = []
    for (product, origin), g in df.groupby(
        [df["product"].astype(str), df["forecast_origin"].astype(int)], sort=False
    ):
        hs = set(int(h) for h in g["horizon"].tolist())
        if hs != set(REQUIRED_HORIZONS):
            incomplete.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "horizons_present": sorted(hs),
                    "missing": sorted(set(REQUIRED_HORIZONS) - hs),
                    "extra": sorted(hs - set(REQUIRED_HORIZONS)),
                }
            )
    return {
        "ok": len(incomplete) == 0,
        "n_incomplete_groups": int(len(incomplete)),
        "incomplete_groups": incomplete[:50],
        "required_horizons": sorted(REQUIRED_HORIZONS),
    }


def _score_method(df: pd.DataFrame, col: str) -> dict[str, float]:
    m = score_frame(df["actual"], df[col])
    err = df[col].astype(float) - df["actual"].astype(float)
    ae = np.abs(err.to_numpy(dtype=float))
    m["p99_ae"] = float(np.quantile(ae, 0.99)) if len(ae) else float("nan")
    m["max_ae"] = float(np.nanmax(ae)) if len(ae) else float("nan")
    return m


def load_pooled_predictions(run_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for o in EXPECTED_ORIGINS:
        p = run_dir / f"oof_predictions_origin_{o}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"missing predictions: {p}")
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def join_router_features(
    panel: pd.DataFrame,
    router_csv: Path,
) -> pd.DataFrame:
    """Left-join whitelist-only router columns; refuse extra selected columns."""
    raw = pd.read_csv(router_csv)
    missing = [c for c in ROUTER_WHITELIST if c not in raw.columns]
    if missing:
        raise ValueError(f"router CSV missing whitelist columns: {missing}")
    slim = raw.loc[:, list(ROUTER_WHITELIST)].copy()
    leaked = [c for c in slim.columns if c not in ROUTER_WHITELIST]
    if leaked:
        raise ValueError(f"router join leaked non-whitelist columns: {leaked}")
    out = panel.merge(
        slim,
        on=["product", "forecast_origin"],
        how="left",
        validate="many_to_one",
    )
    extra = [c for c in out.columns if c in raw.columns and c not in ROUTER_WHITELIST and c not in panel.columns]
    if extra:
        raise ValueError(f"router join introduced non-whitelist columns: {extra}")
    return out


def assign_segments(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    n = out["train_n_months"]
    out["history_len_bin"] = np.where(
        n.isna(),
        "UNKNOWN",
        np.where(n <= 12, "hist_le_12", np.where(n <= 24, "hist_13_24", "hist_ge_25")),
    )
    z = out["train_zero_share"]
    out["sparsity_bin"] = np.where(
        z.isna(),
        "UNKNOWN",
        np.where(z < 0.3, "sparse_lt_0.3", np.where(z < 0.7, "sparse_0.3_0.7", "sparse_ge_0.7")),
    )
    mean = out["train_mean_sales"]
    valid = mean.dropna()
    if len(valid) >= 3:
        t1, t2 = float(valid.quantile(1 / 3)), float(valid.quantile(2 / 3))
        out["demand_intensity_bin"] = np.where(
            mean.isna(),
            "UNKNOWN",
            np.where(
                mean <= t1,
                "demand_low",
                np.where(mean <= t2, "demand_mid", "demand_high"),
            ),
        )
        out.attrs["demand_tertiles"] = {"t1": t1, "t2": t2}
    else:
        out["demand_intensity_bin"] = np.where(mean.isna(), "UNKNOWN", "demand_all")
    return out


def build_bootstrap_copy(panel: pd.DataFrame) -> pd.DataFrame:
    """Separate copy for lock bootstrap helpers; never mutates panel anchor."""
    if HYBRID_ANCHOR not in panel.columns:
        raise ValueError(f"panel missing anchor column {HYBRID_ANCHOR}")
    anchor_before = panel[HYBRID_ANCHOR].astype(float).to_numpy().copy()

    boot = panel.copy()
    boot["f_v2_policy"] = panel[METHOD_FROZEN_TS].astype(float)
    boot[HYBRID_ANCHOR] = panel[METHOD_XGB].astype(float)

    anchor_after = panel[HYBRID_ANCHOR].astype(float).to_numpy()
    if not np.allclose(anchor_before, anchor_after, equal_nan=True):
        raise AssertionError("panel f_a0_hybrid anchor was mutated")

    # Lift direction: positive lift <=> WMAPE(FROZEN_TS) - WMAPE(XGB) > 0
    s_ts = score_frame(panel["actual"], panel[METHOD_FROZEN_TS])
    s_x = score_frame(panel["actual"], panel[METHOD_XGB])
    lift_direct = float(
        s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
    )
    s_v2 = score_frame(boot["actual"], boot["f_v2_policy"])
    s_hy = score_frame(boot["actual"], boot[HYBRID_ANCHOR])
    lift_helper = float(s_v2["portfolio_wmape_pct"] - s_hy["portfolio_wmape_pct"])
    if not np.isclose(lift_direct, lift_helper, rtol=0, atol=1e-9):
        raise AssertionError(
            f"lift direction mismatch: direct={lift_direct} helper={lift_helper}"
        )
    boot.attrs["lift_wmape_pp_asserted"] = lift_direct
    boot.attrs["positive_lift_means"] = "WMAPE(FROZEN_TS)-WMAPE(XGB)>0"
    return boot


def add_correction_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    anchor = out[HYBRID_ANCHOR].astype(float)
    out["corr_xgb"] = out[METHOD_XGB].astype(float) - anchor
    out["corr_median"] = out[METHOD_MEDIAN].astype(float) - anchor
    denom_anchor = np.maximum(np.abs(anchor.to_numpy(dtype=float)), CORR_EPS)
    denom_actual = np.maximum(
        np.abs(out["actual"].astype(float).to_numpy(dtype=float)), CORR_EPS
    )
    out["corr_xgb_norm_anchor"] = np.abs(out["corr_xgb"].to_numpy(dtype=float)) / denom_anchor
    out["corr_xgb_norm_actual"] = np.abs(out["corr_xgb"].to_numpy(dtype=float)) / denom_actual
    out["corr_median_norm_anchor"] = (
        np.abs(out["corr_median"].to_numpy(dtype=float)) / denom_anchor
    )
    out["corr_median_norm_actual"] = (
        np.abs(out["corr_median"].to_numpy(dtype=float)) / denom_actual
    )
    if "f_v2_policy" not in out.columns:
        out["f_v2_policy"] = out[METHOD_FROZEN_TS].astype(float)
    return out


def run_integrity_checks(
    *,
    repo_root: Path,
    family_dir: Path,
    run_dir: Path,
    lock_dir: Path,
    pooled: pd.DataFrame | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    summary_path = family_dir / "prepare_summary_v1_1.json"
    contract_path = family_dir / "residual_study_contract_v1_1.json"
    gates_path = family_dir / "residual_promotion_gates_v1.json"
    lock_path = lock_dir / "ts_architecture_lock.json"
    prod_path = lock_dir / "ts_architecture_production_contract_v1.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}

    add(
        "prepare_summary_present",
        summary_path.is_file(),
        str(summary_path.as_posix()),
    )
    add(
        "contract_sha_matches_prepare",
        contract_path.is_file()
        and file_sha256(contract_path) == summary.get("residual_study_contract_v1_1_sha256"),
        {
            "live": file_sha256(contract_path) if contract_path.is_file() else None,
            "expected": summary.get("residual_study_contract_v1_1_sha256"),
        },
    )
    add(
        "gates_sha_matches_prepare",
        gates_path.is_file()
        and file_sha256(gates_path) == summary.get("promotion_gates_sha256"),
        {
            "live": file_sha256(gates_path) if gates_path.is_file() else None,
            "expected": summary.get("promotion_gates_sha256"),
        },
    )
    add(
        "lock_sha_matches_prepare",
        lock_path.is_file() and file_sha256(lock_path) == summary.get("lock_sha256"),
        {
            "live": file_sha256(lock_path) if lock_path.is_file() else None,
            "expected": summary.get("lock_sha256"),
        },
    )
    add(
        "production_contract_sha_matches_prepare",
        prod_path.is_file()
        and file_sha256(prod_path) == summary.get("production_contract_sha256"),
        {
            "live": file_sha256(prod_path) if prod_path.is_file() else None,
            "expected": summary.get("production_contract_sha256"),
        },
    )

    for key in (
        "feature_list_hash",
        "feature_semantics_hash",
        "xgb_params_hash",
    ):
        cval = None
        if isinstance(contract, dict):
            cval = contract.get(key) or (contract.get("hashes") or {}).get(key)
            if cval is None and "feature_set" in contract:
                cval = (contract.get("feature_set") or {}).get(key)
        add(
            f"hash_align_{key}",
            summary.get(key) is not None
            and (cval is None or cval == summary.get(key)),
            {"prepare": summary.get(key), "contract": cval},
        )

    pred_ok = True
    meta_details: list[dict[str, Any]] = []
    for o in EXPECTED_ORIGINS:
        pred = run_dir / f"oof_predictions_origin_{o}.parquet"
        meta = run_dir / f"oof_meta_origin_{o}.json"
        ok_files = pred.is_file() and meta.is_file()
        detail: dict[str, Any] = {"pred": pred.is_file(), "meta": meta.is_file()}
        if meta.is_file():
            mj = json.loads(meta.read_text(encoding="utf-8"))
            detail["hb_outcomes_inspected"] = mj.get("hb_outcomes_inspected")
            detail["origin_minus_2"] = mj.get("origin_minus_2")
            detail["max_sales_feature_date_test"] = mj.get("max_sales_feature_date_test")
            detail["max_sales_feature_date_train"] = mj.get("max_sales_feature_date_train")
            o2 = int(last_complete_month(o))
            sales_ok = (
                mj.get("hb_outcomes_inspected") is False
                and int(mj.get("origin_minus_2", -1)) == o2
                and int(mj.get("max_sales_feature_date_test", 10**9)) <= o2
                and int(mj.get("max_sales_feature_date_train", 10**9)) <= o2
            )
            detail["pit_sales_ok"] = sales_ok
            if not sales_ok:
                pred_ok = False
        if not ok_files:
            pred_ok = False
        meta_details.append({"origin": o, **detail})
    add(
        "seven_origins_complete_jobs",
        pred_ok and len(EXPECTED_ORIGINS) == 7,
        meta_details,
    )

    if pooled is None:
        try:
            pooled = load_pooled_predictions(run_dir)
        except FileNotFoundError as e:
            add("load_pooled_predictions", False, str(e))
            return {
                "ok": False,
                "checks": checks,
                "hb_outcomes_inspected": False,
                "n_rows": 0,
            }

    add("n_rows_positive", len(pooled) > 0, {"n_rows": int(len(pooled))})

    origins_in = sorted(int(x) for x in pooled["forecast_origin"].unique())
    add(
        "origins_exact_evaluable_set",
        origins_in == list(EXPECTED_ORIGINS),
        {"found": origins_in, "expected": list(EXPECTED_ORIGINS)},
    )

    fo = pooled["forecast_origin"].astype(int)
    bad_hb = sorted(set(fo.tolist()) & set(HB_ORIGINS | POST_HB_ORIGINS))
    add("zero_hb_post_hb_rows", len(bad_hb) == 0, {"forbidden_origins": bad_hb})

    dup = int(pooled.duplicated(list(PRED_KEYS)).sum())
    add("no_duplicate_prediction_keys", dup == 0, {"duplicate_rows": dup, "keys": list(PRED_KEYS)})

    finite_ok = True
    for col in ("actual",) + METHOD_COLS:
        arr = pooled[col].to_numpy(dtype=float)
        ok = bool(np.isfinite(arr).all())
        if col != "actual":
            ok = ok and bool((arr >= 0).all())
        if not ok:
            finite_ok = False
        add(f"finite_nonneg_{col}", ok, {"n": int(len(arr))})

    hz = check_complete_horizons(pooled)
    add("complete_horizons_1_to_15", hz["ok"], hz)

    eligible = summary.get("eligible_evaluation_origins") or EXPECTED_ORIGINS
    add(
        "eligible_origins_match",
        [int(x) for x in eligible] == list(EXPECTED_ORIGINS),
        {"eligible": eligible},
    )

    ok_all = all(bool(c["pass"]) for c in checks)
    return {
        "ok": bool(ok_all),
        "checks": checks,
        "hb_outcomes_inspected": False,
        "n_rows": int(len(pooled)),
        "complete_horizons": hz,
        "repo_root": str(repo_root),
    }


def apply_promotion_gates(
    *,
    scores: dict[str, dict[str, float]],
    lift_xgb: float,
    n_improved: int,
    hz_lifts: dict[str, float],
    drop3: dict[str, Any],
    boot: dict[str, Any],
) -> tuple[dict[str, bool | None], list[str], bool]:
    """Return (gates_result, non_computable, all_pass). None value => non-computable."""
    ts = scores[METHOD_FROZEN_TS]
    xg = scores[METHOD_XGB]
    thr = float(PROMOTION_GATES["no_horizon_group_wmape_regression_gt_pp"])
    boot_thr = float(PROMOTION_GATES["sku_cluster_bootstrap_frac_positive_ge"])
    origins_ge = int(PROMOTION_GATES["origins_improved_ge"])

    def _bool_or_none(ok: bool, computable: bool) -> bool | None:
        return bool(ok) if computable else None

    raw: dict[str, tuple[bool, bool]] = {}  # name -> (value, computable)

    raw["pooled_oof_wmape_improvement_over_frozen_ts"] = (
        lift_xgb > 0,
        lift_xgb == lift_xgb,
    )
    raw["origins_improved_ge_4"] = (n_improved >= origins_ge, True)
    raw["drop_top3_lift_positive"] = (
        bool(drop3.get("lift_positive")),
        drop3.get("lift_wmape_pp") == drop3.get("lift_wmape_pp"),
    )

    for name, key_ts, key_xg in (
        ("rmse_not_more_than_2pct_worse", "rmse", "rmse"),
        ("p95_not_more_than_2pct_worse", "p95_ae", "p95_ae"),
        ("p99_not_more_than_2pct_worse", "p99_ae", "p99_ae"),
    ):
        a, b = ts.get(key_ts), xg.get(key_xg)
        computable = a == a and b == b and a is not None and b is not None
        raw[name] = (
            bool(b <= a * 1.02) if computable else False,
            computable,
        )

    hz_vals = list(hz_lifts.values())
    hz_ok = all(v == v for v in hz_vals) and len(hz_vals) == len(HORIZON_GROUPS)
    raw["no_horizon_group_regression_gt_2pp"] = (
        all(v >= -thr for v in hz_vals) if hz_ok else False,
        hz_ok,
    )

    frac = boot.get("frac_positive")
    boot_ok = frac == frac and frac is not None
    raw["sku_cluster_bootstrap_frac_positive_ge_90"] = (
        bool(float(frac) >= boot_thr) if boot_ok else False,
        boot_ok,
    )

    gates_result: dict[str, bool | None] = {}
    non_computable: list[str] = []
    for k, (val, computable) in raw.items():
        if not computable:
            gates_result[k] = None
            non_computable.append(k)
        else:
            gates_result[k] = bool(val)

    computable_vals = [v for v in gates_result.values() if v is not None]
    all_pass = (not non_computable) and all(computable_vals)
    return gates_result, non_computable, bool(all_pass)


def _metrics_table(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_cols:
        for keys, g in df.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            for method in METHOD_COLS:
                s = _score_method(g, method)
                rows.append({**base, "method": method, **s})
    else:
        for method in METHOD_COLS:
            s = _score_method(df, method)
            rows.append({"method": method, **s})
    return pd.DataFrame(rows)


def build_sku_origin_wins_table(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (product, origin), g in boot.groupby(["product", "forecast_origin"], sort=False):
        m = g[["actual", "f_v2_policy", HYBRID_ANCHOR]].dropna()
        if m.empty:
            continue
        ae_ts = float(np.mean(np.abs(m["f_v2_policy"] - m["actual"])))
        ae_x = float(np.mean(np.abs(m[HYBRID_ANCHOR] - m["actual"])))
        if abs(ae_x - ae_ts) < 1e-9:
            winner = "tie"
        elif ae_x < ae_ts:
            winner = "TS_PLUS_RESIDUAL_XGB"
        else:
            winner = "FROZEN_TS"
        s_ts = score_frame(m["actual"], m["f_v2_policy"])
        s_x = score_frame(m["actual"], m[HYBRID_ANCHOR])
        rows.append(
            {
                "product": str(product),
                "forecast_origin": int(origin),
                "winner": winner,
                "ae_frozen_ts": ae_ts,
                "ae_xgb": ae_x,
                "wmape_frozen_ts": s_ts["portfolio_wmape_pct"],
                "wmape_xgb": s_x["portfolio_wmape_pct"],
                "lift_wmape_pp": float(
                    s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_correction_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, corr, n_a, n_act in (
        (
            METHOD_XGB,
            "corr_xgb",
            "corr_xgb_norm_anchor",
            "corr_xgb_norm_actual",
        ),
        (
            METHOD_MEDIAN,
            "corr_median",
            "corr_median_norm_anchor",
            "corr_median_norm_actual",
        ),
    ):
        c = panel[corr].astype(float)
        rows.append(
            {
                "method": method,
                "n_rows": int(len(c)),
                "mean_corr": float(c.mean()),
                "median_corr": float(c.median()),
                "frac_positive": float((c > 0).mean()),
                "frac_negative": float((c < 0).mean()),
                "frac_zero": float((c == 0).mean()),
                "mean_abs_corr": float(c.abs().mean()),
                "mean_abs_corr_norm_anchor": float(panel[n_a].mean()),
                "mean_abs_corr_norm_actual": float(panel[n_act].mean()),
                "p90_abs_corr_norm_anchor": float(panel[n_a].quantile(0.90)),
                "normalization_note": (
                    "abs(corr)/max(|anchor|,eps) and abs(corr)/max(|actual|,eps); "
                    f"eps={CORR_EPS}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_magnification_table(panel: pd.DataFrame) -> pd.DataFrame:
    if "xgb_magnifies_ts_error" not in panel.columns:
        ae_ts = (panel[METHOD_FROZEN_TS] - panel["actual"]).abs()
        ae_x = (panel[METHOD_XGB] - panel["actual"]).abs()
        mag = panel.copy()
        mag["ae_delta_xgb_minus_ts"] = ae_x - ae_ts
        mag["xgb_magnifies_ts_error"] = mag["ae_delta_xgb_minus_ts"] > 0
    else:
        mag = panel
    bad = mag.loc[mag["xgb_magnifies_ts_error"]].copy()
    if bad.empty:
        return pd.DataFrame(
            columns=[
                "product",
                "forecast_origin",
                "n_rows",
                "sum_ae_delta",
                "mean_ae_delta",
            ]
        )
    g = (
        bad.groupby(["product", "forecast_origin"], sort=False)
        .agg(
            n_rows=("ae_delta_xgb_minus_ts", "size"),
            sum_ae_delta=("ae_delta_xgb_minus_ts", "sum"),
            mean_ae_delta=("ae_delta_xgb_minus_ts", "mean"),
        )
        .reset_index()
        .sort_values("sum_ae_delta", ascending=False)
    )
    return g


def add_oracles(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell + SKU×origin hindsight oracles (TS vs XGB only). Non-deployable."""
    p = panel.copy()
    ae_ts = (p[METHOD_FROZEN_TS] - p["actual"]).abs()
    ae_x = (p[METHOD_XGB] - p["actual"]).abs()
    use_ts = ae_ts <= ae_x
    p["f_cell_oracle"] = np.where(use_ts, p[METHOD_FROZEN_TS], p[METHOD_XGB])
    p["cell_oracle_source"] = np.where(use_ts, METHOD_FROZEN_TS, METHOD_XGB)
    p["ae_cell_oracle"] = np.where(use_ts, ae_ts, ae_x)

    picks = []
    for (product, origin), g in p.groupby(["product", "forecast_origin"], sort=False):
        s_ts = score_frame(g["actual"], g[METHOD_FROZEN_TS])
        s_x = score_frame(g["actual"], g[METHOD_XGB])
        pick = (
            METHOD_FROZEN_TS
            if s_ts["portfolio_wmape_pct"] <= s_x["portfolio_wmape_pct"]
            else METHOD_XGB
        )
        picks.append(
            {
                "product": str(product),
                "forecast_origin": int(origin),
                "sku_origin_oracle_source": pick,
            }
        )
    pick_df = pd.DataFrame(picks)
    p = p.merge(pick_df, on=["product", "forecast_origin"], how="left")
    p["f_sku_origin_oracle"] = np.where(
        p["sku_origin_oracle_source"] == METHOD_FROZEN_TS,
        p[METHOD_FROZEN_TS],
        p[METHOD_XGB],
    )
    p["ae_sku_origin_oracle"] = (p["f_sku_origin_oracle"] - p["actual"]).abs()

    summary_rows = []
    for name, col in (
        ("FROZEN_TS", METHOD_FROZEN_TS),
        ("TS_PLUS_RESIDUAL_XGB", METHOD_XGB),
        ("cell_oracle", "f_cell_oracle"),
        ("sku_origin_oracle", "f_sku_origin_oracle"),
    ):
        s = _score_method(p, col)
        summary_rows.append(
            {
                "method": name,
                "deployable": name in (METHOD_FROZEN_TS, METHOD_XGB),
                "hindsight_ceiling_only": name
                in ("cell_oracle", "sku_origin_oracle"),
                **s,
            }
        )
    return p, pd.DataFrame(summary_rows)


def ensure_out_dir(out_dir: Path, *, force: bool) -> None:
    if out_dir.exists():
        nonempty = any(out_dir.iterdir())
        if nonempty and not force:
            raise EvaluationOverwriteError(
                f"Refusing to overwrite non-empty {out_dir}; pass --force"
            )


def atomic_publish(staging: Path, out_dir: Path, *, force: bool) -> None:
    ensure_out_dir(out_dir, force=force)
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    final_tmp = parent / f".{out_dir.name}.publish_tmp"
    if final_tmp.exists():
        shutil.rmtree(final_tmp)
    shutil.move(str(staging), str(final_tmp))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    final_tmp.rename(out_dir)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def render_report(
    *,
    integrity: dict[str, Any],
    recommendation: str | None,
    gates_result: dict[str, Any] | None,
    promotion_gates_pass: bool | None,
    lift_xgb: float | None,
    scores: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    hier: dict[str, Any] | None,
    drop3: dict[str, Any] | None,
    non_computable: list[str],
) -> str:
    lines = [
        "# Residual XGBoost v1.1 evaluation (no refit)",
        "",
        f"- `hb_outcomes_inspected`: false",
        f"- `residual_lock_written`: false",
        f"- Integrity ok: **{integrity.get('ok')}**",
        f"- Recommendation: **{recommendation}**",
        "",
        "## Integrity",
        "",
    ]
    for c in integrity.get("checks", []):
        mark = "PASS" if c.get("pass") else "FAIL"
        lines.append(f"- [{mark}] `{c.get('check')}`")
    lines += ["", "## Promotion gates", ""]
    if gates_result is None:
        lines.append("_Skipped (integrity failure)._")
    else:
        lines.append(f"- `promotion_gates_pass`: {promotion_gates_pass}")
        if non_computable:
            lines.append(f"- Non-computable gates: {non_computable}")
        for k, v in sorted(gates_result.items()):
            lines.append(f"- `{k}`: {v}")
    lines += ["", "## Overall metrics", ""]
    if scores:
        lines.append(f"- Lift WMAPE pp (TS − XGB): {lift_xgb}")
        if lift_xgb is not None and abs(lift_xgb - VPS_RECONCILE_LIFT_PP) < 0.05:
            lines.append(
                f"- Reconciliation note: recomputed lift is close to prior VPS "
                f"evaluate ({VPS_RECONCILE_LIFT_PP:.4f} pp); not used for recommendation."
            )
        for m, s in scores.items():
            lines.append(
                f"- `{m}`: WMAPE={s.get('portfolio_wmape_pct')}, "
                f"MAE={s.get('mae')}, RMSE={s.get('rmse')}, bias={s.get('bias')}"
            )
    else:
        lines.append("_Skipped._")
    lines += ["", "## Robustness", ""]
    if boot:
        lines.append(f"- SKU bootstrap frac_positive: {boot.get('frac_positive')}")
    if hier:
        lines.append(
            f"- Hierarchical bootstrap frac_positive: {hier.get('frac_positive')}"
        )
    if drop3:
        lines.append(
            f"- Drop-top3: lift={drop3.get('lift_wmape_pp')} "
            f"dropped={drop3.get('dropped_skus')}"
        )
    lines += [
        "",
        "## Feature importance",
        "",
        "Status: `UNAVAILABLE_MODELS_NOT_PERSISTED` (no refit).",
        "",
        "## Oracles",
        "",
        "Cell and SKU×origin oracles are **hindsight ceilings only** (non-deployable).",
        "Median correction is benchmark-only and excluded from oracles.",
        "",
        "## Stop",
        "",
        "Stop before writing a residual lock. Do not open HB outcomes.",
        "",
    ]
    return "\n".join(lines)


def run_evaluation(
    *,
    repo_root: Path,
    force: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    family_dir = repo_root / FAMILY_DIR_REL
    lock_dir = repo_root / LOCK_DIR_REL
    run_dir = family_dir / "runs" / FULL_RUN_ID
    out_dir = Path(out_dir) if out_dir is not None else family_dir / EVAL_DIR_NAME
    router_csv = lock_dir / "router_features_sku_origin.csv"

    ensure_out_dir(out_dir, force=force)

    staging: Path | None = Path(
        tempfile.mkdtemp(prefix="evaluation_v1_stage_", dir=str(out_dir.parent))
    )
    try:
        assert staging is not None
        pooled = None
        try:
            pooled = load_pooled_predictions(run_dir)
        except FileNotFoundError:
            pooled = None

        integrity = run_integrity_checks(
            repo_root=repo_root,
            family_dir=family_dir,
            run_dir=run_dir,
            lock_dir=lock_dir,
            pooled=pooled,
        )
        _write_json(staging / "integrity_report.json", integrity)

        if not integrity["ok"] or pooled is None:
            rec = decide_recommendation(
                integrity_ok=False,
                non_computable_gates=[],
                promotion_gates_pass=False,
            )
            preview = {
                "recommendation": rec,
                "promotion_gates_pass": None,
                "gates_result": None,
                "non_computable_gates": [],
                "lift_wmape_pp_xgb_vs_ts": None,
                "median_benchmark_only": True,
                "hb_outcomes_inspected": False,
                "residual_lock_written": False,
                "rationale_bullets": ["Integrity checks failed; analysis aborted."],
            }
            _write_json(staging / "recommendation_preview.json", preview)
            _write_json(
                staging / "feature_importance_note.json",
                {
                    "feature_importance_status": "UNAVAILABLE_MODELS_NOT_PERSISTED",
                    "hb_outcomes_inspected": False,
                },
            )
            (staging / "report.md").write_text(
                render_report(
                    integrity=integrity,
                    recommendation=rec,
                    gates_result=None,
                    promotion_gates_pass=None,
                    lift_xgb=None,
                    scores=None,
                    boot=None,
                    hier=None,
                    drop3=None,
                    non_computable=[],
                ),
                encoding="utf-8",
            )
            atomic_publish(staging, out_dir, force=force)
            staging = None  # published; do not rmtree
            return {
                "ok": False,
                "out_dir": str(out_dir),
                "recommendation": rec,
                "integrity_ok": False,
            }

        panel = add_correction_columns(pooled)
        panel = join_router_features(panel, router_csv)
        panel = assign_segments(panel)
        boot_df = build_bootstrap_copy(panel)

        scores = {
            METHOD_FROZEN_TS: _score_method(panel, METHOD_FROZEN_TS),
            METHOD_MEDIAN: _score_method(panel, METHOD_MEDIAN),
            METHOD_XGB: _score_method(panel, METHOD_XGB),
        }
        lift_xgb = float(
            scores[METHOD_FROZEN_TS]["portfolio_wmape_pct"]
            - scores[METHOD_XGB]["portfolio_wmape_pct"]
        )

        by_origin_rows = []
        for o, g in panel.groupby(panel["forecast_origin"].astype(int)):
            s_ts = _score_method(g, METHOD_FROZEN_TS)
            s_x = _score_method(g, METHOD_XGB)
            by_origin_rows.append(
                {
                    "forecast_origin": int(o),
                    "lift_wmape_pp": float(
                        s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
                    ),
                    "frozen_ts_wmape": s_ts["portfolio_wmape_pct"],
                    "xgb_wmape": s_x["portfolio_wmape_pct"],
                }
            )
        n_improved = sum(1 for r in by_origin_rows if r["lift_wmape_pp"] > 0)

        hz = horizon_group_wmape_lifts(
            boot_df,
            HORIZON_GROUPS,
            v2_col="f_v2_policy",
            hy_col=HYBRID_ANCHOR,
        )
        B = int(PROMOTION_GATES["bootstrap_B"])
        seed = int(PROMOTION_GATES["bootstrap_seed"])
        boot = sku_cluster_bootstrap(boot_df, B=B, seed=seed)
        hier = hierarchical_origin_then_sku_bootstrap(boot_df, B=B, seed=seed)
        loo_o = leave_one_origin(boot_df)
        loo_s = leave_one_sku(boot_df)
        drop3 = drop_top_k_hybrid_gainers(boot_df, k=3)
        wins_summary = sku_origin_wins(
            boot_df, v2_col="f_v2_policy", hy_col=HYBRID_ANCHOR
        )

        gates_result, non_computable, all_pass = apply_promotion_gates(
            scores=scores,
            lift_xgb=lift_xgb,
            n_improved=n_improved,
            hz_lifts=hz,
            drop3=drop3,
            boot=boot,
        )
        recommendation = decide_recommendation(
            integrity_ok=True,
            non_computable_gates=non_computable,
            promotion_gates_pass=all_pass,
        )

        metrics_overall = _metrics_table(panel)
        metrics_by_origin = _metrics_table(panel, ["forecast_origin"])
        # horizon groups
        hz_rows = []
        for name, hs in HORIZON_GROUPS:
            g = panel.loc[panel["horizon"].astype(int).isin(hs)]
            for method in METHOD_COLS:
                s = _score_method(g, method)
                hz_rows.append({"horizon_group": name, "method": method, **s})
            s_ts = _score_method(g, METHOD_FROZEN_TS)
            s_x = _score_method(g, METHOD_XGB)
            hz_rows.append(
                {
                    "horizon_group": name,
                    "method": "lift_ts_minus_xgb",
                    "portfolio_wmape_pct": float(
                        s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
                    ),
                }
            )
        metrics_by_hz = pd.DataFrame(hz_rows)

        seg_frames = []
        for col in ("history_len_bin", "demand_intensity_bin", "sparsity_bin"):
            t = _metrics_table(panel, [col])
            t.insert(0, "segment_type", col)
            t = t.rename(columns={col: "segment_value"})
            seg_frames.append(t)
        metrics_by_segment = pd.concat(seg_frames, ignore_index=True)
        metrics_by_sku = _metrics_table(panel, ["product"])
        # add lift column for sku
        sku_lift = []
        for product, g in panel.groupby(panel["product"].astype(str)):
            s_ts = _score_method(g, METHOD_FROZEN_TS)
            s_x = _score_method(g, METHOD_XGB)
            sku_lift.append(
                {
                    "product": product,
                    "lift_wmape_pp": float(
                        s_ts["portfolio_wmape_pct"] - s_x["portfolio_wmape_pct"]
                    ),
                }
            )
        sku_lift_df = pd.DataFrame(sku_lift)
        metrics_by_sku = metrics_by_sku.merge(sku_lift_df, on="product", how="left")

        wins_table = build_sku_origin_wins_table(boot_df)
        corr_table = build_correction_summary(panel)
        mag_table = build_magnification_table(panel)
        panel_oracles, oracle_summary = add_oracles(panel)

        # Persist panel (keep method cols + anchor + segments + corrections)
        panel_oracles.to_parquet(staging / "oof_panel.parquet", index=False)
        metrics_overall.to_csv(staging / "metrics_overall.csv", index=False)
        metrics_by_origin.to_csv(staging / "metrics_by_origin.csv", index=False)
        metrics_by_hz.to_csv(staging / "metrics_by_horizon_group.csv", index=False)
        metrics_by_segment.to_csv(staging / "metrics_by_segment.csv", index=False)
        metrics_by_sku.to_csv(staging / "metrics_by_sku.csv", index=False)
        wins_table.to_csv(staging / "sku_origin_wins.csv", index=False)
        corr_table.to_csv(staging / "correction_magnitude.csv", index=False)
        mag_table.to_csv(staging / "xgb_error_magnification.csv", index=False)
        loo_o.to_csv(staging / "robustness_loo_origin.csv", index=False)
        loo_s.to_csv(staging / "robustness_loo_sku.csv", index=False)
        oracle_summary.to_csv(staging / "oracle_ceilings.csv", index=False)

        _write_json(staging / "robustness_sku_bootstrap.json", boot)
        _write_json(staging / "robustness_hierarchical_bootstrap.json", hier)
        _write_json(staging / "robustness_drop_top3.json", drop3)
        _write_json(
            staging / "gates_result.json",
            {
                "gates_result": gates_result,
                "promotion_gates_pass": all_pass,
                "non_computable_gates": non_computable,
                "lift_wmape_pp_xgb_vs_ts": lift_xgb,
                "n_origins_improved": int(n_improved),
                "by_origin": by_origin_rows,
                "horizon_group_lifts": hz,
                "sku_origin_wins_summary": wins_summary,
                "hb_outcomes_inspected": False,
            },
        )
        preview = {
            "recommendation": recommendation,
            "promotion_gates_pass": all_pass,
            "gates_result": gates_result,
            "non_computable_gates": non_computable,
            "lift_wmape_pp_xgb_vs_ts": lift_xgb,
            "n_origins_improved": int(n_improved),
            "median_benchmark_only": True,
            "hb_outcomes_inspected": False,
            "residual_lock_written": False,
            "rationale_bullets": [
                f"Integrity passed ({integrity['n_rows']} OOF rows).",
                f"Pooled WMAPE lift (TS−XGB)={lift_xgb:.4f} pp.",
                f"Origins improved={n_improved}/7; gates_pass={all_pass}.",
                "Median correction is benchmark-only (not selectable).",
                "Stop before residual lock; HB outcomes not inspected.",
            ],
        }
        _write_json(staging / "recommendation_preview.json", preview)
        _write_json(
            staging / "feature_importance_note.json",
            {
                "feature_importance_status": "UNAVAILABLE_MODELS_NOT_PERSISTED",
                "reason": "XGB models discarded after OOF predict; no refit allowed",
                "hb_outcomes_inspected": False,
            },
        )
        (staging / "report.md").write_text(
            render_report(
                integrity=integrity,
                recommendation=recommendation,
                gates_result=gates_result,
                promotion_gates_pass=all_pass,
                lift_xgb=lift_xgb,
                scores=scores,
                boot=boot,
                hier=hier,
                drop3=drop3,
                non_computable=non_computable,
            ),
            encoding="utf-8",
        )

        atomic_publish(staging, out_dir, force=force)
        staging = None  # published; do not rmtree
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "recommendation": recommendation,
            "promotion_gates_pass": all_pass,
            "lift_wmape_pp_xgb_vs_ts": lift_xgb,
            "integrity_ok": True,
            "hb_outcomes_inspected": False,
            "residual_lock_written": False,
        }
    finally:
        if staging is not None and Path(staging).exists():
            shutil.rmtree(staging, ignore_errors=True)
