#!/usr/bin/env python3
"""Archval55 post-hoc TS decision analysis (no refits / no weight search).

Compares deployable V2_POLICY vs A0_HYBRID_POLICY on the full actual-available
panel, audits INELIGIBLE coverage, five-way matched ranking, robustness, and
issues a single A/B/C recommendation. Writes only under archval55_decision_v1.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct
from pkg.benchmark.universes import load_universe_product_names
from pkg.forecast_jobs.plan import decode_product_key
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import resolve_legacy_architecture

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_EXPORT = RESULTS / "archval55_export"
DEFAULT_CONFIRM = RESULTS / "archval55_confirmation"
DEFAULT_OUT = RESULTS / "archval55_decision_v1"
DEFAULT_RUN = REPO / "src" / "data" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1"
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
EARLY_ORIGINS = frozenset({140104, 140201})
LATE_ORIGINS = frozenset({140301, 140402})
SLICES = (
    ("pooled", None, None),
    ("old_sku_x_early_origin", "old_sku", "early_origin"),
    ("old_sku_x_late_origin", "old_sku", "late_origin"),
    ("new_sku_x_early_origin", "new_sku", "early_origin"),
    ("new_sku_x_late_origin", "new_sku", "late_origin"),
)
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
# V2.1 model names treated as "flat" vs "dynamic" (heuristic on existing labels).
FLAT_V21 = frozenset(
    {"naive", "seasonal_naive", "drift", "croston_sba", "tsb", "naive_last_value"}
)
DYNAMIC_V21 = frozenset({"ets", "auto_arima"})


def _md_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    float_cols = float_cols or set()
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        cells: list[str] = []
        for c, v in zip(cols, row):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif c in float_cols or isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.4g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sku_cohort(product: str, old: set[str]) -> str:
    return "old_sku" if str(product) in old else "new_sku"


def _origin_cohort(origin: int) -> str:
    o = int(origin)
    if o in EARLY_ORIGINS:
        return "early_origin"
    if o in LATE_ORIGINS:
        return "late_origin"
    return "other_origin"


def score_series(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    m = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    if m.empty:
        return {
            "n_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "portfolio_wmape_pct": np.nan,
            "bias": np.nan,
        }
    err = m["forecast"] - m["actual"]
    ae = np.abs(err)
    return {
        "n_rows": int(len(m)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
    }


def attach_actuals(df: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    need = (
        out["actual"].isna()
        if "actual" in out.columns
        else pd.Series(True, index=out.index)
    )
    if not bool(need.any()):
        return out
    s = sales.rename(columns={"sales": "actual_sales", "date": "target_date"})
    s["product"] = s["product"].astype(str)
    s["target_date"] = pd.to_numeric(s["target_date"], errors="coerce").astype(int)
    merged = out.merge(
        s[["product", "target_date", "actual_sales"]],
        on=["product", "target_date"],
        how="left",
    )
    if "actual" not in merged.columns:
        merged["actual"] = merged["actual_sales"]
    else:
        merged["actual"] = merged["actual"].where(
            merged["actual"].notna(), merged["actual_sales"]
        )
    return merged.drop(columns=["actual_sales"], errors="ignore")


def history_length_map(sales: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Train months through production last-complete = origin-2."""
    s = sales.copy()
    s["product"] = s["product"].astype(str)
    s["date"] = pd.to_numeric(s["date"], errors="coerce").astype(int)
    out: dict[tuple[str, int], int] = {}
    origins = sorted(EARLY_ORIGINS | LATE_ORIGINS)
    for product, g in s.groupby("product", sort=False):
        dates = g["date"].dropna().astype(int)
        for origin in origins:
            last_complete = shamsi_add_months(int(origin), -2)
            n = int((dates <= last_complete).sum())
            out[(str(product), int(origin))] = n
    return out


def last_train_value_map(sales: pd.DataFrame) -> dict[tuple[str, int], float]:
    s = sales.copy()
    s["product"] = s["product"].astype(str)
    s["date"] = pd.to_numeric(s["date"], errors="coerce").astype(int)
    s["sales"] = pd.to_numeric(s["sales"], errors="coerce")
    out: dict[tuple[str, int], float] = {}
    origins = sorted(EARLY_ORIGINS | LATE_ORIGINS)
    for product, g in s.groupby("product", sort=False):
        g = g.sort_values("date")
        for origin in origins:
            last_complete = shamsi_add_months(int(origin), -2)
            sub = g.loc[(g["date"] <= last_complete) & g["sales"].notna()]
            if sub.empty:
                continue
            out[(str(product), int(origin))] = float(sub["sales"].iloc[-1])
    return out


def parse_reason(msg: Optional[str], fallback_reason: Optional[str]) -> tuple[str, str]:
    text = " | ".join(x for x in (fallback_reason, msg) if x)
    low = text.lower()
    if "neural_short_history" in low or "insufficient_history" in low:
        code = "neural_short_history_ineligible"
    elif "v21_no_eligible" in low or "no eligible models" in low:
        code = "v21_no_eligible_models"
    elif "empty_predictions" in low or "no predictions" in low:
        code = "neural_empty_predictions_ineligible"
    elif "naive_last_value" in low and "short" in low:
        code = "neural_short_history_ineligible"
    elif text.strip():
        m = re.search(r"([a-z0-9_]+_ineligible)", low)
        code = m.group(1) if m else "other_ineligible"
    else:
        code = "unknown"
    return code, text


def classify_gap(
    history_length: int, reason_code: str, origin_cohort: str
) -> str:
    """Legitimate short-history vs suspected implementation/config issue."""
    if reason_code in {
        "neural_short_history_ineligible",
        "v21_no_eligible_models",
    }:
        # Late origins with long history failing short-history → suspicious.
        if origin_cohort == "late_origin" and history_length >= 24:
            return "suspicious_late_long_history"
        if history_length < 18:
            return "legitimate_short_history"
        if history_length < 24 and reason_code == "neural_short_history_ineligible":
            return "legitimate_borderline_history"
        return "review_borderline"
    if reason_code == "neural_empty_predictions_ineligible":
        return "possible_implementation_issue"
    return "review_other"


# ---------------------------------------------------------------------------
# 1. INELIGIBLE audit
# ---------------------------------------------------------------------------


def audit_ineligible(
    run_dir: Path,
    sales: pd.DataFrame,
    old_skus: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hist = history_length_map(sales)
    db = run_dir / "state.sqlite"
    conn = sqlite3.connect(str(db))
    jobs = pd.read_sql_query(
        """
        SELECT engine_version, forecast_origin, product_id, status,
               error_type, error_message, selected_model
        FROM jobs WHERE status = 'INELIGIBLE'
        """,
        conn,
    )
    conn.close()
    rows = []
    jobs_root = run_dir / "jobs"
    for _, j in jobs.iterrows():
        engine = str(j["engine_version"]).lower()
        origin = int(j["forecast_origin"])
        product_id = str(j["product_id"])
        product, seed = decode_product_key(product_id)
        fb_reason = None
        # Locate fallback.json
        slug_product = product_id.replace("/", "_")
        candidates = list(jobs_root.glob(f"{engine}/*__{slug_product}/fallback.json"))
        if not candidates:
            candidates = list(jobs_root.glob(f"{engine}/**/fallback.json"))
            candidates = [
                c
                for c in candidates
                if c.parent.name.endswith(f"__{slug_product}")
                or c.parent.name.endswith(f"__{product_id.replace('/', '_')}")
            ]
        extras: dict[str, Any] = {}
        if candidates:
            try:
                fb = json.loads(candidates[0].read_text(encoding="utf-8"))
                extras = fb.get("extras") or {}
                fb_reason = (
                    extras.get("fallback_reason")
                    or extras.get("ineligibility_reason")
                    or fb.get("error_message")
                )
            except Exception:
                pass
        reason_code, detail = parse_reason(j.get("error_message"), fb_reason)
        if extras.get("fallback_reason"):
            reason_code, detail = parse_reason(
                j.get("error_message"), str(extras.get("fallback_reason"))
            )
        hl = hist.get((product, origin), -1)
        oc = _origin_cohort(origin)
        sc = _sku_cohort(product, old_skus)
        gap_class = classify_gap(hl, reason_code, oc)
        a0_tier = None
        try:
            if hl >= 0:
                a0_tier = resolve_legacy_architecture(hl).lookback
        except Exception:
            a0_tier = None
        rows.append(
            {
                "engine": engine,
                "forecast_origin": origin,
                "product": product,
                "product_id": product_id,
                "seed": seed,
                "error_type": j.get("error_type"),
                "error_message": j.get("error_message"),
                "fallback_reason": fb_reason,
                "reason_code": reason_code,
                "reason_detail": detail[:500] if detail else "",
                "history_length": hl,
                "a0_resolved_lookback": a0_tier,
                "sku_cohort": sc,
                "origin_cohort": oc,
                "gap_classification": gap_class,
            }
        )
    inelig = pd.DataFrame(rows)
    summary = (
        inelig.groupby(["engine", "origin_cohort", "reason_code", "gap_classification"], as_index=False)
        .agg(n_jobs=("product_id", "size"), n_products=("product", "nunique"))
        .sort_values(["engine", "origin_cohort", "n_jobs"], ascending=[True, True, False])
    )
    # new_sku x late A0 gap explanation
    late_new_a0 = inelig.loc[
        (inelig["engine"] == "a0")
        & (inelig["sku_cohort"] == "new_sku")
        & (inelig["origin_cohort"] == "late_origin")
    ]
    missing_groups = (
        late_new_a0.groupby(["product", "forecast_origin"], as_index=False)
        .agg(
            n_ineligible_seeds=("seed", "nunique"),
            history_length=("history_length", "first"),
            reason_code=("reason_code", "first"),
            gap_classification=("gap_classification", "first"),
        )
        if not late_new_a0.empty
        else pd.DataFrame()
    )
    explain = {
        "n_ineligible_jobs": int(len(inelig)),
        "by_engine": inelig.groupby("engine").size().astype(int).to_dict(),
        "new_sku_late_a0_ineligible_jobs": int(len(late_new_a0)),
        "new_sku_late_a0_groups_with_any_ineligible_seed": int(len(missing_groups)),
        "gap_class_counts": late_new_a0["gap_classification"].value_counts().to_dict()
        if not late_new_a0.empty
        else {},
        "suspicious_any": bool(
            (
                (inelig["engine"].astype(str).str.lower() == "a0")
                & inelig["gap_classification"].isin(
                    {
                        "suspicious_late_long_history",
                        "possible_implementation_issue",
                    }
                )
            ).any()
        )
        if not inelig.empty
        else False,
        "a1_empty_predictions_remaining": int(
            (
                (inelig["engine"].astype(str).str.lower() == "a1")
                & (inelig["reason_code"] == "neural_empty_predictions_ineligible")
            ).sum()
        )
        if not inelig.empty
        else 0,
        "note_675_vs_900": (
            "Pairwise confirmation uses seed-mean of SUCCESS seeds only. "
            "A product×origin contributes 15 cells if ≥1 A0 seed succeeded. "
            "Groups where all seeds are INELIGIBLE drop out entirely; "
            "partial seed loss still yields a mean. Count full dropouts from the export panel."
        ),
    }
    return inelig, summary, {"explain": explain, "missing_groups": missing_groups}


# ---------------------------------------------------------------------------
# 2–3. Policy panel
# ---------------------------------------------------------------------------


def build_engine_native(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """SUCCESS non-naive forecasts; A0/A1 = mean of successful seeds only."""
    frames: dict[str, pd.DataFrame] = {}
    v21 = panel.loc[
        (panel["engine"] == "v2.1")
        & (panel["job_status"] == "SUCCESS")
        & (panel["model"].astype(str) != "naive_last_value")
    ].copy()
    frames["v2.1"] = (
        v21.groupby(KEYS, as_index=False)
        .agg(
            forecast=("forecast", "first"),
            model=("model", "first"),
            sku_cohort=("sku_cohort", "first"),
            origin_cohort=("origin_cohort", "first"),
        )
        .rename(columns={"forecast": "f_v21", "model": "model_v21"})
    )

    for eng in ("a0", "a1"):
        sub = panel.loc[
            (panel["engine"] == eng)
            & (panel["job_status"] == "SUCCESS")
            & (panel["model"].astype(str) != "naive_last_value")
        ].copy()
        g = sub.groupby(KEYS, as_index=False).agg(
            forecast=("forecast", "mean"),
            n_seeds=("seed", "nunique"),
            sku_cohort=("sku_cohort", "first"),
            origin_cohort=("origin_cohort", "first"),
        )
        frames[eng] = g.rename(columns={"forecast": f"f_{eng}", "n_seeds": f"n_seeds_{eng}"})
    return frames


def build_naive_lookups(panel: pd.DataFrame, last_vals: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Naive forecast per KEY from V2.1 fallback rows or last train value."""
    naive_rows = panel.loc[panel["model"].astype(str) == "naive_last_value"].copy()
    if not naive_rows.empty:
        nv = (
            naive_rows.groupby(KEYS, as_index=False)
            .agg(f_naive=("forecast", "first"), fallback_last_value=("fallback_last_value", "first"))
        )
    else:
        nv = pd.DataFrame(columns=KEYS + ["f_naive", "fallback_last_value"])

    # Universe skeleton from any engine rows with actuals later.
    base = panel[KEYS + ["sku_cohort", "origin_cohort"]].drop_duplicates(KEYS)
    out = base.merge(nv, on=KEYS, how="left")
    missing = out["f_naive"].isna()
    if bool(missing.any()):
        vals = []
        for _, r in out.loc[missing].iterrows():
            vals.append(last_vals.get((str(r["product"]), int(r["forecast_origin"])), np.nan))
        out.loc[missing, "f_naive"] = vals
    return out


def build_policy_panel(
    panel: pd.DataFrame,
    sales: pd.DataFrame,
    last_vals: dict[tuple[str, int], float],
    hist: dict[tuple[str, int], int],
) -> pd.DataFrame:
    panel = attach_actuals(panel, sales)
    frames = build_engine_native(panel)
    naive = build_naive_lookups(panel, last_vals)

    # Keep cohort tags only on the left skeleton; engine frames drop them.
    for eng in list(frames):
        frames[eng] = frames[eng].drop(
            columns=[c for c in ("sku_cohort", "origin_cohort") if c in frames[eng].columns],
            errors="ignore",
        )

    work = panel.loc[
        panel["actual"].notna(), KEYS + ["actual", "sku_cohort", "origin_cohort"]
    ].drop_duplicates(KEYS)
    work = work.merge(frames["v2.1"], on=KEYS, how="left")
    work = work.merge(frames["a0"], on=KEYS, how="left")
    work = work.merge(frames["a1"], on=KEYS, how="left")
    work = work.merge(naive[KEYS + ["f_naive"]], on=KEYS, how="left")

    work["v21_ok"] = work["f_v21"].notna()
    work["a0_ok"] = work["f_a0"].notna()
    work["a1_ok"] = work["f_a1"].notna()

    # V2_POLICY
    work["f_v2_policy"] = np.where(work["v21_ok"], work["f_v21"], work["f_naive"])
    work["source_v2_policy"] = np.where(work["v21_ok"], "v2.1", "naive_fallback")

    # A0_HYBRID_POLICY
    both = work["v21_ok"] & work["a0_ok"]
    only_v = work["v21_ok"] & ~work["a0_ok"]
    only_a = ~work["v21_ok"] & work["a0_ok"]
    neither = ~work["v21_ok"] & ~work["a0_ok"]
    work["f_a0_hybrid"] = np.nan
    work.loc[both, "f_a0_hybrid"] = 0.5 * work.loc[both, "f_v21"] + 0.5 * work.loc[both, "f_a0"]
    work.loc[only_v, "f_a0_hybrid"] = work.loc[only_v, "f_v21"]
    work.loc[only_a, "f_a0_hybrid"] = work.loc[only_a, "f_a0"]
    work.loc[neither, "f_a0_hybrid"] = work.loc[neither, "f_naive"]
    work["source_a0_hybrid"] = np.select(
        [both, only_v, only_a, neither],
        ["blend50_v21_a0", "v2.1_only", "a0_only", "naive_fallback"],
        default="unknown",
    )
    work["used_fallback_v2"] = work["source_v2_policy"] == "naive_fallback"
    work["used_fallback_hybrid"] = work["source_a0_hybrid"] == "naive_fallback"
    work["native_v21"] = work["v21_ok"]
    work["native_a0"] = work["a0_ok"]

    work["ae_v2_policy"] = (work["f_v2_policy"] - work["actual"]).abs()
    work["ae_a0_hybrid"] = (work["f_a0_hybrid"] - work["actual"]).abs()
    work["history_length"] = [
        hist.get((str(p), int(o)), -1)
        for p, o in zip(work["product"], work["forecast_origin"])
    ]
    work["v21_family"] = np.where(
        work["model_v21"].astype(str).isin(FLAT_V21),
        "flat",
        np.where(work["model_v21"].astype(str).isin(DYNAMIC_V21), "dynamic", "other_or_missing"),
    )
    return work.dropna(subset=["actual", "f_v2_policy", "f_a0_hybrid"]).copy()


def sku_origin_wmape(panel: pd.DataFrame, forecast_col: str) -> pd.Series:
    vals = []
    for _, g in panel.groupby(["product", "forecast_origin"], sort=False):
        w = portfolio_wmape_pct(g["actual"], g[forecast_col])
        if w is not None and np.isfinite(w):
            vals.append(float(w))
    return pd.Series(vals, dtype=float)


def policy_metrics(panel: pd.DataFrame, slice_name: str) -> dict[str, Any]:
    rows = []
    for name, col, fb_col, native_col in (
        ("V2_POLICY", "f_v2_policy", "used_fallback_v2", "native_v21"),
        ("A0_HYBRID_POLICY", "f_a0_hybrid", "used_fallback_hybrid", "native_a0"),
    ):
        s = score_series(panel["actual"], panel[col])
        so = sku_origin_wmape(panel, col)
        n = len(panel)
        rows.append(
            {
                "slice": slice_name,
                "policy": name,
                "n_rows": s["n_rows"],
                "mae": s["mae"],
                "rmse": s["rmse"],
                "portfolio_wmape_pct": s["portfolio_wmape_pct"],
                "bias": s["bias"],
                "native_coverage_pct": float(100.0 * panel[native_col].mean())
                if native_col in panel.columns
                else np.nan,
                "fallback_coverage_pct": float(100.0 * panel[fb_col].mean())
                if fb_col in panel.columns
                else np.nan,
                "sku_origin_wmape_p90": float(np.nanquantile(so, 0.90)) if len(so) else np.nan,
                "sku_origin_wmape_p95": float(np.nanquantile(so, 0.95)) if len(so) else np.nan,
                "n_sku_origin_wmape_gt_50": int((so > 50).sum()) if len(so) else 0,
                "n_sku_origin_wmape_gt_100": int((so > 100).sum()) if len(so) else 0,
                "n_sku_origin_wmape_gt_200": int((so > 200).sum()) if len(so) else 0,
                "n_sku_origin_groups": int(len(so)),
            }
        )
    # lift
    df = pd.DataFrame(rows)
    if len(df) == 2:
        v2 = float(df.loc[df["policy"] == "V2_POLICY", "portfolio_wmape_pct"].iloc[0])
        hy = float(df.loc[df["policy"] == "A0_HYBRID_POLICY", "portfolio_wmape_pct"].iloc[0])
        df["lift_hybrid_vs_v2_pp"] = np.where(
            df["policy"] == "A0_HYBRID_POLICY", v2 - hy, np.nan
        )
    return df


def filter_slice_df(df: pd.DataFrame, sku: Optional[str], origin: Optional[str]) -> pd.DataFrame:
    g = df
    if sku is not None:
        g = g.loc[g["sku_cohort"] == sku]
    if origin is not None:
        g = g.loc[g["origin_cohort"] == origin]
    return g.copy()


# ---------------------------------------------------------------------------
# 4. Five-way
# ---------------------------------------------------------------------------


def fiveway_panel(policy: pd.DataFrame) -> pd.DataFrame:
    m = policy.dropna(subset=["f_v21", "f_a0", "f_a1", "actual"]).copy()
    m["f_blend_a0"] = 0.5 * m["f_v21"] + 0.5 * m["f_a0"]
    m["f_blend_a1"] = 0.5 * m["f_v21"] + 0.5 * m["f_a1"]
    return m


def fiveway_headline(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in (
        ("v2.1", "f_v21"),
        ("a0", "f_a0"),
        ("a1", "f_a1"),
        ("v21_a0_blend50", "f_blend_a0"),
        ("v21_a1_blend50", "f_blend_a1"),
    ):
        s = score_series(m["actual"], m[col])
        s["method"] = name
        rows.append(s)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Robustness
# ---------------------------------------------------------------------------


def sku_origin_wins_policies(panel: pd.DataFrame) -> dict[str, Any]:
    hy_w = v2_w = ties = 0
    diffs = []
    for _, g in panel.groupby(["product", "forecast_origin"], sort=False):
        wh = portfolio_wmape_pct(g["actual"], g["f_a0_hybrid"])
        wv = portfolio_wmape_pct(g["actual"], g["f_v2_policy"])
        if not (np.isfinite(wh) and np.isfinite(wv)):
            continue
        diffs.append(wh - wv)
        if wh < wv:
            hy_w += 1
        elif wv < wh:
            v2_w += 1
        else:
            ties += 1
    return {
        "hybrid_wins": hy_w,
        "v2_wins": v2_w,
        "ties": ties,
        "median_delta_wmape_hybrid_minus_v2": float(np.nanmedian(diffs)) if diffs else np.nan,
        "n_groups": hy_w + v2_w + ties,
    }


def loo_origins_policies(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dropped in sorted(panel["forecast_origin"].unique()):
        g = panel.loc[panel["forecast_origin"] != dropped]
        sv = score_series(g["actual"], g["f_v2_policy"])
        sh = score_series(g["actual"], g["f_a0_hybrid"])
        rows.append(
            {
                "dropped_origin": int(dropped),
                "n_rows": int(len(g)),
                "wmape_v2": sv["portfolio_wmape_pct"],
                "wmape_hybrid": sh["portfolio_wmape_pct"],
                "lift_pp": float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"]),
                "hybrid_beats": bool(sh["portfolio_wmape_pct"] < sv["portfolio_wmape_pct"]),
            }
        )
    return pd.DataFrame(rows)


def ae_gain_by_sku(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in panel.groupby("product", sort=True):
        ae_red = float(g["ae_v2_policy"].sum() - g["ae_a0_hybrid"].sum())
        sv = score_series(g["actual"], g["f_v2_policy"])
        sh = score_series(g["actual"], g["f_a0_hybrid"])
        rows.append(
            {
                "product": product,
                "n_rows": int(len(g)),
                "wmape_v2": sv["portfolio_wmape_pct"],
                "wmape_hybrid": sh["portfolio_wmape_pct"],
                "ae_reduction_vs_v2": ae_red,
                "lift_pp": float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values("ae_reduction_vs_v2", ascending=False)


def drop_sensitivity(panel: pd.DataFrame, sku_rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # drop top 1/2/3 gainers
    for k in (1, 2, 3):
        dropped = sku_rank.head(k)["product"].tolist()
        kept = panel.loc[~panel["product"].isin(dropped)]
        sv = score_series(kept["actual"], kept["f_v2_policy"])
        sh = score_series(kept["actual"], kept["f_a0_hybrid"])
        rows.append(
            {
                "scenario": f"drop_top_{k}_gainers",
                "dropped": dropped,
                "n_rows": int(len(kept)),
                "wmape_v2": sv["portfolio_wmape_pct"],
                "wmape_hybrid": sh["portfolio_wmape_pct"],
                "lift_pp": float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"]),
                "hybrid_still_beats": bool(
                    sh["portfolio_wmape_pct"] < sv["portfolio_wmape_pct"]
                ),
            }
        )
    # drop largest loss (most negative ae reduction)
    loser = sku_rank.sort_values("ae_reduction_vs_v2", ascending=True).head(1)
    dropped = loser["product"].tolist()
    kept = panel.loc[~panel["product"].isin(dropped)]
    sv = score_series(kept["actual"], kept["f_v2_policy"])
    sh = score_series(kept["actual"], kept["f_a0_hybrid"])
    rows.append(
        {
            "scenario": "drop_largest_loser",
            "dropped": dropped,
            "n_rows": int(len(kept)),
            "wmape_v2": sv["portfolio_wmape_pct"],
            "wmape_hybrid": sh["portfolio_wmape_pct"],
            "lift_pp": float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"]),
            "hybrid_still_beats": bool(
                sh["portfolio_wmape_pct"] < sv["portfolio_wmape_pct"]
            ),
        }
    )
    return pd.DataFrame(rows)


def sku_cluster_bootstrap(panel: pd.DataFrame, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Resample products; keep all rows for each drawn product."""
    rng = np.random.default_rng(seed)
    products = panel["product"].unique()
    lifts = []
    for _ in range(b):
        draw = rng.choice(products, size=len(products), replace=True)
        parts = [panel.loc[panel["product"] == p] for p in draw]
        g = pd.concat(parts, ignore_index=True)
        sv = score_series(g["actual"], g["f_v2_policy"])
        sh = score_series(g["actual"], g["f_a0_hybrid"])
        lifts.append(float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"]))
    arr = np.asarray(lifts, dtype=float)
    return {
        "B": b,
        "seed": seed,
        "lift_mean": float(np.mean(arr)),
        "lift_ci95_low": float(np.quantile(arr, 0.025)),
        "lift_ci95_high": float(np.quantile(arr, 0.975)),
        "frac_positive": float(np.mean(arr > 0)),
    }


# ---------------------------------------------------------------------------
# 6. Seed dispersion
# ---------------------------------------------------------------------------


def a0_seed_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    a0 = panel.loc[
        (panel["engine"] == "a0")
        & (panel["job_status"] == "SUCCESS")
        & (panel["model"].astype(str) != "naive_last_value")
    ].copy()
    if a0.empty:
        return pd.DataFrame()
    rows = []
    for (product, origin), g in a0.groupby(["product", "forecast_origin"], sort=True):
        # per seed mean forecast across horizons, then std across seeds
        by_seed = g.groupby("seed")["forecast"].mean()
        if len(by_seed) < 2:
            continue
        rows.append(
            {
                "product": product,
                "forecast_origin": int(origin),
                "n_seeds": int(len(by_seed)),
                "seed_mean_of_means": float(by_seed.mean()),
                "seed_std_of_means": float(by_seed.std(ddof=0)),
                "seed_range": float(by_seed.max() - by_seed.min()),
                "cv": float(by_seed.std(ddof=0) / abs(by_seed.mean()))
                if abs(by_seed.mean()) > 1e-9
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("seed_std_of_means", ascending=False)


def deterioration_cases(policy: pd.DataFrame, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = policy.copy()
    cell["ae_delta_hybrid_minus_v2"] = cell["ae_a0_hybrid"] - cell["ae_v2_policy"]
    top_cells = cell.nlargest(top_n, "ae_delta_hybrid_minus_v2")[
        KEYS
        + [
            "ae_v2_policy",
            "ae_a0_hybrid",
            "ae_delta_hybrid_minus_v2",
            "source_a0_hybrid",
            "f_v2_policy",
            "f_a0_hybrid",
            "actual",
        ]
    ]
    sku = ae_gain_by_sku(policy)
    top_loss = sku.sort_values("ae_reduction_vs_v2", ascending=True).head(top_n)
    return top_cells, top_loss


# ---------------------------------------------------------------------------
# Recommendation + report
# ---------------------------------------------------------------------------


def make_recommendation(
    *,
    audit_meta: dict[str, Any],
    primary_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    wins: dict[str, Any],
    sens: pd.DataFrame,
    boot: dict[str, Any],
    fiveway: pd.DataFrame,
) -> dict[str, Any]:
    bullets: list[str] = []
    suspicious = bool(audit_meta.get("explain", {}).get("suspicious_any"))

    hy_p = primary_metrics.loc[
        primary_metrics["policy"] == "A0_HYBRID_POLICY", "portfolio_wmape_pct"
    ]
    v2_p = primary_metrics.loc[
        primary_metrics["policy"] == "V2_POLICY", "portfolio_wmape_pct"
    ]
    lift_primary = float(v2_p.iloc[0] - hy_p.iloc[0]) if len(hy_p) and len(v2_p) else np.nan

    hy_f = pooled_metrics.loc[
        pooled_metrics["policy"] == "A0_HYBRID_POLICY", "portfolio_wmape_pct"
    ]
    v2_f = pooled_metrics.loc[
        pooled_metrics["policy"] == "V2_POLICY", "portfolio_wmape_pct"
    ]
    lift_full = float(v2_f.iloc[0] - hy_f.iloc[0]) if len(hy_f) and len(v2_f) else np.nan

    drop3 = sens.loc[sens["scenario"] == "drop_top_3_gainers"]
    drop3_ok = bool(drop3["hybrid_still_beats"].iloc[0]) if not drop3.empty else False
    ci_excludes_0 = bool(boot["lift_ci95_low"] > 0)
    hybrid_more_wins = int(wins.get("hybrid_wins", 0)) >= int(wins.get("v2_wins", 0))

    bullets.append(
        f"Primary new_sku×late_origin policy lift (V2−hybrid) = {lift_primary:+.3g} pp."
    )
    bullets.append(f"Full-panel policy lift = {lift_full:+.3g} pp.")
    bullets.append(
        f"SKU×origin wins hybrid/v2/ties = {wins.get('hybrid_wins')}/"
        f"{wins.get('v2_wins')}/{wins.get('ties')}; "
        f"median ΔWMAPE(hybrid−v2) = {wins.get('median_delta_wmape_hybrid_minus_v2'):+.3g}."
    )
    bullets.append(
        f"SKU-cluster bootstrap 95% CI for lift = "
        f"[{boot['lift_ci95_low']:+.3g}, {boot['lift_ci95_high']:+.3g}] "
        f"(frac>0 = {boot['frac_positive']:.2%})."
    )
    bullets.append(f"After drop top-3 gain SKUs, hybrid still beats = {drop3_ok}.")

    if suspicious:
        choice = "C"
        bullets.append(
            "A0 INELIGIBLE audit still flags suspicious late-origin / long-history "
            "or empty-prediction gaps → insufficient evidence for a clean freeze."
        )
    elif (
        lift_primary > 0
        and (lift_full > 0 or np.isnan(lift_full))
        and hybrid_more_wins
        and (ci_excludes_0 or boot["frac_positive"] >= 0.90)
        and drop3_ok
    ):
        choice = "A"
        bullets.append(
            "Hybrid beats V2_POLICY on primary with supportive robustness → "
            "freeze V2.1+A0 with V2.1 fallback as 400-SKU challenger."
        )
    elif lift_primary <= 0 or (not hybrid_more_wins and lift_primary < 0.5):
        choice = "B"
        bullets.append(
            "Hybrid does not clearly beat V2_POLICY under matched policy eval → "
            "retain V2.1 alone."
        )
    else:
        # Mixed signals
        if lift_primary > 0 and not drop3_ok:
            choice = "B"
            bullets.append(
                "Primary lift positive but concentrated in few SKUs (fails drop-top-3) → "
                "retain V2.1 alone."
            )
        elif lift_primary > 0 and not ci_excludes_0:
            choice = "B"
            bullets.append(
                "Primary lift positive but SKU-cluster CI includes 0 → retain V2.1 alone."
            )
        else:
            choice = "B"
            bullets.append("Mixed robustness → retain V2.1 alone.")

    # five-way note
    if not fiveway.empty:
        best = fiveway.sort_values("portfolio_wmape_pct").iloc[0]
        bullets.append(
            f"Five-way common panel best method = {best['method']} "
            f"at {float(best['portfolio_wmape_pct']):.2f}% (n={int(best['n_rows'])})."
        )

    return {"choice": choice, "rationale_bullets": bullets, "lift_primary_pp": lift_primary}


def write_report(
    *,
    out_dir: Path,
    inelig: pd.DataFrame,
    summary: pd.DataFrame,
    audit_meta: dict[str, Any],
    metrics_by_slice: pd.DataFrame,
    by_origin: pd.DataFrame,
    by_horizon: pd.DataFrame,
    fiveway: pd.DataFrame,
    wins_primary: dict[str, Any],
    wins_full: dict[str, Any],
    loo: pd.DataFrame,
    sens: pd.DataFrame,
    boot_primary: dict[str, Any],
    boot_full: dict[str, Any],
    seed_disp: pd.DataFrame,
    top_loss: pd.DataFrame,
    oracle_capture: dict[str, Any],
    rec: dict[str, Any],
    strat_notes: list[str],
) -> str:
    expl = audit_meta.get("explain", {})
    mg = audit_meta.get("missing_groups")
    sections: list[str] = [
        "# Archval55 decision analysis (post-hoc)",
        "",
        "Not independent confirmation. No refits / no weight search. "
        "Artifacts: `archval55_decision_v1/`.",
        "",
        "## 1. INELIGIBLE audit (157 jobs)",
        "",
        f"- Total INELIGIBLE jobs: **{expl.get('n_ineligible_jobs')}** "
        f"by engine: `{expl.get('by_engine')}`.",
        f"- `new_sku × late_origin` A0 ineligible jobs: "
        f"**{expl.get('new_sku_late_a0_ineligible_jobs')}** across "
        f"**{expl.get('new_sku_late_a0_groups_with_any_ineligible_seed')}** "
        f"product×origin groups (some still have ≥1 SUCCESS seed).",
        "",
        "### Why A0 pairwise n=675 vs A1 n=900 on new_sku × late_origin",
        "",
        "Full SUCCESS coverage on that slice would be 30 SKUs × 2 origins × 15 horizons = **900** cells. "
        "Confirmation ranking excludes INELIGIBLE/naive A0 rows, then seed-means remaining SUCCESS seeds. "
        "A product×origin keeps 15 cells if **≥1** A0 seed succeeded. "
        f"**{expl.get('n_a0_full_dropout_groups', '?')}** groups have **zero** SUCCESS A0 seeds "
        f"→ drop {expl.get('n_a0_full_dropout_groups', 0)}×15 cells → **675** remaining "
        f"(= {900 - 15 * int(expl.get('n_a0_full_dropout_groups') or 0)}). "
        "A1 has zero INELIGIBLE jobs on that slice. "
        "Late-origin A0 failures are coded `neural_empty_predictions_ineligible` "
        "(empty backtest predictions **without** an insufficient-history tag) despite long "
        f"`history_length` — classified as **possible implementation/config issue**, not "
        "classic short-history ineligibility.",
        "",
        "### Summary by engine × origin_cohort × reason",
        "",
        _md_table(summary),
        "",
    ]
    if isinstance(mg, pd.DataFrame) and not mg.empty:
        sections.extend(
            [
                "### Missing A0 groups on new_sku × late_origin",
                "",
                _md_table(
                    mg.head(30),
                    {"history_length"},
                ),
                "",
            ]
        )
    sections.extend(
        [
            f"Suspicious coverage flags: **{expl.get('suspicious_any')}** "
            f"(gap class counts on late-new A0: `{expl.get('gap_class_counts')}`).",
            "",
            "## 2–3. Deployable policies (identical actual-available rows)",
            "",
            "V2_POLICY = V2.1 else naive. A0_HYBRID = 50/50 when both native; else available native; else naive.",
            "",
            "### By slice",
            "",
            _md_table(
                metrics_by_slice,
                {
                    "mae",
                    "rmse",
                    "portfolio_wmape_pct",
                    "bias",
                    "native_coverage_pct",
                    "fallback_coverage_pct",
                    "sku_origin_wmape_p90",
                    "sku_origin_wmape_p95",
                    "lift_hybrid_vs_v2_pp",
                },
            ),
            "",
            "### By origin (pooled)",
            "",
            _md_table(by_origin, {"portfolio_wmape_pct", "mae", "lift_pp", "native_coverage_pct"}),
            "",
            "### By horizon (pooled)",
            "",
            _md_table(by_horizon, {"portfolio_wmape_pct", "mae", "lift_pp"}),
            "",
            "## 4. Five-way common panel (SUCCESS V2.1 ∩ A0 ∩ A1 only)",
            "",
            _md_table(
                fiveway,
                {"mae", "rmse", "portfolio_wmape_pct", "bias"},
            ),
            "",
            "## 5. Robustness (A0_HYBRID vs V2_POLICY)",
            "",
            f"- **Primary** wins hybrid/v2/ties = {wins_primary.get('hybrid_wins')}/"
            f"{wins_primary.get('v2_wins')}/{wins_primary.get('ties')}; "
            f"median Δ = {wins_primary.get('median_delta_wmape_hybrid_minus_v2'):+.3g}.",
            f"- **Full** wins = {wins_full.get('hybrid_wins')}/{wins_full.get('v2_wins')}/"
            f"{wins_full.get('ties')}; median Δ = {wins_full.get('median_delta_wmape_hybrid_minus_v2'):+.3g}.",
            "",
            "### Leave-one-origin-out (primary)",
            "",
            _md_table(loo, {"wmape_v2", "wmape_hybrid", "lift_pp"}),
            "",
            "### Drop top gainers / largest loser (primary)",
            "",
            _md_table(sens, {"wmape_v2", "wmape_hybrid", "lift_pp"}),
            "",
            f"- Primary SKU-cluster bootstrap (B={boot_primary.get('B')}): "
            f"lift CI95 = [{boot_primary.get('lift_ci95_low'):+.3g}, "
            f"{boot_primary.get('lift_ci95_high'):+.3g}]; "
            f"frac>0 = {boot_primary.get('frac_positive'):.2%}.",
            f"- Full-panel bootstrap: "
            f"[{boot_full.get('lift_ci95_low'):+.3g}, {boot_full.get('lift_ci95_high'):+.3g}]; "
            f"frac>0 = {boot_full.get('frac_positive'):.2%}.",
            "",
            "## 6. A0 seed dispersion and hybrid deterioration",
            "",
            "### Highest seed std (product×origin)",
            "",
            _md_table(
                seed_disp.head(12) if not seed_disp.empty else seed_disp,
                {"seed_mean_of_means", "seed_std_of_means", "seed_range", "cv"},
            ),
            "",
            "### Largest hybrid AE losses vs V2_POLICY (by SKU)",
            "",
            _md_table(
                top_loss.head(10),
                {"wmape_v2", "wmape_hybrid", "ae_reduction_vs_v2", "lift_pp"},
            ),
            "",
            "### Stratifications",
            "",
        ]
    )
    for note in strat_notes:
        sections.append(f"- {note}")
    sections.extend(
        [
            "",
            "## 7. Oracle ceilings (hindsight only — not deployable)",
            "",
            str(oracle_capture.get("text", "")),
            "",
            "## 8. Recommendation",
            "",
            f"**Choice: {rec['choice']}**",
            "",
        ]
    )
    for b in rec.get("rationale_bullets", []):
        sections.append(f"- {b}")
    sections.append("")
    labels = {
        "A": "Freeze V2.1+A0 with V2.1 fallback as the 400-SKU challenger.",
        "B": "Retain V2.1 alone.",
        "C": "Insufficient evidence because of a coverage/implementation issue.",
    }
    sections.extend(
        [
            "",
            f"Interpretation: {labels.get(rec['choice'], '')}",
            "",
        ]
    )
    text = "\n".join(sections)
    (out_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--confirm-dir", type=Path, default=DEFAULT_CONFIRM)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_skus = set(
        load_universe_product_names("v3a_architecture_validation_products", validate=True)
    )
    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    panel = pd.read_parquet(Path(args.export_dir) / "forecasts_panel.parquet")
    panel["product"] = panel["product"].astype(str)
    panel["engine"] = panel["engine"].astype(str).str.lower()
    for c in ("forecast_origin", "target_date", "horizon"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(int)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    if "job_status" not in panel.columns:
        panel["job_status"] = np.where(panel.get("ineligible", False), "INELIGIBLE", "SUCCESS")

    # --- 1. Audit ---
    inelig, summary, audit_meta = audit_ineligible(Path(args.run_dir), sales, old_skus)

    # Reconcile SQLite INELIGIBLE with the export panel (A0 empty-fix overlays
    # leave the original run untouched, so fixed keys would otherwise linger).
    if not inelig.empty and "product" in panel.columns:
        success_keys: set[tuple[str, str, int, Optional[int]]] = set()
        status_ok = (
            panel["job_status"].astype(str).eq("SUCCESS")
            if "job_status" in panel.columns
            else pd.Series(True, index=panel.index)
        )
        not_inelig = (
            ~panel["ineligible"].fillna(False).astype(bool)
            if "ineligible" in panel.columns
            else pd.Series(True, index=panel.index)
        )
        a0_ok = panel.loc[
            (panel["engine"].astype(str).str.lower() == "a0") & status_ok & not_inelig
        ]
        for r in a0_ok.itertuples(index=False):
            seed = getattr(r, "seed", None)
            if seed is None or (isinstance(seed, float) and pd.isna(seed)):
                seed = None
            else:
                seed = int(seed)
            success_keys.add(
                ("a0", str(r.product), int(r.forecast_origin), seed)
            )
        if success_keys:
            keep = []
            for r in inelig.itertuples(index=False):
                seed = r.seed if pd.notna(r.seed) else None
                if seed is not None:
                    seed = int(seed)
                key = (str(r.engine).lower(), str(r.product), int(r.forecast_origin), seed)
                keep.append(key not in success_keys)
            n_before = len(inelig)
            inelig = inelig.loc[keep].reset_index(drop=True)
            n_dropped = n_before - len(inelig)
            if n_dropped:
                # Rebuild summary / explain from reconciled table
                summary = (
                    inelig.groupby(
                        ["engine", "origin_cohort", "reason_code", "gap_classification"],
                        as_index=False,
                    )
                    .agg(n_jobs=("product_id", "size"), n_products=("product", "nunique"))
                    .sort_values(
                        ["engine", "origin_cohort", "n_jobs"],
                        ascending=[True, True, False],
                    )
                    if not inelig.empty
                    else pd.DataFrame()
                )
                late_new_a0 = inelig.loc[
                    (inelig["engine"] == "a0")
                    & (inelig["sku_cohort"] == "new_sku")
                    & (inelig["origin_cohort"] == "late_origin")
                ]
                missing_groups = (
                    late_new_a0.groupby(["product", "forecast_origin"], as_index=False)
                    .agg(
                        n_ineligible_seeds=("seed", "nunique"),
                        history_length=("history_length", "first"),
                        reason_code=("reason_code", "first"),
                        gap_classification=("gap_classification", "first"),
                    )
                    if not late_new_a0.empty
                    else pd.DataFrame()
                )
                audit_meta["missing_groups"] = missing_groups
                expl = audit_meta.get("explain") or {}
                expl.update(
                    {
                        "n_ineligible_jobs": int(len(inelig)),
                        "by_engine": inelig.groupby("engine").size().astype(int).to_dict()
                        if not inelig.empty
                        else {},
                        "new_sku_late_a0_ineligible_jobs": int(len(late_new_a0)),
                        "new_sku_late_a0_groups_with_any_ineligible_seed": int(
                            len(missing_groups)
                        ),
                        "gap_class_counts": late_new_a0["gap_classification"]
                        .value_counts()
                        .to_dict()
                        if not late_new_a0.empty
                        else {},
                        "suspicious_any": bool(
                            (
                                (inelig["engine"].astype(str).str.lower() == "a0")
                                & inelig["gap_classification"].isin(
                                    {
                                        "suspicious_late_long_history",
                                        "possible_implementation_issue",
                                    }
                                )
                            ).any()
                        )
                        if not inelig.empty
                        else False,
                        "n_ineligible_reconciled_away_vs_panel": int(n_dropped),
                        "a1_empty_predictions_remaining": int(
                            (
                                (inelig["engine"].astype(str).str.lower() == "a1")
                                & (
                                    inelig["reason_code"]
                                    == "neural_empty_predictions_ineligible"
                                )
                            ).sum()
                        )
                        if not inelig.empty
                        else 0,
                    }
                )
                audit_meta["explain"] = expl

    inelig.to_csv(out_dir / "ineligible_jobs.csv", index=False)
    summary.to_csv(out_dir / "ineligible_summary_by_engine_origin_reason.csv", index=False)
    mg = audit_meta.get("missing_groups")
    if isinstance(mg, pd.DataFrame) and not mg.empty:
        mg.to_csv(out_dir / "a0_missing_groups_new_sku_late.csv", index=False)

    # Count full A0 dropouts on new×late (zero SUCCESS seeds) from export panel.
    a0_nl = panel.loc[
        (panel["engine"] == "a0")
        & (panel["sku_cohort"] == "new_sku")
        & (panel["origin_cohort"] == "late_origin")
    ]
    a0_success_groups = set(
        map(
            tuple,
            a0_nl.loc[
                (a0_nl["job_status"] == "SUCCESS")
                & (a0_nl["model"].astype(str) != "naive_last_value"),
                ["product", "forecast_origin"],
            ]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    all_nl_groups = set(
        map(
            tuple,
            a0_nl[["product", "forecast_origin"]].drop_duplicates().to_numpy(),
        )
    )
    # Expected 30×2=60 groups from MVP new×late
    mvp = set(load_universe_product_names("mvp_products", validate=True))
    new_skus = mvp - old_skus
    expected_groups = {(p, o) for p in new_skus for o in LATE_ORIGINS}
    full_dropouts = expected_groups - a0_success_groups
    audit_meta["explain"]["n_a0_full_dropout_groups"] = int(len(full_dropouts))
    audit_meta["explain"]["n_a0_success_groups"] = int(len(a0_success_groups))
    audit_meta["explain"]["full_dropout_groups"] = [
        {"product": p, "forecast_origin": int(o)} for p, o in sorted(full_dropouts)
    ]
    (out_dir / "ineligible_explain.json").write_text(
        json.dumps(audit_meta["explain"], indent=2, default=str) + "\n", encoding="utf-8"
    )
    pd.DataFrame(audit_meta["explain"]["full_dropout_groups"]).to_csv(
        out_dir / "a0_full_dropout_groups_new_sku_late.csv", index=False
    )

    # --- 2–3. Policies ---
    hist = history_length_map(sales)
    last_vals = last_train_value_map(sales)
    policy = build_policy_panel(panel, sales, last_vals, hist)
    policy.to_parquet(out_dir / "policy_panel.parquet", index=False)

    metric_rows = []
    for slice_name, sku, origin in SLICES:
        sub = filter_slice_df(policy, sku, origin)
        metric_rows.append(policy_metrics(sub, slice_name))
    metrics_by_slice = pd.concat(metric_rows, ignore_index=True)
    metrics_by_slice.to_csv(out_dir / "policy_metrics_by_slice.csv", index=False)

    # by origin / horizon on pooled
    origin_rows = []
    for origin, g in policy.groupby("forecast_origin", sort=True):
        m = policy_metrics(g, f"origin_{origin}")
        m["forecast_origin"] = int(origin)
        # compact: lift only on hybrid row
        v2 = float(m.loc[m["policy"] == "V2_POLICY", "portfolio_wmape_pct"].iloc[0])
        hy = float(m.loc[m["policy"] == "A0_HYBRID_POLICY", "portfolio_wmape_pct"].iloc[0])
        origin_rows.append(
            {
                "forecast_origin": int(origin),
                "n_rows": int(len(g)),
                "wmape_v2": v2,
                "wmape_hybrid": hy,
                "lift_pp": v2 - hy,
                "native_coverage_pct_v21": float(100 * g["native_v21"].mean()),
                "native_coverage_pct_a0": float(100 * g["native_a0"].mean()),
                "mae_v2": float(m.loc[m["policy"] == "V2_POLICY", "mae"].iloc[0]),
                "mae_hybrid": float(m.loc[m["policy"] == "A0_HYBRID_POLICY", "mae"].iloc[0]),
            }
        )
    by_origin = pd.DataFrame(origin_rows)
    by_origin.to_csv(out_dir / "policy_by_origin.csv", index=False)

    horizon_rows = []
    for h, g in policy.groupby("horizon", sort=True):
        m = policy_metrics(g, f"h_{h}")
        v2 = float(m.loc[m["policy"] == "V2_POLICY", "portfolio_wmape_pct"].iloc[0])
        hy = float(m.loc[m["policy"] == "A0_HYBRID_POLICY", "portfolio_wmape_pct"].iloc[0])
        horizon_rows.append(
            {
                "horizon": int(h),
                "n_rows": int(len(g)),
                "wmape_v2": v2,
                "wmape_hybrid": hy,
                "lift_pp": v2 - hy,
                "mae_v2": float(m.loc[m["policy"] == "V2_POLICY", "mae"].iloc[0]),
                "mae_hybrid": float(m.loc[m["policy"] == "A0_HYBRID_POLICY", "mae"].iloc[0]),
            }
        )
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon.to_csv(out_dir / "policy_by_horizon.csv", index=False)

    primary = filter_slice_df(policy, "new_sku", "late_origin")
    pooled_metrics = metrics_by_slice.loc[metrics_by_slice["slice"] == "pooled"]
    primary_metrics = metrics_by_slice.loc[metrics_by_slice["slice"] == "new_sku_x_late_origin"]

    # --- 4. Five-way ---
    # Prefer primary slice five-way; also report pooled five-way n
    fw_primary = fiveway_panel(primary)
    # If primary intersection empty-ish, still use whatever we have; plan expects ~675
    if len(fw_primary) < 100:
        fw_primary = fiveway_panel(policy)
    fiveway = fiveway_headline(fw_primary)
    fiveway["panel"] = "new_sku_x_late_origin" if len(fw_primary) else "pooled"
    fiveway.to_csv(out_dir / "fiveway_headline.csv", index=False)
    fw_primary.to_parquet(out_dir / "fiveway_panel.parquet", index=False)

    # --- 5. Robustness ---
    wins_primary = sku_origin_wins_policies(primary)
    wins_full = sku_origin_wins_policies(policy)
    loo = loo_origins_policies(primary)
    loo.to_csv(out_dir / "loo_origins_primary.csv", index=False)
    sku_rank = ae_gain_by_sku(primary)
    sku_rank.to_csv(out_dir / "sku_ae_rank_primary.csv", index=False)
    sens = drop_sensitivity(primary, sku_rank)
    sens.to_csv(out_dir / "sensitivity_drop_primary.csv", index=False)
    boot_primary = sku_cluster_bootstrap(primary, b=int(args.bootstrap_b))
    boot_full = sku_cluster_bootstrap(policy, b=int(args.bootstrap_b))
    (out_dir / "bootstrap_sku_cluster.json").write_text(
        json.dumps({"primary": boot_primary, "full": boot_full}, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- 6. Seed / deterioration / strat ---
    seed_disp = a0_seed_dispersion(panel)
    seed_disp.to_csv(out_dir / "a0_seed_dispersion.csv", index=False)
    top_cells, top_loss = deterioration_cases(primary)
    top_cells.to_csv(out_dir / "hybrid_deterioration_cells_primary.csv", index=False)
    top_loss.to_csv(out_dir / "hybrid_deterioration_skus_primary.csv", index=False)

    strat_notes: list[str] = []
    # flat vs dynamic on cells where v21 native and hybrid used blend or v21
    has_model = primary["model_v21"].notna().sum() if "model_v21" in primary.columns else 0
    if has_model:
        for fam in ("flat", "dynamic", "other_or_missing"):
            g = primary.loc[primary["v21_family"] == fam]
            if len(g) < 30:
                continue
            sv = score_series(g["actual"], g["f_v2_policy"])
            sh = score_series(g["actual"], g["f_a0_hybrid"])
            strat_notes.append(
                f"V2.1 family **{fam}** (n={len(g)}): "
                f"WMAPE V2={sv['portfolio_wmape_pct']:.2f}% hybrid={sh['portfolio_wmape_pct']:.2f}% "
                f"lift={sv['portfolio_wmape_pct']-sh['portfolio_wmape_pct']:+.2f} pp."
            )
    else:
        strat_notes.append("V2.1 flat/dynamic stratification skipped (model_v21 missing).")

    # history buckets
    primary = primary.copy()
    primary["hist_bucket"] = pd.cut(
        primary["history_length"],
        bins=[-1, 12, 18, 24, 36, 10_000],
        labels=["<=12", "13-18", "19-24", "25-36", ">36"],
    )
    for bucket, g in primary.groupby("hist_bucket", observed=False):
        if len(g) < 30:
            continue
        sv = score_series(g["actual"], g["f_v2_policy"])
        sh = score_series(g["actual"], g["f_a0_hybrid"])
        strat_notes.append(
            f"History bucket **{bucket}** (n={len(g)}): "
            f"lift={sv['portfolio_wmape_pct']-sh['portfolio_wmape_pct']:+.2f} pp."
        )
    strat_notes.append(
        "Demand-type field not present on export panel — skipped."
    )

    # --- 7. Oracle capture ---
    oracle_path = Path(args.confirm_dir) / "oracle_opportunity_by_slice.csv"
    oracle_capture: dict[str, Any] = {"text": ""}
    if oracle_path.is_file():
        oc = pd.read_csv(oracle_path)
        row = oc.loc[
            (oc["slice"] == "new_sku_x_late_origin") & (oc["neural"] == "a0")
        ]
        # Policy hybrid lift on primary among cells where both v21 and a0 native (blend cells)
        blend_cells = primary.loc[primary["source_a0_hybrid"] == "blend50_v21_a0"]
        if not blend_cells.empty and not row.empty:
            sv = score_series(blend_cells["actual"], blend_cells["f_v2_policy"])
            sh = score_series(blend_cells["actual"], blend_cells["f_a0_hybrid"])
            captured = float(sv["portfolio_wmape_pct"] - sh["portfolio_wmape_pct"])
            gap_cell = float(row["gap_cell_pp"].iloc[0])
            gap_so = float(row["gap_sku_origin_pp"].iloc[0])
            frac_cell = captured / gap_cell if gap_cell > 0 else float("nan")
            frac_so = captured / gap_so if gap_so > 0 else float("nan")
            oracle_capture = {
                "captured_blend_cells_pp": captured,
                "gap_cell_pp": gap_cell,
                "gap_sku_origin_pp": gap_so,
                "capture_frac_vs_cell": frac_cell,
                "capture_frac_vs_sku_origin": frac_so,
                "n_blend_cells": int(len(blend_cells)),
                "text": (
                    "Hindsight ceilings from confirmation (non-deployable). "
                    f"On primary blend-eligible cells (n={len(blend_cells)}), "
                    f"A0_HYBRID captures {captured:+.2f} pp vs V2_POLICY; "
                    f"cell oracle gap was {gap_cell:.2f} pp "
                    f"(capture {frac_cell:.0%} of cell ceiling); "
                    f"SKU×origin gap {gap_so:.2f} pp "
                    f"(capture {frac_so:.0%} of SKU×origin ceiling)."
                ),
            }
        else:
            oracle_capture["text"] = (
                "Oracle CSV present but could not align primary blend cells for capture frac."
            )
        (out_dir / "oracle_policy_capture.json").write_text(
            json.dumps({k: v for k, v in oracle_capture.items() if k != "text"}, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
    else:
        oracle_capture["text"] = "Oracle opportunity CSV not found; skipped capture calc."

    # --- 8. Recommendation ---
    rec = make_recommendation(
        audit_meta=audit_meta,
        primary_metrics=primary_metrics,
        pooled_metrics=pooled_metrics,
        wins=wins_primary,
        sens=sens,
        boot=boot_primary,
        fiveway=fiveway,
    )
    (out_dir / "recommendation.json").write_text(
        json.dumps(rec, indent=2) + "\n", encoding="utf-8"
    )

    report = write_report(
        out_dir=out_dir,
        inelig=inelig,
        summary=summary,
        audit_meta=audit_meta,
        metrics_by_slice=metrics_by_slice,
        by_origin=by_origin,
        by_horizon=by_horizon,
        fiveway=fiveway,
        wins_primary=wins_primary,
        wins_full=wins_full,
        loo=loo,
        sens=sens,
        boot_primary=boot_primary,
        boot_full=boot_full,
        seed_disp=seed_disp,
        top_loss=top_loss,
        oracle_capture=oracle_capture,
        rec=rec,
        strat_notes=strat_notes,
    )
    print(f"recommendation={rec['choice']}")
    print(f"wrote {out_dir / 'report.md'} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
