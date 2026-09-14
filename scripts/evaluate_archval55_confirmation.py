#!/usr/bin/env python3
"""Archval55 confirmation eval: pairwise engines/blends + 2x2 slices.

Pooled results are descriptive only — not independent confirmation.
Primary confirmation weight: new_sku x late_origin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct
from pkg.benchmark.universes import load_universe_product_names

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_EXPORT = RESULTS / "archval55_export"
DEFAULT_OUT = RESULTS / "archval55_confirmation"
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
ENGINES = ("v2.1", "a0", "a1")
NEURALS = ("a0", "a1")
EARLY_ORIGINS = frozenset({140104, 140201})
LATE_ORIGINS = frozenset({140301, 140402})
SLICES = (
    ("pooled", None, None),
    ("old_sku_x_early_origin", "old_sku", "early_origin"),
    ("old_sku_x_late_origin", "old_sku", "late_origin"),
    ("new_sku_x_early_origin", "new_sku", "early_origin"),
    ("new_sku_x_late_origin", "new_sku", "late_origin"),
)
# Detailed oracle opportunity CSVs / report sections (hindsight ceilings only).
ORACLE_SLICES = ("pooled", "new_sku_x_late_origin")


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


def ensure_cohort_tags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "sku_cohort" not in out.columns or out["sku_cohort"].isna().all():
        old = set(load_universe_product_names("v3a_architecture_validation_products", validate=True))
        out["sku_cohort"] = np.where(out["product"].isin(old), "old_sku", "new_sku")
    if "origin_cohort" not in out.columns or out["origin_cohort"].isna().all():
        out["origin_cohort"] = np.where(
            out["forecast_origin"].isin(EARLY_ORIGINS),
            "early_origin",
            np.where(out["forecast_origin"].isin(LATE_ORIGINS), "late_origin", "other_origin"),
        )
    return out


def load_engine_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    v21 = panel.loc[panel["engine"] == "v2.1"].copy()
    if "ineligible" not in v21.columns:
        v21["ineligible"] = False
    if "model" not in v21.columns:
        v21["model"] = ""
    frames["v2.1"] = (
        v21.groupby(KEYS, as_index=False)
        .agg(
            forecast=("forecast", "first"),
            actual=("actual", "first"),
            model=("model", "first"),
            ineligible=("ineligible", "max"),
            sku_cohort=("sku_cohort", "first"),
            origin_cohort=("origin_cohort", "first"),
        )
        .assign(engine="v2.1")
    )
    for eng in NEURALS:
        sub = panel.loc[panel["engine"] == eng].copy()
        if "ineligible" not in sub.columns:
            sub["ineligible"] = False
        if "seed" not in sub.columns:
            sub["seed"] = np.nan
        if "model" not in sub.columns:
            sub["model"] = ""
        g = sub.groupby(KEYS + ["engine"], as_index=False).agg(
            forecast=("forecast", "mean"),
            actual=("actual", "mean"),
            n_seeds=("seed", "nunique"),
            model=("model", "first"),
            ineligible=("ineligible", "max"),
            sku_cohort=("sku_cohort", "first"),
            origin_cohort=("origin_cohort", "first"),
        )
        frames[eng] = g
    return frames


def filter_slice(
    frames: dict[str, pd.DataFrame],
    sku: Optional[str],
    origin: Optional[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for eng, df in frames.items():
        g = df
        if sku is not None and "sku_cohort" in g.columns:
            g = g.loc[g["sku_cohort"] == sku]
        if origin is not None and "origin_cohort" in g.columns:
            g = g.loc[g["origin_cohort"] == origin]
        out[eng] = g.copy()
    return out


def score_series(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    m = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    if m.empty:
        return {
            "n_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "portfolio_wmape_pct": np.nan,
            "bias": np.nan,
            "ae_p90": np.nan,
            "ae_p95": np.nan,
            "top_decile_ae_share": np.nan,
        }
    err = m["forecast"] - m["actual"]
    ae = np.abs(err)
    p90 = float(np.quantile(ae, 0.90))
    top = ae >= p90
    return {
        "n_rows": int(len(m)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
        "ae_p90": p90,
        "ae_p95": float(np.quantile(ae, 0.95)),
        "top_decile_ae_share": float(ae[top].sum() / ae.sum()) if float(ae.sum()) else np.nan,
    }


def metrics_engine(g: pd.DataFrame, engine: str) -> dict[str, Any]:
    m = g.dropna(subset=["actual", "forecast"])
    base = score_series(m["actual"], m["forecast"]) if not m.empty else score_series(
        pd.Series(dtype=float), pd.Series(dtype=float)
    )
    base["engine"] = engine
    base["n_products"] = int(m["product"].nunique()) if not m.empty else 0
    base["n_origins"] = int(m["forecast_origin"].nunique()) if not m.empty else 0
    return base


def build_pairwise(v21: pd.DataFrame, neural: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    a = v21[KEYS + ["forecast", "actual"]].rename(
        columns={"forecast": "f_v21", "actual": "actual_v21"}
    )
    b = neural[KEYS + ["forecast", "actual"]].rename(
        columns={"forecast": "f_neural", "actual": "actual_neural"}
    )
    m = a.merge(b, on=KEYS, how="inner")
    m["actual"] = m["actual_v21"].where(m["actual_v21"].notna(), m["actual_neural"])
    m = m.dropna(subset=["actual", "f_v21", "f_neural"]).copy()
    m["f_blend50"] = 0.5 * m["f_v21"] + 0.5 * m["f_neural"]
    m["ae_v21"] = (m["f_v21"] - m["actual"]).abs()
    m["ae_neural"] = (m["f_neural"] - m["actual"]).abs()
    m["ae_blend50"] = (m["f_blend50"] - m["actual"]).abs()
    m["err_v21"] = m["f_v21"] - m["actual"]
    m["err_neural"] = m["f_neural"] - m["actual"]
    m["err_blend50"] = m["f_blend50"] - m["actual"]
    # Cell oracle (hindsight): per-row pick by lower abs error — not deployable.
    use_v21 = m["ae_v21"] <= m["ae_neural"]
    m["f_cell_oracle"] = np.where(use_v21, m["f_v21"], m["f_neural"])
    m["ae_cell_oracle"] = np.where(use_v21, m["ae_v21"], m["ae_neural"])
    m["cell_oracle_source"] = np.where(use_v21, "v2.1", neural_name)
    m["neural"] = neural_name
    return m.drop(columns=["actual_v21", "actual_neural"], errors="ignore")


def add_sku_origin_oracle(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    """Per (product, origin), pick engine with better group WMAPE (hindsight ceiling)."""
    out = panel.copy()
    choices: list[pd.DataFrame] = []
    for (_, _), g in out.groupby(["product", "forecast_origin"], sort=False):
        w_v = portfolio_wmape_pct(g["actual"], g["f_v21"])
        w_n = portfolio_wmape_pct(g["actual"], g["f_neural"])
        pick = "v2.1" if w_v <= w_n else neural_name
        part = g[KEYS].copy()
        part["sku_origin_oracle_source"] = pick
        part["f_sku_origin_oracle"] = np.where(
            pick == "v2.1", g["f_v21"].to_numpy(), g["f_neural"].to_numpy()
        )
        choices.append(part)
    if not choices:
        out["sku_origin_oracle_source"] = "v2.1"
        out["f_sku_origin_oracle"] = out["f_v21"]
        out["ae_sku_origin_oracle"] = out["ae_v21"]
        return out
    ch = pd.concat(choices, ignore_index=True)
    out = out.merge(ch, on=KEYS, how="left")
    out["ae_sku_origin_oracle"] = (out["f_sku_origin_oracle"] - out["actual"]).abs()
    return out


def _capture_frac(captured: float, gap: float) -> float:
    if gap is None or (isinstance(gap, float) and (np.isnan(gap) or gap <= 0)):
        return float("nan")
    return float(captured / gap)


def portfolio_opportunity(panel: pd.DataFrame, neural: str, slice_name: str) -> dict[str, Any]:
    """Portfolio WMAPE gaps and blend capture vs hindsight oracles."""
    s_v = score_series(panel["actual"], panel["f_v21"])
    s_n = score_series(panel["actual"], panel["f_neural"])
    s_b = score_series(panel["actual"], panel["f_blend50"])
    s_c = score_series(panel["actual"], panel["f_cell_oracle"])
    s_so = score_series(panel["actual"], panel["f_sku_origin_oracle"])
    wm_v = float(s_v["portfolio_wmape_pct"])
    wm_b = float(s_b["portfolio_wmape_pct"])
    wm_c = float(s_c["portfolio_wmape_pct"])
    wm_so = float(s_so["portfolio_wmape_pct"])
    captured = wm_v - wm_b
    gap_cell = wm_v - wm_c
    gap_so = wm_v - wm_so
    ae_v = float(panel["ae_v21"].sum())
    ae_b = float(panel["ae_blend50"].sum())
    ae_c = float(panel["ae_cell_oracle"].sum())
    ae_so = float(panel["ae_sku_origin_oracle"].sum())
    ae_cap = ae_v - ae_b
    ae_gap_c = ae_v - ae_c
    ae_gap_so = ae_v - ae_so
    return {
        "slice": slice_name,
        "neural": neural,
        "n_rows": int(len(panel)),
        "wmape_v21": wm_v,
        "wmape_neural": float(s_n["portfolio_wmape_pct"]),
        "wmape_blend50": wm_b,
        "wmape_cell_oracle": wm_c,
        "wmape_sku_origin_oracle": wm_so,
        "captured_blend_pp": captured,
        "gap_cell_pp": gap_cell,
        "gap_sku_origin_pp": gap_so,
        "capture_frac_vs_cell": _capture_frac(captured, gap_cell),
        "capture_frac_vs_sku_origin": _capture_frac(captured, gap_so),
        "ae_sum_v21": ae_v,
        "ae_reduction_blend": ae_cap,
        "ae_reduction_cell_oracle": ae_gap_c,
        "ae_reduction_sku_origin_oracle": ae_gap_so,
        "ae_capture_frac_vs_cell": _capture_frac(ae_cap, ae_gap_c),
        "ae_capture_frac_vs_sku_origin": _capture_frac(ae_cap, ae_gap_so),
        "bias_v21": s_v["bias"],
        "bias_blend50": s_b["bias"],
        "bias_cell_oracle": s_c["bias"],
        "bias_sku_origin_oracle": s_so["bias"],
        "n_cell_oracle_pick_neural": int((panel["cell_oracle_source"] == neural).sum()),
        "n_sku_origin_groups_pick_neural": int(
            panel.groupby(["product", "forecast_origin"])["sku_origin_oracle_source"]
            .first()
            .eq(neural)
            .sum()
        ),
    }


def oracle_breakdown(panel: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group WMAPE/AE for v21, neural, blend, and both oracles + capture."""
    rows: list[dict[str, Any]] = []
    for key, g in panel.groupby(group_col, sort=True):
        s_v = score_series(g["actual"], g["f_v21"])
        s_n = score_series(g["actual"], g["f_neural"])
        s_b = score_series(g["actual"], g["f_blend50"])
        s_c = score_series(g["actual"], g["f_cell_oracle"])
        s_so = score_series(g["actual"], g["f_sku_origin_oracle"])
        wm_v = float(s_v["portfolio_wmape_pct"])
        wm_b = float(s_b["portfolio_wmape_pct"])
        wm_c = float(s_c["portfolio_wmape_pct"])
        wm_so = float(s_so["portfolio_wmape_pct"])
        captured = wm_v - wm_b
        gap_cell = wm_v - wm_c
        gap_so = wm_v - wm_so
        ae_v = float(g["ae_v21"].sum())
        ae_b = float(g["ae_blend50"].sum())
        ae_c = float(g["ae_cell_oracle"].sum())
        ae_so = float(g["ae_sku_origin_oracle"].sum())
        ae_cap = ae_v - ae_b
        ae_gap_c = ae_v - ae_c
        row: dict[str, Any] = {
            group_col: int(key) if group_col in ("forecast_origin", "horizon") else key,
            "n_rows": int(len(g)),
            "wmape_v21": wm_v,
            "wmape_neural": float(s_n["portfolio_wmape_pct"]),
            "wmape_blend50": wm_b,
            "wmape_cell_oracle": wm_c,
            "wmape_sku_origin_oracle": wm_so,
            "lift_blend_pp": captured,
            "gap_cell_pp": gap_cell,
            "gap_sku_origin_pp": gap_so,
            "capture_frac_vs_cell": _capture_frac(captured, gap_cell),
            "capture_frac_vs_sku_origin": _capture_frac(captured, gap_so),
            "ae_sum_v21": ae_v,
            "ae_reduction_blend": ae_cap,
            "ae_reduction_cell_oracle": ae_gap_c,
            "ae_reduction_sku_origin_oracle": ae_v - ae_so,
            "ae_unrecovered_vs_cell": ae_gap_c - ae_cap,
            "ae_capture_frac_vs_cell": _capture_frac(ae_cap, ae_gap_c),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "ae_unrecovered_vs_cell" in df.columns:
        df = df.sort_values("ae_unrecovered_vs_cell", ascending=False)
    return df


def triple_matched(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = None
    for eng in ENGINES:
        k = frames[eng][KEYS].drop_duplicates()
        base = k if base is None else base.merge(k, on=KEYS, how="inner")
    assert base is not None
    rows = []
    for eng in ENGINES:
        part = frames[eng].merge(base, on=KEYS, how="inner")
        part = part.assign(engine=eng)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def sku_origin_wins(panel: pd.DataFrame, left_col: str, right_col: str) -> dict[str, int]:
    """Count (product, origin) groups where left has lower WMAPE than right."""
    left_w = 0
    right_w = 0
    ties = 0
    for _, g in panel.groupby(["product", "forecast_origin"], sort=False):
        wl = portfolio_wmape_pct(g["actual"], g[left_col])
        wr = portfolio_wmape_pct(g["actual"], g[right_col])
        if wl < wr:
            left_w += 1
        elif wr < wl:
            right_w += 1
        else:
            ties += 1
    return {"left_sku_origin_wins": left_w, "right_sku_origin_wins": right_w, "ties": ties}


def method_block(panel: pd.DataFrame, methods: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, col in methods:
        s = score_series(panel["actual"], panel[col])
        s["method"] = name
        # cell wins vs v2.1 when applicable
        if "ae_v21" in panel.columns and col.startswith("f_") and col != "f_v21":
            ae_col = "ae_" + col[2:] if ("ae_" + col[2:]) in panel.columns else None
            if ae_col:
                s["cell_wins_vs_v21"] = int((panel[ae_col] < panel["ae_v21"]).sum())
                s["cell_losses_vs_v21"] = int((panel[ae_col] > panel["ae_v21"]).sum())
        rows.append(s)
    return pd.DataFrame(rows)


def by_sku_blend(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in panel.groupby("product", sort=True):
        s_v = score_series(g["actual"], g["f_v21"])
        s_n = score_series(g["actual"], g["f_neural"])
        s_b = score_series(g["actual"], g["f_blend50"])
        ae_red = float(g["ae_v21"].sum() - g["ae_blend50"].sum())
        rows.append(
            {
                "product": product,
                "n_rows": int(len(g)),
                "wmape_v21": s_v["portfolio_wmape_pct"],
                "wmape_neural": s_n["portfolio_wmape_pct"],
                "wmape_blend50": s_b["portfolio_wmape_pct"],
                "bias_v21": s_v["bias"],
                "bias_neural": s_n["bias"],
                "bias_blend50": s_b["bias"],
                "ae_sum_v21": float(g["ae_v21"].sum()),
                "ae_sum_blend50": float(g["ae_blend50"].sum()),
                "ae_reduction_vs_v21": ae_red,
                "blend_beats_v21": bool(s_b["portfolio_wmape_pct"] < s_v["portfolio_wmape_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values("ae_reduction_vs_v21", ascending=False)


def by_origin_blend(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, g in panel.groupby("forecast_origin", sort=True):
        s_v = score_series(g["actual"], g["f_v21"])
        s_n = score_series(g["actual"], g["f_neural"])
        s_b = score_series(g["actual"], g["f_blend50"])
        wins = sku_origin_wins(g, "f_blend50", "f_v21")
        rows.append(
            {
                "forecast_origin": int(origin),
                "n_rows": int(len(g)),
                "wmape_v21": s_v["portfolio_wmape_pct"],
                "wmape_neural": s_n["portfolio_wmape_pct"],
                "wmape_blend50": s_b["portfolio_wmape_pct"],
                "bias_v21": s_v["bias"],
                "bias_blend50": s_b["bias"],
                "lift_vs_v21_pp": float(s_v["portfolio_wmape_pct"] - s_b["portfolio_wmape_pct"]),
                "blend_sku_origin_wins": wins["left_sku_origin_wins"],
                "v21_sku_origin_wins": wins["right_sku_origin_wins"],
            }
        )
    return pd.DataFrame(rows)


def drop_top_gainers(panel: pd.DataFrame, k: int) -> dict[str, Any]:
    sku = by_sku_blend(panel)
    if sku.empty or k <= 0:
        return {"k": k, "dropped": [], "n_rows": 0}
    top = sku.head(k)
    dropped = top["product"].tolist()
    kept = panel.loc[~panel["product"].isin(dropped)]
    s_v = score_series(kept["actual"], kept["f_v21"])
    s_b = score_series(kept["actual"], kept["f_blend50"])
    return {
        "k": k,
        "dropped": dropped,
        "dropped_ae_reduction": top["ae_reduction_vs_v21"].tolist(),
        "n_rows": int(len(kept)),
        "wmape_v21": s_v["portfolio_wmape_pct"],
        "wmape_blend50": s_b["portfolio_wmape_pct"],
        "lift_vs_v21_pp": float(s_v["portfolio_wmape_pct"] - s_b["portfolio_wmape_pct"]),
        "blend_still_beats_v21": bool(
            s_b["portfolio_wmape_pct"] < s_v["portfolio_wmape_pct"]
        ),
    }


def coverage_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = panel.copy()
    work["is_naive"] = work["model"].astype(str).eq("naive_last_value")
    work["has_actual"] = work["actual"].notna() if "actual" in work.columns else False
    work["ineligible"] = work.get("ineligible", False)
    if not isinstance(work["ineligible"], pd.Series):
        work["ineligible"] = False

    by_eng = (
        work.groupby("engine", as_index=False)
        .agg(
            n_rows=("forecast", "size"),
            n_with_actual=("has_actual", "sum"),
            n_ineligible_rows=("ineligible", "sum"),
            n_naive_model=("is_naive", "sum"),
            n_products=("product", "nunique"),
            n_origins=("forecast_origin", "nunique"),
        )
    )
    by_slice = (
        work.groupby(["sku_cohort", "origin_cohort", "engine"], as_index=False)
        .agg(
            n_rows=("forecast", "size"),
            n_with_actual=("has_actual", "sum"),
            n_ineligible_rows=("ineligible", "sum"),
            n_naive_model=("is_naive", "sum"),
        )
    )
    return {"by_engine": by_eng, "by_slice_engine": by_slice}


def evaluate_slice(
    frames: dict[str, pd.DataFrame], slice_name: str
) -> dict[str, Any]:
    out: dict[str, Any] = {"slice": slice_name}

    # Standalone on all-available rows with actuals
    standalone = [metrics_engine(frames[e].dropna(subset=["actual", "forecast"]), e) for e in ENGINES]
    out["standalone_all"] = pd.DataFrame(standalone)

    # Triple-matched
    try:
        trip = triple_matched(frames)
        trip = trip.dropna(subset=["actual", "forecast"])
    except Exception:
        trip = pd.DataFrame()
    if trip.empty:
        out["triple"] = pd.DataFrame()
        out["triple_n"] = 0
    else:
        trip_metrics = [
            metrics_engine(trip.loc[trip["engine"] == e], e) for e in ENGINES
        ]
        out["triple"] = pd.DataFrame(trip_metrics)
        out["triple_n"] = int(trip.loc[trip["engine"] == "v2.1"].shape[0])

    # Pairwise blends + hindsight oracles
    blend_summaries = []
    opportunity_rows = []
    for neural in NEURALS:
        pair = build_pairwise(frames["v2.1"], frames[neural], neural)
        if pair.empty:
            continue
        pair = add_sku_origin_oracle(pair, neural)
        methods = method_block(
            pair,
            [
                ("v2.1", "f_v21"),
                (neural, "f_neural"),
                ("blend50", "f_blend50"),
                ("cell_oracle", "f_cell_oracle"),
                ("sku_origin_oracle", "f_sku_origin_oracle"),
            ],
        )
        methods["neural"] = neural
        methods["slice"] = slice_name
        so_wins = sku_origin_wins(pair, "f_blend50", "f_v21")
        v21 = methods.loc[methods["method"] == "v2.1"].iloc[0]
        b50 = methods.loc[methods["method"] == "blend50"].iloc[0]
        cell = methods.loc[methods["method"] == "cell_oracle"].iloc[0]
        so_or = methods.loc[methods["method"] == "sku_origin_oracle"].iloc[0]
        wm_v = float(v21["portfolio_wmape_pct"])
        wm_b = float(b50["portfolio_wmape_pct"])
        wm_c = float(cell["portfolio_wmape_pct"])
        wm_so = float(so_or["portfolio_wmape_pct"])
        captured = wm_v - wm_b
        gap_cell = wm_v - wm_c
        gap_so = wm_v - wm_so
        opp = portfolio_opportunity(pair, neural, slice_name)
        opportunity_rows.append(opp)
        blend_summaries.append(
            {
                "slice": slice_name,
                "neural": neural,
                "n_rows": int(len(pair)),
                "wmape_v21": wm_v,
                "wmape_neural": float(
                    methods.loc[methods["method"] == neural, "portfolio_wmape_pct"].iloc[0]
                ),
                "wmape_blend50": wm_b,
                "wmape_cell_oracle": wm_c,
                "wmape_sku_origin_oracle": wm_so,
                "lift_vs_v21_pp": captured,
                "gap_cell_pp": gap_cell,
                "gap_sku_origin_pp": gap_so,
                "capture_frac_vs_cell": _capture_frac(captured, gap_cell),
                "capture_frac_vs_sku_origin": _capture_frac(captured, gap_so),
                "bias_v21": v21["bias"],
                "bias_blend50": b50["bias"],
                "bias_cell_oracle": cell["bias"],
                "bias_sku_origin_oracle": so_or["bias"],
                "ae_p90_v21": v21["ae_p90"],
                "ae_p90_blend50": b50["ae_p90"],
                "ae_p90_cell_oracle": cell["ae_p90"],
                "ae_p95_v21": v21["ae_p95"],
                "ae_p95_blend50": b50["ae_p95"],
                "ae_p95_cell_oracle": cell["ae_p95"],
                "top_decile_ae_share_v21": v21["top_decile_ae_share"],
                "top_decile_ae_share_blend50": b50["top_decile_ae_share"],
                "cell_wins_blend_vs_v21": int(b50.get("cell_wins_vs_v21") or 0),
                "cell_losses_blend_vs_v21": int(b50.get("cell_losses_vs_v21") or 0),
                **so_wins,
            }
        )
        out[f"pairwise_{neural}"] = pair
        out[f"methods_{neural}"] = methods
        out[f"by_sku_{neural}"] = by_sku_blend(pair)
        out[f"by_origin_{neural}"] = by_origin_blend(pair)
        out[f"sens_drop2_{neural}"] = drop_top_gainers(pair, 2)
        out[f"sens_drop3_{neural}"] = drop_top_gainers(pair, 3)
        out[f"opportunity_{neural}"] = opp
        if slice_name in ORACLE_SLICES:
            out[f"oracle_by_sku_{neural}"] = oracle_breakdown(pair, "product")
            out[f"oracle_by_origin_{neural}"] = oracle_breakdown(pair, "forecast_origin")
            out[f"oracle_by_horizon_{neural}"] = oracle_breakdown(pair, "horizon")

    out["blend_summary"] = pd.DataFrame(blend_summaries)
    out["opportunity"] = pd.DataFrame(opportunity_rows)
    return out


def decision_text(results: dict[str, dict[str, Any]]) -> str:
    """Report-only confirmation call weighted on new_sku x late_origin."""
    key = "new_sku_x_late_origin"
    r = results.get(key) or {}
    trip = r.get("triple")
    lines = [
        "## Confirmation call (report-only; weighted on new_sku × late_origin)",
        "",
        "Pooled / old-SKU slices are **not** independent confirmation of the 25-SKU study.",
        "",
    ]
    if trip is None or (isinstance(trip, pd.DataFrame) and trip.empty):
        lines.append("- Insufficient triple-matched rows on the primary slice.")
        return "\n".join(lines)

    best = trip.sort_values("portfolio_wmape_pct").iloc[0]
    lines.append(
        f"- Standalone on **{key}** triple-matched: best = **{best['engine']}** "
        f"at {float(best['portfolio_wmape_pct']):.2f}% WMAPE (n={int(best['n_rows'])})."
    )
    bs = r.get("blend_summary")
    if isinstance(bs, pd.DataFrame) and not bs.empty:
        for _, row in bs.iterrows():
            neural = row["neural"]
            lift = float(row["lift_vs_v21_pp"])
            still = True
            d3 = r.get(f"sens_drop3_{neural}") or {}
            if d3:
                still = bool(d3.get("blend_still_beats_v21"))
            lines.append(
                f"- V2.1+{str(neural).upper()} blend50 lift vs V2.1 = **{lift:+.2f} pp** "
                f"(n={int(row['n_rows'])}); after drop top-3 gain SKUs: "
                f"{'still beats' if still else 'does not beat'} V2.1 "
                f"(lift {float(d3.get('lift_vs_v21_pp', float('nan'))):+.2f} pp)."
            )
    return "\n".join(lines)


def write_oracle_sections(results: dict[str, dict[str, Any]]) -> list[str]:
    """Hindsight ceiling opportunity for pooled + primary confirmation slice."""
    sections: list[str] = [
        "## Hindsight oracles (ceilings only — not deployable)",
        "",
        "**Hindsight ceilings — not deployable; actuals used only for ceiling measurement.**",
        "Neither cell nor SKU×origin oracle may produce a production forecast or choose",
        "that origin’s model. They bound how much of the V2.1-vs-neural gap a routing",
        "policy could ever capture.",
        "",
        "- **Cell oracle:** per-row pick of V2.1 vs neural by lower abs error.",
        "- **SKU×origin oracle:** per `(product, forecast_origin)` pick by lower group WMAPE.",
        "- **Capture frac:** `captured_blend_pp / gap_*_pp` (null when gap ≤ 0).",
        "",
    ]
    float_cols = {
        "wmape_v21",
        "wmape_neural",
        "wmape_blend50",
        "wmape_cell_oracle",
        "wmape_sku_origin_oracle",
        "captured_blend_pp",
        "gap_cell_pp",
        "gap_sku_origin_pp",
        "capture_frac_vs_cell",
        "capture_frac_vs_sku_origin",
        "ae_reduction_blend",
        "ae_reduction_cell_oracle",
        "ae_reduction_sku_origin_oracle",
        "ae_capture_frac_vs_cell",
        "ae_capture_frac_vs_sku_origin",
        "lift_blend_pp",
        "ae_unrecovered_vs_cell",
    }
    for slice_name in ORACLE_SLICES:
        r = results.get(slice_name) or {}
        opp = r.get("opportunity")
        weight = " **(PRIMARY)**" if slice_name == "new_sku_x_late_origin" else " *(descriptive)*"
        sections.extend([f"### {slice_name}{weight}", ""])
        if not isinstance(opp, pd.DataFrame) or opp.empty:
            sections.append("_No oracle opportunity rows._\n")
            continue
        show = opp[
            [
                "neural",
                "n_rows",
                "wmape_v21",
                "wmape_neural",
                "wmape_blend50",
                "wmape_cell_oracle",
                "wmape_sku_origin_oracle",
                "captured_blend_pp",
                "gap_cell_pp",
                "gap_sku_origin_pp",
                "capture_frac_vs_cell",
                "capture_frac_vs_sku_origin",
                "ae_capture_frac_vs_cell",
                "ae_capture_frac_vs_sku_origin",
            ]
        ]
        sections.extend([_md_table(show, float_cols), ""])
        for neural in NEURALS:
            by_sku = r.get(f"oracle_by_sku_{neural}")
            if not isinstance(by_sku, pd.DataFrame) or by_sku.empty:
                continue
            top = by_sku.head(8)
            cols = [
                "product",
                "n_rows",
                "wmape_v21",
                "wmape_blend50",
                "wmape_cell_oracle",
                "gap_cell_pp",
                "lift_blend_pp",
                "capture_frac_vs_cell",
                "ae_unrecovered_vs_cell",
            ]
            sections.extend(
                [
                    f"#### Top unrecovered cell-oracle AE — V2.1+{neural.upper()} ({slice_name})",
                    "",
                    _md_table(top[cols], float_cols),
                    "",
                ]
            )
    return sections


def write_report(
    *,
    coverage: dict[str, pd.DataFrame],
    results: dict[str, dict[str, Any]],
    out_dir: Path,
) -> str:
    sections: list[str] = [
        "# Archval55 confirmation — V2.1 / A0 / A1 + fixed 50/50 blends",
        "",
        "**Pooled panel is descriptive only — not independent confirmation** "
        "(reuses the 25-SKU archval grid and early-origin stress already studied).",
        "",
        "**Primary confirmation weight:** `new_sku` × `late_origin` "
        "({140301, 140402} on the 30 MVP-only SKUs).",
        "",
        "Slice definitions:",
        "",
        "- `old_sku` = 25 `v3a_architecture_validation_products`",
        "- `new_sku` = 30 MVP-only products",
        "- `early_origin` = {140104, 140201}",
        "- `late_origin` = {140301, 140402}",
        "",
        "## A. Full-cohort fallback coverage (not for ranking)",
        "",
        "Includes INELIGIBLE `naive_last_value` deliveries. Do not treat these as model accuracy.",
        "",
        "### By engine",
        "",
        _md_table(coverage["by_engine"]),
        "",
        "### By slice × engine",
        "",
        _md_table(coverage["by_slice_engine"]),
        "",
    ]

    # Pooled first with disclaimer
    for slice_name, _, _ in SLICES:
        r = results[slice_name]
        weight = " **(PRIMARY CONFIRMATION)**" if slice_name == "new_sku_x_late_origin" else ""
        disclaimer = (
            " — *descriptive only; not independent confirmation*"
            if slice_name == "pooled"
            else ""
        )
        sections.extend(
            [
                f"## {slice_name}{weight}{disclaimer}",
                "",
            ]
        )
        trip = r.get("triple")
        if isinstance(trip, pd.DataFrame) and not trip.empty:
            sections.extend(
                [
                    f"### Triple-matched standalone (n={r.get('triple_n', 0)} cells)",
                    "",
                    _md_table(
                        trip,
                        {
                            "mae",
                            "rmse",
                            "portfolio_wmape_pct",
                            "bias",
                            "ae_p90",
                            "ae_p95",
                            "top_decile_ae_share",
                        },
                    ),
                    "",
                ]
            )
        else:
            sections.append("_No triple-matched rows._\n")

        bs = r.get("blend_summary")
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            sections.extend(
                [
                    "### Pairwise blends (identical V2.1 ∩ neural rows)",
                    "",
                    _md_table(
                        bs[
                            [
                                c
                                for c in (
                                    "slice",
                                    "neural",
                                    "n_rows",
                                    "wmape_v21",
                                    "wmape_neural",
                                    "wmape_blend50",
                                    "wmape_cell_oracle",
                                    "wmape_sku_origin_oracle",
                                    "lift_vs_v21_pp",
                                    "gap_cell_pp",
                                    "gap_sku_origin_pp",
                                    "capture_frac_vs_cell",
                                    "capture_frac_vs_sku_origin",
                                    "bias_v21",
                                    "bias_blend50",
                                    "ae_p90_v21",
                                    "ae_p90_blend50",
                                    "cell_wins_blend_vs_v21",
                                    "cell_losses_blend_vs_v21",
                                    "left_sku_origin_wins",
                                    "right_sku_origin_wins",
                                    "ties",
                                )
                                if c in bs.columns
                            ]
                        ],
                        {
                            "wmape_v21",
                            "wmape_neural",
                            "wmape_blend50",
                            "wmape_cell_oracle",
                            "wmape_sku_origin_oracle",
                            "lift_vs_v21_pp",
                            "gap_cell_pp",
                            "gap_sku_origin_pp",
                            "capture_frac_vs_cell",
                            "capture_frac_vs_sku_origin",
                            "bias_v21",
                            "bias_blend50",
                            "ae_p90_v21",
                            "ae_p90_blend50",
                        },
                    ),
                    "",
                ]
            )
            for neural in NEURALS:
                sku = r.get(f"by_sku_{neural}")
                if isinstance(sku, pd.DataFrame) and not sku.empty:
                    top = sku.head(8)
                    sections.extend(
                        [
                            f"#### Top SKU AE reductions — V2.1+{neural.upper()}",
                            "",
                            _md_table(
                                top[
                                    [
                                        "product",
                                        "n_rows",
                                        "wmape_v21",
                                        "wmape_blend50",
                                        "ae_reduction_vs_v21",
                                        "bias_blend50",
                                    ]
                                ],
                                {
                                    "wmape_v21",
                                    "wmape_blend50",
                                    "ae_reduction_vs_v21",
                                    "bias_blend50",
                                },
                            ),
                            "",
                        ]
                    )
                for label, key in (("drop top-2", f"sens_drop2_{neural}"), ("drop top-3", f"sens_drop3_{neural}")):
                    sens = r.get(key) or {}
                    if not sens:
                        continue
                    sections.append(
                        f"- Sensitivity {label} ({neural}): dropped {sens.get('dropped')}; "
                        f"lift={float(sens.get('lift_vs_v21_pp', float('nan'))):+.3g} pp; "
                        f"still beats V2.1 = **{sens.get('blend_still_beats_v21')}**"
                    )
                sections.append("")

    sections.extend(write_oracle_sections(results))
    sections.extend(["", decision_text(results), ""])
    text = "\n".join(sections)
    (out_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = export_dir / "forecasts_panel.parquet"
    if not panel_path.is_file():
        panel_path = export_dir / "forecasts_panel.csv"
    if not panel_path.is_file():
        raise SystemExit(f"missing panel under {export_dir}; run export_archval55_panel.py first")

    panel = (
        pd.read_parquet(panel_path)
        if panel_path.suffix == ".parquet"
        else pd.read_csv(panel_path)
    )
    panel["product"] = panel["product"].astype(str)
    for c in ("forecast_origin", "target_date", "horizon"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(int)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    panel["engine"] = panel["engine"].astype(str).str.lower()
    if "ineligible" not in panel.columns:
        panel["ineligible"] = False
    else:
        panel["ineligible"] = panel["ineligible"].fillna(False).astype(bool)

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    panel = attach_actuals(panel, sales)
    panel = ensure_cohort_tags(panel)

    coverage = coverage_tables(panel)
    coverage["by_engine"].to_csv(out_dir / "coverage_by_engine.csv", index=False)
    coverage["by_slice_engine"].to_csv(out_dir / "coverage_by_slice_engine.csv", index=False)

    # Ranking panels exclude INELIGIBLE / naive fills (coverage section uses full cohort).
    ranking = panel.copy()
    is_naive = ranking["model"].astype(str).eq("naive_last_value") if "model" in ranking.columns else False
    ranking = ranking.loc[~ranking["ineligible"].fillna(False).astype(bool) & ~is_naive].copy()
    base_frames = load_engine_frames(ranking)
    results: dict[str, dict[str, Any]] = {}
    all_blend_rows: list[pd.DataFrame] = []
    all_triple: list[pd.DataFrame] = []
    all_opp_rows: list[pd.DataFrame] = []

    for slice_name, sku, origin in SLICES:
        frames = filter_slice(base_frames, sku, origin)
        r = evaluate_slice(frames, slice_name)
        results[slice_name] = r
        if isinstance(r.get("blend_summary"), pd.DataFrame) and not r["blend_summary"].empty:
            all_blend_rows.append(r["blend_summary"])
        if isinstance(r.get("opportunity"), pd.DataFrame) and not r["opportunity"].empty:
            if slice_name in ORACLE_SLICES:
                all_opp_rows.append(r["opportunity"])
        if isinstance(r.get("triple"), pd.DataFrame) and not r["triple"].empty:
            t = r["triple"].copy()
            t["slice"] = slice_name
            all_triple.append(t)
        # Persist per-slice artifacts
        for neural in NEURALS:
            sku_df = r.get(f"by_sku_{neural}")
            if isinstance(sku_df, pd.DataFrame) and not sku_df.empty:
                sku_df.to_csv(out_dir / f"by_sku_{slice_name}_v21_{neural}.csv", index=False)
            org = r.get(f"by_origin_{neural}")
            if isinstance(org, pd.DataFrame) and not org.empty:
                org.to_csv(out_dir / f"by_origin_{slice_name}_v21_{neural}.csv", index=False)
            pair = r.get(f"pairwise_{neural}")
            if isinstance(pair, pd.DataFrame) and not pair.empty:
                pair.to_parquet(out_dir / f"pairwise_{slice_name}_v21_{neural}.parquet", index=False)
            if slice_name in ORACLE_SLICES:
                for dim in ("sku", "origin", "horizon"):
                    odf = r.get(f"oracle_by_{dim}_{neural}")
                    if isinstance(odf, pd.DataFrame) and not odf.empty:
                        odf.to_csv(
                            out_dir / f"oracle_by_{dim}_{slice_name}_v21_{neural}.csv",
                            index=False,
                        )

    if all_blend_rows:
        pd.concat(all_blend_rows, ignore_index=True).to_csv(
            out_dir / "blend_summary_by_slice.csv", index=False
        )
    if all_opp_rows:
        pd.concat(all_opp_rows, ignore_index=True).to_csv(
            out_dir / "oracle_opportunity_by_slice.csv", index=False
        )
    if all_triple:
        pd.concat(all_triple, ignore_index=True).to_csv(
            out_dir / "triple_matched_by_slice.csv", index=False
        )

    # Sensitivity JSON
    sens_payload = {}
    for slice_name in ("pooled", "new_sku_x_late_origin"):
        r = results.get(slice_name) or {}
        sens_payload[slice_name] = {
            neural: {
                "drop2": r.get(f"sens_drop2_{neural}"),
                "drop3": r.get(f"sens_drop3_{neural}"),
            }
            for neural in NEURALS
        }
    (out_dir / "sensitivity_drop_top_gainers.json").write_text(
        json.dumps(sens_payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = write_report(coverage=coverage, results=results, out_dir=out_dir)
    print(f"wrote {out_dir / 'report.md'} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
