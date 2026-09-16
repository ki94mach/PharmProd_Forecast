#!/usr/bin/env python3
"""A2–A5 late-origin decision eval vs V2.1 / corrected A0 / A1.

Fixed blends only (no weight search). Writes versioned artifacts under
``archval55_a2_a5_late_prod_eval_v2/`` by default. Does not mutate eval_v1
or baseline export trees.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct
from pkg.benchmark.universes import load_universe_product_names

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_A25 = RESULTS / "archval55_a2_a5_late_prod_export_v1"
DEFAULT_BASE = RESULTS / "archval55_export_a0fix_v1"
DEFAULT_OUT = RESULTS / "archval55_a2_a5_late_prod_eval_v2"
DEFAULT_RUN = REPO / "src" / "data" / "forecast_jobs" / "archval55_a2_a5_late_prod_v1"
DEFAULT_BASE_RUN = (
    REPO / "src" / "data" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1"
)
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
LATE_ORIGINS = frozenset({140301, 140402})
AX = ("a2", "a3", "a4", "a5")
SEEDS = (41, 42, 43)
PLANNED_PER_ARCH = 55 * 2 * 3  # 330
HORIZON_GROUPS = (
    ("h1_3", frozenset({1, 2, 3})),
    ("h4_6", frozenset({4, 5, 6})),
    ("h7_9", frozenset({7, 8, 9})),
    ("h10_12", frozenset({10, 11, 12})),
    ("h13_15", frozenset({13, 14, 15})),
)
TAIL_P95_TOL_PP = 2.0
TAIL_GT100_TOL = 2
COVERAGE_MIN_FRAC = 0.95
DROPOUT_TOL = 1


def _load_panel(path: Path) -> pd.DataFrame:
    pq = path / "forecasts_panel.parquet"
    csv = path / "forecasts_panel.csv"
    if pq.is_file():
        df = pd.read_parquet(pq)
    elif csv.is_file():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"missing panel under {path}")
    df["product"] = df["product"].astype(str)
    df["engine"] = df["engine"].astype(str).str.lower()
    for c in ("forecast_origin", "target_date", "horizon"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    if "ineligible" not in df.columns:
        df["ineligible"] = False
    else:
        df["ineligible"] = df["ineligible"].fillna(False).astype(bool)
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    if "model" not in df.columns:
        df["model"] = ""
    return df


def _attach_actuals(df: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
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


def _tag_cohorts(df: pd.DataFrame, old_skus: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["sku_cohort"] = np.where(out["product"].isin(old_skus), "old_sku", "new_sku")
    return out


def _horizon_group(h: int) -> str:
    for name, members in HORIZON_GROUPS:
        if int(h) in members:
            return name
    return "other"


def _score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
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
        }
    err = m["forecast"] - m["actual"]
    ae = np.abs(err)
    return {
        "n_rows": int(len(m)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
        "ae_p90": float(np.quantile(ae, 0.90)),
        "ae_p95": float(np.quantile(ae, 0.95)),
    }


def _seed_mean(panel: pd.DataFrame, engine: str) -> pd.DataFrame:
    sub = panel.loc[
        (panel["engine"] == engine)
        & (~panel["ineligible"])
        & (panel["model"].astype(str) != "naive_last_value")
    ].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=KEYS + ["forecast", "actual", "engine", "n_seeds", "sku_cohort"]
        )
    aggs: dict[str, Any] = {
        "forecast": ("forecast", "mean"),
        "actual": ("actual", "mean"),
        "n_seeds": ("seed", "nunique") if "seed" in sub.columns else ("forecast", "size"),
    }
    if "sku_cohort" in sub.columns:
        aggs["sku_cohort"] = ("sku_cohort", "first")
    g = sub.groupby(KEYS, as_index=False).agg(**aggs)
    g["engine"] = engine
    return g


def _v21_frame(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel.loc[(panel["engine"] == "v2.1") & (~panel["ineligible"])].copy()
    aggs: dict[str, Any] = {
        "forecast": ("forecast", "first"),
        "actual": ("actual", "first"),
    }
    if "sku_cohort" in sub.columns:
        aggs["sku_cohort"] = ("sku_cohort", "first")
    g = sub.groupby(KEYS, as_index=False).agg(**aggs)
    g["engine"] = "v2.1"
    g["n_seeds"] = 1
    return g


def _blend(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    engines = list(weights)
    out = frames[engines[0]][KEYS + ["actual"]].drop_duplicates(KEYS).copy()
    if "sku_cohort" in frames[engines[0]].columns:
        out = out.merge(
            frames[engines[0]][KEYS + ["sku_cohort"]].drop_duplicates(KEYS),
            on=KEYS,
            how="left",
        )
    out["forecast"] = 0.0
    for eng, w in weights.items():
        f = frames[eng][KEYS + ["forecast"]].rename(columns={"forecast": f"f_{eng}"})
        out = out.merge(f, on=KEYS, how="inner")
        out["forecast"] = out["forecast"] + float(w) * out[f"f_{eng}"]
    return out


def _filter_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "pooled_55":
        return df
    if slice_name == "old_sku_25":
        return df.loc[df["sku_cohort"] == "old_sku"]
    if slice_name == "new_sku_30":
        return df.loc[df["sku_cohort"] == "new_sku"]
    raise ValueError(slice_name)


def _sku_origin_wmape(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (product, origin), g in df.groupby(["product", "forecast_origin"]):
        m = g.dropna(subset=["actual", "forecast"])
        if m.empty:
            continue
        rows.append(
            {
                "product": product,
                "forecast_origin": int(origin),
                "portfolio_wmape_pct": float(
                    portfolio_wmape_pct(m["actual"], m["forecast"])
                ),
                "n_rows": int(len(m)),
            }
        )
    return pd.DataFrame(rows)


def _tail_stats(df: pd.DataFrame) -> dict[str, float]:
    so = _sku_origin_wmape(df)
    if so.empty:
        return {
            "n_groups": 0,
            "wmape_p90": np.nan,
            "wmape_p95": np.nan,
            "n_gt_50": 0,
            "n_gt_100": 0,
        }
    return {
        "n_groups": int(len(so)),
        "wmape_p90": float(so["portfolio_wmape_pct"].quantile(0.9)),
        "wmape_p95": float(so["portfolio_wmape_pct"].quantile(0.95)),
        "n_gt_50": int((so["portfolio_wmape_pct"] > 50).sum()),
        "n_gt_100": int((so["portfolio_wmape_pct"] > 100).sum()),
    }


def _seed_dispersion(panel: pd.DataFrame, engine: str) -> dict[str, Any]:
    sub = panel.loc[(panel["engine"] == engine) & (~panel["ineligible"])].copy()
    if sub.empty or "seed" not in sub.columns:
        return {"engine": engine, "n_cells": 0, "mean_seed_std": np.nan, "p90_seed_std": np.nan}
    g = sub.groupby(KEYS)["forecast"].agg(["std", "count"])
    g = g.loc[g["count"] >= 2]
    return {
        "engine": engine,
        "n_cells": int(len(g)),
        "mean_seed_std": float(g["std"].mean()) if not g.empty else np.nan,
        "p90_seed_std": float(g["std"].quantile(0.9)) if not g.empty else np.nan,
    }


def _error_corr(
    a: pd.DataFrame, b: pd.DataFrame, *, absolute: bool
) -> float:
    left = a[KEYS + ["forecast", "actual"]].rename(columns={"forecast": "fa"})
    right = b[KEYS + ["forecast"]].rename(columns={"forecast": "fb"})
    m = left.merge(right, on=KEYS, how="inner").dropna()
    if m.empty:
        return float("nan")
    ea = m["fa"] - m["actual"]
    eb = m["fb"] - m["actual"]
    if absolute:
        ea = ea.abs()
        eb = eb.abs()
    if float(ea.std()) == 0.0 or float(eb.std()) == 0.0:
        return float("nan")
    return float(ea.corr(eb))


def _oracle_ceilings(
    candidates: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Hindsight cell + SKU×origin oracles — non-deployable."""
    names = list(candidates)
    base = candidates[names[0]][KEYS + ["actual"]].drop_duplicates(KEYS)
    wide = base.copy()
    for n, df in candidates.items():
        wide = wide.merge(
            df[KEYS + ["forecast"]].rename(columns={"forecast": n}),
            on=KEYS,
            how="inner",
        )
    if wide.empty:
        return {"n_cells": 0, "note": "non-deployable hindsight only"}, pd.DataFrame()

    def _best_ae(row: pd.Series) -> tuple[float, str]:
        act = float(row["actual"])
        best_ae = None
        best_n = ""
        for n in names:
            if pd.isna(row[n]):
                continue
            ae = abs(float(row[n]) - act)
            if best_ae is None or ae < best_ae:
                best_ae = ae
                best_n = n
        return float(best_ae) if best_ae is not None else np.nan, best_n

    picks = wide.apply(lambda r: pd.Series(_best_ae(r), index=["oracle_ae", "pick"]), axis=1)
    wide = pd.concat([wide, picks], axis=1)
    cell = {
        "n_cells": int(len(wide)),
        "cell_oracle_mae": float(wide["oracle_ae"].mean()),
        "note": "NON-DEPLOYABLE hindsight ceiling",
    }
    so_rows = []
    for (product, origin), g in wide.groupby(["product", "forecast_origin"]):
        best_w = None
        best_name = None
        for n in names:
            m = g.dropna(subset=["actual", n])
            if m.empty:
                continue
            w = float(portfolio_wmape_pct(m["actual"], m[n]))
            if best_w is None or w < best_w:
                best_w = w
                best_name = n
        if best_w is not None:
            so_rows.append(
                {
                    "product": product,
                    "forecast_origin": int(origin),
                    "best_method": best_name,
                    "oracle_wmape_pct": best_w,
                }
            )
    so = pd.DataFrame(so_rows)
    cell["n_sku_origin"] = int(len(so))
    cell["mean_sku_origin_oracle_wmape_pct"] = (
        float(so["oracle_wmape_pct"].mean()) if not so.empty else np.nan
    )
    return cell, so


def _read_sqlite_status(run_dir: Path) -> pd.DataFrame:
    db = run_dir / "state.sqlite"
    if not db.is_file():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db))
    jobs = pd.read_sql_query(
        """
        SELECT engine_version AS engine, forecast_origin, product_id, status,
               error_type, error_message
        FROM jobs
        """,
        conn,
    )
    conn.close()
    jobs["engine"] = jobs["engine"].astype(str).str.lower()
    return jobs


def _coverage_report(
    a25_jobs: pd.DataFrame,
    a25_panel: pd.DataFrame,
    base_panel: pd.DataFrame,
    old_skus: set[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows = []
    detail = []
    mvp = set(load_universe_product_names("mvp_products", validate=True))
    planned_po = {(p, o) for p in mvp for o in LATE_ORIGINS}
    explain: dict[str, Any] = {
        "planned_per_arch": PLANNED_PER_ARCH,
        "seeds": list(SEEDS),
        "n_planned_product_origin": len(planned_po),
    }

    if not a25_jobs.empty:
        for eng in AX:
            e = a25_jobs.loc[a25_jobs["engine"] == eng]
            counts = e["status"].value_counts().to_dict()
            rows.append(
                {
                    "source": "a25_run",
                    "engine": eng,
                    "n_jobs": int(len(e)),
                    "n_success": int(counts.get("SUCCESS", 0)),
                    "n_ineligible": int(counts.get("INELIGIBLE", 0)),
                    "n_failed": int(counts.get("FAILED", 0)),
                    "n_pending": int(counts.get("PENDING", 0)),
                    "n_running": int(counts.get("RUNNING", 0)),
                }
            )

    # Corrected baseline coverage from a0fix panel (not pre-fix sqlite empties)
    for eng in ("v2.1", "a0", "a1"):
        sub = base_panel.loc[
            (base_panel["engine"] == eng) & (~base_panel["ineligible"])
        ]
        if eng == "v2.1":
            success_po = set(
                map(
                    tuple,
                    sub[["product", "forecast_origin"]].drop_duplicates().to_numpy(),
                )
            )
            n_success_jobs = int(len(success_po))
        else:
            sc0 = (
                sub.groupby(["product", "forecast_origin"])["seed"]
                .nunique()
                .reset_index(name="n_success_seeds")
            )
            success_po = set(
                map(
                    tuple,
                    sc0.loc[
                        sc0["n_success_seeds"] > 0, ["product", "forecast_origin"]
                    ].to_numpy(),
                )
            )
            n_success_jobs = int(
                sub.groupby(["product", "forecast_origin", "seed"]).ngroups
            )
        rows.append(
            {
                "source": "baseline_panel_a0fix",
                "engine": eng,
                "n_jobs": n_success_jobs,
                "n_success": n_success_jobs,
                "n_ineligible": 0,
                "n_failed": 0,
                "n_pending": 0,
                "n_running": 0,
            }
        )
        if eng == "a0":
            explain["a0_late"] = {
                "n_full_dropout_product_origin": int(len(planned_po - success_po)),
                "n_success_product_origin": int(len(success_po)),
                "n_success_seed_jobs_approx": n_success_jobs,
            }

    for eng in AX:
        sub = a25_panel.loc[a25_panel["engine"] == eng].copy()
        if sub.empty:
            continue
        job_keys = sub.groupby(
            ["product", "forecast_origin", "seed"], as_index=False
        ).agg(ineligible=("ineligible", "max"), n_rows=("forecast", "size"))
        success_jobs = job_keys.loc[~job_keys["ineligible"]]
        sc = (
            success_jobs.groupby(["product", "forecast_origin"])["seed"]
            .nunique()
            .reset_index(name="n_success_seeds")
        )
        sc["engine"] = eng
        sc["seed_completeness"] = sc["n_success_seeds"] / float(len(SEEDS))
        sc["sku_cohort"] = np.where(sc["product"].isin(old_skus), "old_sku", "new_sku")
        detail.append(sc)
        success_po = set(
            map(
                tuple,
                sc.loc[sc["n_success_seeds"] > 0, ["product", "forecast_origin"]].to_numpy(),
            )
        )
        full_drop = planned_po - success_po
        n_success = (
            int(
                a25_jobs.loc[
                    (a25_jobs["engine"] == eng) & (a25_jobs["status"] == "SUCCESS")
                ].shape[0]
            )
            if not a25_jobs.empty
            else int(len(success_jobs))
        )
        explain[eng] = {
            "n_success_jobs": n_success,
            "planned_jobs": PLANNED_PER_ARCH,
            "success_frac": n_success / float(PLANNED_PER_ARCH),
            "n_full_dropout_product_origin": int(len(full_drop)),
            "full_dropout_sample": [
                {"product": p, "forecast_origin": int(o)}
                for p, o in sorted(full_drop)[:20]
            ],
            "mean_seed_completeness": float(sc["seed_completeness"].mean())
            if not sc.empty
            else 0.0,
        }

    seed_detail = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return pd.DataFrame(rows), explain, seed_detail


def _drop_top3(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    v2: pd.DataFrame,
) -> dict[str, Any]:
    """Drop top-3 SKUs by lift of candidate vs reference (V21_A0_50)."""
    ref_so = _sku_origin_wmape(reference).rename(
        columns={"portfolio_wmape_pct": "wmape_ref"}
    )
    cand_so = _sku_origin_wmape(candidate).rename(
        columns={"portfolio_wmape_pct": "wmape_cand"}
    )
    m = ref_so.merge(cand_so, on=["product", "forecast_origin"], how="inner")
    m["lift_pp"] = m["wmape_ref"] - m["wmape_cand"]
    # Aggregate lift by product (mean across origins)
    by_sku = (
        m.groupby("product", as_index=False)["lift_pp"].mean().sort_values(
            "lift_pp", ascending=False
        )
    )
    top = by_sku.head(3)
    drop_skus = set(top["product"].tolist())
    c_kept = candidate.loc[~candidate["product"].isin(drop_skus)]
    r_kept = reference.loc[~reference["product"].isin(drop_skus)]
    v_kept = v2.loc[~v2["product"].isin(drop_skus)]
    # Rematch on keys for fair compare
    common = c_kept.merge(
        r_kept[KEYS + ["forecast"]].rename(columns={"forecast": "f_ref"}),
        on=KEYS,
        how="inner",
    ).merge(
        v_kept[KEYS + ["forecast"]].rename(columns={"forecast": "f_v2"}),
        on=KEYS,
        how="inner",
    )
    sc_c = _score(common["actual"], common["forecast"])
    sc_r = _score(common["actual"], common["f_ref"])
    sc_v = _score(common["actual"], common["f_v2"])
    return {
        "dropped_skus": sorted(drop_skus),
        "candidate_wmape_full": _score(candidate["actual"], candidate["forecast"])[
            "portfolio_wmape_pct"
        ],
        "ref_wmape_full": _score(reference["actual"], reference["forecast"])[
            "portfolio_wmape_pct"
        ],
        "v2_wmape_full": _score(v2["actual"], v2["forecast"])["portfolio_wmape_pct"],
        "candidate_wmape_drop3": sc_c["portfolio_wmape_pct"],
        "ref_wmape_drop3": sc_r["portfolio_wmape_pct"],
        "v2_wmape_drop3": sc_v["portfolio_wmape_pct"],
        "beats_v2_drop3": bool(
            sc_c["n_rows"]
            and sc_v["n_rows"]
            and sc_c["portfolio_wmape_pct"] < sc_v["portfolio_wmape_pct"]
        ),
        "beats_ref_drop3": bool(
            sc_c["n_rows"]
            and sc_r["n_rows"]
            and sc_c["portfolio_wmape_pct"] < sc_r["portfolio_wmape_pct"]
        ),
    }


def _loo_origin(methods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    origins = sorted(LATE_ORIGINS)
    for held in origins:
        for name, df in methods.items():
            kept = df.loc[df["forecast_origin"] != held]
            sc = _score(kept["actual"], kept["forecast"])
            rows.append(
                {
                    "held_out_origin": int(held),
                    "method": name,
                    **sc,
                }
            )
    return pd.DataFrame(rows)


def _loo_sku(methods: dict[str, pd.DataFrame], max_skus: int = 55) -> pd.DataFrame:
    # Mean absolute LOO impact: for each SKU, score without that SKU
    products = sorted(
        set().union(*[set(df["product"].unique()) for df in methods.values()])
    )[:max_skus]
    rows = []
    for sku in products:
        for name, df in methods.items():
            kept = df.loc[df["product"] != sku]
            sc = _score(kept["actual"], kept["forecast"])
            full = _score(df["actual"], df["forecast"])
            rows.append(
                {
                    "held_out_sku": sku,
                    "method": name,
                    "wmape_loo": sc["portfolio_wmape_pct"],
                    "wmape_full": full["portfolio_wmape_pct"],
                    "delta_pp": (
                        float(sc["portfolio_wmape_pct"] - full["portfolio_wmape_pct"])
                        if sc["n_rows"] and full["n_rows"]
                        else np.nan
                    ),
                    "n_rows": sc["n_rows"],
                }
            )
    return pd.DataFrame(rows)


def _advancement(
    *,
    ax: str,
    candidate: pd.DataFrame,
    v2: pd.DataFrame,
    v21_a0: pd.DataFrame,
    cov_explain: dict[str, Any],
    drop3: dict[str, Any],
) -> dict[str, Any]:
    sc_c = _score(candidate["actual"], candidate["forecast"])
    sc_v = _score(v2["actual"], v2["forecast"])
    sc_r = _score(v21_a0["actual"], v21_a0["forecast"])
    wm_c = sc_c["portfolio_wmape_pct"]
    wm_v = sc_v["portfolio_wmape_pct"]
    wm_r = sc_r["portfolio_wmape_pct"]

    gate1 = bool(
        sc_c["n_rows"]
        and sc_v["n_rows"]
        and sc_r["n_rows"]
        and wm_c < wm_v
        and wm_c < wm_r
    )
    gate2 = bool(drop3.get("beats_v2_drop3") and drop3.get("beats_ref_drop3"))

    ax_cov = cov_explain.get(ax, {})
    success_frac = float(ax_cov.get("success_frac", 0.0))
    a0_drop = int(
        cov_explain.get("a0_late", {}).get("n_full_dropout_product_origin", 0)
    )
    ax_drop = int(ax_cov.get("n_full_dropout_product_origin", 999))
    gate3 = bool(
        success_frac >= COVERAGE_MIN_FRAC
        and ax_drop <= a0_drop + DROPOUT_TOL
    )

    tail_c = _tail_stats(candidate)
    tail_r = _tail_stats(v21_a0)
    gate4 = bool(
        (
            np.isnan(tail_c["wmape_p95"])
            or np.isnan(tail_r["wmape_p95"])
            or tail_c["wmape_p95"] <= tail_r["wmape_p95"] + TAIL_P95_TOL_PP
        )
        and (tail_c["n_gt_100"] <= tail_r["n_gt_100"] + TAIL_GT100_TOL)
    )

    advance = bool(gate1 and gate2 and gate3 and gate4)
    return {
        "ax": ax,
        "primary_candidate": f"V21_A0_{ax.upper()}_50_25_25",
        "wmape_candidate": wm_c,
        "wmape_v2": wm_v,
        "wmape_v21_a0_50": wm_r,
        "gate_beats_v2_and_v21_a0": gate1,
        "gate_drop_top3": gate2,
        "gate_coverage": gate3,
        "gate_tail": gate4,
        "success_frac": success_frac,
        "n_full_dropout_ax": ax_drop,
        "n_full_dropout_a0": a0_drop,
        "tail_p95_candidate": tail_c["wmape_p95"],
        "tail_p95_v21_a0": tail_r["wmape_p95"],
        "n_gt_100_candidate": tail_c["n_gt_100"],
        "n_gt_100_v21_a0": tail_r["n_gt_100"],
        "verdict": "ADVANCE" if advance else "HOLD",
        "drop3": drop3,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a25-export", type=Path, default=DEFAULT_A25)
    p.add_argument("--baseline-export", type=Path, default=DEFAULT_BASE)
    p.add_argument("--a25-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--baseline-run", type=Path, default=DEFAULT_BASE_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_skus = set(
        load_universe_product_names(
            "v3a_architecture_validation_products", validate=True
        )
    )

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    a25 = _tag_cohorts(
        _attach_actuals(_load_panel(Path(args.a25_export)), sales),
        old_skus,
    )
    a25 = a25.loc[a25["forecast_origin"].isin(LATE_ORIGINS)].copy()

    base = _tag_cohorts(
        _attach_actuals(_load_panel(Path(args.baseline_export)), sales),
        old_skus,
    )
    base = base.loc[base["forecast_origin"].isin(LATE_ORIGINS)].copy()

    a25_jobs = _read_sqlite_status(Path(args.a25_run))
    cov_df, cov_explain, seed_detail = _coverage_report(
        a25_jobs, a25, base, old_skus
    )
    cov_df.to_csv(out / "coverage_ineligibility.csv", index=False)
    (out / "coverage_ineligibility.json").write_text(
        json.dumps(cov_explain, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not seed_detail.empty:
        seed_detail.to_csv(out / "seed_completeness_by_product_origin.csv", index=False)

    frames: dict[str, pd.DataFrame] = {
        "v2.1": _v21_frame(base),
        "a0": _seed_mean(base, "a0"),
        "a1": _seed_mean(base, "a1"),
    }
    for eng in AX:
        frames[eng] = _seed_mean(a25, eng)

    # Ensure sku_cohort on all frames
    for k, df in list(frames.items()):
        if "sku_cohort" not in df.columns and not df.empty:
            frames[k] = _tag_cohorts(df, old_skus)

    # Deployable methods
    methods: dict[str, pd.DataFrame] = {
        "V2_POLICY": frames["v2.1"],
        "V21_A0_50": _blend(frames, {"v2.1": 0.5, "a0": 0.5}),
    }
    for ax in AX:
        methods[f"V21_{ax.upper()}_50"] = _blend(frames, {"v2.1": 0.5, ax: 0.5})
        methods[f"V21_A0_{ax.upper()}_50_25_25"] = _blend(
            frames, {"v2.1": 0.5, "a0": 0.25, ax: 0.25}
        )
    for k, df in list(methods.items()):
        if "sku_cohort" not in df.columns and not df.empty:
            methods[k] = _tag_cohorts(df, old_skus)

    slices = ("pooled_55", "old_sku_25", "new_sku_30")

    # Standalone by slice
    stand_rows = []
    for slice_name in slices:
        for eng, df in frames.items():
            sub = _filter_slice(df, slice_name)
            row = _score(sub["actual"], sub["forecast"])
            row.update({"slice": slice_name, "engine": eng})
            if "runtime_seconds" in a25.columns and eng in AX:
                rt = a25.loc[a25["engine"] == eng, "runtime_seconds"]
                row["mean_runtime_s"] = float(rt.mean()) if rt.notna().any() else np.nan
            else:
                row["mean_runtime_s"] = np.nan
            stand_rows.append(row)
    pd.DataFrame(stand_rows).to_csv(out / "standalone_metrics_by_slice.csv", index=False)

    # Seed dispersion
    disp = [_seed_dispersion(base, "a0"), _seed_dispersion(base, "a1")]
    for eng in AX:
        disp.append(_seed_dispersion(a25, eng))
    pd.DataFrame(disp).to_csv(out / "seed_dispersion.csv", index=False)

    # Blend metrics by slice
    blend_rows = []
    for slice_name in slices:
        for name, df in methods.items():
            if name == "V2_POLICY":
                continue
            sub = _filter_slice(df, slice_name)
            row = _score(sub["actual"], sub["forecast"])
            row.update({"slice": slice_name, "method": name})
            blend_rows.append(row)
        # also V2 for reference
        sub_v = _filter_slice(methods["V2_POLICY"], slice_name)
        row_v = _score(sub_v["actual"], sub_v["forecast"])
        row_v.update({"slice": slice_name, "method": "V2_POLICY"})
        blend_rows.append(row_v)
    pd.DataFrame(blend_rows).to_csv(out / "blend_metrics_by_slice.csv", index=False)

    # By origin / horizon group / SKU (deployable methods, pooled)
    by_o, by_hg, by_sku = [], [], []
    for name, df in methods.items():
        for o, g in df.groupby("forecast_origin"):
            r = _score(g["actual"], g["forecast"])
            r.update({"method": name, "forecast_origin": int(o)})
            by_o.append(r)
        tmp = df.copy()
        tmp["horizon_group"] = tmp["horizon"].map(_horizon_group)
        for hg, g in tmp.groupby("horizon_group"):
            r = _score(g["actual"], g["forecast"])
            r.update({"method": name, "horizon_group": hg})
            by_hg.append(r)
        for sku, g in df.groupby("product"):
            r = _score(g["actual"], g["forecast"])
            r.update({"method": name, "product": sku})
            by_sku.append(r)
    pd.DataFrame(by_o).to_csv(out / "metrics_by_origin.csv", index=False)
    pd.DataFrame(by_hg).to_csv(out / "metrics_by_horizon_group.csv", index=False)
    pd.DataFrame(by_sku).to_csv(out / "metrics_by_sku.csv", index=False)

    # Error correlations vs V2.1 and A0 (signed + abs)
    corr_signed_rows = []
    corr_abs_rows = []
    for eng in ("a0", "a1", *AX):
        corr_signed_rows.append(
            {
                "engine": eng,
                "corr_vs_v21": _error_corr(frames[eng], frames["v2.1"], absolute=False),
                "corr_vs_a0": _error_corr(frames[eng], frames["a0"], absolute=False)
                if eng != "a0"
                else 1.0,
            }
        )
        corr_abs_rows.append(
            {
                "engine": eng,
                "corr_vs_v21": _error_corr(frames[eng], frames["v2.1"], absolute=True),
                "corr_vs_a0": _error_corr(frames[eng], frames["a0"], absolute=True)
                if eng != "a0"
                else 1.0,
            }
        )
    pd.DataFrame(corr_signed_rows).to_csv(
        out / "error_correlation_signed.csv", index=False
    )
    pd.DataFrame(corr_abs_rows).to_csv(out / "error_correlation_abs.csv", index=False)

    # Oracles (non-deployable)
    oracle_summary = {
        "_note": "Hindsight oracles only — NON-DEPLOYABLE; not for production selection."
    }
    for ax in AX:
        cands = {"v2.1": frames["v2.1"], "a0": frames["a0"], ax: frames[ax]}
        cell, so = _oracle_ceilings(cands)
        oracle_summary[ax] = cell
        if not so.empty:
            so.to_csv(out / f"oracle_sku_origin_{ax}.csv", index=False)
    (out / "oracle_ceilings.json").write_text(
        json.dumps(oracle_summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # LOO
    loo_methods = {
        "V2_POLICY": methods["V2_POLICY"],
        "V21_A0_50": methods["V21_A0_50"],
    }
    for ax in AX:
        loo_methods[f"V21_A0_{ax.upper()}_50_25_25"] = methods[
            f"V21_A0_{ax.upper()}_50_25_25"
        ]
    _loo_origin(loo_methods).to_csv(out / "loo_origin.csv", index=False)
    _loo_sku(loo_methods).to_csv(out / "loo_sku.csv", index=False)

    # Drop-top3 vs V21_A0_50 + advancement
    drop_rows = []
    adv_list = []
    for ax in AX:
        cand_name = f"V21_A0_{ax.upper()}_50_25_25"
        cand = methods[cand_name]
        d3 = _drop_top3(cand, methods["V21_A0_50"], methods["V2_POLICY"])
        d3_row = {"ax": ax, "candidate": cand_name, **{k: v for k, v in d3.items() if k != "dropped_skus"}}
        d3_row["dropped_skus"] = ",".join(d3["dropped_skus"])
        drop_rows.append(d3_row)
        adv_list.append(
            _advancement(
                ax=ax,
                candidate=cand,
                v2=methods["V2_POLICY"],
                v21_a0=methods["V21_A0_50"],
                cov_explain=cov_explain,
                drop3=d3,
            )
        )
    pd.DataFrame(drop_rows).to_csv(out / "drop_top3_vs_v21_a0.csv", index=False)

    # Tail by slice for key methods
    tail_rows = []
    for slice_name in slices:
        for name in (
            "V2_POLICY",
            "V21_A0_50",
            *[f"V21_A0_{ax.upper()}_50_25_25" for ax in AX],
            *[f"V21_{ax.upper()}_50" for ax in AX],
        ):
            sub = _filter_slice(methods[name], slice_name)
            t = _tail_stats(sub)
            tail_rows.append({"slice": slice_name, "method": name, **t})
    pd.DataFrame(tail_rows).to_csv(out / "tail_errors_by_slice.csv", index=False)

    overall = "HOLD"
    if all(a["verdict"] == "ADVANCE" for a in adv_list):
        overall = "ADVANCE_ALL"
    elif any(a["verdict"] == "ADVANCE" for a in adv_list):
        overall = "ADVANCE_PARTIAL"
    advancement = {
        "overall": overall,
        "rule": (
            "Primary candidate V21_A0_Ax_50_25_25 on pooled_55 must beat V2.1 and "
            "V21_A0_50, survive drop-top-3, meet coverage (>=95% seed jobs; dropout "
            "vs A0 within +1 group), and not regress tails beyond +2pp p95 / +2 "
            "groups WMAPE>100."
        ),
        "per_ax": adv_list,
    }
    # JSON-serializable drop3
    for a in advancement["per_ax"]:
        d3 = a.get("drop3") or {}
        a["drop3"] = {
            k: (list(v) if isinstance(v, set) else v) for k, v in d3.items()
        }
    (out / "advancement.json").write_text(
        json.dumps(advancement, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Report
    pooled_blends = pd.DataFrame(blend_rows)
    pooled_blends = pooled_blends.loc[pooled_blends["slice"] == "pooled_55"]
    lines = [
        "# Archval55 A2–A5 late-origin decision evaluation (v2)",
        "",
        "Baseline: `archval55_export_a0fix_v1` (late origins). "
        "Challengers: A2–A5 production-contract run. **No weight search.**",
        "",
        "Hindsight cell / SKU×origin oracles are **NON-DEPLOYABLE** ceilings only.",
        "",
        f"## Advancement overall: **{overall}**",
        "",
    ]
    for a in adv_list:
        lines.append(
            f"- **{a['ax'].upper()}**: `{a['verdict']}` — candidate WMAPE "
            f"{a['wmape_candidate']:.3f}% vs V2.1 {a['wmape_v2']:.3f}% / "
            f"V21_A0_50 {a['wmape_v21_a0_50']:.3f}% "
            f"(gates beats={a['gate_beats_v2_and_v21_a0']}, "
            f"drop3={a['gate_drop_top3']}, cov={a['gate_coverage']}, "
            f"tail={a['gate_tail']})"
        )
    lines.extend(["", "## Pooled_55 deployable WMAPE", "", "```"])
    lines.append(pooled_blends[["method", "n_rows", "portfolio_wmape_pct", "bias"]].to_string(index=False))
    lines.extend(["```", "", "## Coverage (A2–A5 run)", "", "```"])
    lines.append(cov_df.to_string(index=False))
    lines.extend(["```", ""])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "out_dir": str(out),
        "overall_advancement": overall,
        "per_ax_verdict": {a["ax"]: a["verdict"] for a in adv_list},
        "pooled_55_wmape": {
            r["method"]: r["portfolio_wmape_pct"]
            for r in blend_rows
            if r["slice"] == "pooled_55"
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
