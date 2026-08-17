"""Evaluate F2A / F2B / F2C on frozen matched PRIMARY rows."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.config import PANEL_FILES, PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, horizon_bucket
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.benchmark.models import fit_xgb
from pkg.research.evaluate_features import (
    ROW_KEYS,
    _horizon_bucket_table,
    _origins_improved,
    _rel_wmape,
    assert_same_eval_rows,
)
from pkg.research.f2.config import (
    CONCENTRATION_ONE_PRODUCT,
    CONCENTRATION_TOP5,
    F0_DRIFT_TOL,
    F0_WMAPE_TOL,
    FILLNA_EXTRA,
    F2A,
    F2B,
    F2C,
    F2Experiment,
    HIGH_VOLUME_WATCHLIST,
    LOCKED_F0_WMAPE,
    f2_output_dir,
    get_f2_experiment,
)
from pkg.research.features.demand import load_frozen_sales
from pkg.research.features.demand_f2 import add_demand_f2_features
from pkg.research.features.human_f2 import add_human_f2_features

PROMOTION_BIAS_WMAPE_FRAC = 0.25  # |bias| worsening vs |F0 bias| relative to WMAPE scale


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def freeze_checksums(ds: BenchmarkDataset) -> dict[str, str]:
    out = {}
    for name in PANEL_FILES:
        p = ds.root / name
        if p.exists():
            out[name] = _file_sha256(p)
    man = ds.root / "manifest.json"
    if man.exists():
        out["manifest.json"] = _file_sha256(man)
    return out


def assert_freeze_unchanged(ds: BenchmarkDataset, before: dict[str, str]) -> None:
    after = freeze_checksums(ds)
    if after != before:
        raise AssertionError(
            "F2 evaluation modified frozen benchmark files "
            f"(before={before} after={after})"
        )


def make_f2_residual_model(anchor: str, feature_cols: Sequence[str]):
    """Residual XGB using frozen XGB_PARAMS / fit_xgb; fillna for F2 extras."""
    cols = list(feature_cols)
    if anchor == "ts":
        forecast_col = "ts_forecast"
        name = "ts_xgb_f2"
    elif anchor == "human":
        forecast_col = "budget_forecast"
        name = "human_xgb_f2"
    else:
        raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")

    def _predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        tr = train_df.copy()
        te = test_df.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
        missing = [c for c in cols if c not in tr.columns or c not in te.columns]
        if missing:
            raise KeyError(f"missing feature columns for {name}: {missing}")
        for c in cols:
            if (
                c.startswith("sales_")
                or c.startswith("human_")
                or c in FILLNA_EXTRA
            ):
                tr[c] = tr[c].fillna(0)
                te[c] = te[c].fillna(0)
        if "horizon" not in tr.columns:
            raise KeyError("train_df needs horizon for sample weights")
        model = fit_xgb(cols, tr)
        resid = model.predict(te[cols])
        return np.maximum(0.0, te[forecast_col].astype(float).to_numpy() + resid)

    _predict.__name__ = name
    return _predict


def enrich_f2_dataset(ds: BenchmarkDataset, experiment: F2Experiment) -> BenchmarkDataset:
    """Copy freeze panels and attach F2 feature groups. Does not write to disk."""
    if not experiment.groups:
        return BenchmarkDataset(
            version=ds.version,
            root=ds.root,
            ts_universe=ds.ts_universe.copy(),
            budget_universe=ds.budget_universe.copy(),
            matched_universe=ds.matched_universe.copy(),
            manifest=ds.manifest,
        )

    sales_hist = None
    if "demand_f2" in experiment.groups:
        sales_hist = load_frozen_sales(ds.root)

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        origin_col = (
            "origin"
            if "origin" in out.columns
            else ("ts_origin" if "ts_origin" in out.columns else "budget_origin")
        )
        if "demand_f2" in experiment.groups:
            out = add_demand_f2_features(out, sales_hist, origin_col=origin_col)
        if "human_f2" in experiment.groups:
            out = add_human_f2_features(out, ds.budget_universe, origin_col=origin_col)
        return out

    return BenchmarkDataset(
        version=ds.version,
        root=ds.root,
        ts_universe=_enrich(ds.ts_universe),
        budget_universe=_enrich(ds.budget_universe),
        matched_universe=_enrich(ds.matched_universe),
        manifest=ds.manifest,
    )


def run_frozen_f0(ds: BenchmarkDataset, anchor: str) -> BacktestResult:
    name = "ts_xgb" if anchor == "ts" else "human_xgb"
    return backtest(name, dataset=ds, universe="matched", eligibility="primary")


def confirm_canonical_f0(ds: BenchmarkDataset) -> dict:
    """Reproduce frozen F0; return canonical metrics used by F2.

    Does not rewrite locked EXPECTED_ANALYSIS_B_PRIMARY. If the current
    environment's XGBoost disagrees with freeze-time WMAPE, F2 uses the
    *currently reproduced* frozen backtest as canonical and records the gap.
    Refuses to run if n / origins do not match the contract.
    """
    rows = []
    f0_results: dict[str, BacktestResult] = {}
    for anchor, key in (("ts", "ts_xgb"), ("human", "human_xgb")):
        res = run_frozen_f0(ds, anchor)
        f0_results[anchor] = res
        got = float(res.overall["wmape"].iloc[0])
        n = int(res.overall["n"].iloc[0])
        n_origins = len(res.origins)
        locked = LOCKED_F0_WMAPE[key]
        rows.append(
            {
                "anchor": anchor,
                "frozen_name": key,
                "wmape_reproduced": got,
                "wmape_locked_contract": locked,
                "wmape_gap": got - locked,
                "n": n,
                "n_locked": LOCKED_F0_WMAPE["n"],
                "n_origins": n_origins,
                "n_origins_locked": LOCKED_F0_WMAPE["n_origins"],
                "matches_locked_wmape": abs(got - locked) <= F0_WMAPE_TOL,
            }
        )
        if n != LOCKED_F0_WMAPE["n"]:
            raise AssertionError(
                f"F0 {anchor} n={n} != locked contract n={LOCKED_F0_WMAPE['n']}"
            )
        if n_origins != LOCKED_F0_WMAPE["n_origins"]:
            raise AssertionError(
                f"F0 {anchor} n_origins={n_origins} != "
                f"{LOCKED_F0_WMAPE['n_origins']}"
            )
        if sorted(int(o) for o in res.origins) != list(PRIMARY_ORIGINS):
            raise AssertionError(
                f"F0 {anchor} origins {res.origins} != PRIMARY {PRIMARY_ORIGINS}"
            )

    summary = pd.DataFrame(rows)
    canonical = {
        "ts": float(summary.loc[summary["anchor"] == "ts", "wmape_reproduced"].iloc[0]),
        "human": float(
            summary.loc[summary["anchor"] == "human", "wmape_reproduced"].iloc[0]
        ),
        "n": LOCKED_F0_WMAPE["n"],
        "n_origins": LOCKED_F0_WMAPE["n_origins"],
        "source": "current frozen backtest(ts_xgb/human_xgb) on pkg.benchmark v1",
        "locked_contract_matches": bool(summary["matches_locked_wmape"].all()),
    }
    return {"summary": summary, "canonical": canonical, "results": f0_results}


def _product_stats_full(base: BacktestResult, cand: BacktestResult) -> dict:
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    rows = []
    for product, g in m.groupby("product"):
        if len(g) < 3:
            continue
        w0 = wmape(g["actual"], g["pred_f0"])
        w1 = wmape(g["actual"], g["pred_new"])
        rows.append(
            {
                "product": product,
                "n": len(g),
                "wmape_f0": w0,
                "wmape_new": w1,
                "rel_improvement_pct": _rel_wmape(w0, w1),
            }
        )
    if not rows:
        return {
            "product_win_rate": float("nan"),
            "median_product_improvement_pct": float("nan"),
            "p25_product_improvement_pct": float("nan"),
            "p75_product_improvement_pct": float("nan"),
            "n_products": 0,
            "table": pd.DataFrame(),
        }
    pdf = pd.DataFrame(rows)
    return {
        "product_win_rate": float((pdf["rel_improvement_pct"] > 0).mean()),
        "median_product_improvement_pct": float(pdf["rel_improvement_pct"].median()),
        "p25_product_improvement_pct": float(pdf["rel_improvement_pct"].quantile(0.25)),
        "p75_product_improvement_pct": float(pdf["rel_improvement_pct"].quantile(0.75)),
        "n_products": int(len(pdf)),
        "table": pdf,
    }


def _merge_ae(f0: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    b = f0.predictions[
        ["product", "qrt", "target_date", "test_origin", "horizon", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_cand"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    m["ae_f0"] = (m["actual"] - m["pred_f0"]).abs()
    m["ae_cand"] = (m["actual"] - m["pred_cand"]).abs()
    m["delta_ae"] = m["ae_cand"] - m["ae_f0"]
    m["horizon_bucket"] = m["horizon"].map(horizon_bucket)
    return m


def _error_concentration(m: pd.DataFrame, experiment: str, anchor: str) -> dict:
    net = float(m["delta_ae"].sum())
    det = m.loc[m["delta_ae"] > 0, "delta_ae"].sum()
    imp = -m.loc[m["delta_ae"] < 0, "delta_ae"].sum()
    by_p = (
        m.groupby("product")
        .agg(
            delta_ae=("delta_ae", "sum"),
            actual_volume=("actual", lambda s: float(np.abs(s).sum())),
            n=("delta_ae", "count"),
        )
        .reset_index()
        .sort_values("delta_ae", ascending=False)
    )
    det_sorted = by_p.loc[by_p["delta_ae"] > 0].sort_values("delta_ae", ascending=False)
    imp_sorted = by_p.loc[by_p["delta_ae"] < 0].sort_values("delta_ae")
    top5_det_share = (
        float(det_sorted.head(5)["delta_ae"].sum() / det) if det > 0 else 0.0
    )
    top10_det_share = (
        float(det_sorted.head(10)["delta_ae"].sum() / det) if det > 0 else 0.0
    )
    top5_imp_share = (
        float((-imp_sorted.head(5)["delta_ae"]).sum() / imp) if imp > 0 else 0.0
    )
    top1_share_of_net_det = 0.0
    if det > 0 and len(det_sorted):
        top1_share_of_net_det = float(det_sorted.iloc[0]["delta_ae"] / det)

    flags = []
    if top1_share_of_net_det > CONCENTRATION_ONE_PRODUCT:
        flags.append("one_product_gt_25pct_deterioration")
    if top5_det_share > CONCENTRATION_TOP5:
        flags.append("top5_gt_50pct_deterioration")

    return {
        "net_delta_ae": net,
        "total_deterioration": float(det),
        "total_improvement": float(imp),
        "top5_deterioration_share": top5_det_share,
        "top10_deterioration_share": top10_det_share,
        "top5_improvement_share": top5_imp_share,
        "top1_deterioration_share": top1_share_of_net_det,
        "flags": flags,
        "by_product": by_p,
        "top5_det": det_sorted.head(5),
        "top10_det": det_sorted.head(10),
        "top5_imp": imp_sorted.head(5),
    }


def classify_outcome(
    *,
    wmape_f0: float,
    wmape_new: float,
    product_win_rate: float,
    median_product_improvement_pct: float,
    origins_improved: int,
    origins_total: int,
    bias_f0: float,
    bias_new: float,
    concentration_flags: list[str],
) -> str:
    """PROMOTE / PROMISING_BUT_UNSTABLE / REJECT — not tuned on test WMAPE."""
    wmape_better = wmape_new < wmape_f0
    median_ok = median_product_improvement_pct > 0
    win_ok = product_win_rate > 0.50
    origins_ok = origins_improved > origins_total / 2
    bias_ok = abs(bias_new) <= abs(bias_f0) * (1.0 + PROMOTION_BIAS_WMAPE_FRAC) or abs(
        bias_new
    ) <= abs(bias_f0) + 200.0
    concentrated = bool(concentration_flags)

    if (
        wmape_better
        and median_ok
        and win_ok
        and origins_ok
        and bias_ok
        and not concentrated
    ):
        return "PROMOTE"
    if wmape_better and (not median_ok or not win_ok or not origins_ok or concentrated):
        return "PROMISING_BUT_UNSTABLE"
    if (not wmape_better) and (median_ok or win_ok):
        return "PROMISING_BUT_UNSTABLE"
    return "REJECT"


def _run_experiment(
    ds: BenchmarkDataset,
    experiment: F2Experiment,
    anchor: str,
    f0: BacktestResult,
) -> BacktestResult:
    if experiment.name == "F0":
        return f0
    enriched = enrich_f2_dataset(ds, experiment)
    feats = experiment.features_for(anchor)  # type: ignore[arg-type]
    model = make_f2_residual_model(anchor, feats)
    train_u = experiment.train_universe[anchor]
    result = backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_u,
    )
    assert_same_eval_rows(f0, result)
    return result


def _train_diag_rows(res: BacktestResult, experiment: str, anchor: str) -> pd.DataFrame:
    fd = res.fold_diagnostics.copy()
    fd["experiment"] = experiment
    fd["anchor"] = anchor
    fd["train_universe"] = "ts" if anchor == "ts" else "budget"
    return fd


def evaluate_f2(
    *,
    experiments: Optional[Sequence[str]] = None,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
    skip_diagnostics: bool = False,
) -> dict:
    """Run F2 experiments independently (F2A, F2B) then F2C if justified."""
    out_dir = out_dir or f2_output_dir()
    ds = dataset or load_benchmark(verify_checksums=verify_checksums)
    freeze_before = freeze_checksums(ds)

    canon = confirm_canonical_f0(ds)
    f0_results: dict[str, BacktestResult] = canon["results"]
    canon["summary"].to_csv(out_dir / "f0_canonical.csv", index=False)

    requested = list(experiments) if experiments is not None else ["F2A", "F2B", "F2C"]

    if not skip_diagnostics:
        from pkg.research.f2.diagnostics import run_demand_diagnostics, run_human_diagnostics

        if any(x in requested for x in ("F2A", "F2C")):
            run_demand_diagnostics(ds, out_dir=out_dir)
        if any(x in requested for x in ("F2B", "F2C")):
            run_human_diagnostics(ds, out_dir=out_dir)

    overall_rows = []
    origin_rows = []
    horizon_rows = []
    train_rows = []
    conc_rows = []
    watch_rows = []
    product_tables = []
    classifications: dict[tuple[str, str], str] = {}
    all_results: dict[tuple[str, str], BacktestResult] = {}

    def _record(exp: F2Experiment, anchor: str, res: BacktestResult) -> None:
        f0 = f0_results[anchor]
        # Drift check vs canonical reproduced F0
        f0_w = float(f0.overall["wmape"].iloc[0])
        expected_f0 = canon["canonical"][anchor]
        if abs(f0_w - expected_f0) > F0_DRIFT_TOL:
            raise AssertionError(
                f"F0 {anchor} WMAPE drifted during F2 run: {f0_w} vs canonical {expected_f0}"
            )

        o = res.overall.iloc[0]
        n_imp, n_tot = _origins_improved(f0, res)
        pstats = _product_stats_full(f0, res)
        m = _merge_ae(f0, res)
        conc = _error_concentration(m, exp.name, anchor)
        if exp.name == "F0":
            verdict = "CONTROL"
        else:
            verdict = classify_outcome(
            wmape_f0=f0_w,
            wmape_new=float(o["wmape"]),
            product_win_rate=pstats["product_win_rate"],
            median_product_improvement_pct=pstats["median_product_improvement_pct"],
            origins_improved=n_imp,
            origins_total=n_tot,
            bias_f0=float(f0.overall["bias"].iloc[0]),
            bias_new=float(o["bias"]),
            concentration_flags=conc["flags"],
        )
        classifications[(exp.name, anchor)] = verdict
        all_results[(exp.name, anchor)] = res

        overall_rows.append(
            {
                "experiment": exp.name,
                "anchor": anchor,
                "groups": "+".join(exp.groups) or "f0",
                "n_features": len(exp.features_for(anchor)),  # type: ignore[arg-type]
                "wmape": float(o["wmape"]),
                "wmape_f0": f0_w,
                "rel_wmape_vs_f0_pct": _rel_wmape(f0_w, float(o["wmape"])),
                "rmse": float(o["rmse"]),
                "mae": float(o["mae"]),
                "bias": float(o["bias"]),
                "bias_f0": float(f0.overall["bias"].iloc[0]),
                "n": int(o["n"]),
                "origins_improved": n_imp,
                "origins_total": n_tot,
                "product_win_rate": pstats["product_win_rate"],
                "median_product_improvement_pct": pstats["median_product_improvement_pct"],
                "p25_product_improvement_pct": pstats["p25_product_improvement_pct"],
                "p75_product_improvement_pct": pstats["p75_product_improvement_pct"],
                "n_products": pstats["n_products"],
                "net_delta_ae": conc["net_delta_ae"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top5_improvement_share": conc["top5_improvement_share"],
                "concentration_flags": ";".join(conc["flags"]),
                "verdict": verdict,
            }
        )

        for _, row in res.by_origin.iterrows():
            o_id = int(row["origin"])
            f0_ow = float(f0.by_origin.loc[f0.by_origin["origin"] == o_id, "wmape"].iloc[0])
            origin_rows.append(
                {
                    "experiment": exp.name,
                    "anchor": anchor,
                    "origin": o_id,
                    "wmape": float(row["wmape"]),
                    "wmape_f0": f0_ow,
                    "rel_wmape_vs_f0_pct": _rel_wmape(f0_ow, float(row["wmape"])),
                    "n": int(row["n"]),
                }
            )

        hb = _horizon_bucket_table(f0, res)
        for _, row in hb.iterrows():
            horizon_rows.append({"experiment": exp.name, "anchor": anchor, **row.to_dict()})

        train_rows.append(_train_diag_rows(res, exp.name, anchor))

        conc_rows.append(
            {
                "experiment": exp.name,
                "anchor": anchor,
                "net_delta_ae": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "top5_improvement_share": conc["top5_improvement_share"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "flags": ";".join(conc["flags"]),
            }
        )
        by_p = conc["by_product"].copy()
        by_p["experiment"] = exp.name
        by_p["anchor"] = anchor
        product_tables.append(by_p)

        for _, row in conc["top10_det"].iterrows():
            watch_rows.append(
                {
                    "experiment": exp.name,
                    "anchor": anchor,
                    "direction": "deterioration",
                    **row.to_dict(),
                }
            )
        for _, row in conc["top5_imp"].iterrows():
            watch_rows.append(
                {
                    "experiment": exp.name,
                    "anchor": anchor,
                    "direction": "improvement",
                    **row.to_dict(),
                }
            )

        # Watchlist
        for sku in HIGH_VOLUME_WATCHLIST:
            sub = m.loc[m["product"] == sku]
            if sub.empty:
                continue
            watch_rows.append(
                {
                    "experiment": exp.name,
                    "anchor": anchor,
                    "direction": "watchlist",
                    "product": sku,
                    "delta_ae": float(sub["delta_ae"].sum()),
                    "actual_volume": float(np.abs(sub["actual"]).sum()),
                    "n": len(sub),
                    "wmape_f0": wmape(sub["actual"], sub["pred_f0"]),
                    "wmape_cand": wmape(sub["actual"], sub["pred_cand"]),
                }
            )

        by_origin_ae = m.groupby("test_origin")["delta_ae"].sum().reset_index()
        by_origin_ae["experiment"] = exp.name
        by_origin_ae["anchor"] = anchor
        by_origin_ae.to_csv(
            out_dir / f"delta_ae_by_origin_{exp.name}_{anchor}.csv", index=False
        )
        by_hz = m.groupby("horizon_bucket")["delta_ae"].sum().reset_index()
        by_hz["experiment"] = exp.name
        by_hz["anchor"] = anchor
        by_hz.to_csv(out_dir / f"delta_ae_by_horizon_{exp.name}_{anchor}.csv", index=False)
        by_po = (
            m.groupby(["product", "test_origin"])["delta_ae"]
            .sum()
            .reset_index()
            .sort_values("delta_ae", ascending=False)
        )
        by_po["experiment"] = exp.name
        by_po["anchor"] = anchor
        by_po.to_csv(
            out_dir / f"delta_ae_by_product_origin_{exp.name}_{anchor}.csv", index=False
        )

    # Always record F0 for the scoreboard
    for anchor in ("ts", "human"):
        _record(get_f2_experiment("F0"), anchor, f0_results[anchor])

    if "F2A" in requested:
        for anchor in F2A.anchors:
            res = _run_experiment(ds, F2A, anchor, f0_results[anchor])
            _record(F2A, anchor, res)

    if "F2B" in requested:
        for anchor in F2B.anchors:
            res = _run_experiment(ds, F2B, anchor, f0_results[anchor])
            _record(F2B, anchor, res)

    run_f2c = "F2C" in requested
    if run_f2c and experiments is None:
        # Independent-first: skip F2C if a family is REJECT
        f2a_h = classifications.get(("F2A", "human"))
        f2b_h = classifications.get(("F2B", "human"))
        if f2a_h == "REJECT" or f2b_h == "REJECT" or f2a_h is None or f2b_h is None:
            run_f2c = False
            pd.DataFrame(
                [
                    {
                        "skipped": True,
                        "reason": "F2C not forced: a family is REJECT or missing",
                        "F2A_human": f2a_h,
                        "F2B_human": f2b_h,
                    }
                ]
            ).to_csv(out_dir / "f2c_skip.csv", index=False)

    if run_f2c and "F2C" in requested:
        for anchor in F2C.anchors:
            res = _run_experiment(ds, F2C, anchor, f0_results[anchor])
            _record(F2C, anchor, res)

    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    train_diag = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    conc_df = pd.DataFrame(conc_rows)
    products_df = pd.concat(product_tables, ignore_index=True) if product_tables else pd.DataFrame()
    watch_df = pd.DataFrame(watch_rows)

    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon_bucket.csv", index=False)
    train_diag.to_csv(out_dir / "train_diagnostics.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    products_df.to_csv(out_dir / "by_product.csv", index=False)
    watch_df.to_csv(out_dir / "watchlist_and_top.csv", index=False)

    assert_freeze_unchanged(ds, freeze_before)

    return {
        "canonical_f0": canon,
        "overall": overall,
        "by_origin": by_origin,
        "by_horizon_bucket": by_horizon,
        "train_diagnostics": train_diag,
        "error_concentration": conc_df,
        "by_product": products_df,
        "watchlist": watch_df,
        "classifications": classifications,
        "results": all_results,
        "f0_results": f0_results,
        "out_dir": out_dir,
    }
