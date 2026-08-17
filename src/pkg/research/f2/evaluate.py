"""Evaluate F2A / F2B / F2C on frozen matched PRIMARY rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import (
    F0_DRIFT_TOL,
    FILLNA_EXTRA,
    F2A,
    F2B,
    F2C,
    F2Experiment,
    HIGH_VOLUME_WATCHLIST,
    f2_output_dir,
    get_f2_experiment,
)
from pkg.research.features.demand import load_frozen_sales
from pkg.research.features.demand_f2 import add_demand_f2_features
from pkg.research.features.human_f2 import add_human_f2_features
from pkg.research.harness.dataset import copy_dataset, enrich_dataset, resolve_origin_col
from pkg.research.harness.gates import (  # noqa: F401 — re-exported for leftover imports
    assert_freeze_unchanged,
    confirm_canonical_f0,
    freeze_checksums,
)
from pkg.research.harness.metrics import (
    error_concentration,
    error_concentration as _error_concentration,
    horizon_bucket_table,
    merge_ae,
    merge_ae as _merge_ae,
    origins_improved,
    product_stats_full,
    product_stats_full as _product_stats_full,
    rel_wmape,
)
from pkg.research.harness.residual import make_residual_model
from pkg.research.harness.run import FamilySession
from pkg.research.harness.spec import ExperimentSpec

PROMOTION_BIAS_WMAPE_FRAC = 0.25


def make_f2_residual_model(anchor: str, feature_cols: Sequence[str]):
    """Residual XGB using frozen XGB_PARAMS / fit_xgb; fillna for F2 extras."""
    name = "ts_xgb_f2" if anchor == "ts" else "human_xgb_f2"
    return make_residual_model(
        anchor,
        feature_cols,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=(),
        name=name,
    )


def enrich_f2_dataset(ds: BenchmarkDataset, experiment: F2Experiment) -> BenchmarkDataset:
    """Copy freeze panels and attach F2 feature groups. Does not write to disk."""
    if not experiment.groups:
        return copy_dataset(ds)

    sales_hist = None
    if "demand_f2" in experiment.groups:
        sales_hist = load_frozen_sales(ds.root)

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        origin_col = resolve_origin_col(out)
        if "demand_f2" in experiment.groups:
            out = add_demand_f2_features(out, sales_hist, origin_col=origin_col)
        if "human_f2" in experiment.groups:
            out = add_human_f2_features(out, ds.budget_universe, origin_col=origin_col)
        return out

    return enrich_dataset(ds, _enrich)


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


def _spec_for(exp: F2Experiment, anchor: str) -> ExperimentSpec:
    enrich = "+".join(exp.groups) if exp.groups else None
    return ExperimentSpec(
        name=exp.name,
        anchor=anchor,
        features=exp.features_for(anchor),
        train_universe=exp.train_universe[anchor],
        control="F0",
        use_frozen_adapter=not bool(exp.groups),
        enrich=enrich,
    )


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
    enrichers = {
        "demand_f2": lambda ds: enrich_f2_dataset(ds, F2A),
        "human_f2": lambda ds: enrich_f2_dataset(ds, F2B),
        "demand_f2+human_f2": lambda ds: enrich_f2_dataset(ds, F2C),
    }
    session = FamilySession(
        "f2",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=frozenset(),
        enrichers=enrichers,
        model_name_prefix="xgb",
    )
    ds = session.ds
    canon = session.canon
    f0_results = session.f0_results

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
        f0_w = float(f0.overall["wmape"].iloc[0])
        expected_f0 = canon["canonical"][anchor]
        if abs(f0_w - expected_f0) > F0_DRIFT_TOL:
            raise AssertionError(
                f"F0 {anchor} WMAPE drifted during F2 run: {f0_w} vs canonical {expected_f0}"
            )

        o = res.overall.iloc[0]
        n_imp, n_tot = origins_improved(f0, res)
        pstats = product_stats_full(f0, res)
        m = merge_ae(f0, res)
        conc = error_concentration(m, exp.name, anchor)
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
                "n_features": len(exp.features_for(anchor)),
                "wmape": float(o["wmape"]),
                "wmape_f0": f0_w,
                "rel_wmape_vs_f0_pct": rel_wmape(f0_w, float(o["wmape"])),
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
                    "rel_wmape_vs_f0_pct": rel_wmape(f0_ow, float(row["wmape"])),
                    "n": int(row["n"]),
                }
            )

        hb = horizon_bucket_table(f0, res)
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

    for anchor in ("ts", "human"):
        _record(get_f2_experiment("F0"), anchor, f0_results[anchor])

    if "F2A" in requested:
        for anchor in F2A.anchors:
            res = session.run(_spec_for(F2A, anchor))
            _record(F2A, anchor, res)

    if "F2B" in requested:
        for anchor in F2B.anchors:
            res = session.run(_spec_for(F2B, anchor))
            _record(F2B, anchor, res)

    run_f2c = "F2C" in requested
    if run_f2c and experiments is None:
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
            res = session.run(_spec_for(F2C, anchor))
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

    session.finish()

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
