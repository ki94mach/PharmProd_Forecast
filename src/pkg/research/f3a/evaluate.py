"""Evaluate F3A (observed product tenure) on frozen matched PRIMARY rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.features.demand import load_frozen_sales
from pkg.research.features.lifecycle import SCORED_FEATURE, add_lifecycle_features
from pkg.research.f3a.audit import assign_age_group, audit_lifecycle
from pkg.research.f3a.config import (
    ALL_EXPERIMENTS,
    CORE_TS_WMAPE_REF,
    CURRENT_ENV_F0_WMAPE,
    FILLNA_EXTRA,
    NEVER_FILLNA,
    PAIRS,
    F3AExperiment,
    H1,
    T1,
    T2,
    T3,
    f3a_output_dir,
)
from pkg.research.harness.dataset import enrich_dataset, resolve_origin_col
from pkg.research.harness.gates import assert_wmape_gate, wmape_gate_row
from pkg.research.harness.metrics import (
    ROW_KEYS,
    error_concentration,
    horizon_bucket_table,
    merge_ae,
    origin_pair_table,
    origin_summary,
    product_pair_table,
    product_summary,
    rel_wmape,
)
from pkg.research.harness.residual import make_residual_model
from pkg.research.harness.run import FamilySession
from pkg.research.harness.spec import ExperimentSpec

PROMOTION_BIAS_WMAPE_FRAC = 0.25


def make_f3a_residual_model(anchor: str, feature_cols: Sequence[str]):
    """Residual XGB using frozen XGB_PARAMS / fit_xgb; never fillna lifecycle age."""
    name = "ts_xgb_f3a" if anchor == "ts" else "human_xgb_f3a"
    return make_residual_model(
        anchor,
        feature_cols,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        name=name,
    )


def enrich_lifecycle_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    sales_hist = load_frozen_sales(ds.root)

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        return add_lifecycle_features(
            panel, sales_hist, origin_col=resolve_origin_col(panel)
        )

    return enrich_dataset(ds, _enrich)


def _spec_for(exp: F3AExperiment) -> ExperimentSpec:
    enrich = "lifecycle" if exp.include_lifecycle else None
    return ExperimentSpec(
        name=exp.name,
        anchor=exp.anchor,
        features=exp.features(),
        train_universe=exp.train_universe,
        control=exp.control,
        use_frozen_adapter=exp.feature_source == "frozen" and not exp.include_lifecycle,
        enrich=enrich,
    )


def _bias_ok(bias_control: float, bias_cand: float) -> bool:
    return abs(bias_cand) <= abs(bias_control) * (1.0 + PROMOTION_BIAS_WMAPE_FRAC) or abs(
        bias_cand
    ) <= abs(bias_control) + 200.0


def _anchor_helps(
    *,
    wmape_control: float,
    wmape_cand: float,
    origins_improved: int,
    origins_total: int,
    product_win_rate: float,
    median_product_improvement_pct: float,
    bias_control: float,
    bias_cand: float,
    concentration_flags: list[str],
) -> bool:
    pooled = wmape_cand < wmape_control
    origins_ok = origins_total > 0 and origins_improved > origins_total / 2.0
    product_ok = (
        np.isfinite(product_win_rate) and product_win_rate > 0.50
    ) or (
        np.isfinite(median_product_improvement_pct)
        and median_product_improvement_pct > 0
    )
    return (
        pooled
        and origins_ok
        and product_ok
        and _bias_ok(bias_control, bias_cand)
        and not bool(concentration_flags)
    )


def _segmentation_signal(age_groups: pd.DataFrame) -> bool:
    """True if WMAPE varies materially across observed-age groups (not missing)."""
    if age_groups is None or age_groups.empty:
        return False
    g = age_groups.loc[age_groups["age_group"] != "age_missing"]
    if len(g) < 2:
        return False
    for col in ("wmape_T0", "wmape_T1", "wmape_H0", "wmape_H1"):
        if col not in g.columns:
            continue
        vals = g[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        if float(np.max(vals) - np.min(vals)) >= 1.0:
            return True
    for a, b in (("wmape_T1", "wmape_T0"), ("wmape_H1", "wmape_H0")):
        if a not in g.columns or b not in g.columns:
            continue
        delta = g[b].to_numpy(dtype=float) - g[a].to_numpy(dtype=float)
        finite = delta[np.isfinite(delta)]
        if len(finite) >= 2 and (finite > 0).any() and (finite < 0).any():
            return True
    return False


def classify_f3a(
    overall: pd.DataFrame,
    conc: pd.DataFrame,
    age_groups: pd.DataFrame,
) -> str:
    def row(name: str):
        sub = overall.loc[overall["experiment"] == name]
        return sub.iloc[0] if len(sub) else None

    def flags(name: str) -> list[str]:
        if conc is None or conc.empty or "flags" not in conc.columns:
            return []
        sub = conc.loc[conc["experiment"] == name]
        if sub.empty:
            return []
        raw = sub["flags"].iloc[0]
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)) or raw == "":
            return []
        return [x for x in str(raw).split(";") if x]

    t1, h1 = row("T1"), row("H1")
    if t1 is None or h1 is None:
        return "E"
    ts_helps = _anchor_helps(
        wmape_control=float(t1["wmape_control"]),
        wmape_cand=float(t1["wmape"]),
        origins_improved=int(t1["origins_improved"]),
        origins_total=int(t1["origins_total"]),
        product_win_rate=float(t1["product_win_rate"]),
        median_product_improvement_pct=float(t1["median_product_improvement_pct"]),
        bias_control=float(t1["bias_control"]),
        bias_cand=float(t1["bias"]),
        concentration_flags=flags("T1"),
    )
    human_helps = _anchor_helps(
        wmape_control=float(h1["wmape_control"]),
        wmape_cand=float(h1["wmape"]),
        origins_improved=int(h1["origins_improved"]),
        origins_total=int(h1["origins_total"]),
        product_win_rate=float(h1["product_win_rate"]),
        median_product_improvement_pct=float(h1["median_product_improvement_pct"]),
        bias_control=float(h1["bias_control"]),
        bias_cand=float(h1["bias"]),
        concentration_flags=flags("H1"),
    )
    if ts_helps and human_helps:
        return "A"
    if human_helps and not ts_helps:
        return "B"
    if ts_helps and not human_helps:
        return "C"
    if _segmentation_signal(age_groups):
        return "D"
    return "E"


def _age_group_table(
    results: dict[str, BacktestResult],
    age_lookup: pd.DataFrame,
    edges,
) -> pd.DataFrame:
    t0 = results["T0"].predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_T0"})
    t1 = results["T1"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_T1"}
    )
    h0 = results["H0"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_H0"}
    )
    h1 = results["H1"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_H1"}
    )
    m = (
        t0.merge(t1, on=list(ROW_KEYS))
        .merge(h0, on=list(ROW_KEYS))
        .merge(h1, on=list(ROW_KEYS))
    )
    look = age_lookup.copy()
    look["product"] = look["product"].astype(str)
    look["test_origin"] = look["test_origin"].astype(int)
    m["product"] = m["product"].astype(str)
    m["test_origin"] = m["test_origin"].astype(int)
    m = m.merge(look, on=["product", "test_origin"], how="left")
    m["age_group"] = [
        assign_age_group(a, edges) for a in m[SCORED_FEATURE].to_numpy(dtype=float)
    ]
    rows = []
    for grp in ("Q1_youngest", "Q2", "Q3", "Q4_oldest", "age_available", "age_missing"):
        g = m.loc[m["age_group"] == grp]
        if g.empty:
            continue
        rows.append(
            {
                "age_group": grp,
                "n": int(len(g)),
                "actual_volume": float(np.abs(g["actual"]).sum()),
                "wmape_T0": wmape(g["actual"], g["pred_T0"]),
                "wmape_T1": wmape(g["actual"], g["pred_T1"]),
                "wmape_H0": wmape(g["actual"], g["pred_H0"]),
                "wmape_H1": wmape(g["actual"], g["pred_H1"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_f3a(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f3a_output_dir()
    session = FamilySession(
        "f3a",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        enrichers={"lifecycle": enrich_lifecycle_dataset},
        model_name_prefix="xgb",
    )
    t0 = session.f0("ts")
    h0 = session.f0("human")
    canon = session.canon

    gate_rows = []
    t0_w = float(t0.overall["wmape"].iloc[0])
    h0_w = float(h0.overall["wmape"].iloc[0])
    gate_rows.append(
        wmape_gate_row(
            "T0 vs current-env TS F0",
            t0_w,
            CURRENT_ENV_F0_WMAPE["ts"],
            int(t0.overall["n"].iloc[0]),
            len(t0.origins),
        )
    )
    gate_rows.append(
        wmape_gate_row(
            "H0 vs current-env Human F0",
            h0_w,
            CURRENT_ENV_F0_WMAPE["human"],
            int(h0.overall["n"].iloc[0]),
            len(h0.origins),
        )
    )
    if sorted(int(o) for o in t0.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(f"T0 origins {t0.origins} != PRIMARY {PRIMARY_ORIGINS}")
    if sorted(int(o) for o in h0.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(f"H0 origins {h0.origins} != PRIMARY {PRIMARY_ORIGINS}")
    for row in gate_rows:
        assert_wmape_gate(row)

    sales_hist = load_frozen_sales(session.ds.root)
    audit_panel = t0.predictions[["product", "test_origin"]].copy()
    audit_panel["product"] = audit_panel["product"].astype(str)
    audit_panel["origin"] = audit_panel["test_origin"].astype(int)
    audit = audit_lifecycle(audit_panel, sales_hist, origin_col="origin")
    audit["product_audit"].to_csv(out_dir / "lifecycle_audit.csv", index=False)
    audit["coverage"].to_csv(out_dir / "lifecycle_coverage.csv", index=False)
    audit["by_origin"].to_csv(out_dir / "lifecycle_audit_by_origin.csv", index=False)
    audit["first_nonzero_audit"].to_csv(out_dir / "first_nonzero_audit.csv", index=False)

    print("=== T2 CORE_TS control ===")
    t2 = session.run(_spec_for(T2))
    t2_w = float(t2.overall["wmape"].iloc[0])
    t2_gate = wmape_gate_row(
        "T2 vs prior CORE_TS diagnostic",
        t2_w,
        CORE_TS_WMAPE_REF,
        int(t2.overall["n"].iloc[0]),
        len(t2.origins),
    )
    gate_rows.append(t2_gate)
    assert_wmape_gate(t2_gate)
    pd.DataFrame(gate_rows).to_csv(out_dir / "reproduction_gates.csv", index=False)

    print("=== T1 F0 + F3A ===")
    t1 = session.run(_spec_for(T1))
    print("=== T3 CORE_TS + F3A ===")
    t3 = session.run(_spec_for(T3))
    print("=== H1 F0_HUMAN + F3A ===")
    h1 = session.run(_spec_for(H1))

    results: dict[str, BacktestResult] = {
        "T0": t0,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "H0": h0,
        "H1": h1,
    }

    pair_map = {cand: ctrl for cand, ctrl in PAIRS}
    overall_rows = []
    origin_rows = []
    product_rows = []
    horizon_rows = []
    conc_rows = []
    watch_rows = []

    for name, exp in ALL_EXPERIMENTS.items():
        res = results[name]
        control_name = pair_map.get(name, name)
        control = results[control_name]
        o = res.overall.iloc[0]
        co = control.overall.iloc[0]
        odf = origin_pair_table(control, res)
        osu = origin_summary(odf)
        pdf = product_pair_table(control, res)
        psu = product_summary(pdf)
        m = merge_ae(control, res)
        conc = error_concentration(m, name, exp.anchor)
        hb = horizon_bucket_table(control, res)

        overall_rows.append(
            {
                "experiment": name,
                "anchor": exp.anchor,
                "control": control_name,
                "n_features": len(exp.features()),
                "include_lifecycle": exp.include_lifecycle,
                "train_universe": exp.train_universe,
                "wmape": float(o["wmape"]),
                "wmape_control": float(co["wmape"]),
                "rel_wmape_vs_control_pct": rel_wmape(
                    float(co["wmape"]), float(o["wmape"])
                ),
                "rmse": float(o["rmse"]),
                "mae": float(o["mae"]),
                "bias": float(o["bias"]),
                "bias_control": float(co["bias"]),
                "n": int(o["n"]),
                **osu,
                **psu,
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
            }
        )
        for _, r in odf.iterrows():
            origin_rows.append(
                {"experiment": name, "control": control_name, **r.to_dict()}
            )
        for _, r in pdf.iterrows():
            product_rows.append(
                {"experiment": name, "control": control_name, **r.to_dict()}
            )
        for _, r in hb.iterrows():
            horizon_rows.append(
                {
                    "experiment": name,
                    "control": control_name,
                    "horizon_bucket": r["horizon_bucket"],
                    "n": r["n"],
                    "wmape_control": r["wmape_f0"],
                    "wmape_candidate": r["wmape_new"],
                    "relative_improvement_pct": r["rel_wmape_vs_f0_pct"],
                }
            )
        conc_rows.append(
            {
                "experiment": name,
                "anchor": exp.anchor,
                "control": control_name,
                "net_delta_ae": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "top5_improvement_share": conc["top5_improvement_share"],
                "flags": ";".join(conc["flags"]),
            }
        )
        for sku in HIGH_VOLUME_WATCHLIST:
            sub = m.loc[m["product"] == sku]
            if sub.empty:
                continue
            watch_rows.append(
                {
                    "experiment": name,
                    "control": control_name,
                    "product": sku,
                    "delta_ae": float(sub["delta_ae"].sum()),
                    "actual_volume": float(np.abs(sub["actual"]).sum()),
                    "n": len(sub),
                    "wmape_control": wmape(sub["actual"], sub["pred_f0"]),
                    "wmape_candidate": wmape(sub["actual"], sub["pred_cand"]),
                }
            )

        fd = res.fold_diagnostics.copy()
        fd["experiment"] = name
        fd["anchor"] = exp.anchor
        fd["train_universe"] = exp.train_universe
        fd.to_csv(out_dir / f"train_diagnostics_{name}.csv", index=False)

    age_lookup = audit["enriched_panel"][["product", "origin", SCORED_FEATURE]].copy()
    age_lookup = age_lookup.rename(columns={"origin": "test_origin"})
    age_lookup["product"] = age_lookup["product"].astype(str)
    age_lookup["test_origin"] = age_lookup["test_origin"].astype(int)
    age_lookup = age_lookup.drop_duplicates(["product", "test_origin"])
    age_groups = _age_group_table(results, age_lookup, audit["age_edges"])

    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_product = pd.DataFrame(product_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    conc_df = pd.DataFrame(conc_rows)
    watch_df = pd.DataFrame(watch_rows)
    gates = pd.DataFrame(gate_rows)

    verdict = classify_f3a(overall, conc_df, age_groups)

    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_product.to_csv(out_dir / "by_product.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    watch_df.to_csv(out_dir / "watchlist.csv", index=False)
    age_groups.to_csv(out_dir / "age_groups.csv", index=False)
    gates.to_csv(out_dir / "reproduction_gates.csv", index=False)
    pd.DataFrame([{"verdict": verdict}]).to_csv(out_dir / "verdict.csv", index=False)

    session.finish()

    return {
        "overall": overall,
        "by_origin": by_origin,
        "by_product": by_product,
        "by_horizon": by_horizon,
        "error_concentration": conc_df,
        "watchlist": watch_df,
        "age_groups": age_groups,
        "gates": gates,
        "lifecycle_audit": audit["product_audit"],
        "lifecycle_coverage": audit["coverage"],
        "lifecycle_by_origin": audit["by_origin"],
        "first_nonzero_audit": audit["first_nonzero_audit"],
        "canonical_f0": canon,
        "results": results,
        "verdict": verdict,
        "out_dir": out_dir,
    }
