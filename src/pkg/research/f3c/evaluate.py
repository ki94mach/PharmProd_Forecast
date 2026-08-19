"""Evaluate F3C (point-in-time month-end inventory) on frozen matched PRIMARY rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.f3c.config import (
    ALL_EXPERIMENTS,
    CURRENT_ENV_F0_WMAPE,
    FILLNA_EXTRA,
    INVENTORY_FEATURE_NAMES,
    NEVER_FILLNA,
    PAIRS,
    F3CExperiment,
    I1_TS,
    I2_TS,
    I1_HUMAN,
    I2_HUMAN,
    f3c_output_dir,
)
from pkg.research.features.inventory import (
    FEATURE_NAMES,
    add_inventory_features,
    load_frozen_distributor_inventory,
    load_frozen_factory_inventory,
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


def enrich_inventory_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    """Attach PIT inventory features from frozen parquets. No SQL."""
    dist_hist = load_frozen_distributor_inventory()
    fact_hist = load_frozen_factory_inventory()

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        return add_inventory_features(
            panel, dist_hist, fact_hist,
            origin_col=resolve_origin_col(panel),
        )

    return enrich_dataset(ds, _enrich)


def _spec_for(exp: F3CExperiment) -> ExperimentSpec:
    return ExperimentSpec(
        name=exp.name,
        anchor=exp.anchor,
        features=exp.features(),
        train_universe=exp.train_universe,
        control=exp.control,
        use_frozen_adapter=exp.use_frozen_adapter,
        enrich="inventory" if exp.inventory_features else None,
    )


# ---------------------------------------------------------------------------
# Inventory regime analysis (pre-model quartiles, descriptive only)
# ---------------------------------------------------------------------------

def _quartile_edges(vals: np.ndarray):
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        return None
    q1, q2, q3 = np.quantile(finite, [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def _assign_group(value: float, edges) -> str:
    if not np.isfinite(value):
        return "missing"
    if edges is None:
        return "available"
    q1, q2, q3 = edges
    if value <= q1:
        return "Q1"
    if value <= q2:
        return "Q2"
    if value <= q3:
        return "Q3"
    return "Q4"


def _assign_zero_group(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value == 0:
        return "=0"
    return ">0"


def inventory_regime_table(
    results: dict[str, BacktestResult],
    enriched_panel: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE by pre-model distributor/factory quartiles and zero/nonzero/missing."""
    lookup = enriched_panel[["product", "origin", "distributor_inventory_qty", "factory_inventory_qty"]].copy()
    lookup["product"] = lookup["product"].astype(str)
    lookup["test_origin"] = lookup["origin"].astype(int)
    lookup = lookup.drop(columns=["origin"]).drop_duplicates(["product", "test_origin"])

    dist_edges = _quartile_edges(lookup["distributor_inventory_qty"].to_numpy(dtype=float))
    fact_edges = _quartile_edges(lookup["factory_inventory_qty"].to_numpy(dtype=float))

    # merge predictions
    preds = {}
    for name in results:
        p = results[name].predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p = p.rename(columns={"prediction": f"pred_{name}"})
        preds[name] = p

    base = preds["I0_TS"][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in preds.items():
        if name == "I0_TS":
            base[f"pred_{name}"] = p[f"pred_{name}"]
        else:
            base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base["test_origin"] = base["test_origin"].astype(int)
    m = base.merge(lookup, on=["product", "test_origin"], how="left")

    dist_q = pd.to_numeric(m["distributor_inventory_qty"], errors="coerce").to_numpy(dtype=float)
    fact_q = pd.to_numeric(m["factory_inventory_qty"], errors="coerce").to_numpy(dtype=float)
    m["dist_quartile"] = [_assign_group(v, dist_edges) for v in dist_q]
    m["dist_zero_group"] = [_assign_zero_group(v) for v in dist_q]
    m["fact_zero_group"] = [_assign_zero_group(v) for v in fact_q]

    rows = []
    slices = [
        ("distributor_quartile", "dist_quartile", ("Q1", "Q2", "Q3", "Q4", "available", "missing")),
        ("distributor_zero", "dist_zero_group", ("=0", ">0", "missing")),
        ("factory_zero", "fact_zero_group", ("=0", ">0", "missing")),
    ]

    pred_cols = [c for c in m.columns if c.startswith("pred_")]
    for slice_name, col, groups in slices:
        for grp in groups:
            g = m.loc[m[col] == grp]
            if g.empty:
                continue
            row = {"slice": slice_name, "group": grp, "n": int(len(g)),
                   "actual_volume": float(np.abs(g["actual"]).sum())}
            for pc in pred_cols:
                exp_name = pc.replace("pred_", "")
                row[f"wmape_{exp_name}"] = wmape(g["actual"], g[pc])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature importance (diagnostic only)
# ---------------------------------------------------------------------------

def _feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    weight_map = booster.get_score(importance_type="weight")
    importances = getattr(model, "feature_importances_", None)
    rows = []
    for i, name in enumerate(feature_names):
        gain = gain_map.get(name, gain_map.get(f"f{i}", 0.0))
        weight = weight_map.get(name, weight_map.get(f"f{i}", 0))
        if importances is not None and i < len(importances):
            gain = float(importances[i]) if float(gain) == 0.0 else float(gain)
        rows.append({"feature": name, "gain": float(gain), "weight": int(weight)})
    return pd.DataFrame(rows)


def diagnostic_feature_importance(enriched: BenchmarkDataset) -> pd.DataFrame:
    from pkg.benchmark.dataset import prep_lags
    from pkg.benchmark.evaluate import _fold_eligible_primary, _train_slice
    from pkg.benchmark.models import fit_xgb

    ts_u = prep_lags(enriched.ts_universe)
    bud_u = prep_lags(enriched.budget_universe)
    matched_u = prep_lags(enriched.matched_universe)
    extra = frozenset(FILLNA_EXTRA)
    skip = frozenset(NEVER_FILLNA)
    rows = []
    specs = [
        ("I1_TS_DISTRIBUTOR", "ts", "ts", list(I1_TS.features()), "ts_forecast"),
        ("I2_TS_DISTRIBUTOR_FACTORY", "ts", "ts", list(I2_TS.features()), "ts_forecast"),
        ("I1_HUMAN_DISTRIBUTOR", "human", "budget", list(I1_HUMAN.features()), "budget_forecast"),
        ("I2_HUMAN_DISTRIBUTOR_FACTORY", "human", "budget", list(I2_HUMAN.features()), "budget_forecast"),
    ]
    inv_set = set(INVENTORY_FEATURE_NAMES)
    for exp_name, anchor, train_key, feats, forecast_col in specs:
        for O in sorted(int(o) for o in PRIMARY_ORIGINS):
            train_bud = bud_u.loc[bud_u["target_date"].astype(int) < O]
            if not _fold_eligible_primary(train_bud):
                continue
            train = _train_slice(
                lambda _t, _te: None,
                train_key, O, ts_u, bud_u, matched_u,
            )
            if train is None or train.empty:
                continue
            tr = train.copy()
            tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
            missing = [c for c in feats if c not in tr.columns]
            if missing:
                raise KeyError(f"importance missing columns {missing}")
            for c in feats:
                if c in skip:
                    continue
                if c.startswith("sales_") or c.startswith("human_") or c in extra:
                    tr[c] = tr[c].fillna(0)
            model = fit_xgb(feats, tr)
            imp = _feature_importance(model, feats)
            total_gain = float(imp["gain"].sum())
            inv_gain = float(imp.loc[imp["feature"].isin(inv_set), "gain"].sum())
            for _, row in imp.iterrows():
                is_inv = row["feature"] in inv_set
                rows.append({
                    "experiment": exp_name,
                    "anchor": anchor,
                    "origin": int(O),
                    "feature": row["feature"],
                    "is_inventory_feature": is_inv,
                    "gain": float(row["gain"]),
                    "weight": int(row["weight"]),
                    "inventory_gain_share": (inv_gain / total_gain) if total_gain > 0 else 0.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

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
    bias_ok = abs(bias_cand) <= abs(bias_control) * 1.25 or abs(bias_cand) <= abs(bias_control) + 200.0
    return pooled and origins_ok and product_ok and bias_ok and not bool(concentration_flags)


def classify_f3c(overall: pd.DataFrame, conc: pd.DataFrame) -> str:
    def row(name):
        sub = overall.loc[overall["experiment"] == name]
        return sub.iloc[0] if len(sub) else None

    def flags(name):
        if conc is None or conc.empty or "flags" not in conc.columns:
            return []
        sub = conc.loc[conc["experiment"] == name]
        if sub.empty:
            return []
        raw = sub["flags"].iloc[0]
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)) or raw == "":
            return []
        return [x for x in str(raw).split(";") if x]

    i1t, i1h = row("I1_TS_DISTRIBUTOR"), row("I1_HUMAN_DISTRIBUTOR")
    i2t, i2h = row("I2_TS_DISTRIBUTOR_FACTORY"), row("I2_HUMAN_DISTRIBUTOR_FACTORY")
    if i1t is None or i1h is None:
        return "E"

    def helps(r, ctrl_name):
        c = row(ctrl_name)
        if r is None or c is None:
            return False
        return _anchor_helps(
            wmape_control=float(c["wmape"]),
            wmape_cand=float(r["wmape"]),
            origins_improved=int(r["origins_improved"]),
            origins_total=int(r["origins_total"]),
            product_win_rate=float(r["product_win_rate"]),
            median_product_improvement_pct=float(r["median_product_improvement_pct"]),
            bias_control=float(c["bias"]),
            bias_cand=float(r["bias"]),
            concentration_flags=flags(r["experiment"]),
        )

    ts_dist = helps(i1t, "I0_TS")
    human_dist = helps(i1h, "I0_HUMAN")

    if ts_dist and human_dist:
        verdict = "A"
    elif ts_dist and not human_dist:
        verdict = "C"
    elif human_dist and not ts_dist:
        verdict = "C"
    else:
        verdict = "E"

    # Check if factory adds incremental value
    if i2t is not None and i2h is not None:
        ts_fact = helps(i2t, "I0_TS")
        human_fact = helps(i2h, "I0_HUMAN")
        if (ts_fact or human_fact) and verdict in ("A", "C"):
            verdict = "B" if verdict == "A" else verdict

    return verdict


# ---------------------------------------------------------------------------
# Main evaluate
# ---------------------------------------------------------------------------

def evaluate_f3c(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f3c_output_dir()
    session = FamilySession(
        "f3c",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        enrichers={"inventory": enrich_inventory_dataset},
        model_name_prefix="xgb",
    )
    i0_ts = session.f0("ts")
    i0_human = session.f0("human")
    canon = session.canon

    # F0 reproduction gates
    gate_rows = []
    ts_w = float(i0_ts.overall["wmape"].iloc[0])
    h_w = float(i0_human.overall["wmape"].iloc[0])
    gate_rows.append(wmape_gate_row(
        "I0_TS vs current-env TS F0", ts_w,
        CURRENT_ENV_F0_WMAPE["ts"],
        int(i0_ts.overall["n"].iloc[0]), len(i0_ts.origins),
    ))
    gate_rows.append(wmape_gate_row(
        "I0_HUMAN vs current-env Human F0", h_w,
        CURRENT_ENV_F0_WMAPE["human"],
        int(i0_human.overall["n"].iloc[0]), len(i0_human.origins),
    ))
    if sorted(int(o) for o in i0_ts.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(f"I0_TS origins {i0_ts.origins} != PRIMARY {PRIMARY_ORIGINS}")
    if sorted(int(o) for o in i0_human.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(f"I0_HUMAN origins {i0_human.origins} != PRIMARY {PRIMARY_ORIGINS}")
    for row in gate_rows:
        assert_wmape_gate(row)
    pd.DataFrame(gate_rows).to_csv(out_dir / "reproduction_gates.csv", index=False)

    # Run candidate experiments
    print("=== I1_TS_DISTRIBUTOR ===")
    i1_ts = session.run(_spec_for(I1_TS))
    print("=== I2_TS_DISTRIBUTOR_FACTORY ===")
    i2_ts = session.run(_spec_for(I2_TS))
    print("=== I1_HUMAN_DISTRIBUTOR ===")
    i1_human = session.run(_spec_for(I1_HUMAN))
    print("=== I2_HUMAN_DISTRIBUTOR_FACTORY ===")
    i2_human = session.run(_spec_for(I2_HUMAN))

    results: dict[str, BacktestResult] = {
        "I0_TS": i0_ts,
        "I1_TS_DISTRIBUTOR": i1_ts,
        "I2_TS_DISTRIBUTOR_FACTORY": i2_ts,
        "I0_HUMAN": i0_human,
        "I1_HUMAN_DISTRIBUTOR": i1_human,
        "I2_HUMAN_DISTRIBUTOR_FACTORY": i2_human,
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

        overall_rows.append({
            "experiment": name,
            "anchor": exp.anchor,
            "control": control_name,
            "n_features": len(exp.features()),
            "inventory_features": ", ".join(exp.inventory_features) if exp.inventory_features else "none",
            "train_universe": exp.train_universe,
            "wmape": float(o["wmape"]),
            "wmape_control": float(co["wmape"]),
            "rel_wmape_vs_control_pct": rel_wmape(float(co["wmape"]), float(o["wmape"])),
            "rmse": float(o["rmse"]),
            "mae": float(o["mae"]),
            "bias": float(o["bias"]),
            "bias_control": float(co["bias"]),
            "n": int(o["n"]),
            **osu,
            "median_origin_improvement_pct": osu.get("median_origin_improvement", float("nan")),
            **psu,
            "top1_deterioration_share": conc["top1_deterioration_share"],
            "top5_deterioration_share": conc["top5_deterioration_share"],
            "top10_deterioration_share": conc["top10_deterioration_share"],
        })
        for _, r in odf.iterrows():
            origin_rows.append({"experiment": name, "control": control_name, **r.to_dict()})
        for _, r in pdf.iterrows():
            product_rows.append({"experiment": name, "control": control_name, **r.to_dict()})
        for _, r in hb.iterrows():
            horizon_rows.append({
                "experiment": name, "control": control_name,
                "horizon_bucket": r["horizon_bucket"],
                "n": r["n"],
                "wmape_control": r["wmape_f0"],
                "wmape_candidate": r["wmape_new"],
                "relative_improvement_pct": r["rel_wmape_vs_f0_pct"],
            })
        conc_rows.append({
            "experiment": name, "anchor": exp.anchor, "control": control_name,
            "net_delta_ae": conc["net_delta_ae"],
            "total_deterioration": conc["total_deterioration"],
            "total_improvement": conc["total_improvement"],
            "top1_deterioration_share": conc["top1_deterioration_share"],
            "top5_deterioration_share": conc["top5_deterioration_share"],
            "top10_deterioration_share": conc["top10_deterioration_share"],
            "top5_improvement_share": conc.get("top5_improvement_share", float("nan")),
            "flags": ";".join(conc["flags"]),
        })
        for sku in HIGH_VOLUME_WATCHLIST:
            sub = m.loc[m["product"] == sku]
            if sub.empty:
                continue
            watch_rows.append({
                "experiment": name, "control": control_name,
                "product": sku,
                "delta_ae": float(sub["delta_ae"].sum()),
                "actual_volume": float(np.abs(sub["actual"]).sum()),
                "n": len(sub),
                "wmape_control": wmape(sub["actual"], sub["pred_f0"]),
                "wmape_candidate": wmape(sub["actual"], sub["pred_cand"]),
            })

    # Inventory regime analysis
    enriched = enrich_inventory_dataset(session.ds)
    matched_enriched = enriched.matched_universe
    regimes = inventory_regime_table(results, matched_enriched)

    # Feature importance (diagnostic)
    print("=== diagnostic XGB gain (not used for promotion) ===")
    importance = diagnostic_feature_importance(enriched)

    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_product = pd.DataFrame(product_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    conc_df = pd.DataFrame(conc_rows)
    watch_df = pd.DataFrame(watch_rows)
    gates = pd.DataFrame(gate_rows)

    verdict = classify_f3c(overall, conc_df)

    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_product.to_csv(out_dir / "by_product.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon.csv", index=False)
    regimes.to_csv(out_dir / "inventory_regime_analysis.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    watch_df.to_csv(out_dir / "high_volume_watchlist.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    gates.to_csv(out_dir / "reproduction_gates.csv", index=False)
    pd.DataFrame([{"verdict": verdict}]).to_csv(out_dir / "verdict.csv", index=False)

    session.finish()

    return {
        "overall": overall,
        "by_origin": by_origin,
        "by_product": by_product,
        "by_horizon": by_horizon,
        "inventory_regime_analysis": regimes,
        "error_concentration": conc_df,
        "watchlist": watch_df,
        "feature_importance": importance,
        "gates": gates,
        "canonical_f0": canon,
        "results": results,
        "verdict": verdict,
        "out_dir": out_dir,
    }
