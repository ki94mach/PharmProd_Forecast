"""Evaluate F3D (patient-consumption profile) on frozen matched PRIMARY rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.f3d.config import (
    ALL_EXPERIMENTS,
    CURRENT_ENV_F0_WMAPE,
    FILLNA_EXTRA,
    NEVER_FILLNA,
    PAIRS,
    F3DExperiment,
    D1_TS,
    D2_TS,
    D1_HUMAN,
    D2_HUMAN,
    f3d_output_dir,
)
from pkg.research.features.patient_consumption import (
    FEATURE_NAMES,
    add_patient_consumption_features,
    load_frozen_profile,
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


# ---------------------------------------------------------------------------
# Dataset enricher
# ---------------------------------------------------------------------------

def enrich_profile_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    """Attach PIT patient-consumption features from frozen parquet.  No SQL."""
    profile = load_frozen_profile()

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        return add_patient_consumption_features(panel, profile)

    return enrich_dataset(ds, _enrich)


# ---------------------------------------------------------------------------
# ExperimentSpec builder
# ---------------------------------------------------------------------------

def _spec_for(exp: F3DExperiment) -> ExperimentSpec:
    return ExperimentSpec(
        name=exp.name,
        anchor=exp.anchor,
        features=exp.features(),
        train_universe=exp.train_universe,
        control=exp.control,
        use_frozen_adapter=exp.use_frozen_adapter,
        enrich="profile" if exp.profile_features else None,
    )


# ---------------------------------------------------------------------------
# Diagnostic: performance by PatientConsumeType
# ---------------------------------------------------------------------------

def _consume_type_table(
    results: dict[str, BacktestResult],
    enriched_panel: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE by PatientConsumeType for all experiments (diagnostic only)."""
    lookup = (
        enriched_panel[["product", "is_continuous_consumption"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)

    # Build label
    def _label(v) -> str:
        if pd.isna(v):
            return "missing"
        return "Continuous" if float(v) == 1.0 else "SinglePeriod"

    lookup["consume_type_label"] = lookup["is_continuous_consumption"].map(_label)

    base_key = "D0_TS"
    if base_key not in results:
        base_key = next(iter(results))

    pred_cols = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "consume_type_label"]], on="product", how="left")

    rows = []
    for label in ("Continuous", "SinglePeriod", "missing"):
        g = base.loc[base["consume_type_label"] == label]
        if g.empty:
            continue
        row = {
            "consume_type": label,
            "n": int(len(g)),
            "actual_volume": float(np.abs(g["actual"]).sum()),
        }
        for name in results:
            pc = f"pred_{name}"
            if pc in g.columns:
                row[f"wmape_{name}"] = wmape(g["actual"], g[pc])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic: performance by pre-model annual consumption quartile
# ---------------------------------------------------------------------------

def _quartile_edges(vals: np.ndarray):
    finite = vals[np.isfinite(vals)]
    if len(finite) < 4:
        return None
    q1, q2, q3 = np.quantile(finite, [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def _assign_quartile(value: float, edges) -> str:
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


def _consumption_quartile_table(
    results: dict[str, BacktestResult],
    enriched_panel: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE by pre-model patient_annual_consumption quartile (diagnostic only)."""
    lookup = (
        enriched_panel[["product", "patient_annual_consumption"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)
    annual_vals = pd.to_numeric(lookup["patient_annual_consumption"], errors="coerce").to_numpy(float)
    edges = _quartile_edges(annual_vals)

    lookup["quartile"] = [_assign_quartile(v, edges) for v in annual_vals]

    base_key = "D0_TS"
    if base_key not in results:
        base_key = next(iter(results))

    pred_cols = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "quartile"]], on="product", how="left")

    rows = []
    for grp in ("Q1", "Q2", "Q3", "Q4", "available", "missing"):
        g = base.loc[base["quartile"] == grp]
        if g.empty:
            continue
        row = {
            "consumption_quartile": grp,
            "n": int(len(g)),
            "actual_volume": float(np.abs(g["actual"]).sum()),
        }
        for name in results:
            pc = f"pred_{name}"
            if pc in g.columns:
                row[f"wmape_{name}"] = wmape(g["actual"], g[pc])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature importance (diagnostic only, no SHAP)
# ---------------------------------------------------------------------------

def _feature_importance_from_model(model, feature_names: list[str]) -> dict:
    try:
        booster = model.get_booster()
        gain_map = booster.get_score(importance_type="gain")
        weight_map = booster.get_score(importance_type="weight")
    except Exception:
        return {}
    rows = {}
    for i, name in enumerate(feature_names):
        gain = gain_map.get(name, gain_map.get(f"f{i}", 0.0))
        weight = weight_map.get(name, weight_map.get(f"f{i}", 0))
        rows[name] = {"gain": float(gain), "weight": int(weight)}
    return rows


def _diagnostic_importance(
    ds: BenchmarkDataset,
    enriched_ds: BenchmarkDataset,
    experiments: list[F3DExperiment],
) -> pd.DataFrame:
    """Train one fold per origin/experiment to extract XGB gain for F3D features."""
    from pkg.benchmark.dataset import prep_lags
    from pkg.benchmark.evaluate import _fold_eligible_primary, _train_slice
    from pkg.benchmark.models import fit_xgb

    profile_set = set(FEATURE_NAMES)

    ts_u = prep_lags(enriched_ds.ts_universe)
    bud_u = prep_lags(enriched_ds.budget_universe)
    matched_u = prep_lags(enriched_ds.matched_universe)

    extra = frozenset(FILLNA_EXTRA)
    skip = frozenset(NEVER_FILLNA)

    rows = []
    for exp in experiments:
        if not exp.profile_features:
            continue
        feats = list(exp.features())
        forecast_col = "ts_forecast" if exp.anchor == "ts" else "budget_forecast"
        train_key = exp.train_universe

        for O in sorted(int(o) for o in PRIMARY_ORIGINS):
            # _fold_eligible_primary always checks budget_origin, so always
            # pass the budget universe slice regardless of anchor.
            train_check = bud_u.loc[bud_u["target_date"].astype(int) < O]

            if not _fold_eligible_primary(train_check):
                continue

            train = _train_slice(
                lambda _t, _te: None,
                train_key, O, ts_u, bud_u, matched_u,
            )
            if train is None or train.empty:
                continue

            tr = train.copy()
            tr["residual"] = (
                tr["sales"].astype(float) - tr[forecast_col].astype(float)
            )
            missing = [c for c in feats if c not in tr.columns]
            if missing:
                continue

            for c in feats:
                if c in skip:
                    continue
                if c.startswith("sales_") or c.startswith("human_") or c in extra:
                    tr[c] = tr[c].fillna(0)

            model = fit_xgb(feats, tr)
            imp = _feature_importance_from_model(model, feats)
            total_gain = sum(v["gain"] for v in imp.values())
            f3d_gain = sum(
                v["gain"] for k, v in imp.items() if k in profile_set
            )

            for feat_name, vals in imp.items():
                is_f3d = feat_name in profile_set
                rows.append(
                    {
                        "experiment": exp.name,
                        "anchor": exp.anchor,
                        "origin": int(O),
                        "feature": feat_name,
                        "is_f3d_feature": is_f3d,
                        "gain": float(vals["gain"]),
                        "weight": int(vals["weight"]),
                        "f3d_gain_share": (
                            f3d_gain / total_gain if total_gain > 0 else 0.0
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verdict classification
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
    bias_ok = (
        abs(bias_cand) <= abs(bias_control) * 1.25
        or abs(bias_cand) <= abs(bias_control) + 200.0
    )
    return pooled and origins_ok and product_ok and bias_ok and not bool(
        concentration_flags
    )


def classify_f3d(overall: pd.DataFrame, conc: pd.DataFrame) -> str:
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

    def helps(name: str, ctrl_name: str) -> bool:
        r = row(name)
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
            concentration_flags=flags(name),
        )

    ts_type = helps("D1_TS_TYPE", "D0_TS")
    human_type = helps("D1_HUMAN_TYPE", "D0_HUMAN")

    if ts_type and human_type:
        return "A"
    if ts_type and not human_type:
        return "B"
    if human_type and not ts_type:
        return "C"

    # D — check if there is regime heterogeneity (type/quartile varies materially)
    # We use a simple heuristic: if the type table shows >1pp WMAPE spread
    return "D" if _has_segmentation_signal(overall) else "E"


def _has_segmentation_signal(overall: pd.DataFrame) -> bool:
    """Rough check: D1 improves at least one anchor though not majority-origins."""
    for name in ("D1_TS_TYPE", "D1_HUMAN_TYPE"):
        sub = overall.loc[overall["experiment"] == name]
        if sub.empty:
            continue
        rel = float(sub.iloc[0]["rel_wmape_vs_control_pct"])
        if rel > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Main evaluate
# ---------------------------------------------------------------------------

def evaluate_f3d(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f3d_output_dir()
    session = FamilySession(
        "f3d",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        enrichers={"profile": enrich_profile_dataset},
        model_name_prefix="xgb",
    )
    d0_ts = session.f0("ts")
    d0_human = session.f0("human")
    canon = session.canon

    # F0 reproduction gates
    gate_rows = []
    ts_w = float(d0_ts.overall["wmape"].iloc[0])
    h_w = float(d0_human.overall["wmape"].iloc[0])
    gate_rows.append(
        wmape_gate_row(
            "D0_TS vs current-env TS F0",
            ts_w,
            CURRENT_ENV_F0_WMAPE["ts"],
            int(d0_ts.overall["n"].iloc[0]),
            len(d0_ts.origins),
        )
    )
    gate_rows.append(
        wmape_gate_row(
            "D0_HUMAN vs current-env Human F0",
            h_w,
            CURRENT_ENV_F0_WMAPE["human"],
            int(d0_human.overall["n"].iloc[0]),
            len(d0_human.origins),
        )
    )
    if sorted(int(o) for o in d0_ts.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(
            f"D0_TS origins {d0_ts.origins} != PRIMARY {PRIMARY_ORIGINS}"
        )
    if sorted(int(o) for o in d0_human.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(
            f"D0_HUMAN origins {d0_human.origins} != PRIMARY {PRIMARY_ORIGINS}"
        )
    for row in gate_rows:
        assert_wmape_gate(row)
    pd.DataFrame(gate_rows).to_csv(out_dir / "reproduction_gates.csv", index=False)

    # Run candidate experiments
    print("=== D1_TS_TYPE ===")
    d1_ts = session.run(_spec_for(D1_TS))
    print("=== D2_TS_PROFILE ===")
    d2_ts = session.run(_spec_for(D2_TS))
    print("=== D1_HUMAN_TYPE ===")
    d1_human = session.run(_spec_for(D1_HUMAN))
    print("=== D2_HUMAN_PROFILE ===")
    d2_human = session.run(_spec_for(D2_HUMAN))

    results: dict[str, BacktestResult] = {
        "D0_TS": d0_ts,
        "D1_TS_TYPE": d1_ts,
        "D2_TS_PROFILE": d2_ts,
        "D0_HUMAN": d0_human,
        "D1_HUMAN_TYPE": d1_human,
        "D2_HUMAN_PROFILE": d2_human,
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

        # Compute rel_wmape vs D0 for D2 rows (for "also D2 vs D0" question)
        d0_name = "D0_TS" if exp.anchor == "ts" else "D0_HUMAN"
        d0_wmape = float(results[d0_name].overall.iloc[0]["wmape"])
        rel_vs_d0 = rel_wmape(d0_wmape, float(o["wmape"])) if name.startswith("D2") else np.nan

        overall_rows.append(
            {
                "experiment": name,
                "anchor": exp.anchor,
                "control": control_name,
                "n_features": len(exp.features()),
                "profile_features": (
                    ", ".join(exp.profile_features) if exp.profile_features else "none"
                ),
                "train_universe": exp.train_universe,
                "wmape": float(o["wmape"]),
                "wmape_control": float(co["wmape"]),
                "rel_wmape_vs_control_pct": rel_wmape(
                    float(co["wmape"]), float(o["wmape"])
                ),
                "rel_wmape_vs_d0_pct": rel_vs_d0,
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
                "top5_improvement_share": conc.get("top5_improvement_share", np.nan),
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

    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_product = pd.DataFrame(product_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    conc_df = pd.DataFrame(conc_rows)
    watch_df = pd.DataFrame(watch_rows)
    gates = pd.DataFrame(gate_rows)

    verdict = classify_f3d(overall, conc_df)

    # Diagnostics: enrich the matched universe for profile-specific tables
    enriched_ds = enrich_profile_dataset(session.ds)
    enriched_matched = enriched_ds.matched_universe

    consume_type_table = _consume_type_table(results, enriched_matched)
    quartile_table = _consumption_quartile_table(results, enriched_matched)

    # Feature importance (gain for F3D features only)
    print("=== diagnostic XGB gain (F3D features, not used for promotion) ===")
    candidate_exps = [
        ALL_EXPERIMENTS[n]
        for n in ("D1_TS_TYPE", "D2_TS_PROFILE", "D1_HUMAN_TYPE", "D2_HUMAN_PROFILE")
    ]
    importance = _diagnostic_importance(session.ds, enriched_ds, candidate_exps)

    # Write outputs
    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_product.to_csv(out_dir / "by_product.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon.csv", index=False)
    consume_type_table.to_csv(out_dir / "by_patient_consume_type.csv", index=False)
    quartile_table.to_csv(out_dir / "by_consumption_quartile.csv", index=False)
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
        "by_patient_consume_type": consume_type_table,
        "by_consumption_quartile": quartile_table,
        "error_concentration": conc_df,
        "high_volume_watchlist": watch_df,
        "feature_importance": importance,
        "gates": gates,
        "canonical_f0": canon,
        "results": results,
        "verdict": verdict,
        "out_dir": out_dir,
    }
