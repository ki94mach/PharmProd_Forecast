"""Evaluate F3E (peer demand) on frozen matched PRIMARY rows.

Step 3 of the F3E research pipeline.
Reads ONLY frozen artifacts from src/data/results/f3e/source/.
Does NOT query SQL.
Does NOT tune any feature definitions, normalizations, or XGBoost params.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.f3e.config import (
    ALL_EXPERIMENTS,
    CURRENT_ENV_F0_WMAPE,
    F3E_A_FEATURES,
    F3E_B_FEATURES,
    FILLNA_EXTRA,
    NEVER_FILLNA,
    NORMALIZED_MONTHLY_SALES_PARQUET,
    PRODUCT_PEER_PROFILE_PARQUET,
    F3EExperiment,
    E0_TS,
    E1_TS_GENERIC,
    E2_TS_GENERIC_CROSS_PATIENT,
    E0_HUMAN,
    E1_HUMAN_GENERIC,
    E2_HUMAN_GENERIC_CROSS_PATIENT,
    PAIRS,
    f3e_output_dir,
    f3e_source_dir,
    f3e_feature_audit_dir,
)
from pkg.research.f3e.features import build_f3e_features
from pkg.research.harness.dataset import enrich_dataset, resolve_origin_col
from pkg.research.harness.gates import assert_wmape_gate, wmape_gate_row
from pkg.research.harness.metrics import (
    ROW_KEYS,
    assert_same_eval_rows,
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

def enrich_peer_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    """Attach PIT F3E peer-demand features from frozen Step-1 parquets. No SQL."""
    panel = pd.read_parquet(f3e_source_dir() / NORMALIZED_MONTHLY_SALES_PARQUET)
    profile = pd.read_parquet(f3e_source_dir() / PRODUCT_PEER_PROFILE_PARQUET)

    def _enrich(rows: pd.DataFrame) -> pd.DataFrame:
        return build_f3e_features(panel, profile, rows)

    return enrich_dataset(ds, _enrich)


# ---------------------------------------------------------------------------
# ExperimentSpec builder
# ---------------------------------------------------------------------------

def _spec_for(exp: F3EExperiment) -> ExperimentSpec:
    return ExperimentSpec(
        name=exp.name,
        anchor=exp.anchor,
        features=exp.features(),
        train_universe=exp.train_universe,
        control=exp.control,
        use_frozen_adapter=exp.use_frozen_adapter,
        enrich="peer" if exp.peer_features else None,
    )


# ---------------------------------------------------------------------------
# Diagnostic: generic peer count groups
# ---------------------------------------------------------------------------

def _generic_peer_count_table(
    results: dict[str, BacktestResult],
    peer_group_df: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE grouped by number of same-generic peers (0, 1, 2+). Diagnostic only."""
    lookup = (
        peer_group_df[["product", "n_generic_peers"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)

    def _bucket(n) -> str:
        if pd.isna(n):
            return "unknown"
        n = int(n)
        if n == 0:
            return "0_no_peers"
        if n == 1:
            return "1_peer"
        return "2plus_peers"

    lookup["peer_bucket"] = lookup["n_generic_peers"].map(_bucket)

    base_key = "E0_TS" if "E0_TS" in results else next(iter(results))
    pred_cols: dict[str, pd.DataFrame] = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "peer_bucket"]], on="product", how="left")
    base["peer_bucket"] = base["peer_bucket"].fillna("unknown")

    rows = []
    for bucket in ("0_no_peers", "1_peer", "2plus_peers", "unknown"):
        g = base.loc[base["peer_bucket"] == bucket]
        if g.empty:
            continue
        row: dict = {
            "generic_peer_count_group": bucket,
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
# Diagnostic: cross-generic patient peer count groups (tertiles)
# ---------------------------------------------------------------------------

def _cross_generic_peer_count_table(
    results: dict[str, BacktestResult],
    peer_group_df: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE grouped by n_cross_generic_patient_convertible_peers tertiles. Diagnostic only."""
    lookup = (
        peer_group_df[["product", "n_cross_generic_patient_convertible_peers"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)
    vals = pd.to_numeric(lookup["n_cross_generic_patient_convertible_peers"], errors="coerce")
    finite = vals.dropna()
    if len(finite) >= 3:
        t1, t2 = float(np.quantile(finite, 1 / 3)), float(np.quantile(finite, 2 / 3))
    else:
        t1, t2 = 0.0, 1.0

    def _bucket(n) -> str:
        if pd.isna(n):
            return "unknown"
        n = int(n)
        if n == 0:
            return "none"
        if n <= t1:
            return "low"
        if n <= t2:
            return "medium"
        return "high"

    lookup["cross_peer_bucket"] = vals.map(_bucket)

    base_key = "E1_TS_GENERIC" if "E1_TS_GENERIC" in results else next(iter(results))
    pred_cols: dict[str, pd.DataFrame] = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "cross_peer_bucket"]], on="product", how="left")
    base["cross_peer_bucket"] = base["cross_peer_bucket"].fillna("unknown")

    rows = []
    for bucket in ("none", "low", "medium", "high", "unknown"):
        g = base.loc[base["cross_peer_bucket"] == bucket]
        if g.empty:
            continue
        row: dict = {
            "cross_generic_peer_count_group": bucket,
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
# Diagnostic: PatientConsumeType
# ---------------------------------------------------------------------------

def _consume_type_table(
    results: dict[str, BacktestResult],
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE by PatientConsumeType (Continuous / SinglePeriod). Diagnostic only."""
    lookup = (
        profile_df[["product", "PatientConsumeType"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)
    lookup["consume_type"] = lookup["PatientConsumeType"].where(
        lookup["PatientConsumeType"].isin({"Continuous", "SinglePeriod"}), other="missing"
    )

    base_key = "E0_TS" if "E0_TS" in results else next(iter(results))
    pred_cols: dict[str, pd.DataFrame] = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "consume_type"]], on="product", how="left")
    base["consume_type"] = base["consume_type"].fillna("missing")

    rows = []
    for label in ("Continuous", "SinglePeriod", "missing"):
        g = base.loc[base["consume_type"] == label]
        if g.empty:
            continue
        row: dict = {
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
# Diagnostic: Field × PatientConsumeType groups
# ---------------------------------------------------------------------------

def _field_consume_group_table(
    results: dict[str, BacktestResult],
    profile_df: pd.DataFrame,
    min_n: int = 10,
) -> pd.DataFrame:
    """WMAPE by Field×PatientConsumeType segment (rows with n >= min_n only). Diagnostic only."""
    lookup = (
        profile_df[["product", "Field", "PatientConsumeType"]]
        .drop_duplicates("product")
        .copy()
    )
    lookup["product"] = lookup["product"].astype(str)
    lookup["field_consume_group"] = (
        lookup["Field"].fillna("unknown").astype(str)
        + " × "
        + lookup["PatientConsumeType"].fillna("unknown").astype(str)
    )

    base_key = "E1_TS_GENERIC" if "E1_TS_GENERIC" in results else next(iter(results))
    pred_cols: dict[str, pd.DataFrame] = {}
    for name, res in results.items():
        p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
        p["product"] = p["product"].astype(str)
        pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

    base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
    for name, p in pred_cols.items():
        base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
    base["product"] = base["product"].astype(str)
    base = base.merge(lookup[["product", "field_consume_group"]], on="product", how="left")
    base["field_consume_group"] = base["field_consume_group"].fillna("unknown")

    rows = []
    for grp, g in base.groupby("field_consume_group"):
        if len(g) < min_n:
            continue
        row: dict = {
            "field_consume_group": grp,
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
# Diagnostic: peer demand magnitude quartiles
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


def _peer_demand_quartile_table(
    results: dict[str, BacktestResult],
    enriched_panel: pd.DataFrame,
) -> pd.DataFrame:
    """WMAPE by pre-model generic/cross-generic demand quartile. Diagnostic only."""
    rows = []

    for feat_col, label in (
        ("generic_peer_dqtyunit_3m_mean", "generic_3m_mean"),
        ("cross_generic_field_consume_patients_3m_mean", "cross_generic_3m_mean"),
    ):
        if feat_col not in enriched_panel.columns:
            continue

        lookup = (
            enriched_panel[["product", feat_col]]
            .drop_duplicates("product")
            .copy()
        )
        lookup["product"] = lookup["product"].astype(str)
        feat_vals = pd.to_numeric(lookup[feat_col], errors="coerce").to_numpy(float)
        edges = _quartile_edges(feat_vals)
        lookup["quartile"] = [_assign_quartile(v, edges) for v in feat_vals]

        base_key = "E0_TS" if "E0_TS" in results else next(iter(results))
        pred_cols: dict[str, pd.DataFrame] = {}
        for name, res in results.items():
            p = res.predictions[list(ROW_KEYS) + ["actual", "prediction"]].copy()
            p["product"] = p["product"].astype(str)
            pred_cols[name] = p.rename(columns={"prediction": f"pred_{name}"})

        base = pred_cols[base_key][list(ROW_KEYS) + ["actual"]].copy()
        for name, p in pred_cols.items():
            base = base.merge(p[list(ROW_KEYS) + [f"pred_{name}"]], on=list(ROW_KEYS), how="left")
        base["product"] = base["product"].astype(str)
        base = base.merge(lookup[["product", "quartile"]], on="product", how="left")
        base["quartile"] = base["quartile"].fillna("missing")

        for grp in ("Q1", "Q2", "Q3", "Q4", "available", "missing"):
            g = base.loc[base["quartile"] == grp]
            if g.empty:
                continue
            row: dict = {
                "peer_demand_feature": label,
                "quartile": grp,
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
    result = {}
    for i, name in enumerate(feature_names):
        gain = gain_map.get(name, gain_map.get(f"f{i}", 0.0))
        weight = weight_map.get(name, weight_map.get(f"f{i}", 0))
        result[name] = {"gain": float(gain), "weight": int(weight)}
    return result


def _diagnostic_importance(
    ds: BenchmarkDataset,
    enriched_ds: BenchmarkDataset,
    experiments: list[F3EExperiment],
) -> pd.DataFrame:
    """Train one fold per origin/experiment to extract XGBoost gain for F3E features."""
    from pkg.benchmark.dataset import prep_lags
    from pkg.benchmark.evaluate import _fold_eligible_primary, _train_slice
    from pkg.benchmark.models import fit_xgb

    f3e_feature_set = set(F3E_B_FEATURES)

    ts_u = prep_lags(enriched_ds.ts_universe)
    bud_u = prep_lags(enriched_ds.budget_universe)
    matched_u = prep_lags(enriched_ds.matched_universe)

    extra = frozenset(FILLNA_EXTRA)
    skip = frozenset(NEVER_FILLNA)

    rows = []
    for exp in experiments:
        if not exp.peer_features:
            continue
        feats = list(exp.features())
        forecast_col = "ts_forecast" if exp.anchor == "ts" else "budget_forecast"
        train_key = exp.train_universe

        for O in sorted(int(o) for o in PRIMARY_ORIGINS):
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
            f3e_gain = sum(
                v["gain"] for k, v in imp.items() if k in f3e_feature_set
            )

            for feat_name, vals in imp.items():
                is_f3e = feat_name in f3e_feature_set
                rows.append(
                    {
                        "experiment": exp.name,
                        "anchor": exp.anchor,
                        "origin": int(O),
                        "feature": feat_name,
                        "is_f3e_feature": is_f3e,
                        "gain": float(vals["gain"]),
                        "weight": int(vals["weight"]),
                        "f3e_gain_share": (
                            f3e_gain / total_gain if total_gain > 0 else 0.0
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


def classify_f3e(overall: pd.DataFrame, conc: pd.DataFrame) -> tuple[str, str]:
    """Return (primary_verdict, e2_vs_e1_verdict).

    Primary verdict (E1 level):
      A — generic demand helps both anchors
      B — generic demand helps TS only
      C — generic demand helps Human only
      D — weak / regime-specific signal
      E — F3E peer-demand representation fails

    e2_vs_e1_verdict:
      + — cross-generic adds incremental value over same-generic for both
      ts_only — adds for TS only
      human_only — adds for Human only
      none — cross-generic does not add beyond same-generic
    """
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

    def helps(cand_name: str, ctrl_name: str) -> bool:
        r = row(cand_name)
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
            concentration_flags=flags(cand_name),
        )

    # E1 vs E0
    e1_ts_helps = helps("E1_TS_GENERIC", "E0_TS")
    e1_human_helps = helps("E1_HUMAN_GENERIC", "E0_HUMAN")

    if e1_ts_helps and e1_human_helps:
        primary = "A"
    elif e1_ts_helps and not e1_human_helps:
        primary = "B"
    elif e1_human_helps and not e1_ts_helps:
        primary = "C"
    else:
        # D vs E: any signal?
        primary = "D" if _has_segmentation_signal(overall, "E1") else "E"

    # E2 vs E1 (incremental cross-generic value)
    e2_ts_adds = helps("E2_TS_GENERIC_CROSS_PATIENT", "E1_TS_GENERIC")
    e2_human_adds = helps("E2_HUMAN_GENERIC_CROSS_PATIENT", "E1_HUMAN_GENERIC")

    if e2_ts_adds and e2_human_adds:
        e2_verdict = "cross_generic_adds_both"
    elif e2_ts_adds:
        e2_verdict = "cross_generic_adds_ts_only"
    elif e2_human_adds:
        e2_verdict = "cross_generic_adds_human_only"
    else:
        e2_verdict = "cross_generic_no_incremental_value"

    return primary, e2_verdict


def _has_segmentation_signal(overall: pd.DataFrame, prefix: str) -> bool:
    """Rough check: any E1 or E2 experiment improves at least one anchor."""
    for name in overall["experiment"].tolist():
        if not name.startswith(prefix):
            continue
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

def evaluate_f3e(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f3e_output_dir()
    session = FamilySession(
        "f3e",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        enrichers={"peer": enrich_peer_dataset},
        model_name_prefix="xgb",
    )

    print("=== E0_TS (F0 baseline) ===")
    e0_ts = session.f0("ts")
    print("=== E0_HUMAN (F0 baseline) ===")
    e0_human = session.f0("human")
    canon = session.canon

    # F0 reproduction gates
    gate_rows = []
    ts_w = float(e0_ts.overall["wmape"].iloc[0])
    h_w = float(e0_human.overall["wmape"].iloc[0])
    gate_rows.append(
        wmape_gate_row(
            "E0_TS vs current-env TS F0",
            ts_w,
            CURRENT_ENV_F0_WMAPE["ts"],
            int(e0_ts.overall["n"].iloc[0]),
            len(e0_ts.origins),
        )
    )
    gate_rows.append(
        wmape_gate_row(
            "E0_HUMAN vs current-env Human F0",
            h_w,
            CURRENT_ENV_F0_WMAPE["human"],
            int(e0_human.overall["n"].iloc[0]),
            len(e0_human.origins),
        )
    )
    if sorted(int(o) for o in e0_ts.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(
            f"E0_TS origins {e0_ts.origins} != PRIMARY {PRIMARY_ORIGINS}"
        )
    if sorted(int(o) for o in e0_human.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(
            f"E0_HUMAN origins {e0_human.origins} != PRIMARY {PRIMARY_ORIGINS}"
        )
    for grow in gate_rows:
        assert_wmape_gate(grow)
    pd.DataFrame(gate_rows).to_csv(out_dir / "reproduction_gates.csv", index=False)
    print(f"F0 gate passed: TS={ts_w:.4f}, Human={h_w:.4f}")

    # Run candidate experiments (all use peer enricher)
    print("=== E1_TS_GENERIC ===")
    e1_ts = session.run(_spec_for(E1_TS_GENERIC))
    print("=== E2_TS_GENERIC_CROSS_PATIENT ===")
    e2_ts = session.run(_spec_for(E2_TS_GENERIC_CROSS_PATIENT))
    print("=== E1_HUMAN_GENERIC ===")
    e1_human = session.run(_spec_for(E1_HUMAN_GENERIC))
    print("=== E2_HUMAN_GENERIC_CROSS_PATIENT ===")
    e2_human = session.run(_spec_for(E2_HUMAN_GENERIC_CROSS_PATIENT))

    results: dict[str, BacktestResult] = {
        "E0_TS": e0_ts,
        "E1_TS_GENERIC": e1_ts,
        "E2_TS_GENERIC_CROSS_PATIENT": e2_ts,
        "E0_HUMAN": e0_human,
        "E1_HUMAN_GENERIC": e1_human,
        "E2_HUMAN_GENERIC_CROSS_PATIENT": e2_human,
    }

    # Assert same eval rows across TS group and Human group
    assert_same_eval_rows(e0_ts, e1_ts)
    assert_same_eval_rows(e0_ts, e2_ts)
    assert_same_eval_rows(e0_human, e1_human)
    assert_same_eval_rows(e0_human, e2_human)

    # Also build an E0-level E2-vs-E0 comparison pair (for Q5 reporting)
    extra_pairs = (
        ("E2_TS_GENERIC_CROSS_PATIENT", "E0_TS"),
        ("E2_HUMAN_GENERIC_CROSS_PATIENT", "E0_HUMAN"),
    )

    pair_map = {cand: ctrl for cand, ctrl in PAIRS}
    # Add E2-vs-E0 for the "also report E2 vs E0" requirement
    pair_map_e2_vs_e0 = {cand: ctrl for cand, ctrl in extra_pairs}

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

        # Compute rel_wmape vs E0 for E2 rows (secondary comparison)
        e0_name = "E0_TS" if exp.anchor == "ts" else "E0_HUMAN"
        e0_wmape_val = float(results[e0_name].overall.iloc[0]["wmape"])
        rel_vs_e0 = (
            rel_wmape(e0_wmape_val, float(o["wmape"]))
            if name.startswith("E2")
            else np.nan
        )

        overall_rows.append(
            {
                "experiment": name,
                "anchor": exp.anchor,
                "control": control_name,
                "n_features": len(exp.features()),
                "peer_features": (
                    ", ".join(exp.peer_features) if exp.peer_features else "none"
                ),
                "train_universe": exp.train_universe,
                "wmape": float(o["wmape"]),
                "wmape_control": float(co["wmape"]),
                "rel_wmape_vs_control_pct": rel_wmape(
                    float(co["wmape"]), float(o["wmape"])
                ),
                "rel_wmape_vs_e0_pct": rel_vs_e0,
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

    primary_verdict, e2_verdict = classify_f3e(overall, conc_df)

    # Diagnostics — enrich matched universe for feature-specific tables
    print("=== Building peer-enriched dataset for diagnostics ===")
    enriched_ds = enrich_peer_dataset(session.ds)
    enriched_matched = enriched_ds.matched_universe

    # Load peer group audit for peer-count tables
    peer_group_path = f3e_feature_audit_dir() / "peer_group_audit.csv"
    if peer_group_path.exists():
        peer_group_df = pd.read_csv(peer_group_path)
    else:
        peer_group_df = pd.DataFrame(
            columns=["product", "n_generic_peers", "n_cross_generic_patient_convertible_peers"]
        )

    # Load product profile for consume-type and field tables
    profile_path = f3e_source_dir() / PRODUCT_PEER_PROFILE_PARQUET
    profile_df = pd.read_parquet(profile_path) if profile_path.exists() else pd.DataFrame()

    generic_peer_table = _generic_peer_count_table(results, peer_group_df)
    cross_peer_table = _cross_generic_peer_count_table(results, peer_group_df)
    consume_type_table = _consume_type_table(results, profile_df)
    field_consume_table = _field_consume_group_table(results, profile_df)
    quartile_table = _peer_demand_quartile_table(results, enriched_matched)

    # Feature importance (diagnostic, not for promotion)
    print("=== Diagnostic XGB gain (F3E features, not used for promotion) ===")
    candidate_exps = [
        ALL_EXPERIMENTS[n]
        for n in (
            "E1_TS_GENERIC",
            "E2_TS_GENERIC_CROSS_PATIENT",
            "E1_HUMAN_GENERIC",
            "E2_HUMAN_GENERIC_CROSS_PATIENT",
        )
    ]
    importance = _diagnostic_importance(session.ds, enriched_ds, candidate_exps)

    # Write all outputs
    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_product.to_csv(out_dir / "by_product.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon.csv", index=False)
    generic_peer_table.to_csv(out_dir / "by_generic_peer_count.csv", index=False)
    cross_peer_table.to_csv(out_dir / "by_cross_generic_peer_count.csv", index=False)
    consume_type_table.to_csv(out_dir / "by_patient_consume_type.csv", index=False)
    field_consume_table.to_csv(out_dir / "by_field_consume_group.csv", index=False)
    quartile_table.to_csv(out_dir / "by_peer_demand_quartile.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    watch_df.to_csv(out_dir / "high_volume_watchlist.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    gates.to_csv(out_dir / "reproduction_gates.csv", index=False)
    pd.DataFrame(
        [{"primary_verdict": primary_verdict, "e2_vs_e1_verdict": e2_verdict}]
    ).to_csv(out_dir / "verdict.csv", index=False)

    session.finish()

    return {
        "overall": overall,
        "by_origin": by_origin,
        "by_product": by_product,
        "by_horizon": by_horizon,
        "by_generic_peer_count": generic_peer_table,
        "by_cross_generic_peer_count": cross_peer_table,
        "by_patient_consume_type": consume_type_table,
        "by_field_consume_group": field_consume_table,
        "by_peer_demand_quartile": quartile_table,
        "error_concentration": conc_df,
        "high_volume_watchlist": watch_df,
        "feature_importance": importance,
        "gates": gates,
        "canonical_f0": canon,
        "results": results,
        "primary_verdict": primary_verdict,
        "e2_verdict": e2_verdict,
        "out_dir": out_dir,
    }
