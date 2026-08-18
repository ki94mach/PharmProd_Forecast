"""Evaluate F3B (point-in-time consumer price) on frozen matched PRIMARY rows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, prep_lags
from pkg.benchmark.evaluate import BacktestResult, _fold_eligible_primary, _train_slice, wmape
from pkg.benchmark.models import fit_xgb
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.f3b.config import (
    ALL_EXPERIMENTS,
    CURRENT_ENV_F0_WMAPE,
    FILLNA_EXTRA,
    NEVER_FILLNA,
    PAIRS,
    F3BExperiment,
    P1_HUMAN,
    P1_TS,
    f3b_output_dir,
)
from pkg.research.features.price import (
    FEATURE_NAMES,
    add_price_features,
    load_frozen_price_history,
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
CHANGE_COL = "last_consumer_price_change_pct"
RECENCY_COL = "months_since_last_consumer_price_change"


def make_f3b_residual_model(anchor: str, feature_cols: Sequence[str]):
    """Residual XGB using frozen XGB_PARAMS / fit_xgb; never fillna price features."""
    name = "ts_xgb_f3b" if anchor == "ts" else "human_xgb_f3b"
    return make_residual_model(
        anchor,
        feature_cols,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        name=name,
    )


def enrich_price_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    """Attach PIT price features from frozen parquet only. No SQL / Excel."""
    hist = load_frozen_price_history()

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        return add_price_features(panel, hist, origin_col=resolve_origin_col(panel))

    return enrich_dataset(ds, _enrich)


def _spec_for(exp: F3BExperiment) -> ExperimentSpec:
    return ExperimentSpec(
        name=exp.name,
        anchor=exp.anchor,
        features=exp.features(),
        train_universe=exp.train_universe,
        control=exp.control,
        use_frozen_adapter=exp.use_frozen_adapter,
        enrich="price" if exp.include_price else None,
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


def quartile_edges(vals: np.ndarray) -> Optional[tuple[float, float, float]]:
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        return None
    q1, q2, q3 = np.quantile(finite, [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def assign_quartile(value: float, edges: Optional[tuple[float, float, float]]) -> str:
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


def assign_direction(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value > 0:
        return ">0"
    if value < 0:
        return "<0"
    return "=0"


def lock_regime_edges(panel: pd.DataFrame) -> dict[str, Optional[tuple[float, float, float]]]:
    """Pre-model quartile edges from PRIMARY attached features. Not tuned on WMAPE."""
    return {
        "change_magnitude": quartile_edges(
            pd.to_numeric(panel[CHANGE_COL], errors="coerce").to_numpy(dtype=float)
        ),
        "recency": quartile_edges(
            pd.to_numeric(panel[RECENCY_COL], errors="coerce").to_numpy(dtype=float)
        ),
    }


def _price_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    keep = ["product", "origin", CHANGE_COL, RECENCY_COL]
    out = panel[keep].copy()
    out["product"] = out["product"].astype(str)
    out["test_origin"] = out["origin"].astype(int)
    return out.drop(columns=["origin"]).drop_duplicates(["product", "test_origin"])


def _merged_preds(results: dict[str, BacktestResult]) -> pd.DataFrame:
    p0t = results["P0_TS"].predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_P0_TS"})
    p1t = results["P1_TS"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_P1_TS"}
    )
    p0h = results["P0_HUMAN"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_P0_HUMAN"}
    )
    p1h = results["P1_HUMAN"].predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "pred_P1_HUMAN"}
    )
    m = (
        p0t.merge(p1t, on=list(ROW_KEYS))
        .merge(p0h, on=list(ROW_KEYS))
        .merge(p1h, on=list(ROW_KEYS))
    )
    m["product"] = m["product"].astype(str)
    m["test_origin"] = m["test_origin"].astype(int)
    return m


def price_regime_table(
    results: dict[str, BacktestResult],
    lookup: pd.DataFrame,
    edges: dict[str, Optional[tuple[float, float, float]]],
) -> pd.DataFrame:
    m = _merged_preds(results)
    look = lookup.copy()
    look["product"] = look["product"].astype(str)
    look["test_origin"] = look["test_origin"].astype(int)
    m = m.merge(look, on=["product", "test_origin"], how="left")
    change = pd.to_numeric(m[CHANGE_COL], errors="coerce").to_numpy(dtype=float)
    recency = pd.to_numeric(m[RECENCY_COL], errors="coerce").to_numpy(dtype=float)
    m["change_group"] = [assign_quartile(v, edges["change_magnitude"]) for v in change]
    m["recency_group"] = [assign_quartile(v, edges["recency"]) for v in recency]
    m["direction_group"] = [assign_direction(v) for v in change]

    rows = []
    slices = (
        ("change_magnitude", "change_group", ("Q1", "Q2", "Q3", "Q4", "available", "missing")),
        ("recency", "recency_group", ("Q1", "Q2", "Q3", "Q4", "available", "missing")),
        ("direction", "direction_group", (">0", "=0", "<0", "missing")),
    )
    for slice_name, col, groups in slices:
        for grp in groups:
            g = m.loc[m[col] == grp]
            if g.empty:
                continue
            rows.append(
                {
                    "slice": slice_name,
                    "group": grp,
                    "n": int(len(g)),
                    "actual_volume": float(np.abs(g["actual"]).sum()),
                    "wmape_P0_TS": wmape(g["actual"], g["pred_P0_TS"]),
                    "wmape_P1_TS": wmape(g["actual"], g["pred_P1_TS"]),
                    "wmape_P0_HUMAN": wmape(g["actual"], g["pred_P0_HUMAN"]),
                    "wmape_P1_HUMAN": wmape(g["actual"], g["pred_P1_HUMAN"]),
                }
            )
    return pd.DataFrame(rows)


def _regime_signal(regimes: pd.DataFrame) -> bool:
    """True if WMAPE varies materially across change/recency groups (not missing)."""
    if regimes is None or regimes.empty:
        return False
    for slice_name in ("change_magnitude", "recency"):
        g = regimes.loc[
            (regimes["slice"] == slice_name) & (regimes["group"] != "missing")
        ]
        if len(g) < 2:
            continue
        for col in ("wmape_P0_TS", "wmape_P1_TS", "wmape_P0_HUMAN", "wmape_P1_HUMAN"):
            if col not in g.columns:
                continue
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2 and float(np.max(vals) - np.min(vals)) >= 1.0:
                return True
        for a, b in (("wmape_P1_TS", "wmape_P0_TS"), ("wmape_P1_HUMAN", "wmape_P0_HUMAN")):
            if a not in g.columns or b not in g.columns:
                continue
            delta = g[b].to_numpy(dtype=float) - g[a].to_numpy(dtype=float)
            finite = delta[np.isfinite(delta)]
            if len(finite) >= 2 and (finite > 0).any() and (finite < 0).any():
                return True
    return False


def classify_f3b(
    overall: pd.DataFrame,
    conc: pd.DataFrame,
    regimes: pd.DataFrame,
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

    p1t, p1h = row("P1_TS"), row("P1_HUMAN")
    if p1t is None or p1h is None:
        return "E"
    ts_helps = _anchor_helps(
        wmape_control=float(p1t["wmape_control"]),
        wmape_cand=float(p1t["wmape"]),
        origins_improved=int(p1t["origins_improved"]),
        origins_total=int(p1t["origins_total"]),
        product_win_rate=float(p1t["product_win_rate"]),
        median_product_improvement_pct=float(p1t["median_product_improvement_pct"]),
        bias_control=float(p1t["bias_control"]),
        bias_cand=float(p1t["bias"]),
        concentration_flags=flags("P1_TS"),
    )
    human_helps = _anchor_helps(
        wmape_control=float(p1h["wmape_control"]),
        wmape_cand=float(p1h["wmape"]),
        origins_improved=int(p1h["origins_improved"]),
        origins_total=int(p1h["origins_total"]),
        product_win_rate=float(p1h["product_win_rate"]),
        median_product_improvement_pct=float(p1h["median_product_improvement_pct"]),
        bias_control=float(p1h["bias_control"]),
        bias_cand=float(p1h["bias"]),
        concentration_flags=flags("P1_HUMAN"),
    )
    if ts_helps and human_helps:
        return "A"
    if ts_helps and not human_helps:
        return "B"
    if human_helps and not ts_helps:
        return "C"
    if _regime_signal(regimes):
        return "D"
    return "E"


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
    """Per-origin fit_xgb gain after scoring. Not used for promotion."""
    ts_u = prep_lags(enriched.ts_universe)
    bud_u = prep_lags(enriched.budget_universe)
    matched_u = prep_lags(enriched.matched_universe)
    extra = frozenset(FILLNA_EXTRA)
    skip = frozenset(NEVER_FILLNA)
    rows = []
    specs = (
        ("P1_TS", "ts", "ts", list(P1_TS.features()), "ts_forecast"),
        ("P1_HUMAN", "human", "budget", list(P1_HUMAN.features()), "budget_forecast"),
    )
    for exp_name, anchor, train_key, feats, forecast_col in specs:
        price_set = set(FEATURE_NAMES)
        for O in sorted(int(o) for o in PRIMARY_ORIGINS):
            train_bud = bud_u.loc[bud_u["target_date"].astype(int) < O]
            if not _fold_eligible_primary(train_bud):
                continue
            train = _train_slice(
                lambda _t, _te: None,
                train_key,
                O,
                ts_u,
                bud_u,
                matched_u,
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
            price_gain = float(imp.loc[imp["feature"].isin(price_set), "gain"].sum())
            for _, row in imp.iterrows():
                is_price = row["feature"] in price_set
                rows.append(
                    {
                        "experiment": exp_name,
                        "anchor": anchor,
                        "origin": int(O),
                        "feature": row["feature"],
                        "is_price_feature": is_price,
                        "gain": float(row["gain"]),
                        "weight": int(row["weight"]),
                        "price_gain_share": (price_gain / total_gain) if total_gain > 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _rename_candidate(row: dict) -> dict:
    out = dict(row)
    if "wmape_candidate" in out:
        out["wmape_price"] = out.pop("wmape_candidate")
    return out


def evaluate_f3b(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f3b_output_dir()
    session = FamilySession(
        "f3b",
        out_dir,
        dataset=dataset,
        verify_checksums=verify_checksums,
        fillna_extra=FILLNA_EXTRA,
        never_fillna=NEVER_FILLNA,
        enrichers={"price": enrich_price_dataset},
        model_name_prefix="xgb",
    )
    p0_ts = session.f0("ts")
    p0_human = session.f0("human")
    canon = session.canon

    gate_rows = []
    ts_w = float(p0_ts.overall["wmape"].iloc[0])
    h_w = float(p0_human.overall["wmape"].iloc[0])
    gate_rows.append(
        wmape_gate_row(
            "P0_TS vs current-env TS F0",
            ts_w,
            CURRENT_ENV_F0_WMAPE["ts"],
            int(p0_ts.overall["n"].iloc[0]),
            len(p0_ts.origins),
        )
    )
    gate_rows.append(
        wmape_gate_row(
            "P0_HUMAN vs current-env Human F0",
            h_w,
            CURRENT_ENV_F0_WMAPE["human"],
            int(p0_human.overall["n"].iloc[0]),
            len(p0_human.origins),
        )
    )
    if sorted(int(o) for o in p0_ts.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(f"P0_TS origins {p0_ts.origins} != PRIMARY {PRIMARY_ORIGINS}")
    if sorted(int(o) for o in p0_human.origins) != list(PRIMARY_ORIGINS):
        raise AssertionError(
            f"P0_HUMAN origins {p0_human.origins} != PRIMARY {PRIMARY_ORIGINS}"
        )
    for row in gate_rows:
        assert_wmape_gate(row)
    pd.DataFrame(gate_rows).to_csv(out_dir / "reproduction_gates.csv", index=False)

    hist = load_frozen_price_history()
    audit_panel = p0_ts.predictions[["product", "test_origin"]].copy()
    audit_panel["product"] = audit_panel["product"].astype(str)
    audit_panel["origin"] = audit_panel["test_origin"].astype(int)
    audit_panel = audit_panel.drop_duplicates(["product", "origin"])
    attached = add_price_features(audit_panel, hist, origin_col="origin")
    edges = lock_regime_edges(attached)
    lookup = _price_lookup(attached)

    print("=== P1_TS F0 + price ===")
    p1_ts = session.run(_spec_for(P1_TS))
    print("=== P1_HUMAN F0 + price ===")
    p1_human = session.run(_spec_for(P1_HUMAN))

    results: dict[str, BacktestResult] = {
        "P0_TS": p0_ts,
        "P1_TS": p1_ts,
        "P0_HUMAN": p0_human,
        "P1_HUMAN": p1_human,
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
                "include_price": exp.include_price,
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
                "median_origin_improvement_pct": osu["median_origin_improvement"],
                **psu,
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
            }
        )
        for _, r in odf.iterrows():
            origin_rows.append(
                {
                    "experiment": name,
                    "control": control_name,
                    **_rename_candidate(r.to_dict()),
                }
            )
        for _, r in pdf.iterrows():
            product_rows.append(
                {
                    "experiment": name,
                    "control": control_name,
                    **_rename_candidate(r.to_dict()),
                }
            )
        for _, r in hb.iterrows():
            horizon_rows.append(
                {
                    "experiment": name,
                    "control": control_name,
                    "horizon_bucket": r["horizon_bucket"],
                    "n": r["n"],
                    "wmape_control": r["wmape_f0"],
                    "wmape_price": r["wmape_new"],
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
                    "wmape_price": wmape(sub["actual"], sub["pred_cand"]),
                }
            )

        fd = res.fold_diagnostics.copy()
        fd["experiment"] = name
        fd["anchor"] = exp.anchor
        fd["train_universe"] = exp.train_universe
        fd.to_csv(out_dir / f"train_diagnostics_{name}.csv", index=False)

    regimes = price_regime_table(results, lookup, edges)
    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_product = pd.DataFrame(product_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    conc_df = pd.DataFrame(conc_rows)
    watch_df = pd.DataFrame(watch_rows)
    gates = pd.DataFrame(gate_rows)

    print("=== diagnostic XGB gain (not used for promotion) ===")
    enriched = enrich_price_dataset(session.ds)
    importance = diagnostic_feature_importance(enriched)

    verdict = classify_f3b(overall, conc_df, regimes)

    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_product.to_csv(out_dir / "by_product.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon.csv", index=False)
    regimes.to_csv(out_dir / "price_regime_analysis.csv", index=False)
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
        "price_regime_analysis": regimes,
        "error_concentration": conc_df,
        "watchlist": watch_df,
        "feature_importance": importance,
        "gates": gates,
        "regime_edges": edges,
        "canonical_f0": canon,
        "results": results,
        "verdict": verdict,
        "out_dir": out_dir,
    }
