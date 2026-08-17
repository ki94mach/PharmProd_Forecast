"""Evaluate CORE / F0 / F1 / F2 as replacements vs additions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.benchmark.models import fit_xgb
from pkg.research.ablation.config import (
    CORE_HUMAN,
    CORE_TS,
    F0_DEMAND,
    F1_DEMAND,
    F1_HUMAN,
    F2_DEMAND,
    F2_HUMAN,
    FILLNA_EXTRA,
    MATERIAL_WMAPE_TOL,
    PRED_REPRO_TOL,
    SIMILAR_WMAPE_TOL,
    WMAPE_REPRO_TOL,
    AblationExperiment,
    DEMAND_EXPERIMENTS,
    HUMAN_EXPERIMENTS,
    ablation_output_dir,
    get_ablation,
)
from pkg.research.ablation.decomposition import decompose_vs_f0
from pkg.research.evaluate_features import (
    ROW_KEYS,
    _horizon_bucket_table,
    _origins_improved,
    _rel_wmape,
    assert_same_eval_rows,
)
from pkg.research.experiments import enrich_dataset, get_experiment, make_residual_model
from pkg.research.f2.config import F2A, F2B, F2Experiment
from pkg.research.f2.evaluate import (
    _product_stats_full,
    assert_freeze_unchanged,
    confirm_canonical_f0,
    enrich_f2_dataset,
    freeze_checksums,
    make_f2_residual_model,
)
from pkg.research.features.demand import add_demand_features, load_frozen_sales
from pkg.research.features.demand_f2 import add_demand_f2_features
from pkg.research.features.human import add_human_features
from pkg.research.features.human_f2 import add_human_f2_features


def make_ablation_residual_model(anchor: str, feature_cols: Sequence[str]):
    """Frozen fit_xgb + clip; fillna for F0 lags and F1/F2 extras."""
    cols = list(feature_cols)
    if anchor == "ts":
        forecast_col = "ts_forecast"
        name = "ts_xgb_ablation"
    elif anchor == "human":
        forecast_col = "budget_forecast"
        name = "human_xgb_ablation"
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
            if c.startswith("sales_") or c.startswith("human_") or c in FILLNA_EXTRA:
                tr[c] = tr[c].fillna(0)
                te[c] = te[c].fillna(0)
        if "horizon" not in tr.columns:
            raise KeyError("train_df needs horizon for sample weights")
        model = fit_xgb(cols, tr)
        resid = model.predict(te[cols])
        return np.maximum(0.0, te[forecast_col].astype(float).to_numpy() + resid)

    _predict.__name__ = name
    return _predict


def enrich_ablation_dataset(
    ds: BenchmarkDataset, experiment: AblationExperiment
) -> BenchmarkDataset:
    need_f1d = "f1_demand" in experiment.groups
    need_f2d = "f2_demand" in experiment.groups
    need_f1h = "f1_human" in experiment.groups
    need_f2h = "f2_human" in experiment.groups
    if not (need_f1d or need_f2d or need_f1h or need_f2h):
        return BenchmarkDataset(
            version=ds.version,
            root=ds.root,
            ts_universe=ds.ts_universe.copy(),
            budget_universe=ds.budget_universe.copy(),
            matched_universe=ds.matched_universe.copy(),
            manifest=ds.manifest,
        )

    sales_hist = None
    if need_f1d or need_f2d:
        sales_hist = load_frozen_sales(ds.root)

    def _enrich(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        origin_col = (
            "origin"
            if "origin" in out.columns
            else ("ts_origin" if "ts_origin" in out.columns else "budget_origin")
        )
        if need_f1d:
            out = add_demand_features(out, sales_hist, origin_col=origin_col)
        if need_f2d:
            out = add_demand_f2_features(out, sales_hist, origin_col=origin_col)
        if need_f1h:
            out = add_human_features(
                out, ds.budget_universe, matched_hist=ds.matched_universe, origin_col=origin_col
            )
        if need_f2h:
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


def _pred_diff_stats(a: BacktestResult, b: BacktestResult) -> dict:
    left = a.predictions[list(ROW_KEYS) + ["prediction"]].copy()
    right = b.predictions[list(ROW_KEYS) + ["prediction"]].rename(
        columns={"prediction": "prediction_b"}
    )
    for col in ("product", "qrt"):
        left[col] = left[col].astype(str)
        right[col] = right[col].astype(str)
    for col in ("target_date", "test_origin"):
        left[col] = left[col].astype(int)
        right[col] = right[col].astype(int)
    m = left.merge(right, on=list(ROW_KEYS), how="inner")
    diff = (m["prediction"] - m["prediction_b"]).abs()
    return {
        "n": len(m),
        "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
        "mean_abs_diff": float(diff.mean()) if len(diff) else float("nan"),
        "n_gt_tol": int((diff > PRED_REPRO_TOL).sum()) if len(diff) else 0,
        "wmape_a": wmape(a.predictions["actual"], a.predictions["prediction"]),
        "wmape_b": wmape(b.predictions["actual"], b.predictions["prediction"]),
    }


def _assert_repro(label: str, ablation: BacktestResult, reference: BacktestResult) -> dict:
    assert_same_eval_rows(reference, ablation)
    stats = _pred_diff_stats(ablation, reference)
    wmape_gap = abs(stats["wmape_a"] - stats["wmape_b"])
    ok = stats["max_abs_diff"] <= PRED_REPRO_TOL and wmape_gap <= WMAPE_REPRO_TOL
    stats["label"] = label
    stats["wmape_gap"] = wmape_gap
    stats["ok"] = ok
    if not ok:
        raise AssertionError(
            f"Reproduction gate FAILED for {label}: max_abs_diff={stats['max_abs_diff']} "
            f"wmape_gap={wmape_gap} (tol pred={PRED_REPRO_TOL} wmape={WMAPE_REPRO_TOL})"
        )
    return stats


def _run_ablation(
    ds: BenchmarkDataset, experiment: AblationExperiment, anchor: str, f0: BacktestResult
) -> BacktestResult:
    enriched = enrich_ablation_dataset(ds, experiment)
    feats = experiment.features_for(anchor)
    model = make_ablation_residual_model(anchor, feats)
    train_u = "ts" if anchor == "ts" else "budget"
    result = backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_u,
    )
    assert_same_eval_rows(f0, result)
    return result


def _run_f1(ds: BenchmarkDataset, name: str, anchor: str, f0: BacktestResult) -> BacktestResult:
    experiment = get_experiment(name)
    enriched = enrich_dataset(ds, experiment)
    feats = experiment.features_for(anchor)  # type: ignore[arg-type]
    model = make_residual_model(anchor, feats)  # type: ignore[arg-type]
    train_u = "ts" if anchor == "ts" else "budget"
    result = backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_u,
    )
    assert_same_eval_rows(f0, result)
    return result


def _run_f2(ds: BenchmarkDataset, exp: F2Experiment, anchor: str, f0: BacktestResult) -> BacktestResult:
    enriched = enrich_f2_dataset(ds, exp)
    feats = exp.features_for(anchor)  # type: ignore[arg-type]
    model = make_f2_residual_model(anchor, feats)
    train_u = exp.train_universe[anchor]
    result = backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_u,
    )
    assert_same_eval_rows(f0, result)
    return result


def evaluate_feature_ablation(
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
    skip_secondary: bool = False,
) -> dict:
    out_dir = out_dir or ablation_output_dir()
    ds = dataset or load_benchmark(verify_checksums=verify_checksums)
    freeze_before = freeze_checksums(ds)
    _write_partition(out_dir)

    canon = confirm_canonical_f0(ds)
    f0_results: dict[str, BacktestResult] = canon["results"]
    canon["summary"].to_csv(out_dir / "f0_canonical.csv", index=False)

    overall_rows = []
    origin_rows = []
    horizon_rows = []
    train_parts = []
    conc_rows = []
    top_rows = []
    gate_rows = []
    all_results: dict[tuple[str, str], BacktestResult] = {}

    def _record(exp: AblationExperiment, anchor: str, res: BacktestResult) -> None:
        f0 = f0_results[anchor]
        o = res.overall.iloc[0]
        f0o = f0.overall.iloc[0]
        n_imp, n_tot = _origins_improved(f0, res)
        pstats = _product_stats_full(f0, res)
        decomp = decompose_vs_f0(f0, res, exp.name, anchor, out_dir)
        all_results[(exp.name, anchor)] = res
        overall_rows.append(
            {
                "experiment": exp.name,
                "anchor": anchor,
                "family": exp.family,
                "secondary": exp.secondary,
                "groups": "+".join(exp.groups) or "core",
                "n_features": len(exp.features_for(anchor)),
                "wmape": float(o["wmape"]),
                "wmape_f0": float(f0o["wmape"]),
                "rel_wmape_vs_f0_pct": _rel_wmape(float(f0o["wmape"]), float(o["wmape"])),
                "rmse": float(o["rmse"]),
                "mae": float(o["mae"]),
                "bias": float(o["bias"]),
                "n": int(o["n"]),
                "origins_improved": n_imp,
                "origins_total": n_tot,
                "product_win_rate": pstats["product_win_rate"],
                "median_product_improvement_pct": pstats["median_product_improvement_pct"],
                "p25_product_improvement_pct": pstats["p25_product_improvement_pct"],
                "p75_product_improvement_pct": pstats["p75_product_improvement_pct"],
                "n_products": pstats["n_products"],
                **{k: decomp["summary"][k] for k in (
                    "net_delta_ae",
                    "top1_deterioration_share",
                    "top5_deterioration_share",
                    "top10_deterioration_share",
                )},
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
        fd = res.fold_diagnostics.copy()
        fd["experiment"] = exp.name
        fd["anchor"] = anchor
        fd["train_universe"] = "ts" if anchor == "ts" else "budget"
        train_parts.append(fd)
        conc_rows.append(decomp["summary"])
        top_rows.extend(decomp["top_rows"])

    demand_exps = list(DEMAND_EXPERIMENTS)
    human_exps = [
        e
        for e in HUMAN_EXPERIMENTS
        if e.name not in {"H0_CORE", "H1_F0"}
        and (not skip_secondary or not e.secondary)
    ]

    for exp in demand_exps:
        for anchor in exp.anchors:
            print(f"=== {exp.name} / {anchor} ===")
            res = _run_ablation(ds, exp, anchor, f0_results[anchor])
            _record(exp, anchor, res)

    # Alias H0/H1 from D0/D1 human
    for alias, src in (("H0_CORE", "D0_CORE"), ("H1_F0", "D1_F0")):
        src_res = all_results[(src, "human")]
        alias_exp = get_ablation(alias)
        _record(alias_exp, "human", src_res)

    for exp in human_exps:
        print(f"=== {exp.name} / human ===")
        res = _run_ablation(ds, exp, "human", f0_results["human"])
        _record(exp, "human", res)

    print("=== Reproduction gates ===")
    for anchor in ("ts", "human"):
        gate_rows.append(
            _assert_repro(
                f"D1_F0 vs frozen {anchor}_xgb",
                all_results[("D1_F0", anchor)],
                f0_results[anchor],
            )
        )
        f1a = _run_f1(ds, "F1A", anchor, f0_results[anchor])
        gate_rows.append(
            _assert_repro(f"D4_F1_ADD vs F1A {anchor}", all_results[("D4_F1_ADD", anchor)], f1a)
        )
        f2a = _run_f2(ds, F2A, anchor, f0_results[anchor])
        gate_rows.append(
            _assert_repro(f"D5_F2_ADD vs F2A {anchor}", all_results[("D5_F2_ADD", anchor)], f2a)
        )

    f1b = _run_f1(ds, "F1B", "human", f0_results["human"])
    gate_rows.append(
        _assert_repro("H4_F1_HUMAN_ADD vs F1B", all_results[("H4_F1_HUMAN_ADD", "human")], f1b)
    )
    f2b = _run_f2(ds, F2B, "human", f0_results["human"])
    gate_rows.append(
        _assert_repro("H5_F2_HUMAN_ADD vs F2B", all_results[("H5_F2_HUMAN_ADD", "human")], f2b)
    )

    overall = pd.DataFrame(overall_rows)
    by_origin = pd.DataFrame(origin_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    train_diag = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    conc_df = pd.DataFrame(conc_rows)
    top_df = pd.DataFrame(top_rows)
    gates = pd.DataFrame(gate_rows)
    effects = _replacement_effects(overall)
    cases = _classify_cases(effects)

    overall.to_csv(out_dir / "overall.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin.csv", index=False)
    by_horizon.to_csv(out_dir / "by_horizon_bucket.csv", index=False)
    train_diag.to_csv(out_dir / "train_diagnostics.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    top_df.to_csv(out_dir / "top_products.csv", index=False)
    gates.to_csv(out_dir / "reproduction_gates.csv", index=False)
    effects.to_csv(out_dir / "replacement_effects.csv", index=False)
    cases.to_csv(out_dir / "classifications.csv", index=False)

    assert_freeze_unchanged(ds, freeze_before)

    return {
        "overall": overall,
        "by_origin": by_origin,
        "by_horizon_bucket": by_horizon,
        "train_diagnostics": train_diag,
        "error_concentration": conc_df,
        "top_products": top_df,
        "gates": gates,
        "effects": effects,
        "classifications": cases,
        "canonical_f0": canon,
        "results": all_results,
        "f0_results": f0_results,
        "out_dir": out_dir,
    }


def _replacement_effects(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for anchor in ("ts", "human"):
        def w(name: str) -> float:
            sub = overall.loc[
                (overall["experiment"] == name) & (overall["anchor"] == anchor)
            ]
            if sub.empty:
                return float("nan")
            return float(sub["wmape"].iloc[0])

        d0, d1, d2, d3, d4, d5 = (
            w("D0_CORE"),
            w("D1_F0"),
            w("D2_F1_REPLACE"),
            w("D3_F2_REPLACE"),
            w("D4_F1_ADD"),
            w("D5_F2_ADD"),
        )
        rows.append(
            {
                "anchor": anchor,
                "wmape_core": d0,
                "wmape_core_f0": d1,
                "wmape_core_f1": d2,
                "wmape_core_f2": d3,
                "wmape_core_f0_f1": d4,
                "wmape_core_f0_f2": d5,
                "f0_demand_value": d0 - d1,  # CORE vs CORE+F0; + = F0 demand helps
                "f1_replacement_effect": d1 - d2,
                "f2_replacement_effect": d1 - d3,
                "f1_addition_effect": d1 - d4,
                "f2_addition_effect": d1 - d5,
            }
        )
    # Human reliability effects vs H1/D1 human
    def hw(name: str) -> float:
        sub = overall.loc[
            (overall["experiment"] == name) & (overall["anchor"] == "human")
        ]
        if sub.empty:
            return float("nan")
        return float(sub["wmape"].iloc[0])

    h0, h1, h2, h3, h4, h5 = (
        hw("H0_CORE"),
        hw("H1_F0"),
        hw("H2_F1_HUMAN_ONLY"),
        hw("H3_F2_HUMAN_ONLY"),
        hw("H4_F1_HUMAN_ADD"),
        hw("H5_F2_HUMAN_ADD"),
    )
    rows.append(
        {
            "anchor": "human_reliability",
            "wmape_core": h0,
            "wmape_core_f0": h1,
            "wmape_core_f1": h2,
            "wmape_core_f2": h3,
            "wmape_core_f0_f1": h4,
            "wmape_core_f0_f2": h5,
            "f0_demand_value": h0 - h1,
            "f1_replacement_effect": h1 - h2,  # CORE+F0 vs CORE+F1_HUMAN (not a demand replace)
            "f2_replacement_effect": h1 - h3,
            "f1_addition_effect": h1 - h4,
            "f2_addition_effect": h1 - h5,
        }
    )
    return pd.DataFrame(rows)


def _write_partition(out_dir: Path) -> None:
    rows = []
    for family, names in (
        ("CORE_TS", CORE_TS),
        ("CORE_HUMAN", CORE_HUMAN),
        ("F0_DEMAND", F0_DEMAND),
        ("F1_DEMAND", F1_DEMAND),
        ("F1_HUMAN", F1_HUMAN),
        ("F2_DEMAND", F2_DEMAND),
        ("F2_HUMAN", F2_HUMAN),
    ):
        for i, name in enumerate(names):
            rows.append({"family": family, "position": i, "feature": name})
    pd.DataFrame(rows).to_csv(out_dir / "partition.csv", index=False)


def _demand_case(f0_w: float, replace_w: float, add_w: float, n_family: int) -> str:
    """Classify a demand family as Case A / B / C (or a residual pattern)."""
    if not np.isfinite([f0_w, replace_w, add_w]).all():
        return "missing"
    repl = f0_w - replace_w
    add = f0_w - add_w
    eps = WMAPE_REPRO_TOL
    if repl > eps and add < -eps:
        return "A"
    if repl < -eps and add < -eps:
        return "B"
    if abs(repl) <= SIMILAR_WMAPE_TOL:
        simpler = n_family < len(F0_DEMAND)
        return "C" if simpler else "C_similar_not_simpler"
    if repl > eps and add > eps:
        return "useful_both"
    if repl < -eps and add > eps:
        return "addition_only"
    return "mixed"


def _human_case(f0_w: float, standalone_w: float, add_w: float) -> str:
    """Classify a Human-reliability family as Case D / E (or residual)."""
    if not np.isfinite([f0_w, standalone_w, add_w]).all():
        return "missing"
    standalone_gap = standalone_w - f0_w
    add_gap = add_w - f0_w
    standalone_ok = standalone_gap <= MATERIAL_WMAPE_TOL
    standalone_fails = standalone_gap > MATERIAL_WMAPE_TOL
    add_fails = add_gap > MATERIAL_WMAPE_TOL
    if standalone_ok and add_fails:
        return "D"
    if standalone_fails:
        return "E"
    if standalone_ok and not add_fails:
        return "useful_or_neutral"
    return "mixed"


def _classify_cases(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in effects.iterrows():
        anchor = str(r["anchor"])
        if anchor in ("ts", "human"):
            rows.append(
                {
                    "anchor": anchor,
                    "family": "F1_DEMAND",
                    "case": _demand_case(
                        float(r["wmape_core_f0"]),
                        float(r["wmape_core_f1"]),
                        float(r["wmape_core_f0_f1"]),
                        len(F1_DEMAND),
                    ),
                    "replacement_effect": float(r["f1_replacement_effect"]),
                    "addition_effect": float(r["f1_addition_effect"]),
                }
            )
            rows.append(
                {
                    "anchor": anchor,
                    "family": "F2_DEMAND",
                    "case": _demand_case(
                        float(r["wmape_core_f0"]),
                        float(r["wmape_core_f2"]),
                        float(r["wmape_core_f0_f2"]),
                        len(F2_DEMAND),
                    ),
                    "replacement_effect": float(r["f2_replacement_effect"]),
                    "addition_effect": float(r["f2_addition_effect"]),
                }
            )
        elif anchor == "human_reliability":
            rows.append(
                {
                    "anchor": "human",
                    "family": "F1_HUMAN",
                    "case": _human_case(
                        float(r["wmape_core_f0"]),
                        float(r["wmape_core_f1"]),
                        float(r["wmape_core_f0_f1"]),
                    ),
                    "replacement_effect": float(r["f1_replacement_effect"]),
                    "addition_effect": float(r["f1_addition_effect"]),
                }
            )
            rows.append(
                {
                    "anchor": "human",
                    "family": "F2_HUMAN",
                    "case": _human_case(
                        float(r["wmape_core_f0"]),
                        float(r["wmape_core_f2"]),
                        float(r["wmape_core_f0_f2"]),
                    ),
                    "replacement_effect": float(r["f2_replacement_effect"]),
                    "addition_effect": float(r["f2_addition_effect"]),
                }
            )
    return pd.DataFrame(rows)
