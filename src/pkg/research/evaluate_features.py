"""Compare F0 / F1A / F1B / F1C on identical frozen matched PRIMARY rows.

Run::

    python -m pkg.research.evaluate_features
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.config import EXPECTED_ANALYSIS_B_PRIMARY, HORIZON_BUCKETS
from pkg.benchmark.dataset import BenchmarkDataset, horizon_bucket
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.experiments import (
    EXPERIMENTS,
    FeatureSet,
    enrich_dataset,
    get_experiment,
    make_residual_model,
    train_universe_for,
)

ROW_KEYS = ("product", "qrt", "target_date", "test_origin")


def _row_key_frame(preds: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ROW_KEYS if c not in preds.columns]
    if missing:
        raise KeyError(f"predictions missing keys {missing}")
    return (
        preds[list(ROW_KEYS)]
        .assign(
            product=lambda d: d["product"].astype(str),
            qrt=lambda d: d["qrt"].astype(str),
            target_date=lambda d: d["target_date"].astype(int),
            test_origin=lambda d: d["test_origin"].astype(int),
        )
        .sort_values(list(ROW_KEYS))
        .reset_index(drop=True)
    )


def assert_same_eval_rows(baseline: BacktestResult, candidate: BacktestResult) -> None:
    """Candidate experiment must score exactly the same TEST identities as F0."""
    a = _row_key_frame(baseline.predictions)
    b = _row_key_frame(candidate.predictions)
    if len(a) != len(b):
        raise AssertionError(
            f"eval row count mismatch: F0 n={len(a)} candidate n={len(b)}"
        )
    if not a.equals(b):
        # Show a small diff sample
        merged = a.merge(b, on=list(ROW_KEYS), how="outer", indicator=True)
        bad = merged.loc[merged["_merge"] != "both"]
        raise AssertionError(
            f"eval row keys differ from F0 (diff_rows={len(bad)}). "
            f"sample:\n{bad.head(10)}"
        )


def _rel_wmape(base: float, new: float) -> float:
    if base == 0 or not np.isfinite(base):
        return float("nan")
    return float((base - new) / base * 100.0)


def _origins_improved(base: BacktestResult, cand: BacktestResult) -> tuple[int, int]:
    b = base.by_origin.set_index("origin")["wmape"]
    c = cand.by_origin.set_index("origin")["wmape"]
    common = sorted(set(b.index) & set(c.index))
    improved = sum(1 for o in common if c.loc[o] < b.loc[o])
    return improved, len(common)


def _product_stats(base: BacktestResult, cand: BacktestResult) -> dict:
    """Product win rate / median improvement of candidate vs F0 predictions."""
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=["product", "qrt", "target_date", "test_origin"], how="inner")
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
            "n_products": 0,
        }
    pdf = pd.DataFrame(rows)
    return {
        "product_win_rate": float((pdf["rel_improvement_pct"] > 0).mean()),
        "median_product_improvement_pct": float(pdf["rel_improvement_pct"].median()),
        "n_products": int(len(pdf)),
    }


def _horizon_bucket_table(base: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "horizon", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=["product", "qrt", "target_date", "test_origin"], how="inner")
    m["horizon_bucket"] = m["horizon"].map(horizon_bucket)
    rows = []
    for name, lo, hi in HORIZON_BUCKETS:
        g = m.loc[m["horizon_bucket"] == name]
        if g.empty:
            continue
        w0 = wmape(g["actual"], g["pred_f0"])
        w1 = wmape(g["actual"], g["pred_new"])
        rows.append(
            {
                "horizon_bucket": name,
                "n": len(g),
                "wmape_f0": w0,
                "wmape_new": w1,
                "rel_wmape_vs_f0_pct": _rel_wmape(w0, w1),
            }
        )
    return pd.DataFrame(rows)


def _run_anchor(
    ds: BenchmarkDataset,
    experiment: FeatureSet,
    anchor: str,
    *,
    f0_result: Optional[BacktestResult] = None,
) -> BacktestResult:
    if experiment.name == "F0":
        # Locked frozen adapters — must match expected Analysis B WMAPEs
        frozen_name = "ts_xgb" if anchor == "ts" else "human_xgb"
        return backtest(
            frozen_name,
            dataset=ds,
            universe="matched",
            eligibility="primary",
        )

    enriched = enrich_dataset(ds, experiment)
    feats = experiment.features_for(anchor)  # type: ignore[arg-type]
    model = make_residual_model(anchor, feats)  # type: ignore[arg-type]
    result = backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_universe_for(anchor),  # type: ignore[arg-type]
    )
    if f0_result is not None:
        assert_same_eval_rows(f0_result, result)
    return result


def compare_feature_experiments(
    experiments: Sequence[str] = ("F0", "F1A", "F1B", "F1C"),
    anchors: Sequence[str] = ("ts", "human"),
    *,
    dataset: Optional[BenchmarkDataset] = None,
    verify_checksums: bool = False,
) -> dict:
    """Run feature experiments on matched PRIMARY; return overall + detail tables.

    Returns
    -------
    dict with keys:
      overall, by_origin, by_horizon_bucket, results
    """
    ds = dataset or load_benchmark(verify_checksums=verify_checksums)

    # F0 baselines first (frozen names)
    f0_results: dict[str, BacktestResult] = {}
    if "F0" not in experiments:
        # Still need F0 for assertions / relative metrics
        for a in anchors:
            f0_results[a] = _run_anchor(ds, get_experiment("F0"), a)
    else:
        for a in anchors:
            f0_results[a] = _run_anchor(ds, get_experiment("F0"), a)

    # Sanity: F0 matches locked scoreboard
    for a, exp_key in (("ts", "ts_xgb"), ("human", "human_xgb")):
        if a not in f0_results:
            continue
        got = float(f0_results[a].overall["wmape"].iloc[0])
        exp = EXPECTED_ANALYSIS_B_PRIMARY[exp_key]
        if abs(got - exp) > 0.05:
            raise AssertionError(
                f"F0 {a} WMAPE {got} drifted from locked {exp} (tol=0.05)"
            )

    all_results: dict[tuple[str, str], BacktestResult] = {}
    overall_rows = []
    origin_rows = []
    horizon_rows = []

    for exp_name in experiments:
        experiment = get_experiment(exp_name)
        for anchor in anchors:
            if exp_name == "F0":
                res = f0_results[anchor]
            else:
                res = _run_anchor(
                    ds, experiment, anchor, f0_result=f0_results[anchor]
                )
            all_results[(exp_name, anchor)] = res

            f0 = f0_results[anchor]
            o = res.overall.iloc[0]
            f0o = f0.overall.iloc[0]
            n_imp, n_tot = _origins_improved(f0, res)
            pstats = _product_stats(f0, res)
            overall_rows.append(
                {
                    "experiment": exp_name,
                    "anchor": anchor,
                    "groups": "+".join(experiment.groups),
                    "n_features": len(experiment.features_for(anchor)),  # type: ignore[arg-type]
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
                    "median_product_improvement_pct": pstats[
                        "median_product_improvement_pct"
                    ],
                    "n_products": pstats["n_products"],
                }
            )

            # by origin
            for _, row in res.by_origin.iterrows():
                o_id = int(row["origin"])
                f0_w = float(
                    f0.by_origin.loc[f0.by_origin["origin"] == o_id, "wmape"].iloc[0]
                )
                origin_rows.append(
                    {
                        "experiment": exp_name,
                        "anchor": anchor,
                        "origin": o_id,
                        "wmape": float(row["wmape"]),
                        "wmape_f0": f0_w,
                        "rel_wmape_vs_f0_pct": _rel_wmape(f0_w, float(row["wmape"])),
                        "n": int(row["n"]),
                    }
                )

            hb = _horizon_bucket_table(f0, res)
            for _, row in hb.iterrows():
                horizon_rows.append(
                    {
                        "experiment": exp_name,
                        "anchor": anchor,
                        **row.to_dict(),
                    }
                )

    return {
        "overall": pd.DataFrame(overall_rows),
        "by_origin": pd.DataFrame(origin_rows),
        "by_horizon_bucket": pd.DataFrame(horizon_rows),
        "results": all_results,
        "f0_results": f0_results,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare research feature sets F0/F1A/F1B/F1C on frozen v1"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        help="Experiment names (default: all)",
    )
    parser.add_argument(
        "--anchors",
        nargs="+",
        default=["ts", "human"],
        choices=["ts", "human"],
    )
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = compare_feature_experiments(
        experiments=args.experiments,
        anchors=args.anchors,
        verify_checksums=args.verify_checksums,
    )
    print("=== Feature experiment overall (matched PRIMARY) ===")
    cols = [
        "experiment",
        "anchor",
        "wmape",
        "rel_wmape_vs_f0_pct",
        "rmse",
        "mae",
        "bias",
        "n",
        "origins_improved",
        "origins_total",
        "product_win_rate",
        "median_product_improvement_pct",
    ]
    print(report["overall"][cols].to_string(index=False))
    print("\n=== By horizon bucket ===")
    print(report["by_horizon_bucket"].to_string(index=False))
    print("\n=== By origin ===")
    print(report["by_origin"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
