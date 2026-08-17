"""Compare F0 / F1A / F1B / F1C on identical frozen matched PRIMARY rows.

Run::

    python -m pkg.research.evaluate_features
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Sequence

import pandas as pd

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.config import EXPECTED_ANALYSIS_B_PRIMARY
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult
from pkg.research.experiments import (
    EXPERIMENTS,
    FeatureSet,
    enrich_dataset,
    get_experiment,
    make_residual_model,
    train_universe_for,
)
from pkg.research.harness.metrics import (
    ROW_KEYS,
    assert_same_eval_rows,
    horizon_bucket_table as _horizon_bucket_table,
    origins_improved as _origins_improved,
    product_stats as _product_stats,
    rel_wmape as _rel_wmape,
)


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
