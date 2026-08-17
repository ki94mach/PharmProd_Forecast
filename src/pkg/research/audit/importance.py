"""Section 8: XGBoost built-in feature usage diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, prep_lags
from pkg.benchmark.evaluate import _fold_eligible_primary, _train_slice
from pkg.benchmark.models import fit_xgb
from pkg.research.audit.common import audit_output_dir, save_csv
from pkg.research.experiments import (
    enrich_dataset,
    get_experiment,
    train_universe_for,
)


def _feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Gain and split count per feature (sklearn + booster fallback)."""
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
        rows.append(
            {
                "feature": name,
                "gain": float(gain),
                "weight": int(weight),
            }
        )
    return pd.DataFrame(rows)


def _classify_usage(gain: float, new_gains: np.ndarray) -> str:
    if gain <= 0:
        return "never_used"
    if len(new_gains) == 0:
        return "occasional"
    q75 = float(np.quantile(new_gains[new_gains > 0], 0.75)) if (new_gains > 0).any() else 0.0
    if q75 > 0 and gain >= q75:
        return "heavy"
    return "occasional"


def analyze_xgb_usage(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
    experiments: tuple[str, ...] = ("F1A", "F1B", "F1C"),
) -> dict:
    """Per-origin XGB gain/weight for new F1 features."""
    out_dir = out_dir or audit_output_dir()
    ts_u = prep_lags(ds.ts_universe)
    bud_u = prep_lags(ds.budget_universe)
    matched_u = prep_lags(ds.matched_universe)
    test_panel = matched_u.copy()

    all_rows = []
    summary_rows = []

    for exp_name in experiments:
        experiment = get_experiment(exp_name)
        enriched = enrich_dataset(ds, experiment)

        for anchor in ("ts", "human"):
            f0_feats = set(get_experiment("F0").features_for(anchor))  # type: ignore
            feats = list(experiment.features_for(anchor))  # type: ignore
            new_feats = [f for f in feats if f not in f0_feats]
            forecast_col = "ts_forecast" if anchor == "ts" else "budget_forecast"
            train_key = train_universe_for(anchor)  # type: ignore

            if train_key == "ts":
                train_panel = prep_lags(enriched.ts_universe)
            else:
                train_panel = prep_lags(enriched.budget_universe)

            for O in sorted(int(o) for o in PRIMARY_ORIGINS):
                train_bud = bud_u.loc[bud_u["target_date"].astype(int) < O]
                if not _fold_eligible_primary(train_bud):
                    continue

                train = _train_slice(
                    lambda t, te: None,
                    train_key,
                    O,
                    prep_lags(enriched.ts_universe),
                    prep_lags(enriched.budget_universe),
                    prep_lags(enriched.matched_universe),
                )
                if train is None or train.empty:
                    continue

                tr = train.copy()
                tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
                for c in feats:
                    if c in tr.columns:
                        tr[c] = tr[c].fillna(0)

                model = fit_xgb(feats, tr)
                imp = _feature_importance(model, feats)

                new_gains = imp.loc[imp["feature"].isin(new_feats), "gain"].to_numpy()
                total_gain = imp["gain"].sum()
                new_gain_sum = imp.loc[imp["feature"].isin(new_feats), "gain"].sum()
                f0_gain_sum = imp.loc[imp["feature"].isin(f0_feats), "gain"].sum()

                for _, row in imp.iterrows():
                    is_new = row["feature"] in new_feats
                    all_rows.append(
                        {
                            "experiment": exp_name,
                            "anchor": anchor,
                            "origin": O,
                            "feature": row["feature"],
                            "is_new_feature": is_new,
                            "gain": row["gain"],
                            "weight": row["weight"],
                            "usage_class": _classify_usage(row["gain"], new_gains)
                            if is_new
                            else "f0_baseline",
                        }
                    )

                for nf in new_feats:
                    nf_row = imp.loc[imp["feature"] == nf]
                    g = float(nf_row["gain"].iloc[0]) if len(nf_row) else 0.0
                    w = int(nf_row["weight"].iloc[0]) if len(nf_row) else 0
                    summary_rows.append(
                        {
                            "experiment": exp_name,
                            "anchor": anchor,
                            "origin": O,
                            "feature": nf,
                            "gain": g,
                            "weight": w,
                            "usage_class": _classify_usage(g, new_gains),
                            "new_gain_share": float(new_gain_sum / total_gain)
                            if total_gain > 0
                            else 0.0,
                            "f0_gain_share": float(f0_gain_sum / total_gain)
                            if total_gain > 0
                            else 0.0,
                        }
                    )

    usage_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Aggregate across origins
    if not summary_df.empty:
        agg = (
            summary_df.groupby(["experiment", "anchor", "feature"])
            .agg(
                mean_gain=("gain", "mean"),
                folds_with_splits=("weight", lambda s: int((s > 0).sum())),
                n_origins=("origin", "count"),
                heavy_folds=("usage_class", lambda s: int((s == "heavy").sum())),
                never_used_folds=("usage_class", lambda s: int((s == "never_used").sum())),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame()

    save_csv(usage_df, out_dir, "xgb_feature_usage.csv")
    save_csv(summary_df, out_dir, "xgb_new_feature_by_origin.csv")
    save_csv(agg, out_dir, "xgb_new_feature_summary.csv")

    return {"usage": usage_df, "summary": summary_df, "aggregate": agg}
