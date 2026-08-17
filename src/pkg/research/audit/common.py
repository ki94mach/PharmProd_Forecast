"""Shared helpers for F1 feature audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, horizon_bucket, prep_lags
from pkg.benchmark.evaluate import backtest, wmape
from pkg.research.experiments import (
    enrich_dataset,
    get_experiment,
    make_residual_model,
    train_universe_for,
)

ROW_KEYS = ("product", "qrt", "target_date", "test_origin")


def audit_output_dir(root: Optional[Path] = None) -> Path:
    """Default audit CSV output directory."""
    if root is not None:
        return root
    # src/pkg/research/audit/common.py -> src/data/results/f1_audit
    src_dir = Path(__file__).resolve().parents[3]
    out = src_dir / "data" / "results" / "f1_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def primary_test_predictions(
    ds: BenchmarkDataset,
    *,
    prep: bool = True,
) -> pd.DataFrame:
    """Matched PRIMARY test rows with origin column."""
    panel = ds.matched_universe.copy()
    if prep:
        panel = prep_lags(panel)
    origins = set(int(o) for o in PRIMARY_ORIGINS)
    test = panel.loc[panel["origin"].astype(int).isin(origins)].copy()
    test["test_origin"] = test["origin"].astype(int)
    return test


def align_predictions(
    frozen: pd.DataFrame,
    control: pd.DataFrame,
) -> pd.DataFrame:
    """Merge frozen and control predictions on ROW_KEYS."""
    f = frozen[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_frozen"})
    c = control[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_control"})
    for col in ROW_KEYS:
        if col == "product" or col == "qrt":
            f[col] = f[col].astype(str)
            c[col] = c[col].astype(str)
        else:
            f[col] = f[col].astype(int)
            c[col] = c[col].astype(int)
    m = f.merge(c, on=list(ROW_KEYS), how="inner")
    m["abs_diff"] = (m["pred_frozen"] - m["pred_control"]).abs()
    return m


def run_f0_control_backtest(
    ds: BenchmarkDataset,
    anchor: str,
):
    """Run F0 via research adapter (make_residual_model, no enrichment)."""
    f0 = get_experiment("F0")
    feats = f0.features_for(anchor)  # type: ignore[arg-type]
    model = make_residual_model(anchor, feats)  # type: ignore[arg-type]
    return backtest(
        model,
        dataset=ds,
        universe="matched",
        eligibility="primary",
        train_universe=train_universe_for(anchor),  # type: ignore[arg-type]
    )


def run_frozen_backtest(ds: BenchmarkDataset, anchor: str):
    """Run frozen ts_xgb or human_xgb."""
    name = "ts_xgb" if anchor == "ts" else "human_xgb"
    return backtest(name, dataset=ds, universe="matched", eligibility="primary")


def run_experiment_backtest(ds: BenchmarkDataset, exp_name: str, anchor: str):
    """Run F1 experiment via research adapter."""
    experiment = get_experiment(exp_name)
    enriched = enrich_dataset(ds, experiment)
    feats = experiment.features_for(anchor)  # type: ignore[arg-type]
    model = make_residual_model(anchor, feats)  # type: ignore[arg-type]
    return backtest(
        model,
        dataset=enriched,
        universe="matched",
        eligibility="primary",
        train_universe=train_universe_for(anchor),  # type: ignore[arg-type]
    )


def distribution_stats(series: pd.Series) -> dict:
    """Min / percentiles / max for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "min": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "non_finite_count": int(len(series) - len(s)),
            "n": 0,
        }
    return {
        "min": float(s.min()),
        "p1": float(s.quantile(0.01)),
        "p5": float(s.quantile(0.05)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "non_finite_count": int(len(series) - len(s)),
        "n": int(len(s)),
    }


def save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    df.to_csv(path, index=False)
    return path


def weighted_portfolio_wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Volume-weighted WMAPE: sum(|actual-pred|) / sum(|actual|) * 100."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - pred).sum() / denom * 100.0)


def enrich_test_panel(ds: BenchmarkDataset, experiment_name: str) -> pd.DataFrame:
    """PRIMARY test rows with experiment features attached."""
    experiment = get_experiment(experiment_name)
    enriched = enrich_dataset(ds, experiment)
    test = primary_test_predictions(enriched, prep=False)
    return test
