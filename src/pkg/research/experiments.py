"""Named feature-set experiments (F0 / F1A / F1B / F1C) on frozen benchmark.

F0 is an immutable tuple of frozen benchmark feature names; experiment helpers
never mutate it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.models import fit_xgb
from pkg.research.features.demand import (
    DEMAND_FEATURE_NAMES,
    add_demand_features,
    load_frozen_sales,
)
from pkg.research.features.human import HUMAN_FEATURE_NAMES, add_human_features

Anchor = Literal["ts", "human"]


@dataclass(frozen=True)
class FeatureSet:
    """Immutable named feature configuration."""

    name: str
    groups: tuple[str, ...]  # e.g. ("f0",) or ("f0", "demand")

    def features_for(self, anchor: Anchor) -> tuple[str, ...]:
        """Return the feature column list for a residual-XGB anchor."""
        if anchor == "ts":
            base = tuple(TS_RESID_FEATURES)
        elif anchor == "human":
            base = tuple(BUDGET_RESID_FEATURES)
        else:
            raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")

        extra: list[str] = []
        if "demand" in self.groups:
            extra.extend(DEMAND_FEATURE_NAMES)
        if "human" in self.groups:
            extra.extend(HUMAN_FEATURE_NAMES)
        # Preserve order; drop accidental dups
        seen = set()
        out: list[str] = []
        for c in base + tuple(extra):
            if c not in seen:
                seen.add(c)
                out.append(c)
        return tuple(out)

    @property
    def features(self) -> tuple[str, ...]:
        """Human-anchor feature names (common research default)."""
        return self.features_for("human")


# Immutable registry — never expose mutable lists of F0
_F0 = FeatureSet(name="F0", groups=("f0",))
_F1A = FeatureSet(name="F1A", groups=("f0", "demand"))
_F1B = FeatureSet(name="F1B", groups=("f0", "human"))
_F1C = FeatureSet(name="F1C", groups=("f0", "demand", "human"))

EXPERIMENTS: dict[str, FeatureSet] = {
    "F0": _F0,
    "F1A": _F1A,
    "F1B": _F1B,
    "F1C": _F1C,
}


def get_experiment(name: str) -> FeatureSet:
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}; known={sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def _panel_origin_col(df: pd.DataFrame) -> str:
    if "origin" in df.columns:
        return "origin"
    if "budget_origin" in df.columns:
        return "budget_origin"
    if "ts_origin" in df.columns:
        return "ts_origin"
    raise ValueError("panel missing origin column")


def enrich_panel(
    df: pd.DataFrame,
    experiment: FeatureSet,
    *,
    sales_hist: pd.DataFrame,
    budget_hist: pd.DataFrame,
    matched_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Copy ``df`` and attach feature groups required by ``experiment``."""
    out = df.copy()
    origin_col = _panel_origin_col(out)
    if "demand" in experiment.groups:
        out = add_demand_features(out, sales_hist, origin_col=origin_col)
    if "human" in experiment.groups:
        out = add_human_features(
            out, budget_hist, matched_hist=matched_hist, origin_col=origin_col
        )
    return out


def enrich_dataset(ds: BenchmarkDataset, experiment: FeatureSet | str) -> BenchmarkDataset:
    """Return a **copy** of the frozen dataset with research features attached.

    Does not write to disk or mutate ``ds`` panels in place.
    """
    if isinstance(experiment, str):
        experiment = get_experiment(experiment)

    if experiment.groups == ("f0",):
        # No enrichment needed; still return copies so callers cannot mutate freeze
        return BenchmarkDataset(
            version=ds.version,
            root=ds.root,
            ts_universe=ds.ts_universe.copy(),
            budget_universe=ds.budget_universe.copy(),
            matched_universe=ds.matched_universe.copy(),
            manifest=ds.manifest,
        )

    sales_hist = load_frozen_sales(ds.root)
    budget_hist = ds.budget_universe
    matched_hist = ds.matched_universe

    return BenchmarkDataset(
        version=ds.version,
        root=ds.root,
        ts_universe=enrich_panel(
            ds.ts_universe,
            experiment,
            sales_hist=sales_hist,
            budget_hist=budget_hist,
            matched_hist=matched_hist,
        ),
        budget_universe=enrich_panel(
            ds.budget_universe,
            experiment,
            sales_hist=sales_hist,
            budget_hist=budget_hist,
            matched_hist=matched_hist,
        ),
        matched_universe=enrich_panel(
            ds.matched_universe,
            experiment,
            sales_hist=sales_hist,
            budget_hist=budget_hist,
            matched_hist=matched_hist,
        ),
        manifest=ds.manifest,
    )


def make_residual_model(anchor: Anchor, feature_cols: Sequence[str]):
    """Build a ``backtest``-compatible residual XGB callable.

    Uses frozen ``fit_xgb`` / ``XGB_PARAMS`` unchanged.
    """
    cols = list(feature_cols)
    if anchor == "ts":
        forecast_col = "ts_forecast"
        name = "ts_xgb_research"
    elif anchor == "human":
        forecast_col = "budget_forecast"
        name = "human_xgb_research"
    else:
        raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")

    def _predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        tr = train_df.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
        missing = [c for c in cols if c not in tr.columns or c not in test_df.columns]
        if missing:
            raise KeyError(f"missing feature columns for {name}: {missing}")
        # Fill NaNs like frozen prep_lags for research extras
        for c in cols:
            if c.startswith("sales_") or c.startswith("human_") or c in {
                "trend_3m",
                "trend_6m",
                "recent_growth",
                "recent_acceleration",
                "historical_actual_budget_ratio",
                "mean_human_adjustment",
                "mean_abs_human_adjustment",
            }:
                tr[c] = tr[c].fillna(0)
                test_df = test_df.copy()
                test_df[c] = test_df[c].fillna(0)
        if "horizon" not in tr.columns:
            raise KeyError("train_df needs horizon for sample weights")
        model = fit_xgb(cols, tr)
        resid = model.predict(test_df[cols])
        return np.maximum(0.0, test_df[forecast_col].astype(float).to_numpy() + resid)

    _predict.__name__ = name
    return _predict


def train_universe_for(anchor: Anchor) -> str:
    return "ts" if anchor == "ts" else "budget"
