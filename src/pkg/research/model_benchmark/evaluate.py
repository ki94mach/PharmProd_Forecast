"""Rolling-origin residual backtest for M2 model-class comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.dataset import BenchmarkDataset, horizon_bucket, prep_lags
from pkg.benchmark.evaluate import BacktestResult, _fold_eligible_primary, metrics_block
from pkg.research.model_benchmark.config import (
    ANCHOR_FORECAST_COL,
    FEATURES_BY_ANCHOR,
    ORIGIN_COL,
    PRIMARY_ORIGINS_LOCKED,
    TRAIN_UNIVERSE_BY_ANCHOR,
)
from pkg.research.model_benchmark.models import ResidualLearner, all_learners
from pkg.research.tuning.folds import _human_fold_eligible, _ts_fold_eligible

SliceKind = Literal["broad", "matched_primary"]


@dataclass
class FoldSpec:
    origin: int
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class SuiteResult:
    anchor: str
    slice_kind: SliceKind
    results: dict[str, BacktestResult] = field(default_factory=dict)
    pooled_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    origins_used: list[int] = field(default_factory=list)
    fold_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _train_eligible(anchor: str, train: pd.DataFrame) -> bool:
    if anchor == "ts":
        return _ts_fold_eligible(train)
    return _human_fold_eligible(train)


def _get_panel(ds: BenchmarkDataset, anchor: str, slice_kind: SliceKind) -> pd.DataFrame:
    if slice_kind == "matched_primary":
        return prep_lags(ds.matched_universe)
    if anchor == "ts":
        return prep_lags(ds.ts_universe)
    return prep_lags(ds.budget_universe)


def _get_train_panel(ds: BenchmarkDataset, anchor: str, slice_kind: SliceKind) -> pd.DataFrame:
    key = TRAIN_UNIVERSE_BY_ANCHOR[anchor]
    if key == "ts":
        return prep_lags(ds.ts_universe)
    return prep_lags(ds.budget_universe)


def discover_eligible_origins(
    ds: BenchmarkDataset,
    anchor: str,
    *,
    slice_kind: SliceKind = "broad",
    origin_filter: Optional[Sequence[int]] = None,
) -> list[int]:
    """Discover rolling origins satisfying maturity and non-empty test rules."""
    test_panel = _get_panel(ds, anchor, slice_kind)
    train_panel = (
        _get_train_panel(ds, anchor, slice_kind)
        if slice_kind == "matched_primary"
        else test_panel
    )
    bud_panel = prep_lags(ds.budget_universe)

    if slice_kind == "matched_primary":
        origin_col = ORIGIN_COL["matched"]
        candidates = sorted(test_panel[origin_col].astype(int).unique().tolist())
        candidates = [o for o in candidates if o in PRIMARY_ORIGINS_LOCKED]
    else:
        origin_col = ORIGIN_COL[anchor]
        candidates = sorted(test_panel[origin_col].dropna().astype(int).unique().tolist())

    if origin_filter is not None:
        filt = {int(o) for o in origin_filter}
        candidates = [o for o in candidates if o in filt]

    eligible: list[int] = []
    for o in candidates:
        train = train_panel.loc[train_panel["target_date"].astype(int) < o].copy()
        test = test_panel.loc[
            (test_panel[origin_col].astype(int) == o) & test_panel["sales"].notna()
        ].copy()
        if test.empty or train.empty:
            continue
        if int(train["target_date"].max()) >= o:
            continue
        if slice_kind == "matched_primary":
            train_bud = bud_panel.loc[bud_panel["target_date"].astype(int) < o].copy()
            if not _fold_eligible_primary(train_bud):
                continue
        elif not _train_eligible(anchor, train):
            continue
        eligible.append(int(o))
    return eligible


def build_folds(
    ds: BenchmarkDataset,
    anchor: str,
    origins: Sequence[int],
    *,
    slice_kind: SliceKind = "broad",
) -> list[FoldSpec]:
    test_panel = _get_panel(ds, anchor, slice_kind)
    train_panel = (
        _get_train_panel(ds, anchor, slice_kind)
        if slice_kind == "matched_primary"
        else test_panel
    )
    bud_panel = prep_lags(ds.budget_universe)

    if slice_kind == "matched_primary":
        origin_col = ORIGIN_COL["matched"]
    else:
        origin_col = ORIGIN_COL[anchor]

    folds: list[FoldSpec] = []
    for o in sorted(int(x) for x in origins):
        train = train_panel.loc[train_panel["target_date"].astype(int) < o].copy()
        test = test_panel.loc[
            (test_panel[origin_col].astype(int) == o) & test_panel["sales"].notna()
        ].copy()
        if test.empty or train.empty:
            raise RuntimeError(f"Empty fold at origin={o} anchor={anchor} slice={slice_kind}")
        assert int(train["target_date"].max()) < o, (
            f"leakage: train.target_date.max={train['target_date'].max()} >= origin={o}"
        )
        if slice_kind == "matched_primary":
            train_bud = bud_panel.loc[bud_panel["target_date"].astype(int) < o].copy()
            if not _fold_eligible_primary(train_bud):
                raise RuntimeError(f"PRIMARY ineligible at origin={o}")
        elif not _train_eligible(anchor, train):
            raise RuntimeError(f"Train ineligible at origin={o} anchor={anchor}")
        folds.append(FoldSpec(origin=o, train=train, test=test))
    return folds


def _predictions_to_result(preds: pd.DataFrame, model_name: str) -> BacktestResult:
    if preds.empty:
        raise ValueError(f"No predictions for model={model_name!r}")
    if "horizon_bucket" not in preds.columns:
        preds = preds.copy()
        preds["horizon_bucket"] = preds["horizon"].map(horizon_bucket)

    overall = pd.DataFrame(
        [metrics_block(preds["actual"], preds["prediction"], model_name)]
    )
    by_origin_rows = []
    for o, g in preds.groupby("test_origin"):
        m = metrics_block(g["actual"], g["prediction"], model_name)
        m["origin"] = int(o)
        by_origin_rows.append(m)
    by_horizon_rows = []
    for h, g in preds.groupby("horizon"):
        m = metrics_block(g["actual"], g["prediction"], model_name)
        m["horizon"] = int(h)
        by_horizon_rows.append(m)

    return BacktestResult(
        model_name=model_name,
        overall=overall,
        by_origin=pd.DataFrame(by_origin_rows),
        by_horizon=pd.DataFrame(by_horizon_rows),
        predictions=preds,
        origins=sorted(preds["test_origin"].astype(int).unique().tolist()),
        universe=TRAIN_UNIVERSE_BY_ANCHOR.get(model_name, "custom"),
    )


def _assert_fairness_across_models(fold_preds: dict[str, pd.DataFrame], origin: int) -> None:
    keys = ("product", "qrt", "target_date")
    ref = None
    ref_actual = None
    ref_anchor = None
    for model_name, fold in fold_preds.items():
        k = fold[list(keys)].reset_index(drop=True)
        actual = fold["actual"].reset_index(drop=True)
        anchor = fold["anchor"].reset_index(drop=True)
        if ref is None:
            ref = k
            ref_actual = actual
            ref_anchor = anchor
            continue
        if not ref.equals(k):
            raise AssertionError(
                f"Test keys differ at origin={origin} for model={model_name}"
            )
        if not ref_actual.equals(actual):
            raise AssertionError(f"Actual values differ at origin={origin} model={model_name}")
        if not ref_anchor.equals(anchor):
            raise AssertionError(f"Anchor values differ at origin={origin} model={model_name}")


def rolling_residual_backtest(
    ds: BenchmarkDataset,
    anchor: str,
    learners: Sequence[ResidualLearner],
    origins: Sequence[int],
    *,
    slice_kind: SliceKind = "broad",
) -> SuiteResult:
    """Run all learners on identical rolling folds."""
    anchor_col = ANCHOR_FORECAST_COL[anchor]
    features = FEATURES_BY_ANCHOR[anchor]
    folds = build_folds(ds, anchor, origins, slice_kind=slice_kind)

    parts_by_model: dict[str, list[pd.DataFrame]] = {l.name: [] for l in learners}
    fold_rows = []

    for fold in folds:
        o = fold.origin
        fold_preds: dict[str, pd.DataFrame] = {}
        for learner in learners:
            preds = learner.fit_predict(
                fold.train, fold.test, anchor_col=anchor_col, features=features
            )
            if len(preds) != len(fold.test):
                raise ValueError(
                    f"{learner.name} returned {len(preds)} preds for {len(fold.test)} rows "
                    f"at origin={o}"
                )
            out = fold.test.copy()
            out["prediction"] = preds
            out["actual"] = out["sales"].astype(float)
            out["anchor"] = out[anchor_col].astype(float)
            out["test_origin"] = int(o)
            out["model"] = learner.name
            out["anchor_name"] = anchor
            out["slice"] = slice_kind
            fold_preds[learner.name] = out
            parts_by_model[learner.name].append(out)

        _assert_fairness_across_models(fold_preds, o)
        fold_rows.append(
            {
                "origin": int(o),
                "anchor": anchor,
                "slice": slice_kind,
                "test_rows": len(fold.test),
                "test_products": int(fold.test["product"].nunique()),
                "train_rows": len(fold.train),
            }
        )

    results: dict[str, BacktestResult] = {}
    pooled_parts = []
    for learner in learners:
        preds = pd.concat(parts_by_model[learner.name], ignore_index=True)
        results[learner.name] = _predictions_to_result(preds, learner.name)
        pooled_parts.append(preds)

    return SuiteResult(
        anchor=anchor,
        slice_kind=slice_kind,
        results=results,
        pooled_predictions=pd.concat(pooled_parts, ignore_index=True),
        origins_used=sorted(int(o) for o in origins),
        fold_diagnostics=pd.DataFrame(fold_rows),
    )


def run_benchmark_suite(
    ds: BenchmarkDataset,
    anchor: str,
    *,
    slice_kind: SliceKind = "broad",
    origin_filter: Optional[Sequence[int]] = None,
    learners: Optional[Sequence[ResidualLearner]] = None,
) -> SuiteResult:
    origins = discover_eligible_origins(
        ds, anchor, slice_kind=slice_kind, origin_filter=origin_filter
    )
    if not origins:
        raise RuntimeError(f"No eligible origins for anchor={anchor} slice={slice_kind}")
    return rolling_residual_backtest(
        ds,
        anchor,
        learners or all_learners(),
        origins,
        slice_kind=slice_kind,
    )


def filter_predictions_to_origins(preds: pd.DataFrame, origins: Sequence[int]) -> pd.DataFrame:
    oset = {int(o) for o in origins}
    return preds.loc[preds["test_origin"].astype(int).isin(oset)].copy()
