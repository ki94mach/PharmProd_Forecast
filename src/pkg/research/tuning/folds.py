"""Pre-PRIMARY rolling inner fold builder for M1 Optuna.

Temporal rules match the benchmark anti-leakage contract:
  TRAIN:      target_date < V
  VALIDATION: origin == V, valid realized sales
  ASSERT:     train.target_date.max() < V

No sklearn KFold. No shuffling. No PRIMARY origin ever appears.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pkg.benchmark.dataset import prep_lags
from pkg.research.tuning.config import (
    INNER_MIN_BUDGET_VINTAGES,
    INNER_MIN_HISTORY_MONTHS,
    INNER_MIN_TRAIN_ROWS,
    INNER_ORIGIN_COL,
    MIN_INNER_FOLDS,
    PRE_PRIMARY_CUTOFF,
)


@dataclass
class InnerFold:
    """One rolling validation fold."""

    origin: int
    train: pd.DataFrame
    val: pd.DataFrame


def discover_pre_primary_origins(universe: pd.DataFrame, anchor: str) -> list[int]:
    """Sorted pre-PRIMARY unique origins for one anchor universe."""
    origin_col = INNER_ORIGIN_COL[anchor]
    if origin_col not in universe.columns:
        raise KeyError(
            f"Universe missing origin column {origin_col!r} for anchor={anchor!r}. "
            f"Available: {list(universe.columns)}"
        )
    all_origins = (
        universe[origin_col]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(o for o in all_origins if o < PRE_PRIMARY_CUTOFF)


def _ts_fold_eligible(train: pd.DataFrame) -> bool:
    """TS eligibility: rows >= 500 and unique months >= 12."""
    if len(train) < INNER_MIN_TRAIN_ROWS:
        return False
    months = int(train["target_date"].nunique())
    return months >= INNER_MIN_HISTORY_MONTHS


def _human_fold_eligible(train: pd.DataFrame) -> bool:
    """Human eligibility: rows >= 500, unique months >= 12, vintages >= 4."""
    if len(train) < INNER_MIN_TRAIN_ROWS:
        return False
    months = int(train["target_date"].nunique())
    if months < INNER_MIN_HISTORY_MONTHS:
        return False
    vintages = int(train["budget_origin"].nunique()) if "budget_origin" in train.columns else 0
    return vintages >= INNER_MIN_BUDGET_VINTAGES


def build_inner_folds(
    universe: pd.DataFrame,
    anchor: str,
    *,
    prepped: bool = False,
) -> list[InnerFold]:
    """Build rolling temporal folds from pre-PRIMARY origins.

    Returns only eligible folds (see eligibility rules for each anchor).
    Raises InsufficientFoldsError if < MIN_INNER_FOLDS usable folds.
    """
    if not prepped:
        universe = prep_lags(universe)

    origin_col = INNER_ORIGIN_COL[anchor]
    candidate_origins = discover_pre_primary_origins(universe, anchor)

    # No PRIMARY origins must leak into inner tuning
    for o in candidate_origins:
        assert o < PRE_PRIMARY_CUTOFF, (
            f"BUG: origin {o} >= PRE_PRIMARY_CUTOFF {PRE_PRIMARY_CUTOFF}"
        )

    folds: list[InnerFold] = []
    for V in candidate_origins:
        # Strict temporal split: train rows where target_date < V
        train = universe.loc[universe["target_date"].astype(int) < V].copy()
        # Validation: rows at this origin with valid realized sales
        val = universe.loc[
            (universe[origin_col].astype(int) == V) & universe["sales"].notna()
        ].copy()

        if val.empty:
            continue

        # Anti-leakage assertion
        if not train.empty:
            assert int(train["target_date"].max()) < V, (
                f"Leakage detected: train.target_date.max()={train['target_date'].max()} "
                f">= validation origin V={V}"
            )

        # Eligibility check
        if anchor == "ts":
            if not _ts_fold_eligible(train):
                continue
        else:
            if not _human_fold_eligible(train):
                continue

        folds.append(InnerFold(origin=V, train=train, val=val))

    if len(folds) < MIN_INNER_FOLDS:
        raise InsufficientFoldsError(
            f"Anchor '{anchor}': only {len(folds)} eligible inner folds "
            f"(need >= {MIN_INNER_FOLDS}). "
            f"Candidate origins: {candidate_origins}. "
            "Cannot run Optuna study. Documenting limitation — not weakening temporal rules."
        )

    return folds


class InsufficientFoldsError(RuntimeError):
    """Raised when an anchor has fewer than MIN_INNER_FOLDS eligible pre-PRIMARY folds."""
