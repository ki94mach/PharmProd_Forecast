"""Chronological internal train/validation split for early stopping.

The outer historical CV evaluation window (``forecast_origin`` … horizon)
must never appear in these supervised windows: callers build windows only from
``date < forecast_origin`` history first, then split.

A split with empty validation may still be returned for diagnostics, but
:mod:`pkg.ts_v3a.eligibility` marks those samples ineligible for NeuralTrainer
fitting (no train-only model fitting path).
"""
from __future__ import annotations

import math
from typing import Optional

from pkg.ts_v3a.types import FoldSplit, WindowDataset
from pkg.ts_v3a.windows import assert_targets_before_origin


def _slice_dataset(dataset: WindowDataset, start: int, end: int) -> WindowDataset:
    """Slice samples ``[start:end)`` into a new :class:`WindowDataset`."""
    return WindowDataset(
        X=dataset.X[start:end].copy(),
        y=dataset.y[start:end].copy(),
        end_indices=dataset.end_indices[start:end],
        end_dates=dataset.end_dates[start:end],
        mode=dataset.mode,
        lookback=dataset.lookback,
        horizon=dataset.horizon,
    )


def chronological_train_val_split(
    dataset: WindowDataset,
    *,
    validation_fraction: float = 0.2,
    min_internal_train_windows: int = 8,
    min_internal_validation_windows: int = 2,
) -> FoldSplit:
    """Split supervised windows chronologically for early stopping.

    The last ``validation_fraction`` of samples become validation (at least
    ``min_internal_validation_windows`` when enough samples exist); earlier
    samples become train (prefer retaining at least ``min_internal_train_windows``).

    When there are too few windows to populate both sides, validation may be
    empty or short — callers must consult eligibility before training.
    """
    n = dataset.n_samples
    if n < 1:
        empty = _slice_dataset(dataset, 0, 0)
        return FoldSplit(train=empty, validation=None, n_train=0, n_validation=0)

    # Target validation size from fraction, then clamp to eligibility mins when possible.
    n_val = max(1, int(math.floor(n * float(validation_fraction))))
    n_val = max(n_val, int(min_internal_validation_windows))

    # Prefer keeping min train windows when the series is long enough for both.
    needed = int(min_internal_train_windows) + int(min_internal_validation_windows)
    if n >= needed:
        n_val = max(int(min_internal_validation_windows), n_val)
        if n - n_val < int(min_internal_train_windows):
            n_val = n - int(min_internal_train_windows)
    else:
        # Not enough for eligibility; still produce a diagnostic split.
        # Prefer giving whatever remains after one train window to validation,
        # but never invent validation when n == 1.
        if n == 1:
            train = _slice_dataset(dataset, 0, 1)
            return FoldSplit(train=train, validation=None, n_train=1, n_validation=0)
        n_val = min(n_val, n - 1)
        n_val = max(1, n_val)

    if n_val < 1 or n_val >= n:
        train = _slice_dataset(dataset, 0, n)
        return FoldSplit(train=train, validation=None, n_train=n, n_validation=0)

    split_at = n - n_val
    train = _slice_dataset(dataset, 0, split_at)
    validation = _slice_dataset(dataset, split_at, n)
    return FoldSplit(
        train=train,
        validation=validation,
        n_train=train.n_samples,
        n_validation=validation.n_samples,
    )


def assert_split_before_origin(
    split: FoldSplit,
    *,
    n_history: int,
    forecast_origin: Optional[int] = None,
) -> None:
    """Assert train/val target indices stay inside pre-origin history."""
    assert_targets_before_origin(
        split.train, n_history=n_history, forecast_origin=forecast_origin
    )
    if split.validation is not None:
        assert_targets_before_origin(
            split.validation, n_history=n_history, forecast_origin=forecast_origin
        )
