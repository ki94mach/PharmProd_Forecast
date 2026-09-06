"""Chronological internal train/validation split for early stopping.

The outer historical CV evaluation window (``forecast_origin`` … horizon)
must never appear in these supervised windows: callers build windows only from
``date < forecast_origin`` history first, then split.
"""
from __future__ import annotations

import math
from typing import Optional

from pkg.ts_v3a.types import FoldSplit, WindowDataset
from pkg.ts_v3a.windows import InsufficientHistoryError, assert_targets_before_origin


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
    min_train_windows: int = 1,
) -> FoldSplit:
    """Split supervised windows chronologically for early stopping.

    The last ``validation_fraction`` of samples become validation; earlier
    samples become train. With a single sample, validation is empty (train-only)
    so early stopping is skipped until more history exists.
    """
    n = dataset.n_samples
    if n < min_train_windows:
        raise InsufficientHistoryError(
            f"Need at least {min_train_windows} supervised window(s) after split; got {n}",
            n_history=n,
            lookback=dataset.lookback,
            horizon=dataset.horizon,
            mode=dataset.mode,
            n_samples=n,
        )

    if n == 1:
        train = _slice_dataset(dataset, 0, 1)
        return FoldSplit(train=train, validation=None, n_train=1, n_validation=0)

    n_val = max(1, int(math.floor(n * float(validation_fraction))))
    # Keep at least min_train_windows in train when possible.
    if n - n_val < min_train_windows:
        n_val = max(0, n - min_train_windows)
    if n_val < 1:
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
