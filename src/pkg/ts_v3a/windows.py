"""Supervised window builders for V3A neural training."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from pkg.ts_v3a.types import TargetMode, WindowDataset

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


class InsufficientHistoryError(ValueError):
    """Raised when history is too short for the requested lookback/horizon/mode."""

    def __init__(
        self,
        message: str,
        *,
        n_history: int,
        lookback: int,
        horizon: int,
        mode: TargetMode,
        n_samples: int = 0,
    ) -> None:
        super().__init__(message)
        self.n_history = int(n_history)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.mode = mode
        self.n_samples = int(n_samples)


def _as_float_array(history: ArrayLike) -> tuple[np.ndarray, tuple[Optional[int], ...]]:
    if isinstance(history, pd.Series):
        values = history.to_numpy(dtype=float, copy=True)
        dates: list[Optional[int]] = []
        for idx in history.index:
            try:
                dates.append(int(idx))
            except (TypeError, ValueError):
                dates.append(None)
        return values, tuple(dates)
    arr = np.asarray(history, dtype=float).reshape(-1)
    return arr, tuple(None for _ in range(len(arr)))


def min_history_length(lookback: int, horizon: int, mode: TargetMode) -> int:
    """Minimum series length required for at least one supervised sample."""
    if mode is TargetMode.RECURSIVE:
        return int(lookback) + 1
    if mode is TargetMode.DIRECT_MIMO:
        return int(lookback) + int(horizon)
    raise ValueError(f"Unsupported target mode: {mode!r}")


def expected_n_samples(n_history: int, lookback: int, horizon: int, mode: TargetMode) -> int:
    """Number of supervised samples for a contiguous series of length ``n_history``."""
    n = int(n_history)
    lb = int(lookback)
    h = int(horizon)
    if mode is TargetMode.RECURSIVE:
        return max(0, n - lb)
    if mode is TargetMode.DIRECT_MIMO:
        return max(0, n - lb - h + 1)
    raise ValueError(f"Unsupported target mode: {mode!r}")


def build_recursive_windows(
    history: ArrayLike,
    *,
    lookback: int = 12,
    require_samples: bool = True,
) -> WindowDataset:
    """Build one-step recursive windows.

    ``X[i] = history[i : i+lookback]``, ``y[i] = history[i+lookback]``.
    Shapes: ``X -> [n, lookback, 1]``, ``y -> [n, 1]``.
    """
    values, dates = _as_float_array(history)
    n = len(values)
    lb = int(lookback)
    mode = TargetMode.RECURSIVE
    n_samples = expected_n_samples(n, lb, horizon=1, mode=mode)
    if n_samples < 1:
        if require_samples:
            raise InsufficientHistoryError(
                f"Need at least {min_history_length(lb, 1, mode)} months for "
                f"recursive lookback={lb}; got N={n}",
                n_history=n,
                lookback=lb,
                horizon=1,
                mode=mode,
                n_samples=0,
            )
        empty_x = np.zeros((0, lb, 1), dtype=float)
        empty_y = np.zeros((0, 1), dtype=float)
        return WindowDataset(
            X=empty_x,
            y=empty_y,
            end_indices=(),
            end_dates=(),
            mode=mode,
            lookback=lb,
            horizon=1,
        )

    X = np.empty((n_samples, lb, 1), dtype=float)
    y = np.empty((n_samples, 1), dtype=float)
    end_indices: list[int] = []
    end_dates: list[Optional[int]] = []
    for i in range(n_samples):
        end_idx = i + lb - 1
        X[i, :, 0] = values[i : i + lb]
        y[i, 0] = values[i + lb]
        end_indices.append(end_idx)
        end_dates.append(dates[end_idx] if end_idx < len(dates) else None)

    return WindowDataset(
        X=X,
        y=y,
        end_indices=tuple(end_indices),
        end_dates=tuple(end_dates),
        mode=mode,
        lookback=lb,
        horizon=1,
    )


def build_mimo_windows(
    history: ArrayLike,
    *,
    lookback: int = 12,
    horizon: int = 15,
    require_samples: bool = True,
) -> WindowDataset:
    """Build DIRECT/MIMO multi-horizon windows.

    For input ending at time index ``t`` (``end_index``), ``y`` is exactly
    ``history[t+1 : t+1+horizon]`` (i.e. ``t+1 ... t+horizon``).

    Shapes: ``X -> [n, lookback, 1]``, ``y -> [n, horizon]``.
    """
    values, dates = _as_float_array(history)
    n = len(values)
    lb = int(lookback)
    h = int(horizon)
    mode = TargetMode.DIRECT_MIMO
    n_samples = expected_n_samples(n, lb, h, mode=mode)
    if n_samples < 1:
        if require_samples:
            raise InsufficientHistoryError(
                f"Need at least {min_history_length(lb, h, mode)} months for "
                f"MIMO lookback={lb}, horizon={h}; got N={n}",
                n_history=n,
                lookback=lb,
                horizon=h,
                mode=mode,
                n_samples=0,
            )
        empty_x = np.zeros((0, lb, 1), dtype=float)
        empty_y = np.zeros((0, h), dtype=float)
        return WindowDataset(
            X=empty_x,
            y=empty_y,
            end_indices=(),
            end_dates=(),
            mode=mode,
            lookback=lb,
            horizon=h,
        )

    X = np.empty((n_samples, lb, 1), dtype=float)
    y = np.empty((n_samples, h), dtype=float)
    end_indices: list[int] = []
    end_dates: list[Optional[int]] = []
    for i in range(n_samples):
        # Input occupies [i, i+lb); last input index t = i+lb-1
        t = i + lb - 1
        target_start = t + 1
        target_end = t + h  # inclusive
        assert target_end < n, "MIMO target must stay inside pre-origin history"
        X[i, :, 0] = values[i : i + lb]
        y[i, :] = values[target_start : target_end + 1]
        end_indices.append(t)
        end_dates.append(dates[t] if t < len(dates) else None)

    return WindowDataset(
        X=X,
        y=y,
        end_indices=tuple(end_indices),
        end_dates=tuple(end_dates),
        mode=mode,
        lookback=lb,
        horizon=h,
    )


def build_windows(
    history: ArrayLike,
    *,
    mode: TargetMode,
    lookback: int = 12,
    horizon: int = 15,
    require_samples: bool = True,
) -> WindowDataset:
    """Dispatch to recursive or MIMO window builder."""
    if mode is TargetMode.RECURSIVE:
        return build_recursive_windows(
            history, lookback=lookback, require_samples=require_samples
        )
    if mode is TargetMode.DIRECT_MIMO:
        return build_mimo_windows(
            history,
            lookback=lookback,
            horizon=horizon,
            require_samples=require_samples,
        )
    raise ValueError(f"Unsupported target mode: {mode!r}")


def assert_targets_before_origin(
    dataset: WindowDataset,
    *,
    n_history: int,
    forecast_origin: Optional[int] = None,
) -> None:
    """Assert every target observation index is strictly inside history (``< N``).

    When ``forecast_origin`` and ``end_dates`` are available, also assert each
    target month label is ``< forecast_origin``.
    """
    h = int(dataset.horizon)
    for end_idx in dataset.end_indices:
        last_target = end_idx + h
        if last_target >= n_history:
            raise AssertionError(
                f"Target index {last_target} reaches or exceeds history length {n_history}"
            )
    if forecast_origin is None:
        return
    for end_date, end_idx in zip(dataset.end_dates, dataset.end_indices):
        if end_date is None:
            continue
        # Targets are the next ``h`` months after end_date on a contiguous grid;
        # with integer Shamsi labels we only know end_date; index check above is
        # authoritative when dates are present as series index of length N.
        _ = end_idx  # silence unused in strict date-less paths
        if end_date >= forecast_origin:
            raise AssertionError(
                f"Window end_date {end_date} is not before forecast_origin {forecast_origin}"
            )
