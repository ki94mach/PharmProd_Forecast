"""Fold-local scaling for V3A neural training.

The scaler is fitted **exactly once** on the unique chronological raw
observations belonging to the internal training period (series indices covered
by train windows). Overlapping windows do not re-weight observations.

Never fit on:
- the outer CV test horizon
- internal validation-only observations
- future observations

``fit_on_series`` supports a later full pre-origin refit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pkg.ts_v3a.types import FoldSplit, WindowDataset

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _as_1d(history: ArrayLike) -> np.ndarray:
    if isinstance(history, pd.Series):
        return history.to_numpy(dtype=float, copy=True)
    return np.asarray(history, dtype=float).reshape(-1)


def unique_observation_indices(dataset: WindowDataset) -> np.ndarray:
    """Sorted unique series indices covered by ``dataset`` X and y windows."""
    if dataset.n_samples < 1:
        return np.zeros(0, dtype=int)
    lb = int(dataset.lookback)
    h = int(dataset.horizon)
    indices: set[int] = set()
    for end_idx in dataset.end_indices:
        t = int(end_idx)
        start = t - lb + 1
        indices.update(range(start, t + 1))  # X
        indices.update(range(t + 1, t + h + 1))  # y
    return np.array(sorted(indices), dtype=int)


class FoldScaler:
    """Univariate StandardScaler with unique-observation fold-train fit."""

    def __init__(self, method: str = "standard") -> None:
        if method != "standard":
            raise ValueError(
                f"Unsupported scaling_method {method!r}; only 'standard' is implemented"
            )
        self.method = method
        self._scaler = StandardScaler()
        self._fitted = False
        self._fit_indices: tuple[int, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fit_indices(self) -> tuple[int, ...]:
        """Series indices used for the last fit (unique train or full-series)."""
        return self._fit_indices

    def fit_on_unique_train_observations(
        self,
        history: ArrayLike,
        train: WindowDataset,
    ) -> "FoldScaler":
        """Fit once on unique raw values at indices covered by ``train`` windows."""
        if train.n_samples < 1:
            raise ValueError("Cannot fit FoldScaler on empty train windows")
        values = _as_1d(history)
        idxs = unique_observation_indices(train)
        if idxs.size < 1:
            raise ValueError("Train windows cover no observation indices")
        if int(idxs.max()) >= len(values) or int(idxs.min()) < 0:
            raise ValueError(
                f"Train observation indices {idxs.min()}..{idxs.max()} "
                f"outside history length {len(values)}"
            )
        self._scaler.fit(values[idxs].reshape(-1, 1))
        self._fit_indices = tuple(int(i) for i in idxs.tolist())
        self._fitted = True
        return self

    def fit_on_series(self, history: ArrayLike) -> "FoldScaler":
        """Fit on all values in ``history`` (for final full pre-origin refit)."""
        values = _as_1d(history)
        if values.size < 1:
            raise ValueError("Cannot fit FoldScaler on empty series")
        self._scaler.fit(values.reshape(-1, 1))
        self._fit_indices = tuple(range(len(values)))
        self._fitted = True
        return self

    def fit_on_train_windows(
        self,
        train: WindowDataset,
        *,
        history: ArrayLike,
    ) -> "FoldScaler":
        """Backward-compatible alias requiring ``history`` for unique-obs fit."""
        return self.fit_on_unique_train_observations(history, train)

    def transform_windows(self, dataset: WindowDataset) -> WindowDataset:
        """Return a copy of ``dataset`` with ``X`` and ``y`` scaled."""
        self._ensure_fitted()
        X = self._transform_array(dataset.X)
        y = self._transform_array(dataset.y)
        return WindowDataset(
            X=X,
            y=y,
            end_indices=dataset.end_indices,
            end_dates=dataset.end_dates,
            mode=dataset.mode,
            lookback=dataset.lookback,
            horizon=dataset.horizon,
        )

    def transform_split(self, split: FoldSplit) -> FoldSplit:
        """Scale train and (if present) validation windows with the fitted scaler."""
        train = self.transform_windows(split.train)
        validation = (
            self.transform_windows(split.validation)
            if split.validation is not None
            else None
        )
        return FoldSplit(
            train=train,
            validation=validation,
            n_train=split.n_train,
            n_validation=split.n_validation,
        )

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """Scale an arbitrary array with the fitted univariate scaler."""
        return self._transform_array(arr)

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse-transform predictions / targets before evaluation."""
        self._ensure_fitted()
        arr = np.asarray(y, dtype=float)
        shape = arr.shape
        flat = arr.reshape(-1, 1)
        inv = self._scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def params(self) -> Mapping[str, Any]:
        """Serializable scaler parameters for metadata."""
        self._ensure_fitted()
        mean = self._scaler.mean_
        scale = self._scaler.scale_
        return {
            "method": self.method,
            "mean": float(mean[0]) if mean is not None else None,
            "scale": float(scale[0]) if scale is not None else None,
            "var": float(self._scaler.var_[0]) if self._scaler.var_ is not None else None,
            "n_observations_fit": int(self._scaler.n_samples_seen_),
            "fit_indices": list(self._fit_indices),
        }

    def _transform_array(self, arr: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        shape = arr.shape
        flat = np.asarray(arr, dtype=float).reshape(-1, 1)
        return self._scaler.transform(flat).reshape(shape)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FoldScaler has not been fitted")


def fit_fold_scaler(
    history: ArrayLike,
    train: WindowDataset,
    *,
    method: str = "standard",
) -> FoldScaler:
    """Fit a :class:`FoldScaler` on unique train-period observations in ``history``."""
    return FoldScaler(method=method).fit_on_unique_train_observations(history, train)
