"""Fold-local scaling for V3A neural training.

The scaler is fitted only on values inside the internal-train supervised
windows for the current historical fold. Never fit on full SKU history, the
outer CV test horizon, future observations, or (by default) validation windows.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler

from pkg.ts_v3a.types import FoldSplit, WindowDataset


class FoldScaler:
    """StandardScaler wrapper with explicit fold-train-only fit semantics."""

    def __init__(self, method: str = "standard") -> None:
        if method != "standard":
            raise ValueError(
                f"Unsupported scaling_method {method!r}; only 'standard' is implemented"
            )
        self.method = method
        self._scaler = StandardScaler()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_on_train_windows(self, train: WindowDataset) -> "FoldScaler":
        """Fit using flattened values from train ``X`` and ``y`` only."""
        if train.n_samples < 1:
            raise ValueError("Cannot fit FoldScaler on empty train windows")
        flat = np.concatenate(
            [train.X.reshape(-1), train.y.reshape(-1)]
        ).reshape(-1, 1)
        self._scaler.fit(flat)
        self._fitted = True
        return self

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
            "n_samples_seen": int(self._scaler.n_samples_seen_),
        }

    def _transform_array(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = np.asarray(arr, dtype=float).reshape(-1, 1)
        return self._scaler.transform(flat).reshape(shape)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FoldScaler has not been fitted")


def fit_fold_scaler(train: WindowDataset, *, method: str = "standard") -> FoldScaler:
    """Convenience: construct and fit a :class:`FoldScaler` on train windows."""
    return FoldScaler(method=method).fit_on_train_windows(train)
