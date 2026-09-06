"""Shared types for the V3A neural forecasting experiment package."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np


class TargetMode(str, Enum):
    """Supervised target layout for neural windows."""

    RECURSIVE = "recursive"
    DIRECT_MIMO = "direct_mimo"


@dataclass(frozen=True)
class WindowDataset:
    """Supervised windows built from pre-origin history only.

    Attributes:
        X: Shape ``[n_samples, lookback, 1]``.
        y: Shape ``[n_samples, 1]`` (recursive) or ``[n_samples, horizon]`` (MIMO).
        end_indices: Inclusive series index of the last lookback month for each
            sample (so targets start at ``end_index + 1``).
        end_dates: Optional Shamsi YYYYMM labels aligned with ``end_indices``
            when the input series carried a date index.
        mode: Target layout.
        lookback: Lookback length used to build ``X``.
        horizon: Target horizon (1 for recursive builders; ``H`` for MIMO).
    """

    X: np.ndarray
    y: np.ndarray
    end_indices: tuple[int, ...]
    end_dates: tuple[Optional[int], ...]
    mode: TargetMode
    lookback: int
    horizon: int

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class FoldSplit:
    """Chronological internal train/validation partition of supervised windows.

    Validation windows are used only for early stopping. They never include the
    outer historical CV evaluation horizon (``forecast_origin`` … ``+H-1``).
    """

    train: WindowDataset
    validation: Optional[WindowDataset]
    n_train: int
    n_validation: int


@dataclass(frozen=True)
class TrainMetadata:
    """Training run metadata for one SKU / origin / architecture / seed."""

    architecture: str
    parameters: Mapping[str, Any]
    random_seed: int
    n_train_samples: int
    n_validation_samples: int
    train_window_count: int
    validation_window_count: int
    epochs_ran: Optional[int]
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    parameter_count: Optional[int]
    scaler_params: Mapping[str, Any]
    training_start: Optional[int]
    training_end: Optional[int]
    forecast_origin: int

    # Backward-compatible aliases used by earlier foundation helpers.
    @property
    def epochs_trained(self) -> Optional[int]:
        return self.epochs_ran

    @property
    def best_validation_loss(self) -> Optional[float]:
        return self.best_val_loss


def build_train_metadata(
    *,
    architecture: str,
    parameters: Mapping[str, Any],
    random_seed: int,
    n_train_samples: int,
    n_validation_samples: int,
    forecast_origin: int,
    scaler_params: Optional[Mapping[str, Any]] = None,
    training_start: Optional[int] = None,
    training_end: Optional[int] = None,
    epochs_ran: Optional[int] = None,
    best_epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    parameter_count: Optional[int] = None,
    train_window_count: Optional[int] = None,
    validation_window_count: Optional[int] = None,
    # Deprecated aliases
    epochs_trained: Optional[int] = None,
    best_validation_loss: Optional[float] = None,
) -> TrainMetadata:
    """Assemble :class:`TrainMetadata`."""
    epochs = epochs_ran if epochs_ran is not None else epochs_trained
    best_loss = best_val_loss if best_val_loss is not None else best_validation_loss
    return TrainMetadata(
        architecture=architecture,
        parameters=dict(parameters),
        random_seed=int(random_seed),
        n_train_samples=int(n_train_samples),
        n_validation_samples=int(n_validation_samples),
        train_window_count=int(
            train_window_count if train_window_count is not None else n_train_samples
        ),
        validation_window_count=int(
            validation_window_count
            if validation_window_count is not None
            else n_validation_samples
        ),
        epochs_ran=epochs,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        parameter_count=parameter_count,
        scaler_params=dict(scaler_params or {}),
        training_start=training_start,
        training_end=training_end,
        forecast_origin=int(forecast_origin),
    )


@dataclass(frozen=True)
class PreparedNeuralFold:
    """Scaled train/val windows plus metadata for one historical fold."""

    train_X: np.ndarray
    train_y: np.ndarray
    val_X: Optional[np.ndarray]
    val_y: Optional[np.ndarray]
    metadata: TrainMetadata
    eligible_for_training: bool
    eligibility_reason: Optional[str] = None
    end_indices_train: tuple[int, ...] = field(default_factory=tuple)
    end_indices_val: tuple[int, ...] = field(default_factory=tuple)
