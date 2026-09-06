"""Prepare a scaled train/val fold from pre-origin history (no model training)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.ts_v3a.architectures import target_mode_for
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig, config_parameters_snapshot
from pkg.ts_v3a.scaling import FoldScaler, fit_fold_scaler
from pkg.ts_v3a.split import assert_split_before_origin, chronological_train_val_split
from pkg.ts_v3a.types import PreparedNeuralFold, TargetMode, build_train_metadata
from pkg.ts_v3a.windows import ArrayLike, build_windows


def prepare_neural_fold(
    history: ArrayLike,
    *,
    forecast_origin: int,
    config: Optional[NeuralExperimentConfig] = None,
    seed: int = 42,
    mode: Optional[TargetMode] = None,
) -> tuple[PreparedNeuralFold, FoldScaler]:
    """Build windows, chronological split, fold-local scale, and metadata.

    ``history`` must contain only months with ``date < forecast_origin``.
    The outer evaluation horizon is never used for windows or early stopping.
    """
    cfg = config or DEFAULT_CONFIG
    target_mode = mode or target_mode_for(cfg.architecture_name)
    dataset = build_windows(
        history,
        mode=target_mode,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        require_samples=True,
    )

    training_start: Optional[int] = None
    training_end: Optional[int] = None
    if isinstance(history, pd.Series):
        n_history = len(history)
        if n_history:
            try:
                training_start = int(history.index[0])
                training_end = int(history.index[-1])
            except (TypeError, ValueError):
                training_start = None
                training_end = None
    else:
        n_history = int(np.asarray(history, dtype=float).reshape(-1).shape[0])

    split = chronological_train_val_split(
        dataset,
        validation_fraction=cfg.validation_fraction,
        min_train_windows=cfg.min_train_windows,
    )
    assert_split_before_origin(
        split, n_history=n_history, forecast_origin=forecast_origin
    )
    scaler = fit_fold_scaler(split.train, method=cfg.scaling_method)
    scaled = scaler.transform_split(split)

    metadata = build_train_metadata(
        architecture=cfg.architecture_name,
        parameters=config_parameters_snapshot(cfg),
        random_seed=seed,
        n_train_samples=scaled.n_train,
        n_validation_samples=scaled.n_validation,
        forecast_origin=forecast_origin,
        scaler_params=scaler.params(),
        training_start=training_start,
        training_end=training_end,
    )
    fold = PreparedNeuralFold(
        train_X=scaled.train.X,
        train_y=scaled.train.y,
        val_X=scaled.validation.X if scaled.validation is not None else None,
        val_y=scaled.validation.y if scaled.validation is not None else None,
        metadata=metadata,
        end_indices_train=scaled.train.end_indices,
        end_indices_val=(
            scaled.validation.end_indices if scaled.validation is not None else ()
        ),
    )
    return fold, scaler
