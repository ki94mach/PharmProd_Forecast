"""Prepare a scaled train/val fold from pre-origin history (no model training)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.ts_v3a.architectures import target_mode_for
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig, config_parameters_snapshot
from pkg.ts_v3a.eligibility import evaluate_split_eligibility
from pkg.ts_v3a.scaling import FoldScaler, fit_fold_scaler
from pkg.ts_v3a.split import assert_split_before_origin, chronological_train_val_split
from pkg.ts_v3a.types import PreparedNeuralFold, TargetMode, WindowDataset, build_train_metadata
from pkg.ts_v3a.windows import ArrayLike, build_windows


def _first_last_end_dates(
    dataset: Optional[WindowDataset],
) -> tuple[Optional[int], Optional[int]]:
    """Return first/last non-null ``end_dates`` from a window dataset."""
    if dataset is None or not dataset.end_dates:
        return None, None
    non_null = [int(d) for d in dataset.end_dates if d is not None]
    if not non_null:
        return None, None
    return non_null[0], non_null[-1]


def prepare_neural_fold(
    history: ArrayLike,
    *,
    forecast_origin: int,
    config: Optional[NeuralExperimentConfig] = None,
    seed: int = 42,
    mode: Optional[TargetMode] = None,
    require_eligible: bool = False,
) -> tuple[PreparedNeuralFold, FoldScaler]:
    """Build windows, chronological split, fold-local scale, and metadata.

    ``history`` must contain only months with ``date < forecast_origin``.
    The outer evaluation horizon is never used for windows or early stopping.

    Scaling fits on unique chronological observations in the internal train
    period only. Samples that fail eligibility mins are still prepared for
    diagnostics unless ``require_eligible=True``.
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

    series_start: Optional[int] = None
    series_end: Optional[int] = None
    if isinstance(history, pd.Series):
        n_history = len(history)
        if n_history:
            try:
                series_start = int(history.index[0])
                series_end = int(history.index[-1])
            except (TypeError, ValueError):
                series_start = None
                series_end = None
    else:
        n_history = int(np.asarray(history, dtype=float).reshape(-1).shape[0])

    split = chronological_train_val_split(
        dataset,
        validation_fraction=cfg.validation_fraction,
        min_internal_train_windows=cfg.min_internal_train_windows,
        min_internal_validation_windows=cfg.min_internal_validation_windows,
    )
    assert_split_before_origin(
        split, n_history=n_history, forecast_origin=forecast_origin
    )
    eligibility = evaluate_split_eligibility(
        split, config=cfg, n_windows=dataset.n_samples
    )
    if require_eligible and not eligibility.eligible_for_training:
        from pkg.ts_v3a.eligibility import IneligibleForTrainingError

        raise IneligibleForTrainingError(eligibility)

    scaler = fit_fold_scaler(history, split.train, method=cfg.scaling_method)
    scaled = scaler.transform_split(split)

    training_start, training_end = _first_last_end_dates(scaled.train)
    if training_start is None:
        training_start = series_start
    if training_end is None:
        training_end = series_end
    validation_start, validation_end = _first_last_end_dates(scaled.validation)

    metadata = build_train_metadata(
        architecture=cfg.architecture_name,
        parameters=config_parameters_snapshot(cfg),
        random_seed=seed,
        n_train_samples=scaled.n_train,
        n_validation_samples=scaled.n_validation,
        train_window_count=scaled.n_train,
        validation_window_count=scaled.n_validation,
        forecast_origin=forecast_origin,
        scaler_params=scaler.params(),
        training_start=training_start,
        training_end=training_end,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    fold = PreparedNeuralFold(
        train_X=scaled.train.X,
        train_y=scaled.train.y,
        val_X=scaled.validation.X if scaled.validation is not None else None,
        val_y=scaled.validation.y if scaled.validation is not None else None,
        metadata=metadata,
        eligible_for_training=eligibility.eligible_for_training,
        eligibility_reason=eligibility.reason,
        end_indices_train=scaled.train.end_indices,
        end_indices_val=(
            scaled.validation.end_indices if scaled.validation is not None else ()
        ),
    )
    return fold, scaler
