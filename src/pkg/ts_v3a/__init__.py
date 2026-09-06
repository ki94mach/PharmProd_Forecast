"""V3A experimental local neural forecasting engine (foundation).

Per-SKU neural models under the V2 forecasting contract. This package does not
modify V1 or V2 behavior. LSTM layers are not implemented in this step.
"""
from __future__ import annotations

from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.scaling import FoldScaler, fit_fold_scaler
from pkg.ts_v3a.seeds import set_global_seeds
from pkg.ts_v3a.split import chronological_train_val_split
from pkg.ts_v3a.types import (
    FoldSplit,
    TargetMode,
    TrainMetadata,
    WindowDataset,
    build_train_metadata,
)
from pkg.ts_v3a.windows import (
    InsufficientHistoryError,
    build_mimo_windows,
    build_recursive_windows,
    build_windows,
    expected_n_samples,
)

__all__ = [
    "ArchitectureName",
    "DEFAULT_CONFIG",
    "FoldScaler",
    "FoldSplit",
    "ForecastOrigin",
    "ForecastWindow",
    "InsufficientHistoryError",
    "NeuralExperimentConfig",
    "TargetMode",
    "TrainMetadata",
    "WindowDataset",
    "build_mimo_windows",
    "build_recursive_windows",
    "build_train_metadata",
    "build_windows",
    "chronological_train_val_split",
    "expected_n_samples",
    "fit_fold_scaler",
    "make_forecast_window",
    "set_global_seeds",
    "target_mode_for",
]
