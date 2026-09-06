"""V3A experimental local neural forecasting engine.

Per-SKU neural models under the V2 forecasting contract. This package does not
modify V1 or V2 behavior. A1 ``small_recursive_lstm`` and A2 ``mimo_lstm`` are
implemented; A0/A3–A6 graphs are not yet.
"""
from __future__ import annotations

from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.eligibility import (
    IneligibleForTrainingError,
    SampleEligibility,
    evaluate_history_eligibility,
    evaluate_split_eligibility,
)
from pkg.ts_v3a.models import (
    MimoLSTM,
    SmallRecursiveLSTM,
    build_mimo_lstm,
    build_small_recursive_lstm,
    rollout_recursive_forecast,
)
from pkg.ts_v3a.scaling import FoldScaler, fit_fold_scaler, unique_observation_indices
from pkg.ts_v3a.seeds import set_global_seeds
from pkg.ts_v3a.split import chronological_train_val_split
from pkg.ts_v3a.trainer import NeuralTrainer
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
    "IneligibleForTrainingError",
    "InsufficientHistoryError",
    "MimoLSTM",
    "NeuralExperimentConfig",
    "NeuralTrainer",
    "SampleEligibility",
    "SmallRecursiveLSTM",
    "TargetMode",
    "TrainMetadata",
    "WindowDataset",
    "build_mimo_lstm",
    "build_mimo_windows",
    "build_recursive_windows",
    "build_small_recursive_lstm",
    "build_train_metadata",
    "build_windows",
    "chronological_train_val_split",
    "evaluate_history_eligibility",
    "evaluate_split_eligibility",
    "expected_n_samples",
    "fit_fold_scaler",
    "make_forecast_window",
    "rollout_recursive_forecast",
    "set_global_seeds",
    "target_mode_for",
    "unique_observation_indices",
]
