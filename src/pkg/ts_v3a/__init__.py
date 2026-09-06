"""V3A experimental local neural forecasting engine.

Per-SKU neural models under the V2 forecasting contract. This package does not
modify V1 or V2 behavior. A0–A5 are implemented; A6 is not yet.
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
    BidirectionalMimoLSTM,
    EncoderDecoderLSTM,
    LegacyAdaptiveRecursiveLSTM,
    LegacyArchitectureSpec,
    MimoLSTM,
    SmallRecursiveLSTM,
    StackedMimoLSTM,
    build_bidirectional_mimo_lstm,
    build_encoder_decoder_lstm,
    build_legacy_adaptive_recursive_lstm,
    build_mimo_lstm,
    build_small_recursive_lstm,
    build_stacked_mimo_lstm,
    default_a0_config,
    reshape_mimo_targets,
    resolve_legacy_architecture,
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
    "BidirectionalMimoLSTM",
    "DEFAULT_CONFIG",
    "EncoderDecoderLSTM",
    "FoldScaler",
    "FoldSplit",
    "ForecastOrigin",
    "ForecastWindow",
    "IneligibleForTrainingError",
    "InsufficientHistoryError",
    "LegacyAdaptiveRecursiveLSTM",
    "LegacyArchitectureSpec",
    "MimoLSTM",
    "NeuralExperimentConfig",
    "NeuralTrainer",
    "SampleEligibility",
    "SmallRecursiveLSTM",
    "StackedMimoLSTM",
    "TargetMode",
    "TrainMetadata",
    "WindowDataset",
    "build_bidirectional_mimo_lstm",
    "build_encoder_decoder_lstm",
    "build_legacy_adaptive_recursive_lstm",
    "build_mimo_lstm",
    "build_mimo_windows",
    "build_recursive_windows",
    "build_small_recursive_lstm",
    "build_stacked_mimo_lstm",
    "build_train_metadata",
    "build_windows",
    "chronological_train_val_split",
    "default_a0_config",
    "evaluate_history_eligibility",
    "evaluate_split_eligibility",
    "expected_n_samples",
    "fit_fold_scaler",
    "make_forecast_window",
    "reshape_mimo_targets",
    "resolve_legacy_architecture",
    "rollout_recursive_forecast",
    "set_global_seeds",
    "target_mode_for",
    "unique_observation_indices",
]
