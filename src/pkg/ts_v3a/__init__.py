"""V3A experimental local neural forecasting engine.

Per-SKU neural models under the V2 forecasting contract. This package does not
modify V1 or V2 behavior. A0–A5, outer expanding backtest, multi-seed evaluation,
and immutable screening persistence are implemented; A6, architecture selection,
and full-history refit are not yet.
"""
from __future__ import annotations

from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.backtest import (
    FOLD_METADATA_COLUMNS,
    PREDICTION_COLUMNS,
    NeuralOuterBacktestResult,
    backtest_product_architectures,
    run_outer_backtest,
)
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.eligibility import (
    IneligibleForTrainingError,
    SampleEligibility,
    evaluate_history_eligibility,
    evaluate_split_eligibility,
)
from pkg.ts_v3a.metrics import (
    build_seed_ensemble_predictions,
    ensemble_metrics_table,
    mean_horizon_mae,
    metrics_summary_row,
    seed_metrics_table,
    stability_table,
)
from pkg.ts_v3a.model_factory import (
    IMPLEMENTED_ARCHITECTURES,
    coerce_architecture_list,
    create_neural_model,
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
from pkg.ts_v3a.persistence import (
    V3A_VERSION,
    ExperimentConfigConflictError,
    ExperimentImmutableError,
    LoadedScreeningExperiment,
    ScreeningExperimentCheckpoint,
    assert_compatible_config,
    begin_screening_experiment,
    finalize_screening_experiment,
    load_screening_experiment,
    persist_completed_screening_experiment,
    screening_config_hash,
    write_screening_checkpoint,
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
    "ExperimentConfigConflictError",
    "ExperimentImmutableError",
    "FOLD_METADATA_COLUMNS",
    "FoldScaler",
    "FoldSplit",
    "ForecastOrigin",
    "ForecastWindow",
    "IMPLEMENTED_ARCHITECTURES",
    "IneligibleForTrainingError",
    "InsufficientHistoryError",
    "LegacyAdaptiveRecursiveLSTM",
    "LegacyArchitectureSpec",
    "LoadedScreeningExperiment",
    "MimoLSTM",
    "NeuralExperimentConfig",
    "NeuralOuterBacktestResult",
    "NeuralTrainer",
    "PREDICTION_COLUMNS",
    "SampleEligibility",
    "ScreeningExperimentCheckpoint",
    "SmallRecursiveLSTM",
    "StackedMimoLSTM",
    "TargetMode",
    "TrainMetadata",
    "V3A_VERSION",
    "WindowDataset",
    "assert_compatible_config",
    "backtest_product_architectures",
    "begin_screening_experiment",
    "build_bidirectional_mimo_lstm",
    "build_encoder_decoder_lstm",
    "build_legacy_adaptive_recursive_lstm",
    "build_mimo_lstm",
    "build_mimo_windows",
    "build_recursive_windows",
    "build_seed_ensemble_predictions",
    "build_small_recursive_lstm",
    "build_stacked_mimo_lstm",
    "build_train_metadata",
    "build_windows",
    "chronological_train_val_split",
    "coerce_architecture_list",
    "create_neural_model",
    "default_a0_config",
    "ensemble_metrics_table",
    "evaluate_history_eligibility",
    "evaluate_split_eligibility",
    "expected_n_samples",
    "finalize_screening_experiment",
    "fit_fold_scaler",
    "load_screening_experiment",
    "make_forecast_window",
    "mean_horizon_mae",
    "metrics_summary_row",
    "persist_completed_screening_experiment",
    "reshape_mimo_targets",
    "resolve_legacy_architecture",
    "rollout_recursive_forecast",
    "run_outer_backtest",
    "screening_config_hash",
    "seed_metrics_table",
    "set_global_seeds",
    "stability_table",
    "target_mode_for",
    "unique_observation_indices",
    "write_screening_checkpoint",
]
