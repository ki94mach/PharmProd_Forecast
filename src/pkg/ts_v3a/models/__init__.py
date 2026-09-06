"""V3A neural models: A1–A5."""
from __future__ import annotations

from pkg.ts_v3a.models.a1_small_recursive_lstm import (
    SmallRecursiveLSTM,
    build_small_recursive_lstm,
    default_a1_config,
)
from pkg.ts_v3a.models.a2_mimo_lstm import (
    MimoLSTM,
    build_mimo_lstm,
    default_a2_config,
)
from pkg.ts_v3a.models.a3_stacked_mimo_lstm import (
    StackedMimoLSTM,
    build_stacked_mimo_lstm,
    default_a3_config,
)
from pkg.ts_v3a.models.a4_encoder_decoder_lstm import (
    EncoderDecoderLSTM,
    build_encoder_decoder_lstm,
    default_a4_config,
    reshape_mimo_targets,
)
from pkg.ts_v3a.models.a5_bidirectional_mimo_lstm import (
    BidirectionalMimoLSTM,
    build_bidirectional_mimo_lstm,
    default_a5_config,
)
from pkg.ts_v3a.models.base import BaseNeuralForecastModel, NeuralForecastModel
from pkg.ts_v3a.models.recursive_rollout import rollout_recursive_forecast

__all__ = [
    "BaseNeuralForecastModel",
    "BidirectionalMimoLSTM",
    "EncoderDecoderLSTM",
    "MimoLSTM",
    "NeuralForecastModel",
    "SmallRecursiveLSTM",
    "StackedMimoLSTM",
    "build_bidirectional_mimo_lstm",
    "build_encoder_decoder_lstm",
    "build_mimo_lstm",
    "build_small_recursive_lstm",
    "build_stacked_mimo_lstm",
    "default_a1_config",
    "default_a2_config",
    "default_a3_config",
    "default_a4_config",
    "default_a5_config",
    "reshape_mimo_targets",
    "rollout_recursive_forecast",
]
