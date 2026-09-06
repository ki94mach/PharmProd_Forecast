"""V3A neural models: A1–A3."""
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
from pkg.ts_v3a.models.base import BaseNeuralForecastModel, NeuralForecastModel
from pkg.ts_v3a.models.recursive_rollout import rollout_recursive_forecast

__all__ = [
    "BaseNeuralForecastModel",
    "MimoLSTM",
    "NeuralForecastModel",
    "SmallRecursiveLSTM",
    "StackedMimoLSTM",
    "build_mimo_lstm",
    "build_small_recursive_lstm",
    "build_stacked_mimo_lstm",
    "default_a1_config",
    "default_a2_config",
    "default_a3_config",
    "rollout_recursive_forecast",
]
