"""Neural model protocol stubs for V3A (no LSTM layers yet)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Sequence

import pandas as pd

from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.types import TrainMetadata


class NeuralForecastModel(Protocol):
    """Structural interface for future V3A architecture implementations."""

    name: str
    config: NeuralExperimentConfig

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "NeuralForecastModel":
        ...

    def predict(self, window: ForecastWindow) -> ForecastResult:
        ...


class BaseNeuralForecastModel(ABC):
    """Abstract base for V3A neural models.

    Subclasses will implement Keras graphs in later steps. This scaffold only
    defines the contract: fit on ``date < origin`` history, predict the
    ``ForecastWindow`` targets, and expose training metadata.
    """

    name: str = "unnamed"
    config: NeuralExperimentConfig
    metadata_: Optional[TrainMetadata] = None

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        from pkg.ts_v3a.config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        self.metadata_ = None

    @abstractmethod
    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "BaseNeuralForecastModel":
        """Fit using only history with ``date < window.forecast_origin``."""

    @abstractmethod
    def predict(self, window: ForecastWindow) -> ForecastResult:
        """Emit one prediction per ``window.target_dates`` entry (length H)."""
