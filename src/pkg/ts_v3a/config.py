"""Configuration for V3A neural forecasting experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Optional, Union

from pkg.ts_v3a.architectures import ArchitectureName, coerce_architecture_name

ScalingMethod = Literal["standard"]
OptimizerName = Literal["Adam"]
LossName = Literal["Huber"]


@dataclass(frozen=True)
class NeuralExperimentConfig:
    """Hyperparameters shared across V3A architecture candidates.

    Defaults match the V3A experimental protocol. ``horizon`` must stay aligned
    with the V2 :class:`~pkg.ts_v2.types.ForecastWindow` contract (15 months).
    """

    architecture_name: str = ArchitectureName.A2_MIMO_LSTM.value
    lookback: int = 12
    horizon: int = 15
    hidden_units: int = 32
    second_hidden_units: int = 16
    number_layers: int = 1
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    optimizer: OptimizerName = "Adam"
    learning_rate: float = 0.001
    loss: LossName = "Huber"
    batch_size: int = 8
    max_epochs: int = 500
    early_stopping_patience: int = 30
    random_seeds: tuple[int, ...] = (41, 42, 43)
    scaling_method: ScalingMethod = "standard"
    validation_fraction: float = 0.2
    min_internal_train_windows: int = 8
    min_internal_validation_windows: int = 2

    def __post_init__(self) -> None:
        coerce_architecture_name(self.architecture_name)
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction}"
            )
        if self.min_internal_train_windows < 1:
            raise ValueError(
                "min_internal_train_windows must be >= 1, "
                f"got {self.min_internal_train_windows}"
            )
        if self.min_internal_validation_windows < 1:
            raise ValueError(
                "min_internal_validation_windows must be >= 1, "
                f"got {self.min_internal_validation_windows}"
            )
        if len(self.random_seeds) < 1:
            raise ValueError("random_seeds must contain at least one seed")

    def as_parameters_dict(self) -> dict:
        """Serializable parameter snapshot for :class:`TrainMetadata`."""
        return asdict(self)

    def with_architecture(
        self, architecture_name: Union[str, ArchitectureName]
    ) -> "NeuralExperimentConfig":
        """Return a copy with a different architecture name."""
        name = coerce_architecture_name(architecture_name).value
        return NeuralExperimentConfig(
            architecture_name=name,
            lookback=self.lookback,
            horizon=self.horizon,
            hidden_units=self.hidden_units,
            second_hidden_units=self.second_hidden_units,
            number_layers=self.number_layers,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            loss=self.loss,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            random_seeds=self.random_seeds,
            scaling_method=self.scaling_method,
            validation_fraction=self.validation_fraction,
            min_internal_train_windows=self.min_internal_train_windows,
            min_internal_validation_windows=self.min_internal_validation_windows,
        )


DEFAULT_CONFIG = NeuralExperimentConfig()


def config_parameters_snapshot(
    config: NeuralExperimentConfig,
    *,
    extra: Optional[Mapping[str, object]] = None,
) -> dict:
    """Merge config fields with optional extras for metadata."""
    snap = config.as_parameters_dict()
    if extra:
        snap.update(dict(extra))
    return snap
