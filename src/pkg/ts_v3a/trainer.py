"""Common NeuralTrainer for V3A architectures.

Architecture-specific code only builds a fresh Keras model. Training,
early stopping, metadata, and inverse-transform live here so later A0–A6
candidates do not fork training loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

import numpy as np

from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig, config_parameters_snapshot
from pkg.ts_v3a.eligibility import IneligibleForTrainingError, SampleEligibility, evaluate_split_eligibility
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.seeds import set_global_seeds
from pkg.ts_v3a.types import FoldSplit, TrainMetadata, build_train_metadata


class ArchitectureBuilder(Protocol):
    """Callable that returns a compiled-ready Keras model for ``config``."""

    def __call__(self, config: NeuralExperimentConfig):  # pragma: no cover - protocol
        ...


def count_trainable_parameters(model: Any) -> int:
    """Return trainable parameter count for a Keras model."""
    try:
        return int(model.count_params())
    except Exception:
        total = 0
        for w in getattr(model, "trainable_weights", []):
            total += int(np.prod(w.shape))
        return int(total)


def _build_optimizer(config: NeuralExperimentConfig):
    from keras.optimizers import Adam

    if config.optimizer != "Adam":
        raise ValueError(f"Unsupported optimizer {config.optimizer!r}; V3A uses Adam")
    return Adam(learning_rate=float(config.learning_rate))


def _build_loss(config: NeuralExperimentConfig):
    from keras.losses import Huber

    if config.loss != "Huber":
        raise ValueError(f"Unsupported loss {config.loss!r}; V3A uses Huber")
    return Huber()


@dataclass
class NeuralTrainer:
    """Shared train / predict path for V3A neural models."""

    architecture_builder: ArchitectureBuilder
    config: NeuralExperimentConfig = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        self.model_ = None
        self.scaler_: Optional[FoldScaler] = None
        self.metadata_: Optional[TrainMetadata] = None
        self.eligibility_: Optional[SampleEligibility] = None

    def fit_prepared(
        self,
        *,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        scaler: FoldScaler,
        seed: int,
        forecast_origin: int,
        training_start: Optional[int] = None,
        training_end: Optional[int] = None,
        eligibility: Optional[SampleEligibility] = None,
    ) -> TrainMetadata:
        """Fit on already-scaled train/val arrays with chronological validation."""
        n_train = int(np.asarray(train_X).shape[0])
        n_val = int(np.asarray(val_X).shape[0])
        if eligibility is not None and not eligibility.eligible_for_training:
            raise IneligibleForTrainingError(eligibility)
        if n_val < self.config.min_internal_validation_windows:
            raise IneligibleForTrainingError(
                SampleEligibility(
                    mathematically_constructible=n_train + n_val >= 1,
                    eligible_for_training=False,
                    n_windows=n_train + n_val,
                    n_train=n_train,
                    n_validation=n_val,
                    min_internal_train_windows=self.config.min_internal_train_windows,
                    min_internal_validation_windows=self.config.min_internal_validation_windows,
                    reason="insufficient_internal_validation_windows",
                )
            )
        if n_train < self.config.min_internal_train_windows:
            raise IneligibleForTrainingError(
                SampleEligibility(
                    mathematically_constructible=True,
                    eligible_for_training=False,
                    n_windows=n_train + n_val,
                    n_train=n_train,
                    n_validation=n_val,
                    min_internal_train_windows=self.config.min_internal_train_windows,
                    min_internal_validation_windows=self.config.min_internal_validation_windows,
                    reason="insufficient_internal_train_windows",
                )
            )

        set_global_seeds(int(seed))
        model = self.architecture_builder(self.config)
        model.compile(
            optimizer=_build_optimizer(self.config),
            loss=_build_loss(self.config),
        )

        from keras.callbacks import EarlyStopping

        early = EarlyStopping(
            monitor="val_loss",
            patience=int(self.config.early_stopping_patience),
            restore_best_weights=True,
        )
        history = model.fit(
            np.asarray(train_X, dtype=float),
            np.asarray(train_y, dtype=float),
            validation_data=(
                np.asarray(val_X, dtype=float),
                np.asarray(val_y, dtype=float),
            ),
            epochs=int(self.config.max_epochs),
            batch_size=int(self.config.batch_size),
            callbacks=[early],
            verbose=0,
        )

        epochs_ran = int(len(history.history.get("loss", [])))
        best_epoch: Optional[int] = None
        best_val_loss: Optional[float] = None
        if getattr(early, "best_epoch", None) is not None:
            # Keras EarlyStopping.best_epoch is 0-based in TF 2.x
            best_epoch = int(early.best_epoch) + 1
        elif "val_loss" in history.history and history.history["val_loss"]:
            val_losses = history.history["val_loss"]
            best_epoch = int(np.argmin(val_losses)) + 1
        if getattr(early, "best", None) is not None and early.best is not None:
            best_val_loss = float(early.best)
        elif "val_loss" in history.history and history.history["val_loss"]:
            best_val_loss = float(np.min(history.history["val_loss"]))

        metadata = build_train_metadata(
            architecture=self.config.architecture_name,
            parameters=config_parameters_snapshot(self.config),
            random_seed=int(seed),
            n_train_samples=n_train,
            n_validation_samples=n_val,
            train_window_count=n_train,
            validation_window_count=n_val,
            forecast_origin=int(forecast_origin),
            scaler_params=scaler.params(),
            training_start=training_start,
            training_end=training_end,
            epochs_ran=epochs_ran,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            parameter_count=count_trainable_parameters(model),
        )
        self.model_ = model
        self.scaler_ = scaler
        self.metadata_ = metadata
        self.eligibility_ = eligibility
        return metadata

    def fit_split(
        self,
        split: FoldSplit,
        scaler: FoldScaler,
        *,
        seed: int,
        forecast_origin: int,
        training_start: Optional[int] = None,
        training_end: Optional[int] = None,
    ) -> TrainMetadata:
        """Fit from a scaled :class:`FoldSplit` (requires non-empty validation)."""
        eligibility = evaluate_split_eligibility(split, config=self.config)
        self.eligibility_ = eligibility
        if not eligibility.eligible_for_training:
            raise IneligibleForTrainingError(eligibility)
        assert split.validation is not None
        return self.fit_prepared(
            train_X=split.train.X,
            train_y=split.train.y,
            val_X=split.validation.X,
            val_y=split.validation.y,
            scaler=scaler,
            seed=seed,
            forecast_origin=forecast_origin,
            training_start=training_start,
            training_end=training_end,
            eligibility=eligibility,
        )

    def predict(self, X: np.ndarray, *, inverse_transform: bool = True) -> np.ndarray:
        """Predict; by default inverse-transform to raw sales units.

        Does **not** apply non-negativity or any business postprocessing.
        """
        if self.model_ is None:
            raise RuntimeError("NeuralTrainer.predict called before fit")
        preds = self.model_.predict(np.asarray(X, dtype=float), verbose=0)
        preds = np.asarray(preds, dtype=float)
        if inverse_transform:
            if self.scaler_ is None:
                raise RuntimeError("NeuralTrainer has no scaler for inverse_transform")
            preds = self.scaler_.inverse_transform_y(preds)
        return preds
