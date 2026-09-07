"""Fresh neural model construction for V3A outer CV folds.

Every outer fold must receive a new model instance so weights never carry
across origins, architectures, or seeds.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from pkg.ts_v3a.architectures import ArchitectureName, coerce_architecture_name
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import (
    LegacyAdaptiveRecursiveLSTM,
)
from pkg.ts_v3a.models.a1_small_recursive_lstm import SmallRecursiveLSTM
from pkg.ts_v3a.models.a2_mimo_lstm import MimoLSTM
from pkg.ts_v3a.models.a3_stacked_mimo_lstm import StackedMimoLSTM
from pkg.ts_v3a.models.a4_encoder_decoder_lstm import EncoderDecoderLSTM
from pkg.ts_v3a.models.a5_bidirectional_mimo_lstm import BidirectionalMimoLSTM
from pkg.ts_v3a.models.base import BaseNeuralForecastModel

# A0–A5 only; A6 is intentionally excluded until implemented.
IMPLEMENTED_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,
    ArchitectureName.A1_SMALL_RECURSIVE_LSTM,
    ArchitectureName.A2_MIMO_LSTM,
    ArchitectureName.A3_STACKED_MIMO_LSTM,
    ArchitectureName.A4_ENCODER_DECODER_LSTM,
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM,
)

_BUILDERS: dict[ArchitectureName, type[BaseNeuralForecastModel]] = {
    ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM: LegacyAdaptiveRecursiveLSTM,
    ArchitectureName.A1_SMALL_RECURSIVE_LSTM: SmallRecursiveLSTM,
    ArchitectureName.A2_MIMO_LSTM: MimoLSTM,
    ArchitectureName.A3_STACKED_MIMO_LSTM: StackedMimoLSTM,
    ArchitectureName.A4_ENCODER_DECODER_LSTM: EncoderDecoderLSTM,
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM: BidirectionalMimoLSTM,
}


def create_neural_model(
    architecture: Union[str, ArchitectureName],
    config: Optional[NeuralExperimentConfig] = None,
) -> BaseNeuralForecastModel:
    """Return a fresh A0–A5 model instance for one outer fold.

    Raises:
        ValueError: If ``architecture`` is unknown or A6 (not implemented).
    """
    name = coerce_architecture_name(architecture)
    if name not in _BUILDERS:
        raise ValueError(
            f"Architecture {name.value!r} is not implemented for outer backtest "
            f"(implemented: {[a.value for a in IMPLEMENTED_ARCHITECTURES]})"
        )
    cfg = config or DEFAULT_CONFIG
    if cfg.architecture_name != name.value:
        cfg = cfg.with_architecture(name)
    model = _BUILDERS[name](cfg)
    # Propagate short-epoch / patience overrides into A0 after tier resolution.
    if isinstance(model, LegacyAdaptiveRecursiveLSTM):
        model.with_fit_overrides(
            max_epochs=int(cfg.max_epochs),
            early_stopping_patience=int(cfg.early_stopping_patience),
            batch_size=int(cfg.batch_size),
        )
    return model


def coerce_architecture_list(
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
) -> tuple[ArchitectureName, ...]:
    """Normalize architecture list; default to all implemented A0–A5."""
    if architectures is None:
        return IMPLEMENTED_ARCHITECTURES
    out = tuple(coerce_architecture_name(a) for a in architectures)
    if not out:
        raise ValueError("architectures must contain at least one entry")
    for name in out:
        if name not in _BUILDERS:
            raise ValueError(
                f"Architecture {name.value!r} is not implemented for outer backtest"
            )
    return out
