"""Named neural architecture candidates for V3A bake-offs.

No Keras/LSTM graphs here — only identifiers and target-mode mapping so
RECURSIVE vs DIRECT/MIMO window builders stay aligned with future models.
"""
from __future__ import annotations

from enum import Enum

from pkg.ts_v3a.types import TargetMode


class ArchitectureName(str, Enum):
    """Experimental architecture IDs (A0–A6)."""

    A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM = "legacy_adaptive_recursive_lstm"
    A1_SMALL_RECURSIVE_LSTM = "small_recursive_lstm"
    A2_MIMO_LSTM = "mimo_lstm"
    A3_STACKED_MIMO_LSTM = "stacked_mimo_lstm"
    A4_ENCODER_DECODER_LSTM = "encoder_decoder_lstm"
    A5_BIDIRECTIONAL_MIMO_LSTM = "bidirectional_mimo_lstm"
    A6_ATTENTION_BIDIRECTIONAL_LSTM = "attention_bidirectional_lstm"


_TARGET_MODE: dict[ArchitectureName, TargetMode] = {
    ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM: TargetMode.RECURSIVE,
    ArchitectureName.A1_SMALL_RECURSIVE_LSTM: TargetMode.RECURSIVE,
    ArchitectureName.A2_MIMO_LSTM: TargetMode.DIRECT_MIMO,
    ArchitectureName.A3_STACKED_MIMO_LSTM: TargetMode.DIRECT_MIMO,
    ArchitectureName.A4_ENCODER_DECODER_LSTM: TargetMode.DIRECT_MIMO,
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM: TargetMode.DIRECT_MIMO,
    ArchitectureName.A6_ATTENTION_BIDIRECTIONAL_LSTM: TargetMode.DIRECT_MIMO,
}

# A0/A1 share the V2 production time contract (bridge month + train through
# last_complete). A2–A5 keep the historical screening contract.
_PRODUCTION_TIME_CONTRACT: frozenset[ArchitectureName] = frozenset(
    {
        ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,
        ArchitectureName.A1_SMALL_RECURSIVE_LSTM,
    }
)


def coerce_architecture_name(name: str | ArchitectureName) -> ArchitectureName:
    """Resolve a string or enum to :class:`ArchitectureName`."""
    if isinstance(name, ArchitectureName):
        return name
    try:
        return ArchitectureName(name)
    except ValueError as exc:
        known = ", ".join(a.value for a in ArchitectureName)
        raise ValueError(f"Unknown architecture_name {name!r}; expected one of: {known}") from exc


def target_mode_for(architecture_name: str | ArchitectureName) -> TargetMode:
    """Return RECURSIVE or DIRECT_MIMO for the given architecture."""
    return _TARGET_MODE[coerce_architecture_name(architecture_name)]


def uses_production_time_contract(
    architecture_name: str | ArchitectureName,
) -> bool:
    """True for A0/A1 (production bridge contract); False for A2–A5 screening."""
    return coerce_architecture_name(architecture_name) in _PRODUCTION_TIME_CONTRACT
