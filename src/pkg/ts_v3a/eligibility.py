"""Neural sample eligibility for V3A training.

Distinguishes windows that are mathematically constructible from those that
are eligible for actual neural model fitting (enough internal train and
validation windows for early stopping).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.types import FoldSplit, TargetMode, WindowDataset
from pkg.ts_v3a.windows import expected_n_samples, min_history_length


@dataclass(frozen=True)
class SampleEligibility:
    """Eligibility verdict for one SKU / origin / architecture setup."""

    mathematically_constructible: bool
    eligible_for_training: bool
    n_windows: int
    n_train: int
    n_validation: int
    min_internal_train_windows: int
    min_internal_validation_windows: int
    reason: Optional[str] = None


def evaluate_window_constructibility(
    n_history: int,
    *,
    lookback: int,
    horizon: int,
    mode: TargetMode,
) -> tuple[bool, int]:
    """Return ``(constructible, n_windows)`` for the given history length."""
    n_windows = expected_n_samples(n_history, lookback, horizon, mode)
    return n_windows >= 1, n_windows


def evaluate_split_eligibility(
    split: FoldSplit,
    *,
    config: Optional[NeuralExperimentConfig] = None,
    n_windows: Optional[int] = None,
) -> SampleEligibility:
    """Judge whether a chronological split is eligible for NeuralTrainer.fit."""
    cfg = config or DEFAULT_CONFIG
    total = int(n_windows) if n_windows is not None else (
        split.n_train + split.n_validation
    )
    constructible = total >= 1
    n_train = int(split.n_train)
    n_val = int(split.n_validation)
    min_tr = int(cfg.min_internal_train_windows)
    min_va = int(cfg.min_internal_validation_windows)

    if not constructible:
        return SampleEligibility(
            mathematically_constructible=False,
            eligible_for_training=False,
            n_windows=0,
            n_train=n_train,
            n_validation=n_val,
            min_internal_train_windows=min_tr,
            min_internal_validation_windows=min_va,
            reason="no_supervised_windows",
        )

    if split.validation is None or n_val < min_va:
        return SampleEligibility(
            mathematically_constructible=True,
            eligible_for_training=False,
            n_windows=total,
            n_train=n_train,
            n_validation=n_val,
            min_internal_train_windows=min_tr,
            min_internal_validation_windows=min_va,
            reason="insufficient_internal_validation_windows",
        )

    if n_train < min_tr:
        return SampleEligibility(
            mathematically_constructible=True,
            eligible_for_training=False,
            n_windows=total,
            n_train=n_train,
            n_validation=n_val,
            min_internal_train_windows=min_tr,
            min_internal_validation_windows=min_va,
            reason="insufficient_internal_train_windows",
        )

    return SampleEligibility(
        mathematically_constructible=True,
        eligible_for_training=True,
        n_windows=total,
        n_train=n_train,
        n_validation=n_val,
        min_internal_train_windows=min_tr,
        min_internal_validation_windows=min_va,
        reason=None,
    )


def evaluate_history_eligibility(
    n_history: int,
    *,
    mode: TargetMode,
    config: Optional[NeuralExperimentConfig] = None,
    dataset: Optional[WindowDataset] = None,
    split: Optional[FoldSplit] = None,
) -> SampleEligibility:
    """Full eligibility check from history length and optional split."""
    cfg = config or DEFAULT_CONFIG
    horizon = 1 if mode is TargetMode.RECURSIVE else int(cfg.horizon)
    constructible, n_windows = evaluate_window_constructibility(
        n_history,
        lookback=cfg.lookback,
        horizon=horizon if mode is TargetMode.DIRECT_MIMO else 1,
        mode=mode,
    )
    if dataset is not None:
        n_windows = dataset.n_samples
        constructible = n_windows >= 1

    if not constructible:
        return SampleEligibility(
            mathematically_constructible=False,
            eligible_for_training=False,
            n_windows=0,
            n_train=0,
            n_validation=0,
            min_internal_train_windows=cfg.min_internal_train_windows,
            min_internal_validation_windows=cfg.min_internal_validation_windows,
            reason=(
                "insufficient_history_for_windows:"
                f"need>={min_history_length(cfg.lookback, horizon, mode)}, got={n_history}"
            ),
        )

    if split is None:
        # Without a split, require enough total windows to satisfy both mins.
        needed = cfg.min_internal_train_windows + cfg.min_internal_validation_windows
        eligible = n_windows >= needed
        return SampleEligibility(
            mathematically_constructible=True,
            eligible_for_training=eligible,
            n_windows=n_windows,
            n_train=0,
            n_validation=0,
            min_internal_train_windows=cfg.min_internal_train_windows,
            min_internal_validation_windows=cfg.min_internal_validation_windows,
            reason=None if eligible else "insufficient_total_windows_for_train_val_mins",
        )

    return evaluate_split_eligibility(split, config=cfg, n_windows=n_windows)


class IneligibleForTrainingError(ValueError):
    """Raised when NeuralTrainer is asked to fit an ineligible sample."""

    def __init__(self, eligibility: SampleEligibility) -> None:
        self.eligibility = eligibility
        super().__init__(
            f"Sample not eligible for neural training: {eligibility.reason} "
            f"(train={eligibility.n_train}, val={eligibility.n_validation}, "
            f"need train>={eligibility.min_internal_train_windows}, "
            f"val>={eligibility.min_internal_validation_windows})"
        )
