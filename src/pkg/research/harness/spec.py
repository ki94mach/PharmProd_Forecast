"""Experiment specifications for a research feature family."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from pkg.benchmark.dataset import BenchmarkDataset

Anchor = Literal["ts", "human"]
Enricher = Callable[[BenchmarkDataset], BenchmarkDataset]


@dataclass(frozen=True)
class ExperimentSpec:
    """One residual-XGB (or frozen adapter) run on matched PRIMARY."""

    name: str
    anchor: Anchor
    features: tuple[str, ...]
    train_universe: str  # "ts" | "budget" | "matched"
    control: str
    use_frozen_adapter: bool = False
    enrich: Optional[str] = None


@dataclass
class FamilyConfig:
    """Strategy bag for one feature family (F2, F3A, …)."""

    family: str
    out_dir: Path
    experiments: tuple[ExperimentSpec, ...]
    enrichers: dict[str, Enricher] = field(default_factory=dict)
    fillna_extra: tuple[str, ...] = ()
    never_fillna: frozenset[str] = field(default_factory=frozenset)
    model_name_prefix: str = "xgb"
    pre_model: Optional[Callable[..., None]] = None
    post_score: Optional[Callable[..., None]] = None
