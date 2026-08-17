"""Feature-family ablation: CORE vs F0_DEMAND partition (semantic, not test-tuned).

Asserts CORE + F0_DEMAND == frozen F0 feature set (set equality).
Feature *order* for D1/D4/H1/H4 follows frozen TS_RESID_FEATURES / BUDGET_RESID_FEATURES
so those experiments can reproduce F0 / F1A / F1B.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.features.demand import DEMAND_FEATURE_NAMES
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES
from pkg.research.features.human import HUMAN_FEATURE_NAMES
from pkg.research.features.human_f2 import HUMAN_F2_FEATURE_NAMES

Anchor = Literal["ts", "human"]

F0_DEMAND: tuple[str, ...] = (
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
)

_CORE_STATIC: tuple[str, ...] = (
    "horizon",
    "month",
    "quarter",
    "model_enc",
    "field_enc",
    "form_enc",
    "provider_enc",
)

CORE_TS: tuple[str, ...] = ("ts_forecast",) + _CORE_STATIC
CORE_HUMAN: tuple[str, ...] = ("budget_forecast",) + _CORE_STATIC

F1_DEMAND: tuple[str, ...] = tuple(DEMAND_FEATURE_NAMES)
F1_HUMAN: tuple[str, ...] = tuple(HUMAN_FEATURE_NAMES)
F2_DEMAND: tuple[str, ...] = tuple(DEMAND_F2_FEATURE_NAMES)
F2_HUMAN: tuple[str, ...] = tuple(HUMAN_F2_FEATURE_NAMES)

WMAPE_REPRO_TOL = 0.05
PRED_REPRO_TOL = 1e-3
SIMILAR_WMAPE_TOL = 0.5  # Case C: replacement ≈ F0
MATERIAL_WMAPE_TOL = 1.0  # Human Cases D/E: material WMAPE gap vs F0

FILLNA_EXTRA: tuple[str, ...] = (
    "trend_3m",
    "trend_6m",
    "recent_growth",
    "recent_acceleration",
    "historical_actual_budget_ratio",
    "mean_human_adjustment",
    "mean_abs_human_adjustment",
    "trend_log_3m",
    "trend_log_6m",
    "yoy_log_change",
)


def _assert_partition() -> None:
    f0_d = set(F0_DEMAND)
    if set(CORE_TS) | f0_d != set(TS_RESID_FEATURES):
        raise AssertionError(
            f"CORE_TS ∪ F0_DEMAND != TS_RESID_FEATURES: "
            f"extra={(set(CORE_TS) | f0_d) - set(TS_RESID_FEATURES)} "
            f"missing={set(TS_RESID_FEATURES) - (set(CORE_TS) | f0_d)}"
        )
    if set(CORE_HUMAN) | f0_d != set(BUDGET_RESID_FEATURES):
        raise AssertionError(
            f"CORE_HUMAN ∪ F0_DEMAND != BUDGET_RESID_FEATURES: "
            f"extra={(set(CORE_HUMAN) | f0_d) - set(BUDGET_RESID_FEATURES)} "
            f"missing={set(BUDGET_RESID_FEATURES) - (set(CORE_HUMAN) | f0_d)}"
        )
    if set(CORE_TS) & f0_d:
        raise AssertionError(f"CORE_TS overlaps F0_DEMAND: {set(CORE_TS) & f0_d}")
    if set(CORE_HUMAN) & f0_d:
        raise AssertionError(f"CORE_HUMAN overlaps F0_DEMAND: {set(CORE_HUMAN) & f0_d}")


_assert_partition()


@dataclass(frozen=True)
class AblationExperiment:
    """Feature groups: f0_demand, f1_demand, f2_demand, f1_human, f2_human."""

    name: str
    groups: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    family: str  # "demand" | "human"
    secondary: bool = False

    def features_for(self, anchor: Anchor) -> tuple[str, ...]:
        if anchor == "ts":
            f0_full = tuple(TS_RESID_FEATURES)
            core = tuple(c for c in f0_full if c not in F0_DEMAND)
        elif anchor == "human":
            f0_full = tuple(BUDGET_RESID_FEATURES)
            core = tuple(c for c in f0_full if c not in F0_DEMAND)
        else:
            raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")

        if "f0_demand" in self.groups:
            out: list[str] = list(f0_full)  # frozen F0 order
        else:
            out = list(core)  # F0 order with demand block removed

        extra: list[str] = []
        if "f1_demand" in self.groups:
            extra.extend(F1_DEMAND)
        if "f2_demand" in self.groups:
            extra.extend(F2_DEMAND)
        if "f1_human" in self.groups:
            extra.extend(F1_HUMAN)
        if "f2_human" in self.groups:
            extra.extend(F2_HUMAN)
        seen = set(out)
        for c in extra:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return tuple(out)


DEMAND_EXPERIMENTS: tuple[AblationExperiment, ...] = (
    AblationExperiment("D0_CORE", (), ("ts", "human"), "demand"),
    AblationExperiment("D1_F0", ("f0_demand",), ("ts", "human"), "demand"),
    AblationExperiment("D2_F1_REPLACE", ("f1_demand",), ("ts", "human"), "demand"),
    AblationExperiment("D3_F2_REPLACE", ("f2_demand",), ("ts", "human"), "demand"),
    AblationExperiment(
        "D4_F1_ADD", ("f0_demand", "f1_demand"), ("ts", "human"), "demand"
    ),
    AblationExperiment(
        "D5_F2_ADD", ("f0_demand", "f2_demand"), ("ts", "human"), "demand"
    ),
)

HUMAN_EXPERIMENTS: tuple[AblationExperiment, ...] = (
    AblationExperiment("H0_CORE", (), ("human",), "human"),
    AblationExperiment("H1_F0", ("f0_demand",), ("human",), "human"),
    AblationExperiment("H2_F1_HUMAN_ONLY", ("f1_human",), ("human",), "human"),
    AblationExperiment("H3_F2_HUMAN_ONLY", ("f2_human",), ("human",), "human"),
    AblationExperiment(
        "H4_F1_HUMAN_ADD", ("f0_demand", "f1_human"), ("human",), "human"
    ),
    AblationExperiment(
        "H5_F2_HUMAN_ADD", ("f0_demand", "f2_human"), ("human",), "human"
    ),
    AblationExperiment(
        "H6_F1_DEMAND_HUMAN",
        ("f1_demand", "f1_human"),
        ("human",),
        "human",
        secondary=True,
    ),
    AblationExperiment(
        "H7_F2_DEMAND_HUMAN",
        ("f2_demand", "f2_human"),
        ("human",),
        "human",
        secondary=True,
    ),
)

ALL_EXPERIMENTS: dict[str, AblationExperiment] = {
    e.name: e for e in DEMAND_EXPERIMENTS + HUMAN_EXPERIMENTS
}


def get_ablation(name: str) -> AblationExperiment:
    if name not in ALL_EXPERIMENTS:
        raise KeyError(f"Unknown ablation {name!r}; known={sorted(ALL_EXPERIMENTS)}")
    return ALL_EXPERIMENTS[name]


def ablation_output_dir() -> Path:
    src_dir = Path(__file__).resolve().parents[3]
    out = src_dir / "data" / "results" / "feature_ablation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "docs"
