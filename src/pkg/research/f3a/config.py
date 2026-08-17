"""F3A product-lifecycle experiment configuration.

Does not mutate F0 / F1 / F2 / ablation registries. CORE_TS is imported from
the frozen feature-family ablation partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.ablation.config import CORE_TS
from pkg.research.features.lifecycle import FEATURE_NAMES, SCORED_FEATURE

Anchor = Literal["ts", "human"]

# Current-environment reproduced F0 (not the locked freeze-time contract).
CURRENT_ENV_F0_WMAPE = {
    "ts": 38.2848,
    "human": 36.5602,
}
CORE_TS_WMAPE_REF = 37.7360
WMAPE_REPRO_TOL = 0.05
PRED_REPRO_TOL = 1e-3
F0_N = 1877
F0_N_ORIGINS = 5

# Fillna extras copied from ablation so F0 lag / CORE columns match prior runs.
# Lifecycle age is never in this set and is listed in NEVER_FILLNA.
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

NEVER_FILLNA: frozenset[str] = frozenset(
    {
        SCORED_FEATURE,
        "has_prior_positive_sale",
        "first_sale_left_censored",
        "first_positive_sale_month",
        "first_nonzero_sale_month",
        "earliest_available_sales_month",
    }
)

PAIRS: tuple[tuple[str, str], ...] = (
    ("T1", "T0"),
    ("T3", "T2"),
    ("H1", "H0"),
)


@dataclass(frozen=True)
class F3AExperiment:
    name: str
    anchor: Anchor
    feature_source: str  # "frozen" | "f0" | "core_ts"
    include_lifecycle: bool
    train_universe: str
    control: str  # experiment this is compared against (self for controls)

    def features(self) -> tuple[str, ...]:
        if self.feature_source == "frozen":
            if self.anchor == "ts":
                base = tuple(TS_RESID_FEATURES)
            else:
                base = tuple(BUDGET_RESID_FEATURES)
        elif self.feature_source == "f0":
            if self.anchor == "ts":
                base = tuple(TS_RESID_FEATURES)
            else:
                base = tuple(BUDGET_RESID_FEATURES)
        elif self.feature_source == "core_ts":
            base = tuple(CORE_TS)
        else:
            raise ValueError(f"unknown feature_source {self.feature_source!r}")
        if self.include_lifecycle:
            extra = tuple(c for c in FEATURE_NAMES if c not in base)
            return base + extra
        return base


T0 = F3AExperiment("T0", "ts", "frozen", False, "ts", "T0")
T1 = F3AExperiment("T1", "ts", "f0", True, "ts", "T0")
T2 = F3AExperiment("T2", "ts", "core_ts", False, "ts", "T2")
T3 = F3AExperiment("T3", "ts", "core_ts", True, "ts", "T2")
H0 = F3AExperiment("H0", "human", "frozen", False, "budget", "H0")
H1 = F3AExperiment("H1", "human", "f0", True, "budget", "H0")

ALL_EXPERIMENTS: dict[str, F3AExperiment] = {
    e.name: e for e in (T0, T1, T2, T3, H0, H1)
}


def get_f3a_experiment(name: str) -> F3AExperiment:
    if name not in ALL_EXPERIMENTS:
        raise KeyError(f"Unknown F3A experiment {name!r}; known={sorted(ALL_EXPERIMENTS)}")
    return ALL_EXPERIMENTS[name]


def f3a_output_dir() -> Path:
    src_dir = Path(__file__).resolve().parents[3]
    out = src_dir / "data" / "results" / "f3a"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "docs"
