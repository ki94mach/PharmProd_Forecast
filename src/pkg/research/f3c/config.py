"""F3C paths, experiment specs, and frozen feature lists.

Does not mutate F0/F1/F2/F3A/F3B artifacts, the v1 freeze, or Step 1/2
source/feature definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES

Anchor = Literal["ts", "human"]

CURRENT_ENV_F0_WMAPE = {
    "ts": 37.23014,
    "human": 36.69475,
}
WMAPE_REPRO_TOL = 0.05
F0_N = 1877
F0_N_ORIGINS = 5

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

# ── Scored inventory features (predeclared before any WMAPE) ──
INVENTORY_FEATURE_NAMES: tuple[str, ...] = (
    "log_distributor_inventory_qty",
    "log_factory_inventory_qty",
)

NEVER_FILLNA: frozenset[str] = frozenset(INVENTORY_FEATURE_NAMES + (
    "distributor_inventory_qty",
    "factory_inventory_qty",
))

# ── Predeclared experiment families (frozen before Step 3 results) ──
# F3C-A: distributor only
# F3C-B: distributor + factory
F3C_A_FEATURES: tuple[str, ...] = ("log_distributor_inventory_qty",)
F3C_B_FEATURES: tuple[str, ...] = (
    "log_distributor_inventory_qty",
    "log_factory_inventory_qty",
)

PAIRS: tuple[tuple[str, str], ...] = (
    ("I1_TS_DISTRIBUTOR", "I0_TS"),
    ("I2_TS_DISTRIBUTOR_FACTORY", "I0_TS"),
    ("I1_HUMAN_DISTRIBUTOR", "I0_HUMAN"),
    ("I2_HUMAN_DISTRIBUTOR_FACTORY", "I0_HUMAN"),
)


@dataclass(frozen=True)
class F3CExperiment:
    name: str
    anchor: Anchor
    inventory_features: tuple[str, ...]
    train_universe: str
    control: str
    use_frozen_adapter: bool

    def features(self) -> tuple[str, ...]:
        if self.anchor == "ts":
            base = tuple(TS_RESID_FEATURES)
        else:
            base = tuple(BUDGET_RESID_FEATURES)
        if self.inventory_features:
            extra = tuple(c for c in self.inventory_features if c not in base)
            return base + extra
        return base


I0_TS = F3CExperiment("I0_TS", "ts", (), "ts", "I0_TS", True)
I1_TS = F3CExperiment("I1_TS_DISTRIBUTOR", "ts", F3C_A_FEATURES, "ts", "I0_TS", False)
I2_TS = F3CExperiment("I2_TS_DISTRIBUTOR_FACTORY", "ts", F3C_B_FEATURES, "ts", "I0_TS", False)

I0_HUMAN = F3CExperiment("I0_HUMAN", "human", (), "budget", "I0_HUMAN", True)
I1_HUMAN = F3CExperiment("I1_HUMAN_DISTRIBUTOR", "human", F3C_A_FEATURES, "budget", "I0_HUMAN", False)
I2_HUMAN = F3CExperiment("I2_HUMAN_DISTRIBUTOR_FACTORY", "human", F3C_B_FEATURES, "budget", "I0_HUMAN", False)

ALL_EXPERIMENTS: dict[str, F3CExperiment] = {
    e.name: e for e in (I0_TS, I1_TS, I2_TS, I0_HUMAN, I1_HUMAN, I2_HUMAN)
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def f3c_output_dir() -> Path:
    out = src_dir() / "data" / "results" / "f3c"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3c_source_dir() -> Path:
    out = f3c_output_dir() / "source"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3c_feature_audit_dir() -> Path:
    out = f3c_output_dir() / "feature_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return repo_root() / "docs"
