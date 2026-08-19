"""F3D paths, experiment specs, and frozen feature lists.

Does not mutate F0/F1/F2/F3A/F3B/F3C artifacts, the v1 freeze, or Step 1
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

# Fillna extras — same lag/trend columns as F3C so F0 control baseline matches.
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

# Patient-consumption profile features are never filled; XGBoost sees NaN as
# missing and handles it natively via its internal missing-value branch.
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
    "is_continuous_consumption",
    "log_patient_annual_consumption",
)

NEVER_FILLNA: frozenset[str] = frozenset(
    PROFILE_FEATURE_NAMES
    + (
        "patient_annual_consumption",
        "PatientConsumePerPeriod",
    )
)

# ── Predeclared experiment families (frozen BEFORE any WMAPE) ──────────────
# F3D-A: consume-type indicator only
# F3D-B: consume-type indicator + log annual consumption
F3D_A_FEATURES: tuple[str, ...] = ("is_continuous_consumption",)
F3D_B_FEATURES: tuple[str, ...] = (
    "is_continuous_consumption",
    "log_patient_annual_consumption",
)

PAIRS: tuple[tuple[str, str], ...] = (
    # Question A: type indicator alone
    ("D1_TS_TYPE", "D0_TS"),
    ("D1_HUMAN_TYPE", "D0_HUMAN"),
    # Question B: profile vs type
    ("D2_TS_PROFILE", "D1_TS_TYPE"),
    ("D2_HUMAN_PROFILE", "D1_HUMAN_TYPE"),
)


@dataclass(frozen=True)
class F3DExperiment:
    name: str
    anchor: Anchor
    profile_features: tuple[str, ...]
    train_universe: str
    control: str
    use_frozen_adapter: bool

    def features(self) -> tuple[str, ...]:
        if self.anchor == "ts":
            base = tuple(TS_RESID_FEATURES)
        else:
            base = tuple(BUDGET_RESID_FEATURES)
        if self.profile_features:
            extra = tuple(c for c in self.profile_features if c not in base)
            return base + extra
        return base


D0_TS = F3DExperiment("D0_TS", "ts", (), "ts", "D0_TS", True)
D1_TS = F3DExperiment("D1_TS_TYPE", "ts", F3D_A_FEATURES, "ts", "D0_TS", False)
D2_TS = F3DExperiment("D2_TS_PROFILE", "ts", F3D_B_FEATURES, "ts", "D1_TS_TYPE", False)

D0_HUMAN = F3DExperiment("D0_HUMAN", "human", (), "budget", "D0_HUMAN", True)
D1_HUMAN = F3DExperiment("D1_HUMAN_TYPE", "human", F3D_A_FEATURES, "budget", "D0_HUMAN", False)
D2_HUMAN = F3DExperiment("D2_HUMAN_PROFILE", "human", F3D_B_FEATURES, "budget", "D1_HUMAN_TYPE", False)

ALL_EXPERIMENTS: dict[str, F3DExperiment] = {
    e.name: e for e in (D0_TS, D1_TS, D2_TS, D0_HUMAN, D1_HUMAN, D2_HUMAN)
}


def get_f3d_experiment(name: str) -> F3DExperiment:
    if name not in ALL_EXPERIMENTS:
        raise KeyError(f"Unknown F3D experiment {name!r}; known={sorted(ALL_EXPERIMENTS)}")
    return ALL_EXPERIMENTS[name]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def f3d_output_dir() -> Path:
    out = src_dir() / "data" / "results" / "f3d"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3d_source_dir() -> Path:
    out = f3d_output_dir() / "source"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3d_profile_audit_dir() -> Path:
    out = f3d_output_dir() / "profile_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return repo_root() / "docs"
