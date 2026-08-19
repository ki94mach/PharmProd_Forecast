"""F3E paths, constants, and normalization rules.

Does not mutate F0–F3D artifacts, the v1 freeze, or any scored feature lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import (
    BUDGET_RESID_FEATURES,
    INCOMPLETE_SHAMSI_MONTHS,  # re-export frozen set
    TS_RESID_FEATURES,
)

# ── Source parquet filenames ─────────────────────────────────────────────────
NORMALIZED_MONTHLY_SALES_PARQUET = "normalized_monthly_sales.parquet"
PRODUCT_PEER_PROFILE_PARQUET = "product_peer_profile.parquet"

# ── Known PatientConsumeType values (confirmed business semantics) ───────────
# Continuous:  PatientConsumePerPeriod = quantity consumed per patient per MONTH
# SinglePeriod: PatientConsumePerPeriod = quantity consumed per patient per YEAR
# For F3E patient-equivalent conversion both use:
#     monthly_patient_equivalent = monthly_dqty / PatientConsumePerPeriod
# No ×12 or ÷12 is applied in F3E (unlike F3D annualization).
KNOWN_CONSUME_TYPES: frozenset[str] = frozenset({"Continuous", "SinglePeriod"})

# ── Normalization rule documentation ────────────────────────────────────────
# Same-generic (DQtyUnit):
#   monthly_dqtyunit = monthly_dqty * unit_ratio
#   Valid only when unit_ratio is finite and > 0.
#   Used ONLY within the same FKGeneric.
#   Unit is a within-generic conversion ratio; it is NOT assumed comparable
#   across different generics.
#
# Cross-generic (monthly_patient_equivalent):
#   monthly_patient_equivalent = monthly_dqty / PatientConsumePerPeriod
#   Valid only when PatientConsumeType ∈ KNOWN_CONSUME_TYPES
#   AND PatientConsumePerPeriod is finite and > 0.
#   Used ONLY across generics within the same Field × PatientConsumeType segment.
#   The target product's ENTIRE generic is excluded from cross-generic peers.
#
# These rules are frozen before any WMAPE is observed and must not be changed
# after Step 3 PRIMARY results are seen.


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def f3e_output_dir() -> Path:
    out = src_dir() / "data" / "results" / "f3e"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3e_source_dir() -> Path:
    out = f3e_output_dir() / "source"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return repo_root() / "docs"


# ── Feature families (frozen before any WMAPE) ───────────────────────────────
# F3E-A: same-generic demand only
F3E_A_FEATURES: tuple[str, ...] = (
    "log_generic_peer_dqtyunit_last_month",
    "log_generic_peer_dqtyunit_3m_mean",
)

# F3E-B: same-generic + cross-generic patient context
F3E_B_FEATURES: tuple[str, ...] = F3E_A_FEATURES + (
    "log_cross_generic_field_consume_patients_last_month",
    "log_cross_generic_field_consume_patients_3m_mean",
)

# ── Feature audit output dir ──────────────────────────────────────────────────
FEATURE_AUDIT_DIR_NAME = "feature_audit"


def f3e_feature_audit_dir() -> Path:
    out = f3e_output_dir() / FEATURE_AUDIT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Step 3: evaluation constants ─────────────────────────────────────────────

CURRENT_ENV_F0_WMAPE: dict[str, float] = {
    "ts": 37.201241,
    "human": 36.710510,
}

# Extra columns filled with 0 before XGBoost (same as F3D/F3C).
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

# F3E peer-demand features: XGBoost handles NaN natively (missing-value branch).
# Also include the raw aggregates so they are never zero-filled if present.
NEVER_FILLNA: frozenset[str] = frozenset(
    F3E_B_FEATURES
    + (
        "generic_peer_dqtyunit_last_month",
        "generic_peer_dqtyunit_3m_mean",
        "cross_generic_field_consume_patients_last_month",
        "cross_generic_field_consume_patients_3m_mean",
    )
)

# ── Experiment dataclass ──────────────────────────────────────────────────────

Anchor = Literal["ts", "human"]


@dataclass(frozen=True)
class F3EExperiment:
    name: str
    anchor: Anchor
    peer_features: tuple[str, ...]
    train_universe: str
    control: str
    use_frozen_adapter: bool

    def features(self) -> tuple[str, ...]:
        base: tuple[str, ...] = (
            tuple(TS_RESID_FEATURES) if self.anchor == "ts" else tuple(BUDGET_RESID_FEATURES)
        )
        if self.peer_features:
            extra = tuple(c for c in self.peer_features if c not in base)
            return base + extra
        return base


# ── Six experiments (frozen before any WMAPE) ────────────────────────────────

E0_TS = F3EExperiment(
    "E0_TS", "ts", (), "ts", "E0_TS", True
)
E1_TS_GENERIC = F3EExperiment(
    "E1_TS_GENERIC", "ts", F3E_A_FEATURES, "ts", "E0_TS", False
)
E2_TS_GENERIC_CROSS_PATIENT = F3EExperiment(
    "E2_TS_GENERIC_CROSS_PATIENT", "ts", F3E_B_FEATURES, "ts", "E1_TS_GENERIC", False
)

E0_HUMAN = F3EExperiment(
    "E0_HUMAN", "human", (), "budget", "E0_HUMAN", True
)
E1_HUMAN_GENERIC = F3EExperiment(
    "E1_HUMAN_GENERIC", "human", F3E_A_FEATURES, "budget", "E0_HUMAN", False
)
E2_HUMAN_GENERIC_CROSS_PATIENT = F3EExperiment(
    "E2_HUMAN_GENERIC_CROSS_PATIENT", "human", F3E_B_FEATURES, "budget", "E1_HUMAN_GENERIC", False
)

ALL_EXPERIMENTS: dict[str, F3EExperiment] = {
    e.name: e
    for e in (
        E0_TS,
        E1_TS_GENERIC,
        E2_TS_GENERIC_CROSS_PATIENT,
        E0_HUMAN,
        E1_HUMAN_GENERIC,
        E2_HUMAN_GENERIC_CROSS_PATIENT,
    )
}

# Comparison pairs (candidate, control) for reporting
PAIRS: tuple[tuple[str, str], ...] = (
    # Question A: same-generic demand vs F0
    ("E1_TS_GENERIC", "E0_TS"),
    ("E1_HUMAN_GENERIC", "E0_HUMAN"),
    # Question B: cross-generic vs same-generic
    ("E2_TS_GENERIC_CROSS_PATIENT", "E1_TS_GENERIC"),
    ("E2_HUMAN_GENERIC_CROSS_PATIENT", "E1_HUMAN_GENERIC"),
)
