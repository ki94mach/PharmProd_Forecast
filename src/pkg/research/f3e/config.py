"""F3E paths, constants, and normalization rules.

Does not mutate F0–F3D artifacts, the v1 freeze, or any scored feature lists.
"""
from __future__ import annotations

from pathlib import Path

from pkg.benchmark.config import INCOMPLETE_SHAMSI_MONTHS  # re-export frozen set

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
