"""F3D source/mapping tests (no live SQL, no XGB)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkg.research.f3d.prepare import (
    DuplicateConflictError,
    ZeroOverlapError,
    _validate_no_conflicts,
    build_product_profile,
)


def _make_dim(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exact-match mapping
# ---------------------------------------------------------------------------

class TestExactMapping:
    def test_matched_product_gets_profile(self):
        dim = _make_dim([
            {"ProductTitleEN": "Alpha", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 2.0},
        ])
        profile, audit = build_product_profile(["Alpha", "Beta"], dim)
        alpha = profile.loc[profile["product"] == "Alpha"].iloc[0]
        assert alpha["PatientConsumeType"] == "Continuous"
        assert float(alpha["PatientConsumePerPeriod"]) == pytest.approx(2.0)

    def test_unmatched_product_gets_nan(self):
        dim = _make_dim([
            {"ProductTitleEN": "Alpha", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 2.0},
        ])
        profile, _ = build_product_profile(["Alpha", "Beta"], dim)
        beta = profile.loc[profile["product"] == "Beta"].iloc[0]
        assert pd.isna(beta["PatientConsumeType"])
        assert pd.isna(beta["is_continuous_consumption"])
        assert pd.isna(beta["log_patient_annual_consumption"])

    def test_no_fuzzy_matching(self):
        """'alpha' != 'Alpha' — exact match only; zero overlap raises ZeroOverlapError."""
        dim = _make_dim([
            {"ProductTitleEN": "Alpha", "PatientConsumeType": "SinglePeriod",
             "PatientConsumePerPeriod": 6.0},
            # Add a second product so 'Alpha' is canonical but 'alpha' is not
        ])
        # 'Alpha' matches, 'alpha' does not
        profile, _ = build_product_profile(["Alpha", "alpha_lower"], dim)
        row = profile.loc[profile["product"] == "alpha_lower"].iloc[0]
        assert pd.isna(row["PatientConsumeType"])


# ---------------------------------------------------------------------------
# Duplicate-conflict detection
# ---------------------------------------------------------------------------

class TestDuplicateConflict:
    def test_identical_rows_ok(self):
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 3.0},
            {"ProductTitleEN": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 3.0},
        ])
        # Should not raise — truly identical rows are deduped
        result = _validate_no_conflicts(dim)
        assert len(result) == 1

    def test_conflicting_type_raises(self):
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 3.0},
            {"ProductTitleEN": "A", "PatientConsumeType": "SinglePeriod",
             "PatientConsumePerPeriod": 3.0},
        ])
        with pytest.raises(DuplicateConflictError):
            _validate_no_conflicts(dim)

    def test_conflicting_period_raises(self):
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 3.0},
            {"ProductTitleEN": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 4.0},
        ])
        with pytest.raises(DuplicateConflictError):
            _validate_no_conflicts(dim)


# ---------------------------------------------------------------------------
# Zero overlap
# ---------------------------------------------------------------------------

class TestZeroOverlap:
    def test_zero_overlap_raises(self):
        dim = _make_dim([
            {"ProductTitleEN": "NotInCanonical", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 1.0},
        ])
        with pytest.raises(ZeroOverlapError):
            build_product_profile(["Alpha", "Beta"], dim)


# ---------------------------------------------------------------------------
# Unexpected types not merged or remapped
# ---------------------------------------------------------------------------

class TestUnexpectedTypes:
    def test_unexpected_type_indicator_is_nan(self):
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Weekly",
             "PatientConsumePerPeriod": 5.0},
        ])
        profile, audit = build_product_profile(["A"], dim)
        row = profile.iloc[0]
        assert pd.isna(row["is_continuous_consumption"])
        assert pd.isna(row["patient_annual_consumption"])

    def test_unexpected_type_reported_in_audit(self):
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Weekly",
             "PatientConsumePerPeriod": 5.0},
        ])
        _, audit = build_product_profile(["A"], dim)
        assert int(audit["n_unexpected_types"].iloc[0]) == 1
        assert "Weekly" in str(audit["unexpected_types"].iloc[0])

    def test_unexpected_type_does_not_stop_execution(self):
        """Unexpected types should NOT raise; only DuplicateConflict/ZeroOverlap do."""
        dim = _make_dim([
            {"ProductTitleEN": "A", "PatientConsumeType": "Weekly",
             "PatientConsumePerPeriod": 5.0},
        ])
        # Should not raise
        profile, _ = build_product_profile(["A"], dim)
        assert len(profile) == 1
