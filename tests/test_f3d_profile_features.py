"""F3D feature-transform tests (no live SQL, no XGB)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkg.research.features.patient_consumption import (
    FEATURE_NAMES,
    KNOWN_TYPES,
    _compute_annual,
    _compute_indicator,
    add_patient_consumption_features,
)


def _make_panel(products: list[str]) -> pd.DataFrame:
    rows = []
    for p in products:
        for origin in [140404, 140407]:
            rows.append({"product": p, "origin": origin, "horizon": 1})
    return pd.DataFrame(rows)


def _make_profile(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annualisation logic
# ---------------------------------------------------------------------------

class TestAnnualCompute:
    def test_continuous_x12(self):
        assert _compute_annual("Continuous", 2.0) == pytest.approx(24.0)

    def test_continuous_fraction(self):
        assert _compute_annual("Continuous", 0.5) == pytest.approx(6.0)

    def test_single_period_as_is(self):
        assert _compute_annual("SinglePeriod", 6.0) == pytest.approx(6.0)

    def test_missing_type_is_nan(self):
        assert np.isnan(_compute_annual(None, 3.0))
        assert np.isnan(_compute_annual(float("nan"), 3.0))

    def test_missing_period_is_nan(self):
        assert np.isnan(_compute_annual("Continuous", None))
        assert np.isnan(_compute_annual("Continuous", float("nan")))

    def test_unexpected_type_is_nan(self):
        assert np.isnan(_compute_annual("Weekly", 3.0))
        assert np.isnan(_compute_annual("Daily", 1.0))

    def test_no_extra_multiplier_for_single_period(self):
        """SinglePeriod must NOT be multiplied or divided by 12."""
        result = _compute_annual("SinglePeriod", 6.0)
        assert result != pytest.approx(72.0)
        assert result != pytest.approx(0.5)
        assert result == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Indicator encoding
# ---------------------------------------------------------------------------

class TestIndicator:
    def test_continuous_is_one(self):
        assert _compute_indicator("Continuous") == pytest.approx(1.0)

    def test_single_period_is_zero(self):
        assert _compute_indicator("SinglePeriod") == pytest.approx(0.0)

    def test_missing_is_nan(self):
        assert np.isnan(_compute_indicator(None))
        assert np.isnan(_compute_indicator(float("nan")))

    def test_unexpected_is_nan(self):
        assert np.isnan(_compute_indicator("Weekly"))

    def test_no_ordinal_hardcode(self):
        """Continuous must not be 0; SinglePeriod must not be 1."""
        assert _compute_indicator("Continuous") != pytest.approx(0.0)
        assert _compute_indicator("SinglePeriod") != pytest.approx(1.0)


# ---------------------------------------------------------------------------
# log1p transform
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_log1p_continuous(self):
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 2.0},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        expected = np.log1p(24.0)
        assert float(out["log_patient_annual_consumption"].iloc[0]) == pytest.approx(expected)

    def test_log1p_single_period(self):
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "SinglePeriod",
             "PatientConsumePerPeriod": 6.0},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        expected = np.log1p(6.0)
        assert float(out["log_patient_annual_consumption"].iloc[0]) == pytest.approx(expected)

    def test_missing_period_log_is_nan(self):
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": float("nan")},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        assert pd.isna(out["log_patient_annual_consumption"].iloc[0])

    def test_negative_period_log_is_nan(self):
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": -1.0},
        ])
        panel = _make_panel(["A"])
        neg_report: list = []
        out = add_patient_consumption_features(panel, profile, negative_report=neg_report)
        assert pd.isna(out["log_patient_annual_consumption"].iloc[0])
        assert len(neg_report) >= 1

    def test_zero_period_log_is_log1p_zero(self):
        """Zero is valid non-negative; log1p(0) == 0.0."""
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 0.0},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        assert float(out["log_patient_annual_consumption"].iloc[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Static feature: same value across origins / horizons
# ---------------------------------------------------------------------------

class TestStaticFeature:
    def test_same_annual_across_origins(self):
        """Static product attributes must be identical for every origin."""
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 3.0},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        vals = out["patient_annual_consumption"].unique()
        assert len(vals) == 1
        assert vals[0] == pytest.approx(36.0)

    def test_same_indicator_across_origins(self):
        profile = _make_profile([
            {"product": "A", "PatientConsumeType": "SinglePeriod",
             "PatientConsumePerPeriod": 5.0},
        ])
        panel = _make_panel(["A"])
        out = add_patient_consumption_features(panel, profile)
        inds = out["is_continuous_consumption"].unique()
        assert len(inds) == 1
        assert inds[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Missing product in profile → all NaN
# ---------------------------------------------------------------------------

class TestMissingProduct:
    def test_unmatched_product_all_nan(self):
        profile = _make_profile([
            {"product": "Known", "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 1.0},
        ])
        panel = _make_panel(["Unknown"])
        out = add_patient_consumption_features(panel, profile)
        assert pd.isna(out["is_continuous_consumption"]).all()
        assert pd.isna(out["log_patient_annual_consumption"]).all()
        assert pd.isna(out["patient_annual_consumption"]).all()


# ---------------------------------------------------------------------------
# FEATURE_NAMES contract
# ---------------------------------------------------------------------------

class TestFeatureNamesContract:
    def test_exactly_two_scored_features(self):
        assert len(FEATURE_NAMES) == 2

    def test_feature_names_match(self):
        assert "is_continuous_consumption" in FEATURE_NAMES
        assert "log_patient_annual_consumption" in FEATURE_NAMES

    def test_known_types(self):
        assert "Continuous" in KNOWN_TYPES
        assert "SinglePeriod" in KNOWN_TYPES
        assert len(KNOWN_TYPES) == 2
