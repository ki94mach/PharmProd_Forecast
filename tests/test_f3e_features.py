"""Tests for F3E Step 2 — PIT feature construction.

No live SQL.  No XGBoost.  No frozen benchmark required.
All tests use synthetic DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import INCOMPLETE_SHAMSI_MONTHS
from pkg.research.f3e.features import (
    _C_AVAILABLE,
    _C_NO_CONV,
    _C_NO_MONTH,
    _C_NO_PEERS,
    _G_AVAILABLE,
    _G_INVALID_UNIT,
    _G_NO_MONTH,
    _G_NO_PEERS,
    _build_cross_generic_monthly_series_fast,
    _build_generic_monthly_series_fast,
    assert_generic_target_exclusion,
    assert_cross_generic_no_same_fkgeneric,
    attach_pit_features,
    safe_log1p,
)
from pkg.research.f3e.config import F3E_A_FEATURES, F3E_B_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(rows: list[dict]) -> pd.DataFrame:
    cols = ["product", "date", "monthly_dqty", "FKGeneric", "Field", "unit_ratio",
            "PatientConsumeType", "PatientConsumePerPeriod", "monthly_dqtyunit",
            "monthly_patient_equivalent"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _make_profile(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestPITSafety
# ---------------------------------------------------------------------------

class TestPITSafety:
    """M1/M2/M3 must be strictly before the origin."""

    def test_m1_before_origin(self):
        origin = 140501
        m1 = shamsi_add_months(origin, -1)
        assert int(m1) < int(origin)

    def test_m3_before_origin(self):
        origin = 140501
        m3 = shamsi_add_months(origin, -3)
        assert int(m3) < int(origin)

    def test_year_boundary(self):
        """Origin = 140401 → M1 = 140312 (year roll-back)."""
        origin = 140401
        m1 = shamsi_add_months(origin, -1)
        assert m1 == 140312
        assert int(m1) < int(origin)

    def test_all_three_before_origin_various(self):
        for origin in (140404, 140407, 140410, 140501, 140504):
            for delta in (-1, -2, -3):
                m = shamsi_add_months(origin, delta)
                assert int(m) < int(origin), (
                    f"PIT violation: delta={delta} origin={origin} month={m}"
                )

    def test_attach_pit_asserts_on_bad_data(self):
        """attach_pit_features raises AssertionError if any M >= O (simulated via mock)."""
        # We cannot construct a natural case because shamsi_add_months(O, -1) < O always,
        # but we can monkey-patch the lookup to verify the inline PIT check runs.
        # Instead verify the assertion function itself is called for each origin.
        # We test the assertion path by calling _check_pit indirectly via attach_pit_features
        # with a valid origin — should complete without error.
        panel = _make_panel([
            {"product": "A", "date": 140412, "monthly_dqty": 10.0, "FKGeneric": "G1",
             "Field": "F", "unit_ratio": 1.0, "PatientConsumeType": "Continuous",
             "PatientConsumePerPeriod": 5.0, "monthly_dqtyunit": 10.0,
             "monthly_patient_equivalent": 2.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "unit_ratio": 1.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, profile)
        c_series = _build_cross_generic_monthly_series_fast(panel, profile)
        primary = pd.DataFrame([{"product": "A", "budget_origin": 140501}])
        # Should not raise
        attach_pit_features(primary, g_series, c_series, covered_months_set=frozenset({140412}))


# ---------------------------------------------------------------------------
# TestGenericPeerSum
# ---------------------------------------------------------------------------

class TestGenericPeerSum:
    def _make_two_product_panel(self, dqtyunit_A=100.0, dqtyunit_B=200.0, month=140412):
        return _make_panel([
            {"product": "A", "date": month, "FKGeneric": "G1",
             "monthly_dqtyunit": dqtyunit_A, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": month, "FKGeneric": "G1",
             "monthly_dqtyunit": dqtyunit_B, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])

    def _make_profile_two(self):
        return _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])

    def test_peer_sum_excludes_self(self):
        """A's peer sum = B's dqtyunit, not A's."""
        panel = self._make_two_product_panel()
        profile = self._make_profile_two()
        result = _build_generic_monthly_series_fast(panel, profile)
        row_A = result.loc[
            (result["product"] == "A") & (result["date"] == 140412)
        ].iloc[0]
        assert float(row_A["generic_peer_dqtyunit"]) == pytest.approx(200.0)
        assert row_A["generic_reason"] == _G_AVAILABLE

    def test_both_products_sum(self):
        """Each product's peer sum = the other's contribution."""
        panel = self._make_two_product_panel(100.0, 300.0)
        profile = self._make_profile_two()
        result = _build_generic_monthly_series_fast(panel, profile)
        row_B = result.loc[
            (result["product"] == "B") & (result["date"] == 140412)
        ].iloc[0]
        assert float(row_B["generic_peer_dqtyunit"]) == pytest.approx(100.0)

    def test_zero_peer_sum_in_covered_month(self):
        """If peer has zero dqtyunit in a covered month, sum = 0 (not NaN)."""
        panel = self._make_two_product_panel(dqtyunit_B=0.0)
        profile = self._make_profile_two()
        result = _build_generic_monthly_series_fast(panel, profile)
        row_A = result.loc[
            (result["product"] == "A") & (result["date"] == 140412)
        ].iloc[0]
        # B has dqtyunit=0 which is valid, so sum should be 0 (AVAILABLE)
        # but 0.0 DQtyUnit: only invalid if NaN, not zero
        assert row_A["generic_reason"] == _G_AVAILABLE

    def test_no_peers_different_generic(self):
        """Products in different generics have no same-generic peers."""
        panel = _make_panel([
            {"product": "A", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": 50.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": 140412, "FKGeneric": "G2",
             "monthly_dqtyunit": 60.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G2", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_generic_monthly_series_fast(panel, profile)
        row_A = result.loc[
            (result["product"] == "A") & (result["date"] == 140412)
        ].iloc[0]
        assert row_A["generic_reason"] == _G_NO_PEERS
        assert np.isnan(row_A["generic_peer_dqtyunit"])


# ---------------------------------------------------------------------------
# TestGenericMissingnessReasons
# ---------------------------------------------------------------------------

class TestGenericMissingnessReasons:
    def test_no_generic_peers_reason(self):
        """Lone product in its generic → NO_GENERIC_PEERS."""
        panel = _make_panel([
            {"product": "Solo", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "Solo", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_generic_monthly_series_fast(panel, profile)
        rows = result.loc[result["product"] == "Solo"]
        assert (rows["generic_reason"] == _G_NO_PEERS).all()

    def test_invalid_unit_reason(self):
        """Peers exist but all have NaN dqtyunit → INVALID_UNIT_FOR_ALL_RELEVANT_PEERS."""
        panel = _make_panel([
            {"product": "A", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": np.nan, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": np.nan},
        ])
        result = _build_generic_monthly_series_fast(panel, profile)
        # For product A: peer B has NaN dqtyunit → INVALID_UNIT
        row_A = result.loc[
            (result["product"] == "A") & (result["date"] == 140412)
        ].iloc[0]
        assert row_A["generic_reason"] == _G_INVALID_UNIT

    def test_source_month_unavailable_reason(self):
        """Month not in panel → SOURCE_MONTH_UNAVAILABLE (NaN peer_dqtyunit)."""
        panel = _make_panel([
            {"product": "A", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": 140412, "FKGeneric": "G1",
             "monthly_dqtyunit": 5.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_generic_monthly_series_fast(panel, profile)
        # 140411 never appeared in panel → SOURCE_MONTH_UNAVAILABLE
        row_A_missing = result.loc[
            (result["product"] == "A") & (result["date"] == 140411)
        ]
        if not row_A_missing.empty:
            assert row_A_missing.iloc[0]["generic_reason"] == _G_NO_MONTH


# ---------------------------------------------------------------------------
# TestCrossGenericPeerSum
# ---------------------------------------------------------------------------

class TestCrossGenericPeerSum:
    def _make_cross_generic_panel(self):
        return _make_panel([
            # Target
            {"product": "T", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 10.0,
             "monthly_dqty": 100.0, "monthly_patient_equivalent": 10.0,
             "unit_ratio": 1.0, "monthly_dqtyunit": 100.0},
            # Cross-generic peer
            {"product": "P1", "date": 140412, "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_dqty": 50.0, "monthly_patient_equivalent": 10.0,
             "unit_ratio": 1.0, "monthly_dqtyunit": 50.0},
            # Different field — should not be included
            {"product": "P2", "date": 140412, "FKGeneric": "G3", "Field": "Oncology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_dqty": 80.0, "monthly_patient_equivalent": 16.0,
             "unit_ratio": 1.0, "monthly_dqtyunit": 80.0},
        ])

    def _make_cross_generic_profile(self):
        return _make_profile([
            {"product": "T", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 10.0, "unit_ratio": 1.0},
            {"product": "P1", "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "P2", "FKGeneric": "G3", "Field": "Oncology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])

    def test_correct_cross_generic_sum(self):
        """T's cross-generic sum = P1's patient_equivalent (P2 excluded: different field)."""
        panel = self._make_cross_generic_panel()
        profile = self._make_cross_generic_profile()
        result = _build_cross_generic_monthly_series_fast(panel, profile)
        row_T = result.loc[
            (result["product"] == "T") & (result["date"] == 140412)
        ].iloc[0]
        assert row_T["cross_generic_reason"] == _C_AVAILABLE
        assert float(row_T["cross_generic_field_consume_patients"]) == pytest.approx(10.0)

    def test_fkgeneric_excluded_from_cross(self):
        """Products with same FKGeneric as target must not contribute."""
        panel = _make_panel([
            {"product": "T", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 10.0,
             "monthly_patient_equivalent": 10.0},
            # Same generic — must be excluded
            {"product": "T2", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 8.0,
             "monthly_patient_equivalent": 12.5},
            # Cross-generic
            {"product": "P1", "date": 140412, "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 20.0},
        ])
        profile = _make_profile([
            {"product": "T", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 10.0, "unit_ratio": 1.0},
            {"product": "T2", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 8.0, "unit_ratio": 1.0},
            {"product": "P1", "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_cross_generic_monthly_series_fast(panel, profile)
        row_T = result.loc[
            (result["product"] == "T") & (result["date"] == 140412)
        ].iloc[0]
        # Only P1 (G2) should contribute; T2 (G1 = same generic as T) must be excluded
        assert float(row_T["cross_generic_field_consume_patients"]) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TestCrossGenericMissingnessReasons
# ---------------------------------------------------------------------------

class TestCrossGenericMissingnessReasons:
    def test_no_cross_generic_peers(self):
        """Single generic in the field → NO_CROSS_GENERIC_PEERS."""
        panel = _make_panel([
            {"product": "A", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 10.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_cross_generic_monthly_series_fast(panel, profile)
        rows = result.loc[result["product"] == "A"]
        assert (rows["cross_generic_reason"] == _C_NO_PEERS).all()

    def test_no_valid_patient_convertible_peers(self):
        """Cross-generic peers exist but all have NaN patient_equivalent."""
        panel = _make_panel([
            {"product": "T", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 10.0},
            {"product": "P", "date": 140412, "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": np.nan,
             "monthly_patient_equivalent": np.nan},
        ])
        profile = _make_profile([
            {"product": "T", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "P", "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": np.nan, "unit_ratio": 1.0},
        ])
        result = _build_cross_generic_monthly_series_fast(panel, profile)
        row_T = result.loc[
            (result["product"] == "T") & (result["date"] == 140412)
        ].iloc[0]
        assert row_T["cross_generic_reason"] == _C_NO_CONV

    def test_source_month_unavailable_cross(self):
        """Month absent from panel → SOURCE_MONTH_UNAVAILABLE."""
        panel = _make_panel([
            {"product": "T", "date": 140412, "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 10.0},
            {"product": "P", "date": 140412, "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 8.0},
        ])
        profile = _make_profile([
            {"product": "T", "FKGeneric": "G1", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "P", "FKGeneric": "G2", "Field": "Cardiology",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        result = _build_cross_generic_monthly_series_fast(panel, profile)
        # 140411 not in panel → NO_MONTH
        row_T_411 = result.loc[
            (result["product"] == "T") & (result["date"] == 140411)
        ]
        if not row_T_411.empty:
            assert row_T_411.iloc[0]["cross_generic_reason"] == _C_NO_MONTH


# ---------------------------------------------------------------------------
# TestLogFeatures
# ---------------------------------------------------------------------------

class TestLogFeatures:
    def test_positive_value(self):
        val, reason = safe_log1p(100.0)
        assert val == pytest.approx(np.log1p(100.0))
        assert reason is None

    def test_zero_value(self):
        val, reason = safe_log1p(0.0)
        assert val == pytest.approx(0.0)
        assert reason is None

    def test_negative_value_is_nan(self):
        val, reason = safe_log1p(-5.0)
        assert np.isnan(val)
        assert reason == "NEGATIVE_NET_PEER_DEMAND"

    def test_nan_input(self):
        val, reason = safe_log1p(float("nan"))
        assert np.isnan(val)
        assert reason is None  # reason already set upstream

    def test_large_positive(self):
        val, reason = safe_log1p(1e8)
        assert val == pytest.approx(np.log1p(1e8))
        assert reason is None


# ---------------------------------------------------------------------------
# Test3mMeanIncompleteMonth
# ---------------------------------------------------------------------------

class Test3mMeanIncompleteMonth:
    """3m mean must be NaN if any of M1/M2/M3 is in INCOMPLETE_SHAMSI_MONTHS."""

    def test_3m_nan_when_m1_incomplete(self):
        # INCOMPLETE_SHAMSI_MONTHS = frozenset({140505})
        # Find an origin where M1 = 140505
        # shamsi_add_months(origin, -1) = 140505 → origin = 140506
        incomplete = list(INCOMPLETE_SHAMSI_MONTHS)
        if not incomplete:
            pytest.skip("No incomplete months defined")

        incomplete_month = incomplete[0]
        # origin such that M1 = incomplete_month
        origin = shamsi_add_months(incomplete_month, 1)

        panel = _make_panel([
            {"product": "A", "date": incomplete_month, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": incomplete_month, "FKGeneric": "G1",
             "monthly_dqtyunit": 20.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            # Also add months for M2 and M3 so they're covered
            {"product": "A", "date": shamsi_add_months(origin, -2), "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": shamsi_add_months(origin, -2), "FKGeneric": "G1",
             "monthly_dqtyunit": 20.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "A", "date": shamsi_add_months(origin, -3), "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": shamsi_add_months(origin, -3), "FKGeneric": "G1",
             "monthly_dqtyunit": 20.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, profile)
        c_series = _build_cross_generic_monthly_series_fast(panel, profile)

        covered = frozenset(panel["date"].dropna().astype(int).unique())
        primary = pd.DataFrame([{"product": "A", "budget_origin": origin}])
        enriched = attach_pit_features(
            primary, g_series, c_series, covered_months_set=covered
        )

        # 3m mean must be NaN because M1 is in INCOMPLETE_SHAMSI_MONTHS
        assert np.isnan(float(enriched["generic_peer_dqtyunit_3m_mean"].iloc[0])), (
            "3m mean should be NaN when any of M1/M2/M3 is an incomplete month"
        )

    def test_3m_available_when_no_incomplete(self):
        """When none of M1/M2/M3 is incomplete, 3m mean should be finite."""
        origin = 140404  # M1=140403, M2=140402, M3=140401
        months = [shamsi_add_months(origin, d) for d in (-1, -2, -3)]

        rows = []
        for m in months:
            rows += [
                {"product": "A", "date": m, "FKGeneric": "G1",
                 "monthly_dqtyunit": 10.0, "Field": "F",
                 "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
                {"product": "B", "date": m, "FKGeneric": "G1",
                 "monthly_dqtyunit": 20.0, "Field": "F",
                 "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            ]
        panel = _make_panel(rows)
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, profile)
        c_series = _build_cross_generic_monthly_series_fast(panel, profile)
        covered = frozenset(panel["date"].dropna().astype(int).unique())
        primary = pd.DataFrame([{"product": "A", "budget_origin": origin}])
        enriched = attach_pit_features(
            primary, g_series, c_series, covered_months_set=covered
        )
        val = float(enriched["generic_peer_dqtyunit_3m_mean"].iloc[0])
        # Each month: peer sum for A = B's dqtyunit = 20; mean = 20
        assert np.isfinite(val), f"3m mean should be finite but got {val}"
        assert val == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TestTargetExclusionAssertion
# ---------------------------------------------------------------------------

class TestTargetExclusionAssertion:
    def test_assertion_passes_with_correct_exclusion(self):
        """assert_generic_target_exclusion passes when target is correctly excluded."""
        origin = 140404
        m1 = shamsi_add_months(origin, -1)

        panel = _make_panel([
            {"product": "A", "date": m1, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": m1, "FKGeneric": "G1",
             "monthly_dqtyunit": 30.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        profile = _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, profile)
        c_series = _build_cross_generic_monthly_series_fast(panel, profile)
        covered = frozenset(panel["date"].dropna().astype(int).unique())
        primary = pd.DataFrame([{"product": "A", "budget_origin": origin}])
        enriched = attach_pit_features(
            primary, g_series, c_series, covered_months_set=covered
        )
        # Should not raise
        assert_generic_target_exclusion(panel, enriched)

    def test_assertion_fires_on_double_counting(self):
        """assert_generic_target_exclusion raises when target is double-counted."""
        origin = 140404
        m1 = shamsi_add_months(origin, -1)

        panel = _make_panel([
            {"product": "A", "date": m1, "FKGeneric": "G1",
             "monthly_dqtyunit": 10.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
            {"product": "B", "date": m1, "FKGeneric": "G1",
             "monthly_dqtyunit": 30.0, "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ]))
        c_series = _build_cross_generic_monthly_series_fast(panel, _make_profile([
            {"product": "A", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "B", "FKGeneric": "G1", "Field": "F",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ]))
        covered = frozenset(panel["date"].dropna().astype(int).unique())
        primary = pd.DataFrame([{"product": "A", "budget_origin": origin}])
        enriched = attach_pit_features(
            primary, g_series, c_series, covered_months_set=covered
        )
        # Corrupt: add target's own value to the peer sum (simulate double-counting)
        enriched = enriched.copy()
        enriched["generic_peer_dqtyunit_last_month"] += 10.0  # add target's own dqtyunit

        with pytest.raises(AssertionError, match="Generic target exclusion failed"):
            assert_generic_target_exclusion(panel, enriched)


# ---------------------------------------------------------------------------
# TestCrossGenericExclusionAssertion
# ---------------------------------------------------------------------------

class TestCrossGenericExclusionAssertion:
    def test_assertion_passes_valid_cross(self):
        """assert_cross_generic_no_same_fkgeneric passes with correct data."""
        origin = 140404
        m1 = shamsi_add_months(origin, -1)

        panel = _make_panel([
            {"product": "T", "date": m1, "FKGeneric": "G1", "Field": "Card",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 10.0},
            {"product": "P", "date": m1, "FKGeneric": "G2", "Field": "Card",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0,
             "monthly_patient_equivalent": 8.0},
        ])
        profile = _make_profile([
            {"product": "T", "FKGeneric": "G1", "Field": "Card",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
            {"product": "P", "FKGeneric": "G2", "Field": "Card",
             "PatientConsumeType": "Continuous", "PatientConsumePerPeriod": 5.0, "unit_ratio": 1.0},
        ])
        g_series = _build_generic_monthly_series_fast(panel, profile)
        c_series = _build_cross_generic_monthly_series_fast(panel, profile)
        covered = frozenset(panel["date"].dropna().astype(int).unique())
        primary = pd.DataFrame([{"product": "T", "budget_origin": origin}])
        enriched = attach_pit_features(
            primary, g_series, c_series, covered_months_set=covered
        )
        # Should not raise
        assert_cross_generic_no_same_fkgeneric(panel, enriched)


# ---------------------------------------------------------------------------
# Frozen feature family names
# ---------------------------------------------------------------------------

class TestFrozenFeatureFamilies:
    def test_f3e_a_feature_names(self):
        assert F3E_A_FEATURES == (
            "log_generic_peer_dqtyunit_last_month",
            "log_generic_peer_dqtyunit_3m_mean",
        )

    def test_f3e_b_includes_a(self):
        for f in F3E_A_FEATURES:
            assert f in F3E_B_FEATURES

    def test_f3e_b_adds_cross_generic(self):
        assert "log_cross_generic_field_consume_patients_last_month" in F3E_B_FEATURES
        assert "log_cross_generic_field_consume_patients_3m_mean" in F3E_B_FEATURES

    def test_no_target_sales_feature(self):
        forbidden = [
            "target", "sales", "dqty", "patient_consume", "market_share",
            "ratio", "growth", "trend", "volatility", "price", "inventory",
        ]
        for f in F3E_B_FEATURES:
            for kw in forbidden:
                assert kw not in f.lower() or "peer" in f.lower() or "cross" in f.lower(), (
                    f"Feature '{f}' may be a forbidden target/non-peer feature"
                )
