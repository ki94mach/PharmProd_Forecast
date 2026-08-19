"""F3D experiment spec tests (no live SQL, no XGB)."""
from __future__ import annotations

import pytest

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.f3d.config import (
    ALL_EXPERIMENTS,
    D0_TS,
    D0_HUMAN,
    D1_TS,
    D1_HUMAN,
    D2_TS,
    D2_HUMAN,
    F3D_A_FEATURES,
    F3D_B_FEATURES,
    NEVER_FILLNA,
    PROFILE_FEATURE_NAMES,
    PAIRS,
)


class TestExperimentNames:
    EXPECTED = {
        "D0_TS",
        "D1_TS_TYPE",
        "D2_TS_PROFILE",
        "D0_HUMAN",
        "D1_HUMAN_TYPE",
        "D2_HUMAN_PROFILE",
    }

    def test_exactly_six_experiments(self):
        assert len(ALL_EXPERIMENTS) == 6

    def test_all_expected_names(self):
        assert set(ALL_EXPERIMENTS.keys()) == self.EXPECTED


class TestProfileFeatureLists:
    def test_f3d_a_is_type_only(self):
        assert F3D_A_FEATURES == ("is_continuous_consumption",)

    def test_f3d_b_includes_log_annual(self):
        assert "is_continuous_consumption" in F3D_B_FEATURES
        assert "log_patient_annual_consumption" in F3D_B_FEATURES

    def test_profile_feature_names_two(self):
        assert len(PROFILE_FEATURE_NAMES) == 2


class TestD0FrozenAdapter:
    def test_d0_ts_frozen(self):
        assert D0_TS.use_frozen_adapter is True

    def test_d0_human_frozen(self):
        assert D0_HUMAN.use_frozen_adapter is True

    def test_d1_not_frozen(self):
        assert D1_TS.use_frozen_adapter is False
        assert D1_HUMAN.use_frozen_adapter is False

    def test_d2_not_frozen(self):
        assert D2_TS.use_frozen_adapter is False
        assert D2_HUMAN.use_frozen_adapter is False


class TestTrainUniverses:
    def test_ts_experiments_use_ts(self):
        for exp in (D0_TS, D1_TS, D2_TS):
            assert exp.train_universe == "ts", exp.name

    def test_human_experiments_use_budget(self):
        for exp in (D0_HUMAN, D1_HUMAN, D2_HUMAN):
            assert exp.train_universe == "budget", exp.name


class TestControlChain:
    def test_d1_controls_are_d0(self):
        assert D1_TS.control == "D0_TS"
        assert D1_HUMAN.control == "D0_HUMAN"

    def test_d2_controls_are_d1(self):
        assert D2_TS.control == "D1_TS_TYPE"
        assert D2_HUMAN.control == "D1_HUMAN_TYPE"


class TestFeatureContents:
    def test_d0_ts_is_exactly_f0(self):
        assert tuple(D0_TS.features()) == tuple(TS_RESID_FEATURES)

    def test_d0_human_is_exactly_f0(self):
        assert tuple(D0_HUMAN.features()) == tuple(BUDGET_RESID_FEATURES)

    def test_d1_ts_adds_type_indicator(self):
        feats = D1_TS.features()
        assert set(TS_RESID_FEATURES).issubset(set(feats))
        assert "is_continuous_consumption" in feats
        assert "log_patient_annual_consumption" not in feats

    def test_d2_ts_adds_both(self):
        feats = D2_TS.features()
        assert set(TS_RESID_FEATURES).issubset(set(feats))
        assert "is_continuous_consumption" in feats
        assert "log_patient_annual_consumption" in feats

    def test_no_extra_dim_fields(self):
        """F3D profile_features must not include extra Dim fields beyond the two scored ones.

        Note: field_enc / form_enc / provider_enc are F0 base features and are
        intentionally present; the restriction applies only to F3D-specific additions.
        """
        for name, exp in ALL_EXPERIMENTS.items():
            # Only F3D-specific additions are inspected
            for f in exp.profile_features:
                assert f in ("is_continuous_consumption", "log_patient_annual_consumption"), (
                    f"{name} profile_features contains unexpected feature {f!r}"
                )


class TestNeverFillna:
    def test_profile_features_in_never_fillna(self):
        for f in PROFILE_FEATURE_NAMES:
            assert f in NEVER_FILLNA, f"{f} missing from NEVER_FILLNA"

    def test_patient_annual_consumption_in_never_fillna(self):
        assert "patient_annual_consumption" in NEVER_FILLNA


class TestPairsContract:
    def test_four_pairs(self):
        assert len(PAIRS) == 4

    def test_d1_pairs_are_vs_d0(self):
        pair_dict = dict(PAIRS)
        assert pair_dict["D1_TS_TYPE"] == "D0_TS"
        assert pair_dict["D1_HUMAN_TYPE"] == "D0_HUMAN"

    def test_d2_pairs_are_vs_d1(self):
        pair_dict = dict(PAIRS)
        assert pair_dict["D2_TS_PROFILE"] == "D1_TS_TYPE"
        assert pair_dict["D2_HUMAN_PROFILE"] == "D1_HUMAN_TYPE"
