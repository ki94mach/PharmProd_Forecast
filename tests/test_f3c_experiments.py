"""F3C experiment spec tests (no live SQL, no XGB)."""
from __future__ import annotations

import pytest

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.f3c.config import (
    ALL_EXPERIMENTS,
    F3C_A_FEATURES,
    F3C_B_FEATURES,
    I0_HUMAN,
    I0_TS,
    I1_HUMAN,
    I1_TS,
    I2_HUMAN,
    I2_TS,
    INVENTORY_FEATURE_NAMES,
)


class TestExperimentNames:
    EXPECTED_NAMES = {
        "I0_TS", "I1_TS_DISTRIBUTOR", "I2_TS_DISTRIBUTOR_FACTORY",
        "I0_HUMAN", "I1_HUMAN_DISTRIBUTOR", "I2_HUMAN_DISTRIBUTOR_FACTORY",
    }

    def test_exactly_six_experiments(self):
        assert len(ALL_EXPERIMENTS) == 6

    def test_all_expected_names(self):
        assert set(ALL_EXPERIMENTS.keys()) == self.EXPECTED_NAMES


class TestScoredExtras:
    def test_scored_features_are_two_logs(self):
        assert INVENTORY_FEATURE_NAMES == (
            "log_distributor_inventory_qty",
            "log_factory_inventory_qty",
        )

    def test_f3c_a_is_distributor_only(self):
        assert F3C_A_FEATURES == ("log_distributor_inventory_qty",)

    def test_f3c_b_is_both(self):
        assert F3C_B_FEATURES == (
            "log_distributor_inventory_qty",
            "log_factory_inventory_qty",
        )


class TestI0FrozenAdapter:
    def test_i0_ts_frozen(self):
        assert I0_TS.use_frozen_adapter is True

    def test_i0_human_frozen(self):
        assert I0_HUMAN.use_frozen_adapter is True

    def test_i1_not_frozen(self):
        assert I1_TS.use_frozen_adapter is False
        assert I1_HUMAN.use_frozen_adapter is False


class TestTrainUniverses:
    def test_ts_experiments_use_ts(self):
        assert I0_TS.train_universe == "ts"
        assert I1_TS.train_universe == "ts"
        assert I2_TS.train_universe == "ts"

    def test_human_experiments_use_budget(self):
        assert I0_HUMAN.train_universe == "budget"
        assert I1_HUMAN.train_universe == "budget"
        assert I2_HUMAN.train_universe == "budget"


class TestNoFactoryOnly:
    def test_no_factory_only_experiment(self):
        for name, exp in ALL_EXPERIMENTS.items():
            if exp.inventory_features:
                assert "log_distributor_inventory_qty" in exp.inventory_features, \
                    f"{name} has factory without distributor"


class TestNoF3AOrPrice:
    def test_no_f3a_features(self):
        for name, exp in ALL_EXPERIMENTS.items():
            feats = exp.features()
            for f in feats:
                assert "lifecycle" not in f.lower(), f"{name} has lifecycle/F3A feature {f}"

    def test_no_price_features(self):
        for name, exp in ALL_EXPERIMENTS.items():
            feats = exp.features()
            for f in feats:
                assert "price" not in f.lower() or f in INVENTORY_FEATURE_NAMES, \
                    f"{name} has price feature {f}"


class TestFeatureLists:
    def test_i1_ts_features_are_f0_plus_distributor(self):
        feats = I1_TS.features()
        assert set(TS_RESID_FEATURES).issubset(set(feats))
        assert "log_distributor_inventory_qty" in feats
        assert "log_factory_inventory_qty" not in feats

    def test_i2_ts_features_are_f0_plus_both(self):
        feats = I2_TS.features()
        assert set(TS_RESID_FEATURES).issubset(set(feats))
        assert "log_distributor_inventory_qty" in feats
        assert "log_factory_inventory_qty" in feats

    def test_i0_ts_features_are_exactly_f0(self):
        feats = I0_TS.features()
        assert tuple(feats) == tuple(TS_RESID_FEATURES)

    def test_i0_human_features_are_exactly_f0(self):
        feats = I0_HUMAN.features()
        assert tuple(feats) == tuple(BUDGET_RESID_FEATURES)
