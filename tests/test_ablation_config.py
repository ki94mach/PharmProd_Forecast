"""Unit tests for feature-family ablation partition (no freeze mutation)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.ablation.config import (
    CORE_HUMAN,
    CORE_TS,
    F0_DEMAND,
    ALL_EXPERIMENTS,
    get_ablation,
)


class TestAblationPartition(unittest.TestCase):
    def test_core_union_f0_demand_equals_frozen_ts(self):
        self.assertEqual(set(CORE_TS) | set(F0_DEMAND), set(TS_RESID_FEATURES))

    def test_core_union_f0_demand_equals_frozen_human(self):
        self.assertEqual(set(CORE_HUMAN) | set(F0_DEMAND), set(BUDGET_RESID_FEATURES))

    def test_core_disjoint_from_f0_demand(self):
        self.assertFalse(set(CORE_TS) & set(F0_DEMAND))
        self.assertFalse(set(CORE_HUMAN) & set(F0_DEMAND))

    def test_d1_features_match_frozen_order(self):
        d1 = get_ablation("D1_F0")
        self.assertEqual(d1.features_for("ts"), tuple(TS_RESID_FEATURES))
        self.assertEqual(d1.features_for("human"), tuple(BUDGET_RESID_FEATURES))

    def test_d4_starts_with_frozen_f0(self):
        d4 = get_ablation("D4_F1_ADD")
        ts = d4.features_for("ts")
        self.assertEqual(ts[: len(TS_RESID_FEATURES)], tuple(TS_RESID_FEATURES))

    def test_registry_complete(self):
        expected = {
            "D0_CORE",
            "D1_F0",
            "D2_F1_REPLACE",
            "D3_F2_REPLACE",
            "D4_F1_ADD",
            "D5_F2_ADD",
            "H0_CORE",
            "H1_F0",
            "H2_F1_HUMAN_ONLY",
            "H3_F2_HUMAN_ONLY",
            "H4_F1_HUMAN_ADD",
            "H5_F2_HUMAN_ADD",
            "H6_F1_DEMAND_HUMAN",
            "H7_F2_DEMAND_HUMAN",
        }
        self.assertEqual(set(ALL_EXPERIMENTS), expected)


if __name__ == "__main__":
    unittest.main()
