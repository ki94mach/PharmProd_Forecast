"""Unit tests for the shared research feature-family harness."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.research.ablation.config import CORE_TS
from pkg.research.f2.config import F2A, F2B, F2C
from pkg.research.f2.evaluate import _spec_for as f2_spec_for
from pkg.research.f3a.config import H0, H1, T0, T1, T2, T3
from pkg.research.f3a.evaluate import _spec_for as f3a_spec_for
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES
from pkg.research.features.human_f2 import HUMAN_F2_FEATURE_NAMES
from pkg.research.features.lifecycle import SCORED_FEATURE
from pkg.research.harness.dataset import enrich_dataset, resolve_origin_col
from pkg.research.harness.residual import make_residual_model


def _fake_ds(root: Path) -> BenchmarkDataset:
    panel = pd.DataFrame({"origin": [140404], "product": ["P"]})
    return BenchmarkDataset(
        version="v1",
        root=root,
        ts_universe=panel.copy(),
        budget_universe=panel.copy(),
        matched_universe=panel.copy(),
        manifest={},
    )


class TestOriginCol(unittest.TestCase):
    def test_prefers_origin(self):
        df = pd.DataFrame({"origin": [1], "ts_origin": [2], "budget_origin": [3]})
        self.assertEqual(resolve_origin_col(df), "origin")

    def test_ts_origin(self):
        self.assertEqual(resolve_origin_col(pd.DataFrame({"ts_origin": [1]})), "ts_origin")

    def test_budget_origin(self):
        self.assertEqual(
            resolve_origin_col(pd.DataFrame({"budget_origin": [1]})), "budget_origin"
        )

    def test_explicit_override(self):
        df = pd.DataFrame({"origin": [1], "ts_origin": [2]})
        self.assertEqual(resolve_origin_col(df, origin_col="ts_origin"), "ts_origin")

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            resolve_origin_col(pd.DataFrame({"product": ["P"]}))


class TestEnrichDoesNotWriteFreeze(unittest.TestCase):
    def test_parquet_bytes_and_mtime_unchanged(self):
        tmp = Path(tempfile.mkdtemp())
        freeze = tmp / "matched_universe.parquet"
        freeze.write_bytes(b"frozen-bytes")
        before = freeze.read_bytes()
        mtime = freeze.stat().st_mtime_ns
        ds = _fake_ds(tmp)
        out = enrich_dataset(ds, lambda df: df.assign(extra=1))
        self.assertEqual(freeze.read_bytes(), before)
        self.assertEqual(freeze.stat().st_mtime_ns, mtime)
        self.assertIn("extra", out.matched_universe.columns)
        self.assertNotIn("extra", ds.matched_universe.columns)


class TestResidualNeverFillna(unittest.TestCase):
    def test_age_stays_nan_sales_and_extra_fill_zero(self):
        captured: dict = {}
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.0])

        def fake_fit(cols, tr):
            captured["tr"] = tr.copy()
            captured["te_cols"] = list(cols)
            return mock_model

        cols = ["sales_lag_1", "trend_log_3m", SCORED_FEATURE]
        with patch("pkg.research.harness.residual.fit_xgb", side_effect=fake_fit):
            model = make_residual_model(
                "ts",
                cols,
                fillna_extra=("trend_log_3m",),
                never_fillna=frozenset({SCORED_FEATURE}),
                name="t",
            )
            frame = pd.DataFrame(
                {
                    "sales": [1.0],
                    "ts_forecast": [1.0],
                    "horizon": [1],
                    "sales_lag_1": [np.nan],
                    "trend_log_3m": [np.nan],
                    SCORED_FEATURE: [np.nan],
                }
            )
            preds = model(frame, frame)

        self.assertEqual(len(preds), 1)
        self.assertEqual(float(captured["tr"]["sales_lag_1"].iloc[0]), 0.0)
        self.assertEqual(float(captured["tr"]["trend_log_3m"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(captured["tr"][SCORED_FEATURE].iloc[0]))


class TestExperimentSpecFeatureLists(unittest.TestCase):
    def test_f2_feature_tuples_match_config(self):
        f2a_ts = tuple(TS_RESID_FEATURES) + tuple(DEMAND_F2_FEATURE_NAMES)
        f2a_h = tuple(BUDGET_RESID_FEATURES) + tuple(DEMAND_F2_FEATURE_NAMES)
        f2b_h = tuple(BUDGET_RESID_FEATURES) + tuple(HUMAN_F2_FEATURE_NAMES)
        f2c_h = (
            tuple(BUDGET_RESID_FEATURES)
            + tuple(DEMAND_F2_FEATURE_NAMES)
            + tuple(HUMAN_F2_FEATURE_NAMES)
        )
        self.assertEqual(F2A.features_for("ts"), f2a_ts)
        self.assertEqual(F2A.features_for("human"), f2a_h)
        self.assertEqual(F2B.features_for("human"), f2b_h)
        self.assertEqual(F2C.features_for("human"), f2c_h)
        self.assertEqual(f2_spec_for(F2A, "ts").features, f2a_ts)
        self.assertEqual(f2_spec_for(F2A, "human").features, f2a_h)
        self.assertEqual(f2_spec_for(F2B, "human").features, f2b_h)
        self.assertEqual(f2_spec_for(F2C, "human").features, f2c_h)
        self.assertEqual(f2_spec_for(F2A, "ts").enrich, "demand_f2")
        self.assertEqual(f2_spec_for(F2C, "human").enrich, "demand_f2+human_f2")

    def test_f3a_feature_tuples_match_config(self):
        t0 = tuple(TS_RESID_FEATURES)
        t1 = t0 + (SCORED_FEATURE,)
        t2 = tuple(CORE_TS)
        t3 = t2 + (SCORED_FEATURE,)
        h0 = tuple(BUDGET_RESID_FEATURES)
        h1 = h0 + (SCORED_FEATURE,)
        self.assertEqual(T0.features(), t0)
        self.assertEqual(T1.features(), t1)
        self.assertEqual(T2.features(), t2)
        self.assertEqual(T3.features(), t3)
        self.assertEqual(H0.features(), h0)
        self.assertEqual(H1.features(), h1)
        self.assertEqual(f3a_spec_for(T1).features, t1)
        self.assertEqual(f3a_spec_for(T3).features, t3)
        self.assertEqual(f3a_spec_for(H1).features, h1)
        self.assertEqual(f3a_spec_for(T1).enrich, "lifecycle")
        self.assertIsNone(f3a_spec_for(T2).enrich)
        self.assertFalse(f3a_spec_for(T2).use_frozen_adapter)
        self.assertEqual(len(t1), 14)
        self.assertEqual(len(t3), 9)


class TestReportedWMAPE(unittest.TestCase):
    """WMAPE contract vs the pre-harness F2/F3A reports (tol 0.05)."""

    tol = 0.05

    def _assert_rows(self, path: Path, key_cols: tuple[str, ...], expected: dict):
        self.assertTrue(path.exists(), f"missing {path}")
        df = pd.read_csv(path)
        for key, wmape in expected.items():
            mask = pd.Series(True, index=df.index)
            for col, val in zip(key_cols, key if isinstance(key, tuple) else (key,)):
                mask &= df[col] == val
            sub = df.loc[mask]
            self.assertEqual(len(sub), 1, f"{key} rows={len(sub)}")
            got = float(sub["wmape"].iloc[0])
            self.assertLessEqual(
                abs(got - wmape),
                self.tol,
                f"{key} WMAPE {got} vs {wmape} (tol={self.tol})",
            )

    def test_f2_overall_reproduces_report(self):
        self._assert_rows(
            _SRC / "data" / "results" / "f2" / "overall.csv",
            ("experiment", "anchor"),
            {
                ("F0", "ts"): 38.2848,
                ("F0", "human"): 36.5602,
                ("F2A", "ts"): 40.2870,
                ("F2A", "human"): 37.6795,
                ("F2B", "human"): 45.1544,
            },
        )

    def test_f3a_overall_reproduces_report(self):
        self._assert_rows(
            _SRC / "data" / "results" / "f3a" / "overall.csv",
            ("experiment",),
            {
                ("T0",): 38.2848,
                ("T1",): 38.2891,
                ("T2",): 37.7360,
                ("T3",): 40.4633,
                ("H0",): 36.5602,
                ("H1",): 36.9383,
            },
        )


if __name__ == "__main__":
    unittest.main()
