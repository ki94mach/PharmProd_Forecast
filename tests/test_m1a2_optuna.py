"""Unit tests for M1A2 fixed-200 Optuna diagnostic (synthetic only)."""
from __future__ import annotations

import json
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

from pkg.research.m1a2.config import (
    EXPECTED_INNER_ORIGINS,
    FIXED_N_ESTIMATORS,
    PRE_PRIMARY_CUTOFF,
)
from pkg.research.m1a2.run import (
    _build_xgb_params,
    _fit_predict_fold,
    _verify_folds,
    classify_diagnostic,
    classify_verdict,
)
from pkg.research.tuning.folds import InnerFold


def _synthetic_fold(origin: int) -> InnerFold:
    train_dates = list(range(140101, origin))
    val_dates = list(range(origin, origin + 20))
    feature_cols = [
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_3",
        "sales_lag_12",
        "sales_roll3",
        "month",
        "quarter",
        "model_enc",
        "field_enc",
        "form_enc",
        "provider_enc",
    ]

    def _rows(dates, sales_base):
        rows = []
        for i, d in enumerate(dates):
            row = {
                "target_date": d,
                "sales": float(sales_base + i),
                "ts_forecast": float(sales_base + i - 1),
                "horizon": 1 + (i % 12),
                "product": "P1",
            }
            for c in feature_cols:
                row[c] = float(i % 3)
            rows.append(row)
        return pd.DataFrame(rows)

    train = _rows(train_dates, 100)
    val = _rows(val_dates, 200)
    assert int(train["target_date"].max()) < origin
    return InnerFold(origin=origin, train=train, val=val)


class TestM1A2Config(unittest.TestCase):
    def test_expected_inner_origins_match_m1r_contract(self):
        self.assertEqual(len(EXPECTED_INNER_ORIGINS), 9)
        self.assertEqual(
            list(EXPECTED_INNER_ORIGINS),
            [140201, 140204, 140207, 140210, 140304, 140306, 140307, 140310, 140401],
        )
        for o in EXPECTED_INNER_ORIGINS:
            self.assertLess(o, PRE_PRIMARY_CUTOFF)


class TestM1A2Fixed200Fit(unittest.TestCase):
    def test_build_xgb_params_n_estimators_200(self):
        p = _build_xgb_params({"max_depth": 4, "learning_rate": 0.05})
        self.assertEqual(p["n_estimators"], 200)
        self.assertEqual(p["n_jobs"], 1)
        self.assertEqual(p["tree_method"], "hist")

    @patch("pkg.research.m1a2.run.XGBRegressor")
    def test_fit_predict_fold_no_eval_set(self, mock_xgb_cls):
        mock_model = MagicMock()
        mock_model.get_params.return_value = {"n_estimators": 200}
        mock_model.predict.return_value = np.zeros(20)
        mock_xgb_cls.return_value = mock_model

        fold = _synthetic_fold(140401)
        structural = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        actual, pred, _ = _fit_predict_fold(fold.train, fold.val, structural)

        mock_xgb_cls.assert_called_once()
        call_kwargs = mock_xgb_cls.call_args[1]
        self.assertEqual(call_kwargs["n_estimators"], 200)
        fit_kwargs = mock_model.fit.call_args[1]
        self.assertNotIn("eval_set", fit_kwargs)
        self.assertNotIn("early_stopping_rounds", call_kwargs)
        self.assertEqual(len(actual), len(pred))


class TestM1A2FoldGates(unittest.TestCase):
    def test_verify_folds_passes_expected_origins(self):
        folds = [_synthetic_fold(o) for o in EXPECTED_INNER_ORIGINS]
        origins = _verify_folds(folds)
        self.assertEqual(origins, list(EXPECTED_INNER_ORIGINS))

    def test_verify_folds_stops_on_origin_mismatch(self):
        folds = [_synthetic_fold(140201), _synthetic_fold(140204)]
        with self.assertRaises(AssertionError):
            _verify_folds(folds)

    def test_fold_leakage_assertions(self):
        fold = _synthetic_fold(140310)
        self.assertLess(fold.origin, PRE_PRIMARY_CUTOFF)
        self.assertLess(int(fold.train["target_date"].max()), fold.origin)


class TestM1A2FreezePayload(unittest.TestCase):
    def test_best_params_json_shape(self):
        payload = {
            "n_estimators": FIXED_N_ESTIMATORS,
            "selected_hyperparameters": {"max_depth": 4},
            "best_inner_pooled_wmape": 25.0,
            "inner_f0_pooled_wmape": 27.0,
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ts_best_params.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["n_estimators"], 200)
        self.assertNotIn("frozen_n_estimators", loaded)


class TestM1A2Classification(unittest.TestCase):
    def test_classify_verdict_promote(self):
        v = classify_verdict(
            wmape_baseline=40.0,
            wmape_tuned=38.0,
            product_win_rate=0.6,
            median_product_improvement_pct=1.0,
            origins_improved=3,
            origins_total=5,
            bias_baseline=100.0,
            bias_tuned=90.0,
            concentration_flags=[],
        )
        self.assertEqual(v, "PROMOTE")

    def test_classify_verdict_reject(self):
        v = classify_verdict(
            wmape_baseline=38.0,
            wmape_tuned=43.0,
            product_win_rate=0.3,
            median_product_improvement_pct=-5.0,
            origins_improved=1,
            origins_total=5,
            bias_baseline=50.0,
            bias_tuned=200.0,
            concentration_flags=["top1_high"],
        )
        self.assertEqual(v, "REJECT")

    def test_classify_diagnostic_early_stopping_major(self):
        label = classify_diagnostic(f0_primary=38.6, m1a2_primary=38.5, m1r_primary=43.2)
        self.assertEqual(label, "EARLY_STOPPING_WAS_MAJOR_FAILURE_SOURCE")

    def test_classify_diagnostic_structural_fails(self):
        label = classify_diagnostic(f0_primary=38.6, m1a2_primary=44.0, m1r_primary=43.2)
        self.assertEqual(label, "STRUCTURAL_TUNING_ALSO_FAILS_TO_TRANSFER")


if __name__ == "__main__":
    unittest.main()
