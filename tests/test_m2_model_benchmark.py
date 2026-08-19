"""Unit tests for M2 residual learner benchmark."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.harness.metrics import rel_wmape
from pkg.research.model_benchmark.config import (
    CATBOOST_PARAMS,
    CATEGORICAL_FEATURES,
    ELASTICNET_PARAMS,
    LIGHTGBM_PARAMS,
    MODELS,
    NUMERIC_BASE,
    RIDGE_PARAMS,
    XGBOOST_F0_PARAMS,
)
from pkg.research.model_benchmark.diagnostics import classify_m2_verdict
from pkg.research.model_benchmark.evaluate import build_folds, discover_eligible_origins
from pkg.research.model_benchmark.models import all_learners, make_learner
from pkg.research.model_benchmark.preprocessing import linear_numeric_features


def _synthetic_panel(n_origins: int = 3) -> pd.DataFrame:
    rows = []
    origin_vals = [140201, 140204, 140207][:n_origins]
    for o in origin_vals:
        for td in range(o - 20, o + 5):
            for h in (1, 2, 3):
                rows.append(
                    {
                        "product": "P1",
                        "qrt": "1402Q1",
                        "target_date": td,
                        "ts_origin": o if td >= o else o - 100,
                        "budget_origin": o if td >= o else o - 100,
                        "origin": o if td >= o else o - 100,
                        "horizon": h,
                        "month": 1,
                        "quarter": 1,
                        "sales": 100.0 + td,
                        "ts_forecast": 90.0,
                        "budget_forecast": 92.0,
                        "sales_lag_1": 1.0,
                        "sales_lag_2": 1.0,
                        "sales_lag_3": 1.0,
                        "sales_lag_12": 1.0,
                        "sales_roll3": 3.0,
                        "model_enc": 1,
                        "field_enc": 1,
                        "form_enc": 1,
                        "provider_enc": 1,
                    }
                )
    return pd.DataFrame(rows)


class TestM2Config(unittest.TestCase):
    def test_models_registry(self):
        self.assertEqual(MODELS, ("xgboost", "ridge", "elasticnet", "catboost", "lightgbm"))

    def test_xgb_deterministic_params(self):
        self.assertEqual(XGBOOST_F0_PARAMS["n_jobs"], 1)
        self.assertEqual(XGBOOST_F0_PARAMS["tree_method"], "hist")
        self.assertEqual(XGBOOST_F0_PARAMS["n_estimators"], 200)
        self.assertEqual(XGBOOST_F0_PARAMS["random_state"], 42)

    def test_catboost_thread_count(self):
        self.assertEqual(CATBOOST_PARAMS["thread_count"], 1)

    def test_lightgbm_deterministic(self):
        self.assertTrue(LIGHTGBM_PARAMS["deterministic"])
        self.assertEqual(LIGHTGBM_PARAMS["n_jobs"], 1)

    def test_ridge_params(self):
        self.assertEqual(RIDGE_PARAMS["alpha"], 1.0)

    def test_elasticnet_params(self):
        self.assertEqual(ELASTICNET_PARAMS["alpha"], 0.1)
        self.assertEqual(ELASTICNET_PARAMS["l1_ratio"], 0.5)

    def test_f0_features_only(self):
        ts_feats = set(TS_RESID_FEATURES)
        self.assertTrue(ts_feats.issuperset(set(CATEGORICAL_FEATURES)))
        self.assertTrue(ts_feats.issuperset(set(NUMERIC_BASE) | {"ts_forecast"}))
        bud_feats = set(BUDGET_RESID_FEATURES)
        self.assertIn("budget_forecast", bud_feats)
        for bad in ("lifecycle", "peer_", "inventory", "price_"):
            for f in ts_feats:
                self.assertNotIn(bad, f)


class TestRelWmape(unittest.TestCase):
    def test_formula(self):
        # positive means candidate beats baseline
        self.assertAlmostEqual(rel_wmape(40.0, 38.0), 5.0)


class TestVerdict(unittest.TestCase):
    def test_beats_xgboost(self):
        v = classify_m2_verdict(
            wmape_xgb=40.0,
            wmape_candidate=38.0,
            product_win_rate=0.6,
            median_product_improvement_pct=1.0,
            origins_improved=3,
            origins_total=4,
            bias_xgb=0.0,
            bias_candidate=0.0,
            concentration_flags=[],
            horizon_buckets_improved=2,
        )
        self.assertEqual(v, "BEATS_XGBOOST")

    def test_weaker(self):
        v = classify_m2_verdict(
            wmape_xgb=38.0,
            wmape_candidate=42.0,
            product_win_rate=0.2,
            median_product_improvement_pct=-1.0,
            origins_improved=1,
            origins_total=4,
            bias_xgb=0.0,
            bias_candidate=100.0,
            concentration_flags=["one_product_gt_25pct_deterioration"],
            horizon_buckets_improved=0,
        )
        self.assertEqual(v, "WEAKER_THAN_XGBOOST")


class TestLearners(unittest.TestCase):
    def test_all_learners_predict_length(self):
        panel = _synthetic_panel(3)
        # Expand panel so TS eligibility passes (>=500 rows, >=12 months)
        big_rows = []
        for td in range(140100, 140210):
            for prod in ("P1", "P2", "P3", "P4", "P5"):
                for h in range(1, 4):
                    big_rows.append(
                        {
                            "product": prod,
                            "qrt": "1402Q1",
                            "target_date": td,
                            "ts_origin": 140210,
                            "budget_origin": 140210,
                            "origin": 140210,
                            "horizon": h,
                            "month": td % 12 + 1,
                            "quarter": 1,
                            "sales": float(td + h),
                            "ts_forecast": 50.0,
                            "budget_forecast": 52.0,
                            "sales_lag_1": 1.0,
                            "sales_lag_2": 1.0,
                            "sales_lag_3": 1.0,
                            "sales_lag_12": 1.0,
                            "sales_roll3": 3.0,
                            "model_enc": 1,
                            "field_enc": 1,
                            "form_enc": 1,
                            "provider_enc": 1,
                        }
                    )
        train = pd.DataFrame(big_rows)
        test = train.iloc[:30].copy()
        test["ts_origin"] = 140210
        test["budget_origin"] = 140210
        test["origin"] = 140210
        for learner in all_learners():
            preds = learner.fit_predict(
                train,
                test,
                anchor_col="ts_forecast",
                features=TS_RESID_FEATURES,
            )
            self.assertEqual(len(preds), len(test))

    def test_ridge_uses_preprocessor(self):
        train = _synthetic_panel(1).head(50)
        test = train.head(5)
        with patch(
            "pkg.research.model_benchmark.models.make_linear_preprocessor"
        ) as mock_pre:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            real_pre = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        linear_numeric_features("ts_forecast"),
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                                (
                                    "onehot",
                                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                ),
                            ]
                        ),
                        list(CATEGORICAL_FEATURES),
                    ),
                ]
            )
            mock_pre.return_value = real_pre
            learner = make_learner("ridge")
            preds = learner.fit_predict(
                train, test, anchor_col="ts_forecast", features=TS_RESID_FEATURES
            )
            mock_pre.assert_called_once()
            self.assertEqual(len(preds), len(test))


class TestTemporalRules(unittest.TestCase):
    def test_train_max_lt_origin(self):
        panel = _synthetic_panel(2)
        from pkg.benchmark.dataset import BenchmarkDataset

        ds = BenchmarkDataset(
            version="v1",
            root=Path("."),
            ts_universe=panel,
            budget_universe=panel,
            matched_universe=panel,
            manifest={},
        )
        origins = discover_eligible_origins(ds, "ts", slice_kind="broad")
        if origins:
            folds = build_folds(ds, "ts", [origins[0]], slice_kind="broad")
            self.assertLess(int(folds[0].train["target_date"].max()), folds[0].origin)


class TestNoSearchImports(unittest.TestCase):
    def test_models_module_has_no_optuna(self):
        import pkg.research.model_benchmark.models as m

        src = Path(m.__file__).read_text(encoding="utf-8")
        self.assertNotIn("optuna", src.lower())
        self.assertNotIn("GridSearchCV", src)
        self.assertNotIn("RandomizedSearchCV", src)


if __name__ == "__main__":
    unittest.main()
