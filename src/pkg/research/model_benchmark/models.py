"""Fixed-configuration residual learners for M2."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from pkg.research.model_benchmark.config import (
    CATBOOST_PARAMS,
    CATEGORICAL_FEATURES,
    ELASTICNET_PARAMS,
    LIGHTGBM_PARAMS,
    RIDGE_PARAMS,
    XGBOOST_F0_PARAMS,
)
from pkg.research.model_benchmark.preprocessing import (
    linear_numeric_features,
    make_linear_preprocessor,
    prep_catboost_frame,
    prep_lightgbm_frame,
    prep_xgb_frame,
)


class ResidualLearner(Protocol):
    name: str

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray: ...


def _sample_weight(train: pd.DataFrame) -> np.ndarray:
    return (1.0 / train["horizon"].clip(lower=1).astype(float)).to_numpy()


def _final_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    anchor_col: str,
    resid_hat: np.ndarray,
) -> np.ndarray:
    anchor = test[anchor_col].astype(float).to_numpy()
    return np.maximum(0.0, anchor + np.asarray(resid_hat, dtype=float))


class XGBoostF0Learner:
    name = "xgboost"

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray:
        tr = prep_xgb_frame(train, features)
        te = prep_xgb_frame(test, features)
        tr = tr.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = _sample_weight(tr)
        model = XGBRegressor(**XGBOOST_F0_PARAMS)
        model.fit(tr[list(features)], tr["residual"], sample_weight=sw, verbose=False)
        resid_hat = model.predict(te[list(features)])
        return _final_forecast(tr, te, anchor_col, resid_hat)


class RidgeLearner:
    name = "ridge"

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray:
        tr = train.copy()
        te = test.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = _sample_weight(tr)
        num_feats = linear_numeric_features(anchor_col)
        pre = make_linear_preprocessor(num_feats, CATEGORICAL_FEATURES)
        pipe = Pipeline([("pre", pre), ("model", Ridge(**RIDGE_PARAMS))])
        pipe.fit(tr, tr["residual"], model__sample_weight=sw)
        resid_hat = pipe.predict(te)
        return _final_forecast(tr, te, anchor_col, resid_hat)


class ElasticNetLearner:
    name = "elasticnet"

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray:
        tr = train.copy()
        te = test.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = _sample_weight(tr)
        num_feats = linear_numeric_features(anchor_col)
        pre = make_linear_preprocessor(num_feats, CATEGORICAL_FEATURES)
        enet_params = dict(ELASTICNET_PARAMS)
        pipe = Pipeline([("pre", pre), ("model", ElasticNet(**enet_params))])
        pipe.fit(tr, tr["residual"], model__sample_weight=sw)
        resid_hat = pipe.predict(te)
        return _final_forecast(tr, te, anchor_col, resid_hat)


class CatBoostLearner:
    name = "catboost"

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray:
        tr = prep_catboost_frame(train, features)
        te = prep_catboost_frame(test, features)
        tr = tr.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = _sample_weight(tr)
        cat_idx = [list(features).index(c) for c in CATEGORICAL_FEATURES if c in features]
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        train_pool = Pool(
            tr[list(features)],
            tr["residual"],
            cat_features=cat_idx,
            weight=sw,
        )
        model.fit(train_pool)
        test_pool = Pool(te[list(features)], cat_features=cat_idx)
        resid_hat = model.predict(test_pool)
        return _final_forecast(tr, te, anchor_col, resid_hat)


class LightGBMLearner:
    name = "lightgbm"

    def fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        anchor_col: str,
        features: Sequence[str],
    ) -> np.ndarray:
        tr, te = prep_lightgbm_frame(train, test, features)
        tr = tr.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = _sample_weight(tr)
        cat_names = [c for c in CATEGORICAL_FEATURES if c in features]
        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(
            tr[list(features)],
            tr["residual"],
            sample_weight=sw,
            categorical_feature=cat_names,
        )
        resid_hat = model.predict(te[list(features)])
        return _final_forecast(tr, te, anchor_col, resid_hat)


def make_learner(name: str) -> ResidualLearner:
    registry = {
        "xgboost": XGBoostF0Learner,
        "ridge": RidgeLearner,
        "elasticnet": ElasticNetLearner,
        "catboost": CatBoostLearner,
        "lightgbm": LightGBMLearner,
    }
    if name not in registry:
        raise ValueError(f"Unknown learner {name!r}; expected one of {sorted(registry)}")
    return registry[name]()


def all_learners() -> list[ResidualLearner]:
    return [make_learner(n) for n in ("xgboost", "ridge", "elasticnet", "catboost", "lightgbm")]
