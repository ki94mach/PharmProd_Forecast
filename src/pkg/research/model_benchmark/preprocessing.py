"""Train-fold preprocessing for linear and tree learners."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pkg.research.model_benchmark.config import CATEGORICAL_FEATURES, NUMERIC_BASE


def make_linear_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """ColumnTransformer fit on training rows only (caller responsibility)."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )


def prep_xgb_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Canonical F0 fillna semantics for XGBoost (prep_lags-compatible)."""
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            continue
        if c.startswith("sales_"):
            out[c] = out[c].fillna(0)
        elif c in CATEGORICAL_FEATURES:
            out[c] = out[c].fillna(-1)
    return out


def prep_catboost_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Numeric fill + treat -1 encodings as missing for native categoricals."""
    out = prep_xgb_frame(df, feature_cols)
    for c in categorical_features:
        if c in out.columns:
            out[c] = out[c].replace(-1, pd.NA)
    return out


def prep_lightgbm_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fitted category dtype for LightGBM native categorical handling."""
    tr = prep_xgb_frame(train, feature_cols)
    te = prep_xgb_frame(test, feature_cols)
    for c in categorical_features:
        if c not in tr.columns:
            continue
        tr[c] = tr[c].replace(-1, pd.NA)
        te[c] = te[c].replace(-1, pd.NA)
        cats = pd.Categorical(tr[c].astype("Int64"))
        tr[c] = cats
        te[c] = pd.Categorical(
            te[c].astype("Int64"),
            categories=cats.categories,
        )
    return tr, te


def linear_numeric_features(anchor_col: str) -> list[str]:
    return [anchor_col, *NUMERIC_BASE]
