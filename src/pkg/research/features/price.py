"""Price features — not implemented in this phase."""
from __future__ import annotations

import pandas as pd

FEATURE_NAMES: tuple[str, ...] = ()


def add_price_features(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError(
        "price features are out of scope for F1; placeholder only"
    )
