"""Commercial activity features — not implemented in this phase."""
from __future__ import annotations

import pandas as pd

FEATURE_NAMES: tuple[str, ...] = ()


def add_commercial_features(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError(
        "commercial features are out of scope for F1; placeholder only"
    )
