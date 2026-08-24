"""Data loading and series preparation for V2.

Principles enforced here (once implemented):

- Train on months strictly before the explicit forecast origin.
- Do not silently drop the last warehouse month.
- Fit any transforms only on training history (no preprocessing leakage).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import ForecastOrigin, ProductSeries


def load_monthly_sales() -> pd.DataFrame:
    """Load warehouse monthly sales.

    Not implemented in this scaffold step. Production V1 loaders stay in
    ``pkg.db.query.sales``; V2 will wrap them without changing V1 behavior.
    """
    raise NotImplementedError("V2 sales loading is not implemented yet")


def series_as_of(
    sales: pd.DataFrame,
    product: str,
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ProductSeries:
    """Build a product series available at ``origin`` (``date < origin``).

    Not implemented in this scaffold step.
    """
    raise NotImplementedError("V2 series_as_of is not implemented yet")


def assert_min_history(series: ProductSeries, config: Optional[TSForecastConfig] = None) -> None:
    """Raise if history length is below ``config.min_train_months``."""
    cfg = config or DEFAULT_CONFIG
    n = int(len(series.history))
    if n < cfg.min_train_months:
        raise ValueError(
            f"{series.product!r} at origin {series.origin.shamsi_yyyymm}: "
            f"need >= {cfg.min_train_months} months, got {n}"
        )
