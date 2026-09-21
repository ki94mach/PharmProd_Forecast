"""Portfolio TS architecture lock (v1) — evaluation helpers.

TS freezes before residual-XGBoost research. Human Budget origins are hard-excluded
from scoring paths.
"""

from pkg.portfolio_ts_lock.constants import (
    HB_ORIGINS,
    HORIZON_GROUPS,
    LOCKED_SCORE_END,
    POST_HB_ORIGINS,
    RESIDUAL_DEVELOPMENT_ORIGINS,
    TIE_AE_EPS,
)
from pkg.portfolio_ts_lock.lock_io import (
    LockExistsError,
    assert_lock_immutable_read,
    write_lock_atomic,
)

__all__ = [
    "HB_ORIGINS",
    "HORIZON_GROUPS",
    "LOCKED_SCORE_END",
    "POST_HB_ORIGINS",
    "RESIDUAL_DEVELOPMENT_ORIGINS",
    "TIE_AE_EPS",
    "LockExistsError",
    "assert_lock_immutable_read",
    "write_lock_atomic",
]
