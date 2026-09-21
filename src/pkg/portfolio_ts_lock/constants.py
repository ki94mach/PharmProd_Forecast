"""Constants for portfolio TS architecture lock v1."""

from __future__ import annotations

# Hard holdouts — never attach actuals / score / recommend / router-error on these.
HB_ORIGINS: frozenset[int] = frozenset({140304, 140307, 140310, 140312})

RESIDUAL_DEVELOPMENT_ORIGINS: frozenset[int] = frozenset(
    {140101, 140104, 140107, 140110, 140201, 140204, 140207, 140210, 140301}
)

POST_HB_ORIGINS: frozenset[int] = frozenset({140401, 140402})

LOCKED_SCORE_END = 140504
ACTIVITY_THRESHOLD = 0.0
A0_SEEDS: frozenset[int] = frozenset({41, 42, 43})

# Fixed analysis groups (inclusive horizon ints).
HORIZON_GROUPS: tuple[tuple[str, frozenset[int]], ...] = (
    ("h1_3", frozenset({1, 2, 3})),
    ("h4_6", frozenset({4, 5, 6})),
    ("h7_12", frozenset(range(7, 13))),
    ("h13_15", frozenset({13, 14, 15})),
)

TIE_AE_EPS = 1e-9
KEYS = ("product", "forecast_origin", "horizon", "target_date")

FALLBACK_BOTH_NATIVE = "both_native"
FALLBACK_V21_ONLY = "v21_only"
FALLBACK_A0_ONLY = "a0_only"
FALLBACK_NAIVE = "naive_fallback"

POLICY_V2 = "V2_POLICY"
POLICY_A0 = "A0_ONLY"
POLICY_HYBRID = "A0_HYBRID_POLICY"

CHOICE_HYBRID = "A0_HYBRID_POLICY"
CHOICE_V2 = "V2_POLICY"
CHOICE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
