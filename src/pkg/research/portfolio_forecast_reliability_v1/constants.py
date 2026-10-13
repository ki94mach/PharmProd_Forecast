"""Constants for portfolio_forecast_reliability_v1."""

from __future__ import annotations

from pathlib import Path

FAMILY = "portfolio_forecast_reliability_v1"
FAMILY_DIR_REL = Path("src/data/results") / FAMILY

CONTRACT_VERSION = "reliability_label_contract_v1"
CONTRACT_FILENAME = "reliability_label_contract_v1.json"
REVIEW_FILENAME = "reliability_label_contract_v1_review.json"
MAPPING_AUDIT_FILENAME = "hb_origin_mapping_audit_v1.json"
REPRESENTATIVENESS_AUDIT_FILENAME = "hb_cohort_representativeness_audit_v1.json"
PROTECTED_ORIGIN_VALIDATION_FILENAME = "hb_protected_origin_validation_v1.json"
STALE_COVERAGE_AUDIT_FILENAME = "hb_reliability_coverage_audit_v1.json"

# Upstream evidence only — residual XGB rejected; production remains FROZEN_TS.
UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL = Path(
    "src/data/results/portfolio_residual_xgb_v1/residual_pipeline_lock_v1.json"
)

# Reliability HB join origins after Version→quarter remap (keep 140401; drop 140312).
# Does NOT mutate portfolio_ts_lock.HB_ORIGINS (scientific lock / production holdouts).
PROTECTED_HB_ORIGINS: frozenset[int] = frozenset(
    {140304, 140307, 140310, 140401}
)

# Prospective full-horizon origin (not admitted until maturity gates pass).
ORIGIN_STATUS_PENDING_MATURITY = "PENDING_MATURITY"
PROSPECTIVE_HB_ORIGIN_140404 = 140404
PROSPECTIVE_HB_QUARTER_140404 = "1404Q2"
# Scoring actuals for h1–15 at origin 140404 require sales through origin+14.
PROSPECTIVE_SCORING_SALES_THROUGH = 140506
# Machine training cutoff for origin 140404 (origin−2).
PROSPECTIVE_MACHINE_TRAIN_END = 140402
# Proposed full-horizon portfolio set after 140404 admission (140401 group-specific only).
PROPOSED_FINAL_FULL_ORIGINS: tuple[int, ...] = (140304, 140307, 140310, 140404)
GROUP_SPECIFIC_ONLY_ORIGIN = 140401

MATURITY_AUDIT_FILENAME = "hb_140404_maturity_audit_v1.json"
CANDIDATE_ORIGIN_MANIFEST_FILENAME = "candidate_origin_manifest_v1.json"
SALES_V2_PREPARE_MANIFEST_FILENAME = "sales_full_portfolio_v2_prepare_manifest.json"
VPS_140404_COMMANDS_FILENAME = "vps_140404_commands.json"
IDENTIFIER_TRACE_FILENAME = "hb_140404_identifier_trace_v1.json"

# Versioned HB reliability extract (outside frozen benchmarks/; no INCOMPLETE_SHAMSI drop).
HB_RELIABILITY_EXTRACT_DIR_REL = FAMILY_DIR_REL / "hb_reliability_extract_v1"
HB_RELIABILITY_IDENTIFIERS_PANEL_REL = (
    HB_RELIABILITY_EXTRACT_DIR_REL / "identifiers_panel.parquet"
)
HB_RELIABILITY_EXTRACT_META_REL = HB_RELIABILITY_EXTRACT_DIR_REL / "meta.json"
HB_FORECAST_COVERAGE_AUDIT_FILENAME = "hb_reliability_forecast_coverage_audit_v1.json"
HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME = (
    "hb_reliability_candidate_manifest_v1.json"
)
HB_SCORING_PATH_PREPARE_FILENAME = "hb_reliability_scoring_path_prepare_v1.json"
HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME = (
    "hb_reliability_quantity_retrieval_prepare_v1.json"
)
HB_140404_MACHINE_RECONCILIATION_FILENAME = (
    "hb_140404_machine_backfill_reconciliation_v1.json"
)
HB_140404_READINESS_AUDIT_FILENAME = "hb_140404_readiness_audit_v1.json"
HB_140404_A0_H14_DIAGNOSIS_FILENAME = "hb_140404_a0_h14_export_diagnosis_v1.json"
HB_140404_A0_RERUN_PREPARE_FILENAME = "hb_140404_a0_h14_rerun_prepare_v1.json"
# Versioned repaired A0 export (written only after VPS rerun + re-export).
MACHINE_140404_A0_EXPORT_V2_REL = Path(
    "src/data/results/product_universe/portfolio_a0_backfill_140404_export_v2/"
    "forecasts_panel.parquet"
)

# Origins evaluated for HB forecast coverage (current protected + prospective 140404).
HB_RELIABILITY_COVERAGE_ORIGINS: frozenset[int] = frozenset(
    {140304, 140307, 140310, 140401, 140404}
)

# 140404-only machine backfill runs (not the frozen portfolio_*_backfill_v2 exports).
MACHINE_140404_V21_RUN_ID = "portfolio_v21_backfill_140404"
MACHINE_140404_A0_RUN_ID = "portfolio_a0_backfill_140404"
MACHINE_140404_V21_EXPORT_REL = Path(
    "src/data/results/product_universe/portfolio_v21_backfill_140404_export/"
    "forecasts_panel.parquet"
)
MACHINE_140404_A0_EXPORT_REL = Path(
    "src/data/results/product_universe/portfolio_a0_backfill_140404_export/"
    "forecasts_panel.parquet"
)
MACHINE_140404_V21_CONFIG_REL = Path(
    "configs/forecast_jobs/portfolio_v21_backfill_140404.win.yaml"
)
MACHINE_140404_A0_CONFIG_REL = Path(
    "configs/forecast_jobs/portfolio_a0_backfill_140404.win.yaml"
)
MACHINE_140404_A0_SEEDS: tuple[int, ...] = (41, 42, 43)
# Pinned training sales contract for 140404 machine jobs (frozen v1).
MACHINE_140404_TRAINING_SALES_REL = Path(
    "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet"
)
MACHINE_140404_TRAINING_SALES_SHA256 = (
    "b67382d451974271c209f3c371219505cfa3fe45e3ad43d9769a93dfc5f1cfcc"
)
MACHINE_140404_TRAINING_SALES_N_ROWS = 22041

SALES_FULL_PORTFOLIO_V2_DIR_REL = Path(
    "src/data/results/product_universe/sales_full_portfolio_v2"
)
SALES_FULL_PORTFOLIO_V2_REL = SALES_FULL_PORTFOLIO_V2_DIR_REL / "sales.parquet"
SALES_FULL_PORTFOLIO_V2_META_REL = SALES_FULL_PORTFOLIO_V2_DIR_REL / "meta.json"
# Explicit warehouse closure proof (not inferred from v1 max_date).
SALES_WAREHOUSE_CLOSURE_V2_REL = (
    SALES_FULL_PORTFOLIO_V2_DIR_REL / "warehouse_closure.json"
)

# Immutable TS-lock holdout list — never drives reliability execution.
LEGACY_TS_LOCK_HB_ORIGINS_KEY = "legacy_ts_lock_hb_origins_immutable"

VALIDITY_GROUPS: tuple[tuple[str, frozenset[int]], ...] = (
    ("h1_3", frozenset({1, 2, 3})),
    ("h4_6", frozenset({4, 5, 6})),
    ("h7_12", frozenset(range(7, 13))),
    ("h13_15", frozenset({13, 14, 15})),
)

STALE_COVERAGE_FIGURES_FORBIDDEN = frozenset({55, 50, 167, 161})

# Point-in-time representativeness: earliest HB origin 140304 minus 2 months.
PIT_REPRESENTATIVENESS_CUTOFF = 140302
EARLIEST_HB_ORIGIN_FOR_PIT = 140304

DEPLOYABLE_BRANCH_BOTH = "both_native"
DEPLOYABLE_BRANCH_V21 = "v21_only"
DEPLOYABLE_BRANCH_A0 = "a0_only"
DEPLOYABLE_BRANCH_NEITHER_NAIVE = "neither_naive"
# Not a deployable forecast — no native engine and no sales through origin−2.
NO_MACHINE_FORECAST = "NO_MACHINE_FORECAST"

DEPLOYABLE_PORTFOLIO_COHORT = "DEPLOYABLE_PORTFOLIO_COHORT"
NO_MACHINE_FORECAST_COHORT = "NO_MACHINE_FORECAST_COHORT"

WMAPE_DENOM_EPS = 1e-9
WMAPE_DENOM_RULE = "sum(abs(actual)) > 1e-9"

LABEL_MNIH = "MACHINE_NONINFERIOR_TO_HUMAN"
LABEL_HR = "HUMAN_REVIEW_REQUIRED"
LABEL_IHE = "INSUFFICIENT_HUMAN_EVIDENCE"

# Out of executable v1 scope (retained for future notes only).
LABEL_MR_OUT_OF_SCOPE = "MACHINE_RELIABLE"

EVIDENCE_STRENGTH_LIMITED = "LIMITED_3_OR_4_ORIGINS"

HUMAN_COMPARABLE_COHORT = "HUMAN_COMPARABLE_COHORT"

REQUIRED_HORIZONS: frozenset[int] = frozenset(range(1, 16))

A0_PANEL_REL = Path(
    "src/data/results/product_universe/portfolio_a0_backfill_v2_export/"
    "forecasts_panel.parquet"
)
V21_PANEL_REL = Path(
    "src/data/results/product_universe/portfolio_v21_backfill_v2_export/"
    "forecasts_panel.parquet"
)
BUDGET_UNIVERSE_V2_REL = Path("src/data/benchmarks/v2/budget_universe.parquet")
ELIGIBLE_UNIVERSE_REL = Path(
    "src/data/results/product_universe/backfill_inventory_v2/universe_eligible.csv"
)
POLICY_PANEL_REL = Path(
    "src/data/results/product_universe/portfolio_ts_architecture_lock_v1/"
    "policy_panel_deployable.parquet"
)
# Archval55 snapshot — do NOT use for full-portfolio representativeness.
SALES_ARCHVAL55_REL = Path("src/data/benchmarks/v1/raw/sales.parquet")
SALES_FULL_PORTFOLIO_REL = Path(
    "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet"
)
# Alias kept for callers that must explicitly opt into Archval55 only.
SALES_REL = SALES_ARCHVAL55_REL
POSTLOCK_REL = Path(
    "src/data/results/product_universe/portfolio_ts_architecture_lock_v1/"
    "postlock_140401_140402.json"
)

ID_COLS_MACHINE = ("product", "forecast_origin", "horizon", "target_date")
ID_COLS_HB = (
    "product",
    "source_budget_quarter",
    "budget_origin",
    "target_date",
    "horizon",
)

# Forbidden outcome / forecast qty columns for identifier-only audits.
FORBIDDEN_OUTCOME_COLS = frozenset(
    {
        "budget_forecast",
        "sales",
        "budget_residual",
        "actual",
        "forecast",
        "raw_forecast",
        "f_v21",
        "f_a0",
        "f_naive",
        "f_v2_policy",
        "f_a0_only",
        "f_a0_hybrid",
    }
)
