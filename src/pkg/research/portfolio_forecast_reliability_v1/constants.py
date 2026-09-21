"""Constants for portfolio_forecast_reliability_v1."""

from __future__ import annotations

from pathlib import Path

FAMILY = "portfolio_forecast_reliability_v1"
FAMILY_DIR_REL = Path("src/data/results") / FAMILY

CONTRACT_VERSION = "reliability_label_contract_v1"
CONTRACT_FILENAME = "reliability_label_contract_v1.json"
REVIEW_FILENAME = "reliability_label_contract_v1_review.json"

# Upstream evidence only — residual XGB rejected; production remains FROZEN_TS.
UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL = Path(
    "src/data/results/portfolio_residual_xgb_v1/residual_pipeline_lock_v1.json"
)

WMAPE_DENOM_EPS = 1e-9
WMAPE_DENOM_RULE = "sum(abs(actual)) > 1e-9"

ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY = (
    "allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling"
)
ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT = False
