"""F3B paths and Step 3 experiment specs.

Does not mutate F0 / F1 / F2 / F3A artifacts, the v1 freeze, or Step 1/2
source/feature definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.features.price import DIAGNOSTIC_NAMES, FEATURE_NAMES

Anchor = Literal["ts", "human"]

# Current-environment reproduced F0 (this machine matches locked Analysis B).
CURRENT_ENV_F0_WMAPE = {
    "ts": 37.23014,
    "human": 36.69475,
}
WMAPE_REPRO_TOL = 0.05
F0_N = 1877
F0_N_ORIGINS = 5

# Fillna extras copied from F3A/ablation so F0 lag columns match prior runs.
# Price features are never in this set and are listed in NEVER_FILLNA.
FILLNA_EXTRA: tuple[str, ...] = (
    "trend_3m",
    "trend_6m",
    "recent_growth",
    "recent_acceleration",
    "historical_actual_budget_ratio",
    "mean_human_adjustment",
    "mean_abs_human_adjustment",
    "trend_log_3m",
    "trend_log_6m",
    "yoy_log_change",
)

NEVER_FILLNA: frozenset[str] = frozenset(FEATURE_NAMES + DIAGNOSTIC_NAMES)

PAIRS: tuple[tuple[str, str], ...] = (
    ("P1_TS", "P0_TS"),
    ("P1_HUMAN", "P0_HUMAN"),
)


@dataclass(frozen=True)
class F3BExperiment:
    name: str
    anchor: Anchor
    include_price: bool
    train_universe: str
    control: str
    use_frozen_adapter: bool

    def features(self) -> tuple[str, ...]:
        if self.anchor == "ts":
            base = tuple(TS_RESID_FEATURES)
        else:
            base = tuple(BUDGET_RESID_FEATURES)
        if self.include_price:
            extra = tuple(c for c in FEATURE_NAMES if c not in base)
            return base + extra
        return base


P0_TS = F3BExperiment("P0_TS", "ts", False, "ts", "P0_TS", True)
P1_TS = F3BExperiment("P1_TS", "ts", True, "ts", "P0_TS", False)
P0_HUMAN = F3BExperiment("P0_HUMAN", "human", False, "budget", "P0_HUMAN", True)
P1_HUMAN = F3BExperiment("P1_HUMAN", "human", True, "budget", "P0_HUMAN", False)

ALL_EXPERIMENTS: dict[str, F3BExperiment] = {
    e.name: e for e in (P0_TS, P1_TS, P0_HUMAN, P1_HUMAN)
}


def get_f3b_experiment(name: str) -> F3BExperiment:
    if name not in ALL_EXPERIMENTS:
        raise KeyError(f"Unknown F3B experiment {name!r}; known={sorted(ALL_EXPERIMENTS)}")
    return ALL_EXPERIMENTS[name]

PRICE_SHEET_NAME = "جدول تغییر قیمت ها"
MAP_SHEET_NAME = "map"

SOURCE_PRODUCT_COL = "نام کالا"
PROVIDER_COL = "نام شرکت"
DISTRIBUTOR_PRICE_COL = "بهای فروش به پخش"
PHARMACY_PRICE_COL = "بهای فروش به داروخانه"
CONSUMER_PRICE_COL = "بهای مصرف کننده"
PACK_QTY_COL = "تعداد در بسته"
DATE_COL = "تاریخ"

MAP_SOURCE_COL = "نام محصول در تحویل به پخش"
MAP_TARGET_COL = "Dim Product"

PRICE_HISTORY_COLS = (
    "product_id",
    "product",
    "generic",
    "provider",
    "source_product_fa",
    "mapped_product_fa",
    "effective_date_raw",
    "effective_date",
    "effective_month",
    "distributor_price",
    "pharmacy_price",
    "consumer_price",
    "pack_quantity",
    "mapping_applied",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def external_price_dir() -> Path:
    return src_dir() / "data" / "external" / "f3b_price"


def triple_price_xlsx() -> Path:
    return external_price_dir() / "فرم قیمت سه گانهsc-fr-008 (2).xlsx"


def product_map_xlsx() -> Path:
    return external_price_dir() / "Map Product-Delivery dis.xlsx"


def f3b_output_dir() -> Path:
    out = src_dir() / "data" / "results" / "f3b"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3b_source_dir() -> Path:
    out = f3b_output_dir() / "source"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3b_feature_audit_dir() -> Path:
    out = f3b_output_dir() / "feature_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return repo_root() / "docs"
