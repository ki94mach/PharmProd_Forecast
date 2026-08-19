"""F3C source contract tests (no live SQL, no freeze writes)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkg.benchmark.calendar import shamsi_month_start_gregorian


# ---------------------------------------------------------------------------
# SQL files on disk
# ---------------------------------------------------------------------------

QUERY_DIR = Path(__file__).resolve().parents[1] / "query"


def test_distributor_sql_file_exists():
    assert (QUERY_DIR / "f3c_distributor_inventory.sql").exists()


def test_factory_sql_file_exists():
    assert (QUERY_DIR / "f3c_factory_inventory.sql").exists()


def _read_sql(name: str) -> str:
    return (QUERY_DIR / name).read_text(encoding="utf-8")


class TestDistributorSql:
    sql = _read_sql("f3c_distributor_inventory.sql")

    def test_no_dqty_ne_zero_filter(self):
        assert "DQty <> 0" not in self.sql
        assert "DQty != 0" not in self.sql

    def test_no_islastdate(self):
        assert "IsLastDate" not in self.sql

    def test_blocked_not_in_scored_sum(self):
        # blocked has its own column but must not appear in distributor_inventory_qty
        assert "بلوکه" in self.sql  # audited
        lines = self.sql.split("\n")
        in_scored = False
        for line in lines:
            if "distributor_inventory_qty" in line.lower() and "as" in line.lower():
                in_scored = True
            if in_scored and "بلوکه" in line:
                pytest.fail("بلوکه appears inside the scored distributor_inventory_qty CASE")
            if in_scored and line.strip().startswith(")"):
                break

    def test_status_mojudi_and_dar_rah(self):
        assert "موجودی" in self.sql or "موجودي" in self.sql
        assert "در راه" in self.sql


class TestFactorySql:
    sql = _read_sql("f3c_factory_inventory.sql")

    def test_no_dqty_ne_zero_filter(self):
        assert "DQty <> 0" not in self.sql
        assert "DQty != 0" not in self.sql

    def test_no_islastdate(self):
        assert "IsLastDate" not in self.sql


# ---------------------------------------------------------------------------
# Calendar dates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yyyymm,expected", [
    (140404, date(2025, 6, 22)),
    (140407, date(2025, 9, 23)),
    (140410, date(2025, 12, 22)),
    (140501, date(2026, 3, 21)),
    (140504, date(2026, 6, 22)),
])
def test_shamsi_month_start(yyyymm, expected):
    assert shamsi_month_start_gregorian(yyyymm) == expected


@pytest.mark.parametrize("yyyymm,expected_inv_date", [
    (140404, date(2025, 6, 21)),
    (140407, date(2025, 9, 22)),
    (140410, date(2025, 12, 21)),
    (140501, date(2026, 3, 20)),
    (140504, date(2026, 6, 21)),
])
def test_inventory_month_end_date(yyyymm, expected_inv_date):
    origin_start = shamsi_month_start_gregorian(yyyymm)
    inv_date = origin_start - timedelta(days=1)
    assert inv_date == expected_inv_date


# ---------------------------------------------------------------------------
# Identity: inventory == on_hand + in_transit (synthetic)
# ---------------------------------------------------------------------------

def test_distributor_identity_synthetic():
    """distributor_inventory_qty must equal on_hand + in_transit exactly."""
    df = pd.DataFrame({
        "distributor_on_hand_qty": [100.0, 0.0, 50.0],
        "distributor_in_transit_qty": [20.0, 0.0, 30.0],
        "distributor_inventory_qty": [120.0, 0.0, 80.0],
        "blocked_inventory_qty": [5.0, 0.0, 10.0],
    })
    np.testing.assert_allclose(
        df["distributor_inventory_qty"],
        df["distributor_on_hand_qty"] + df["distributor_in_transit_qty"],
    )


# ---------------------------------------------------------------------------
# Mapping exactness
# ---------------------------------------------------------------------------

def test_mapping_exact_no_fuzzy():
    from pkg.research.f3c.prepare import product_mapping_audit
    df = pd.DataFrame({
        "fk_product": [1, 2, 3],
        "product": ["Alpha", "Beta", None],
    })
    mvp = ["Alpha", "Beta", "Gamma"]
    result = product_mapping_audit(df, mvp, "test")
    assert result["n_mvp_unmapped"].iloc[0] == 1
    assert "Gamma" in result["mvp_unmapped_products"].iloc[0]


# ---------------------------------------------------------------------------
# Missing product-date stays absent (no grid fill)
# ---------------------------------------------------------------------------

def test_missing_product_date_stays_absent():
    from pkg.research.features.inventory import add_inventory_features

    panel = pd.DataFrame({
        "product": ["ProductA"],
        "origin": [140404],
    })
    dist = pd.DataFrame({
        "product": ["ProductB"],
        "snapshot_date": pd.to_datetime(["2025-06-21"]),
        "distributor_inventory_qty": [100.0],
    })
    fact = pd.DataFrame(columns=["product", "snapshot_date", "factory_inventory_qty"])

    enriched = add_inventory_features(panel, dist, fact, origin_col="origin")
    assert pd.isna(enriched["log_distributor_inventory_qty"].iloc[0])
    assert enriched["distributor_missing_reason"].iloc[0] == "NO_EXACT_MONTH_END_PRODUCT_RECORD"
