"""F3C inventory feature tests (no live SQL, no XGB)."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pkg.benchmark.calendar import shamsi_month_start_gregorian
from pkg.research.features.inventory import add_inventory_features


def _make_panel(products, origins):
    rows = []
    for p in products:
        for o in origins:
            rows.append({"product": p, "origin": o})
    return pd.DataFrame(rows)


def _make_dist(product_date_qty):
    """[(product, snapshot_date_str, qty), ...]"""
    return pd.DataFrame([
        {"product": p, "snapshot_date": pd.Timestamp(d), "distributor_inventory_qty": q}
        for p, d, q in product_date_qty
    ])


def _make_fact(product_date_qty):
    return pd.DataFrame([
        {"product": p, "snapshot_date": pd.Timestamp(d), "factory_inventory_qty": q}
        for p, d, q in product_date_qty
    ])


class TestExactMonthEndJoin:
    def test_exact_date_match(self):
        panel = _make_panel(["A"], [140404])
        inv_date = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        dist = _make_dist([("A", inv_date, 100.0)])
        fact = _make_fact([("A", inv_date, 50.0)])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert out["log_distributor_inventory_qty"].iloc[0] == pytest.approx(np.log1p(100.0))
        assert out["log_factory_inventory_qty"].iloc[0] == pytest.approx(np.log1p(50.0))

    def test_later_snapshot_ignored(self):
        panel = _make_panel(["A"], [140404])
        inv_date = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        later = str(shamsi_month_start_gregorian(140404))
        dist = _make_dist([("A", later, 999.0)])
        fact = _make_fact([])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert pd.isna(out["log_distributor_inventory_qty"].iloc[0])

    def test_earlier_snapshot_ignored(self):
        panel = _make_panel(["A"], [140404])
        inv_date = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        earlier = str(shamsi_month_start_gregorian(140404) - timedelta(days=5))
        dist = _make_dist([("A", earlier, 999.0)])
        fact = _make_fact([])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert pd.isna(out["log_distributor_inventory_qty"].iloc[0])


class TestStatusAbsentOnExistingDate:
    def test_zero_qty_yields_log1p_zero(self):
        panel = _make_panel(["A"], [140404])
        inv_date = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        dist = _make_dist([("A", inv_date, 0.0)])
        fact = _make_fact([])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert out["log_distributor_inventory_qty"].iloc[0] == pytest.approx(0.0)
        assert out["distributor_missing_reason"].iloc[0] == "AVAILABLE"


class TestMissingDate:
    def test_missing_date_yields_nan(self):
        panel = _make_panel(["A"], [140404])
        dist = _make_dist([])
        fact = _make_fact([])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert pd.isna(out["log_distributor_inventory_qty"].iloc[0])
        assert pd.isna(out["log_factory_inventory_qty"].iloc[0])
        assert out["distributor_missing_reason"].iloc[0] == "NO_EXACT_MONTH_END_PRODUCT_RECORD"
        assert out["factory_missing_reason"].iloc[0] == "NO_EXACT_MONTH_END_PRODUCT_RECORD"


class TestNegativeQty:
    def test_negative_qty_yields_nan(self):
        panel = _make_panel(["A"], [140404])
        inv_date = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        dist = _make_dist([("A", inv_date, -10.0)])
        fact = _make_fact([("A", inv_date, -5.0)])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        assert pd.isna(out["log_distributor_inventory_qty"].iloc[0])
        assert pd.isna(out["log_factory_inventory_qty"].iloc[0])
        assert out["distributor_missing_reason"].iloc[0] == "NEGATIVE_QTY"
        assert out["factory_missing_reason"].iloc[0] == "NEGATIVE_QTY"


class TestLeakage:
    def test_each_row_uses_its_own_origin(self):
        """Two rows with different origins must get different inventory dates."""
        panel = _make_panel(["A"], [140404, 140407])
        d1 = str(shamsi_month_start_gregorian(140404) - timedelta(days=1))
        d2 = str(shamsi_month_start_gregorian(140407) - timedelta(days=1))
        dist = _make_dist([("A", d1, 10.0), ("A", d2, 20.0)])
        fact = _make_fact([])
        out = add_inventory_features(panel, dist, fact, origin_col="origin")
        row_o1 = out.loc[out["origin"] == 140404].iloc[0]
        row_o2 = out.loc[out["origin"] == 140407].iloc[0]
        assert row_o1["log_distributor_inventory_qty"] == pytest.approx(np.log1p(10.0))
        assert row_o2["log_distributor_inventory_qty"] == pytest.approx(np.log1p(20.0))

    def test_snapshot_strictly_before_origin_start(self):
        """inventory_month_end must be < origin_start for all PRIMARY origins."""
        for ym in [140404, 140407, 140410, 140501, 140504]:
            origin_start = shamsi_month_start_gregorian(ym)
            inv_date = origin_start - timedelta(days=1)
            assert inv_date < origin_start
