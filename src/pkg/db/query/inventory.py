"""F3C inventory SQL wrappers (distributor + factory)."""
from __future__ import annotations

from pkg.db.client import read_sql
from pkg.db.query.sql_file import load_sql


def load_distributor_inventory(**engine_kwargs):
    """Execute f3c_distributor_inventory.sql and return a DataFrame."""
    return read_sql(load_sql("f3c_distributor_inventory.sql"), **engine_kwargs)


def load_factory_inventory(**engine_kwargs):
    """Execute f3c_factory_inventory.sql and return a DataFrame."""
    return read_sql(load_sql("f3c_factory_inventory.sql"), **engine_kwargs)
