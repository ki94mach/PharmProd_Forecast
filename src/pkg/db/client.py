"""Shared SQL Server connection helpers."""
import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

DEFAULT_SERVER = os.getenv("SQL_SERVER", "op-db1-srv")
DEFAULT_DATABASE = os.getenv("SQL_DATABASE", "DWOrchid")
DEFAULT_DRIVER = os.getenv("SQL_DRIVER", "SQL Server")


def get_engine(
    server: str | None = None,
    database: str | None = None,
    driver: str | None = None,
) -> Engine:
    server = server or DEFAULT_SERVER
    database = database or DEFAULT_DATABASE
    driver = driver or DEFAULT_DRIVER
    connection_url = URL.create(
        "mssql+pyodbc",
        query={
            "odbc_connect": (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                "Trusted_Connection=yes;"
            )
        },
    )
    return create_engine(connection_url)


def read_sql(query: str, params=None, **engine_kwargs) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    engine = get_engine(**engine_kwargs)
    with engine.connect() as connection:
        return pd.read_sql(query, connection, params=params)
