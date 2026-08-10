"""Shared SQL Server connection helpers."""
import os
import sys
from typing import Literal

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

AuthMode = Literal["windows", "sql"]

# Walk up from cwd so notebooks/ and src/ both find the repo .env
load_dotenv(find_dotenv(usecwd=True))


def _default_driver() -> str:
    if sys.platform == "win32":
        return "SQL Server"
    return "ODBC Driver 18 for SQL Server"


def _resolve_auth(auth: str | None) -> AuthMode:
    mode = (auth or os.getenv("SQL_AUTH", "windows")).strip().lower()
    if mode not in ("windows", "sql"):
        raise ValueError("SQL_AUTH must be 'windows' or 'sql'")
    return mode  # type: ignore[return-value]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _server_address(server: str | None = None, port: str | None = None) -> str:
    """Build ODBC SERVER= value; accepts hostname or IP, optional port."""
    server = (server or os.getenv("SQL_SERVER", "op-db1-srv")).strip()
    port = (port if port is not None else os.getenv("SQL_PORT", "")).strip()
    # Don't double-append if caller already used host,port or host\instance
    if port and "," not in server and "\\" not in server:
        return f"{server},{port}"
    return server


def _odbc_connect_string(
    server: str,
    database: str,
    driver: str,
    auth: AuthMode,
    username: str | None,
    password: str | None,
) -> str:
    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
    ]
    if auth == "windows":
        parts.append("Trusted_Connection=yes")
    else:
        user = username if username is not None else os.getenv("SQL_USER", "")
        pwd = password if password is not None else os.getenv("SQL_PASSWORD", "")
        if not user or not pwd:
            raise ValueError(
                "SQL login requires SQL_USER and SQL_PASSWORD "
                "(or username/password arguments)"
            )
        parts.extend([f"UID={user}", f"PWD={pwd}", "Trusted_Connection=no"])

    # Driver 17/18 on Linux often need this for corporate SQL Server TLS
    default_trust = sys.platform != "win32"
    if _truthy(os.getenv("SQL_TRUST_SERVER_CERTIFICATE"), default=default_trust):
        parts.append("TrustServerCertificate=yes")

    return ";".join(parts) + ";"


def get_engine(
    server: str | None = None,
    database: str | None = None,
    driver: str | None = None,
    auth: str | None = None,
    username: str | None = None,
    password: str | None = None,
    port: str | None = None,
) -> Engine:
    load_dotenv(find_dotenv(usecwd=True))
    server = _server_address(server, port)
    database = database or os.getenv("SQL_DATABASE", "DWOrchid")
    driver = driver or os.getenv("SQL_DRIVER", _default_driver())
    auth_mode = _resolve_auth(auth)
    connection_url = URL.create(
        "mssql+pyodbc",
        query={
            "odbc_connect": _odbc_connect_string(
                server, database, driver, auth_mode, username, password
            )
        },
    )
    return create_engine(connection_url)


def read_sql(query: str, params=None, **engine_kwargs) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    engine = get_engine(**engine_kwargs)
    with engine.connect() as connection:
        return pd.read_sql(query, connection, params=params)
