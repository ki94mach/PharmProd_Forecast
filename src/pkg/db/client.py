"""Shared SQL Server connection helpers."""
import os
import sys
from typing import Literal

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

AuthMode = Literal["windows", "sql"]


def _load_env() -> None:
    """Find repo .env from cwd or parents (works from notebooks/)."""
    load_dotenv(find_dotenv(usecwd=True))


def _default_driver() -> str:
    return "SQL Server" if sys.platform == "win32" else "FreeTDS"


def _resolve_auth(auth: str | None) -> AuthMode:
    mode = (auth or os.getenv("SQL_AUTH", "windows")).strip().lower()
    if mode not in ("windows", "sql"):
        raise ValueError("SQL_AUTH must be 'windows' or 'sql'")
    return mode  # type: ignore[return-value]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_freetds(driver: str) -> bool:
    return "freetds" in driver.lower()


def _is_ms_odbc(driver: str) -> bool:
    name = driver.lower()
    return "odbc driver" in name and "sql server" in name


def _host_and_port(
    server: str | None = None,
    port: str | None = None,
) -> tuple[str, str]:
    server = (server or os.getenv("SQL_SERVER", "op-db1-srv")).strip()
    port = (port if port is not None else os.getenv("SQL_PORT", "")).strip()
    if "," in server and not port:
        host, _, maybe_port = server.partition(",")
        return host.strip(), maybe_port.strip()
    return server, port


def _odbc_connect_string(
    host: str,
    port: str,
    database: str,
    driver: str,
    auth: AuthMode,
    username: str | None,
    password: str | None,
) -> str:
    parts = [f"DRIVER={{{driver}}}"]

    if _is_freetds(driver):
        parts += [
            f"SERVER={host}",
            f"PORT={port or '1433'}",
            f"TDS_Version={os.getenv('SQL_TDS_VERSION', '7.4').strip()}",
        ]
    else:
        parts.append(f"SERVER={host},{port}" if port else f"SERVER={host}")

    parts.append(f"DATABASE={database}")

    if auth == "windows":
        if _is_freetds(driver):
            raise ValueError(
                "Windows auth is not supported with FreeTDS; "
                "use SQL_AUTH=sql or a Microsoft ODBC driver on Windows"
            )
        parts.append("Trusted_Connection=yes")
    else:
        user = username if username is not None else os.getenv("SQL_USER", "")
        pwd = password if password is not None else os.getenv("SQL_PASSWORD", "")
        if not user or not pwd:
            raise ValueError(
                "SQL login requires SQL_USER and SQL_PASSWORD "
                "(or username/password arguments)"
            )
        parts += [f"UID={user}", f"PWD={pwd}"]
        if _is_ms_odbc(driver):
            parts += ["Trusted_Connection=no", "Authentication=SqlPassword"]

    if _is_ms_odbc(driver) and _truthy(
        os.getenv("SQL_TRUST_SERVER_CERTIFICATE"),
        default=sys.platform != "win32",
    ):
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
    _load_env()
    host, resolved_port = _host_and_port(server, port)
    database = database or os.getenv("SQL_DATABASE", "DWOrchid")
    driver = driver or os.getenv("SQL_DRIVER") or _default_driver()
    connection_url = URL.create(
        "mssql+pyodbc",
        query={
            "odbc_connect": _odbc_connect_string(
                host,
                resolved_port,
                database,
                driver,
                _resolve_auth(auth),
                username,
                password,
            )
        },
    )
    return create_engine(connection_url)


def read_sql(query: str, params=None, **engine_kwargs) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    engine = get_engine(**engine_kwargs)
    with engine.connect() as connection:
        return pd.read_sql(query, connection, params=params)
