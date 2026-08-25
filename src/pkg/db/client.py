"""Shared SQL Server connection helpers.

Authentication matches DistCore's ODBC pattern
(``ODBC Driver 17 for SQL Server`` + Windows Trusted_Connection, or SQL UID/PWD,
always ``TrustServerCertificate=yes`` for the Microsoft driver).

Two profiles live in ``src/.env`` (see DistCore ``db.yml``):

* ``SQL_PROFILE=local``  — hostname ``op-db1-srv``, Windows auth (laptop)
* ``SQL_PROFILE=server`` — IP ``10.20.40.40``, SQL login (Linux deploy)

Switch with ``SQL_PROFILE`` only; keep both credential blocks in ``src/.env``.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from pkg.env import load_project_env

AuthMode = Literal["windows", "sql"]
SqlProfile = Literal["local", "server"]

# Same default driver DistCore uses in db.yml / ConnectionStringBuilder.
DEFAULT_MS_ODBC_DRIVER = "ODBC Driver 17 for SQL Server"

_PROFILE_ALIASES = {
    "local": "local",
    "laptop": "local",
    "dev": "local",
    "windows": "local",
    "server": "server",
    "remote": "server",
    "prod": "server",
    "linux": "server",
}


def _load_env() -> None:
    """Load centralized ``src/.env``."""
    load_project_env()


def active_sql_profile() -> SqlProfile:
    """Return ``local`` or ``server`` from ``SQL_PROFILE`` (default: local)."""
    raw = (os.getenv("SQL_PROFILE") or "local").strip().lower()
    mapped = _PROFILE_ALIASES.get(raw)
    if mapped is None:
        raise ValueError(
            f"SQL_PROFILE must be 'local' or 'server' (got {raw!r})"
        )
    return mapped  # type: ignore[return-value]


def _profile_prefix(profile: Optional[SqlProfile] = None) -> str:
    p = profile or active_sql_profile()
    return "SQL_LOCAL_" if p == "local" else "SQL_SERVER_"


def sql_setting(
    name: str,
    default: str = "",
    *,
    profile: Optional[SqlProfile] = None,
) -> str:
    """Read a profile-scoped setting with legacy unprefixed fallback.

    Example: ``sql_setting("SERVER")`` reads ``SQL_LOCAL_SERVER`` or
    ``SQL_SERVER_SERVER`` depending on ``SQL_PROFILE``, then falls back to
    ``SQL_SERVER`` for older ``.env`` files.
    """
    key = str(name).strip().upper()
    prefixed = f"{_profile_prefix(profile)}{key}"
    value = os.getenv(prefixed)
    if value is not None and str(value).strip() != "":
        return str(value).strip()
    legacy = os.getenv(f"SQL_{key}")
    if legacy is not None and str(legacy).strip() != "":
        return str(legacy).strip()
    return default


def _default_driver() -> str:
    """Prefer Microsoft ODBC 17 (DistCore). FreeTDS only if explicitly configured."""
    return DEFAULT_MS_ODBC_DRIVER


def _resolve_auth(auth: Optional[str]) -> AuthMode:
    default = "windows" if active_sql_profile() == "local" else "sql"
    mode = (auth or sql_setting("AUTH", default)).strip().lower()
    if mode not in ("windows", "sql"):
        raise ValueError("SQL auth must be 'windows' or 'sql'")
    return mode  # type: ignore[return-value]


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_freetds(driver: str) -> bool:
    return "freetds" in driver.lower()


def _is_ms_odbc(driver: str) -> bool:
    name = driver.lower()
    return "odbc driver" in name and "sql server" in name


def _host_and_port(
    server: Optional[str] = None,
    port: Optional[str] = None,
) -> tuple[str, str]:
    default_host = "op-db1-srv" if active_sql_profile() == "local" else "10.20.40.40"
    server = (server if server is not None else sql_setting("SERVER", default_host)).strip()
    port = (port if port is not None else sql_setting("PORT", "")).strip()
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
    username: Optional[str],
    password: Optional[str],
) -> str:
    """Build an ODBC connect string compatible with DistCore.

    DistCore ``ConnectionStringBuilder`` emits::

        DRIVER={ODBC Driver 17 for SQL Server};
        SERVER=...;DATABASE=...;
        Trusted_Connection=yes;   # or UID=...;PWD=...;
        TrustServerCertificate=yes;
    """
    parts = [f"DRIVER={{{driver}}}"]

    if _is_freetds(driver):
        # Optional Linux fallback — not DistCore's default path.
        parts += [
            f"SERVER={host}",
            f"PORT={port or '1433'}",
            f"TDS_Version={sql_setting('TDS_VERSION', '7.4')}",
        ]
    else:
        # DistCore passes SERVER as configured (host or host,port). Prefer
        # host,port only when PORT is set explicitly.
        parts.append(f"SERVER={host},{port}" if port else f"SERVER={host}")

    parts.append(f"DATABASE={database}")

    if auth == "windows":
        if _is_freetds(driver):
            raise ValueError(
                "Windows auth is not supported with FreeTDS; "
                "use SQL_PROFILE=local with ODBC Driver 17, or SQL auth credentials"
            )
        parts.append("Trusted_Connection=yes")
    else:
        user = username if username is not None else sql_setting("USER", "")
        pwd = password if password is not None else sql_setting("PASSWORD", "")
        if not user or not pwd:
            raise ValueError(
                "SQL login requires USER and PASSWORD for the active SQL_PROFILE "
                "(SQL_LOCAL_* or SQL_SERVER_*)"
            )
        parts += [f"UID={user}", f"PWD={pwd}"]

    if _is_ms_odbc(driver) and _truthy(
        sql_setting("TRUST_SERVER_CERTIFICATE") or None,
        default=True,
    ):
        parts.append("TrustServerCertificate=yes")

    return ";".join(parts) + ";"


def get_engine(
    server: Optional[str] = None,
    database: Optional[str] = None,
    driver: Optional[str] = None,
    auth: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[str] = None,
) -> Engine:
    _load_env()
    host, resolved_port = _host_and_port(server, port)
    database = database or sql_setting("DATABASE", "DWOrchid")
    driver = driver or sql_setting("DRIVER") or _default_driver()
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
