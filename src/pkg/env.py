"""Centralized project environment loading.

Canonical file: ``src/.env`` (next to ``main.py`` / package data).

All entry points should call :func:`load_project_env` instead of ad-hoc
``load_dotenv()``. Credentials and SQL settings belong only in that file —
never in systemd unit files or shell wrappers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# pkg/env.py → pkg → src
SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
# Single source of truth for local/server configuration.
DEFAULT_ENV_PATH = SRC_DIR / ".env"

_LOADED = False


def project_env_path() -> Path:
    """Return the env file path (``FORECAST_ENV_FILE`` override or ``src/.env``)."""
    override = os.environ.get("FORECAST_ENV_FILE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ENV_PATH


def load_project_env(*, force: bool = False) -> Optional[Path]:
    """Load the centralized ``src/.env`` into ``os.environ``.

    Returns the path that was loaded, or ``None`` if the file is missing.
    Safe to call multiple times (no-op after the first successful load unless
    ``force=True``).
    """
    global _LOADED
    if _LOADED and not force:
        return project_env_path() if project_env_path().is_file() else None

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    path = project_env_path()
    if path.is_file():
        load_dotenv(path, override=False)
        _LOADED = True
        return path

    # Helpful fallback while migrating: repo-root .env if src/.env is absent.
    legacy = REPO_ROOT / ".env"
    if legacy.is_file():
        load_dotenv(legacy, override=False)
        _LOADED = True
        return legacy

    return None
