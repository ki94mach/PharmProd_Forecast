"""Mark contradictory origin-validation artifact as superseded (preserve file)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    FAMILY,
    FAMILY_DIR_REL,
    MAPPING_AUDIT_FILENAME,
    PROTECTED_HB_ORIGINS,
    PROTECTED_ORIGIN_VALIDATION_FILENAME,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_protected_origin_validation_superseded(*, repo_root: Path) -> dict[str, Any]:
    """Preserve hb_protected_origin_validation_v1.json but prevent it from driving code."""
    repo_root = Path(repo_root).resolve()
    path = repo_root / FAMILY_DIR_REL / PROTECTED_ORIGIN_VALIDATION_FILENAME
    if not path.is_file():
        return {
            "ok": False,
            "exists": False,
            "path": path.as_posix(),
            "note": "Nothing to stamp; file absent",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["supersession"] = {
        "status": "SUPERSEDED",
        "superseded_by": (FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME).as_posix(),
        "drives_reliability_execution": False,
        "drives_review_contract": False,
        "stamped_at": _utc(),
        "family": FAMILY,
        "erratum": {
            "1404Q1_maps_to": 140401,
            "140312_is_not_a_reliability_origin": True,
            "reliability_execution_origins": sorted(PROTECTED_HB_ORIGINS),
            "correct_rule": "Version → version_to_qrt → origin_from_quarter",
            "incorrect_prior_recommendation": [140304, 140307, 140310, 140312],
            "note": (
                "Prior validation recommended protecting 140312 from min(target_date) "
                "HB construction. Corrected mapping: 1404Q1→140401; 140312 is pre-origin "
                "under that vintage, not a reliability forecast origin. Use "
                "hb_origin_mapping_audit_v1.json for reliability execution origins."
            ),
        },
    }
    # Soften contradictory conclusion so readers see supersession first
    payload["conclusion_status"] = "SUPERSEDED_DO_NOT_USE"
    payload["recommended_protected_origins_status"] = "SUPERSEDED"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return {
        "ok": True,
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "drives_reliability_execution": False,
        "superseded_by": (FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME).as_posix(),
        "erratum_1404Q1": 140401,
    }
