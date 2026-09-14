"""Shared Archval55 reuse-gate helpers (inventory + importer)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.benchmark.calendar import quarter_from_origin, shamsi_add_months
from pkg.benchmark.universes import load_universe_product_names
from pkg.forecast_jobs.plan import decode_product_key, encode_product_key

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
LOCK_DIR = RESULTS / "archval55_lock"
ARCHVAL25_EXPORT = RESULTS / "archval25_export"
ARCHVAL25_FALLBACK_EXPORT = RESULTS / "archval25_export_fallback_v1"
SALES_PATH = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

PINNED_SALES_SHA = (
    "1d146d49261357f6b35d743a4c4064b22abe82047f6f5c36bbe4496d03f1ffc7"
)
PINNED_N_ROWS = 6100
PINNED_SIZE_BYTES = 34006

ORIGINS = (140104, 140201, 140301, 140402)
ARCHITECTURES = ("v2.1", "a0", "a1")
SEEDS = (41, 42, 43)
HORIZON = 15
CUTOFF = "production"
FALLBACK_POLICY = "naive_last_value"
FALLBACK_VERSION = "v1"
MVP_UNIVERSE = "mvp_products"
ARCH25_UNIVERSE = "v3a_architecture_validation_products"


@dataclass(frozen=True)
class JobKey:
    product: str
    forecast_origin: int
    engine: str
    seed: Optional[int]

    @property
    def product_id(self) -> str:
        return encode_product_key(self.product, self.seed)

    @property
    def quarter(self) -> str:
        return quarter_from_origin(int(self.forecast_origin))

    def as_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "forecast_origin": int(self.forecast_origin),
            "engine": self.engine,
            "seed": self.seed if self.seed is not None else "",
            "product_id": self.product_id,
            "quarter": self.quarter,
        }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_reuse_gate(
    *,
    sales_path: Path = SALES_PATH,
    archval25_export: Path = ARCHVAL25_EXPORT,
) -> dict[str, Any]:
    """Return gate report; ``ok`` True only if sales + science settings match."""
    report: dict[str, Any] = {
        "ok": False,
        "checks": {},
        "pinned_sales_sha": PINNED_SALES_SHA,
        "sales_path": str(sales_path.as_posix()),
    }
    checks = report["checks"]

    if not sales_path.is_file():
        checks["sales_file"] = {"ok": False, "detail": f"missing {sales_path}"}
        return report
    live_sha = file_sha256(sales_path)
    live_size = sales_path.stat().st_size
    sales_df = pd.read_parquet(sales_path)
    live_rows = int(len(sales_df))
    checks["sales_content_sha256"] = {
        "ok": live_sha.lower() == PINNED_SALES_SHA.lower(),
        "live": live_sha,
        "pinned": PINNED_SALES_SHA,
    }
    checks["sales_n_rows"] = {
        "ok": live_rows == PINNED_N_ROWS,
        "live": live_rows,
        "pinned": PINNED_N_ROWS,
    }
    checks["sales_size_bytes"] = {
        "ok": int(live_size) == PINNED_SIZE_BYTES,
        "live": int(live_size),
        "pinned": PINNED_SIZE_BYTES,
    }

    snap_path = archval25_export / "input_snapshot.json"
    man_path = archval25_export / "manifest.json"
    if snap_path.is_file():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        export_sha = str(snap.get("content_sha256") or "").lower()
        checks["archval25_snapshot_sha"] = {
            "ok": export_sha == PINNED_SALES_SHA.lower(),
            "export": export_sha,
            "pinned": PINNED_SALES_SHA,
        }
    else:
        checks["archval25_snapshot_sha"] = {
            "ok": False,
            "detail": f"missing {snap_path}",
        }

    science_ok = True
    science_detail: dict[str, Any] = {
        "cutoff": CUTOFF,
        "horizon": HORIZON,
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "origins": list(ORIGINS),
        "fallback_policy": FALLBACK_POLICY,
        "fallback_version": FALLBACK_VERSION,
    }
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sci = man.get("scientific_config") or {}
        cutoff = str((sci.get("cutoff_policy") or {}).get("name") or man.get("cutoff_policy") or "")
        horizon = int(man.get("horizon") or sci.get("horizon") or -1)
        arches = [str(a).lower() for a in (man.get("architectures") or sci.get("architectures") or [])]
        seeds = [int(s) for s in (man.get("seeds") or sci.get("seeds") or [])]
        origins = [int(o) for o in (man.get("origins") or sci.get("origins") or [])]
        science_ok = (
            cutoff == CUTOFF
            and horizon == HORIZON
            and arches == list(ARCHITECTURES)
            and seeds == list(SEEDS)
            and origins == list(ORIGINS)
        )
        science_detail["export_cutoff"] = cutoff
        science_detail["export_horizon"] = horizon
        science_detail["export_architectures"] = arches
        science_detail["export_seeds"] = seeds
        science_detail["export_origins"] = origins
        # Original archval25 lacked fallback key; prediction science for SUCCESS
        # is still compatible. Gate records that; importer uses fallback_v1 fills.
        science_detail["export_has_fallback"] = "fallback" in sci
    else:
        science_ok = False
        science_detail["detail"] = f"missing {man_path}"
    checks["scientific_settings"] = {"ok": science_ok, **science_detail}

    report["ok"] = all(bool(c.get("ok")) for c in checks.values())
    return report


def expand_grid(products: list[str]) -> list[JobKey]:
    keys: list[JobKey] = []
    for product in products:
        for origin in ORIGINS:
            for engine in ARCHITECTURES:
                if engine in {"a0", "a1"}:
                    for seed in SEEDS:
                        keys.append(
                            JobKey(
                                product=str(product),
                                forecast_origin=int(origin),
                                engine=engine,
                                seed=int(seed),
                            )
                        )
                else:
                    keys.append(
                        JobKey(
                            product=str(product),
                            forecast_origin=int(origin),
                            engine=engine,
                            seed=None,
                        )
                    )
    return keys


def load_archval25_job_statuses(
    export_dir: Path = ARCHVAL25_EXPORT,
) -> dict[tuple[str, int, str, Optional[int]], dict[str, Any]]:
    """Map logical job key → status row from archval25 state.sqlite."""
    db = export_dir / "state.sqlite"
    out: dict[tuple[str, int, str, Optional[int]], dict[str, Any]] = {}
    if not db.is_file():
        return out
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT engine_version, forecast_origin, product_id, status,
               error_type, error_message, selected_model, output_path
        FROM jobs
        """
    )
    for row in rows:
        engine = str(row["engine_version"]).lower()
        origin = int(row["forecast_origin"])
        product_id = str(row["product_id"])
        if engine in {"a0", "a1"}:
            product, seed = decode_product_key(product_id)
        else:
            product, seed = product_id, None
        key = (str(product), origin, engine, seed)
        out[key] = {
            "status": str(row["status"]),
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "selected_model": row["selected_model"],
            "output_path": row["output_path"],
            "product_id": product_id,
        }
    conn.close()
    return out


def short_history_flag(product: str, origin: int, sales: pd.DataFrame) -> dict[str, Any]:
    """Approx train months through last_complete (origin−2)."""
    lc = shamsi_add_months(int(origin), -2)
    sub = sales.loc[sales["product"].astype(str) == str(product)]
    train_n = int((pd.to_numeric(sub["date"], errors="coerce") <= lc).sum())
    return {
        "train_months_through_last_complete": train_n,
        "a1_short_history_risk": bool(train_n < 22 and int(origin) == 140104),
        "v21_short_history_risk": bool(train_n < 13 and int(origin) == 140104),
    }


def classify_archval25_bucket(
    status_row: Optional[dict[str, Any]],
) -> str:
    """Return reusable | ineligible_fill | missing_from_export."""
    if status_row is None:
        return "missing_from_export"
    status = str(status_row.get("status") or "").upper()
    if status == "SUCCESS":
        return "reusable"
    if status in {"FAILED", "INELIGIBLE"}:
        return "ineligible_fill"
    return "missing_from_export"


def build_inventory(
    *,
    gate: dict[str, Any],
    sales: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return reusable, ineligible_fill, required, all_jobs frames."""
    mvp = list(load_universe_product_names(MVP_UNIVERSE, validate=True))
    arch25 = set(load_universe_product_names(ARCH25_UNIVERSE, validate=True))
    statuses = load_archval25_job_statuses()
    gate_ok = bool(gate.get("ok"))

    reusable_rows: list[dict[str, Any]] = []
    ineligible_rows: list[dict[str, Any]] = []
    required_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for key in expand_grid(mvp):
        in_arch25 = key.product in arch25
        status_row = statuses.get(
            (key.product, key.forecast_origin, key.engine, key.seed)
        )
        hist = short_history_flag(key.product, key.forecast_origin, sales)
        base = {
            **key.as_dict(),
            "in_archval25_cohort": in_arch25,
            "gate_ok": gate_ok,
            **hist,
        }

        if not in_arch25:
            bucket = "required"
            reason = "new_mvp_sku"
            action = "execute"
        else:
            bucket = classify_archval25_bucket(status_row)
            if bucket == "reusable":
                if gate_ok:
                    reason = "archval25_success_gate_pass"
                    action = "import_success"
                else:
                    bucket = "required"
                    reason = "archval25_success_but_gate_failed"
                    action = "execute"
            elif bucket == "ineligible_fill":
                if gate_ok:
                    reason = "archval25_failed_or_ineligible_expected"
                    action = "import_fallback_or_rerun_ineligible"
                else:
                    bucket = "required"
                    reason = "archval25_ineligible_but_gate_failed"
                    action = "execute"
            else:
                bucket = "required"
                reason = "archval25_key_missing_from_export"
                action = "execute"

        row = {
            **base,
            "bucket": bucket,
            "reason": reason,
            "action": action,
            "source_status": (status_row or {}).get("status"),
            "source_error_type": (status_row or {}).get("error_type"),
            "source_error_message": (status_row or {}).get("error_message"),
            "source_selected_model": (status_row or {}).get("selected_model"),
        }
        all_rows.append(row)
        if bucket == "reusable":
            reusable_rows.append(row)
        elif bucket == "ineligible_fill":
            ineligible_rows.append(row)
        else:
            required_rows.append(row)

    return (
        pd.DataFrame(reusable_rows),
        pd.DataFrame(ineligible_rows),
        pd.DataFrame(required_rows),
        pd.DataFrame(all_rows),
    )
