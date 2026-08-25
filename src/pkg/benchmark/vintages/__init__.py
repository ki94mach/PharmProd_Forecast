"""Canonical Shamsi vintage specs for historical TS benchmark backfills.

Quarter → forecast_origin mapping uses
:func:`pkg.benchmark.calendar.origin_from_quarter` (first month of the quarter),
the inverse of :func:`pkg.benchmark.calendar.quarter_from_origin`. Origins are
**not** hard-coded independently of quarter labels.

Training rule (independent of maturity / sales availability)::

    train on months with date < forecast_origin

CLI::

    python -m pkg.benchmark.vintages validate
    python -m pkg.benchmark.vintages build --force
    python -m pkg.benchmark.vintages show
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import pandas as pd

from pkg.benchmark.calendar import (
    iter_shamsi_quarters,
    origin_from_quarter,
    quarter_from_origin,
    shamsi_add_months,
)
from pkg.benchmark.config import (
    ALLOWED_FORECAST_QRTS,
    EXCLUDED_EMPTY_FORECAST_QRTS,
    FORECAST_HORIZON_MONTHS,
    INCOMPLETE_SHAMSI_MONTHS,
    default_benchmark_root,
)

VintageStatus = Literal[
    "forecastable",
    "partially_matured",
    "fully_matured",
    "future_origin",
]

MANIFEST_NAME = "ts_backfill_1401Q1_1405Q2"
START_QUARTER = "1401Q1"
END_QUARTER = "1405Q2"
EXPECTED_N_QUARTERS = 18
DEFAULT_HORIZON = FORECAST_HORIZON_MONTHS  # 15

MANIFEST_COLUMNS = (
    "quarter",
    "forecast_origin",
    "horizon",
    "status",
    "training_cutoff_exclusive",
    "notes",
)

# Historical CSV modal origins that differ from the canonical first-of-quarter
# mapping (observed in frozen ts_csvs / results). Documented only — not used
# as the experiment origin.
HISTORICAL_MODAL_ORIGIN_DEVIATIONS: dict[str, int] = {
    "1403Q1": 140304,
    "1403Q2": 140306,
}


class VintageManifestError(Exception):
    """Raised when the vintage manifest fails validation."""


@dataclass(frozen=True)
class VintageSpec:
    """One historical forecast vintage.

    Attributes:
        quarter: Shamsi quarter label (``YYYYQn``).
        forecast_origin: First forecast month (Shamsi YYYYMM). Always equal to
            ``origin_from_quarter(quarter)`` in the canonical manifest.
        horizon: Number of forecast months (locked to 15 for this experiment).
        status: Lifecycle / evaluation maturity label (see module docstring).
        notes: Optional free-text context (historical deviations, exclusions).
    """

    quarter: str
    forecast_origin: int
    horizon: int = DEFAULT_HORIZON
    status: VintageStatus = "forecastable"
    notes: str = ""

    @property
    def training_cutoff_exclusive(self) -> int:
        """Exclusive training cutoff: keep sales with ``date < this``."""
        return int(self.forecast_origin)

    @property
    def last_target_month(self) -> int:
        return shamsi_add_months(int(self.forecast_origin), int(self.horizon) - 1)

    def target_months(self) -> tuple[int, ...]:
        origin = int(self.forecast_origin)
        return tuple(shamsi_add_months(origin, h) for h in range(int(self.horizon)))


def vintages_dir() -> Path:
    return Path(__file__).resolve().parent


def default_manifest_csv_path() -> Path:
    return vintages_dir() / f"{MANIFEST_NAME}.csv"


def default_manifest_meta_path() -> Path:
    return vintages_dir() / f"{MANIFEST_NAME}.meta.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify_vintage_status(
    forecast_origin: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    as_of_complete_month: int,
) -> VintageStatus:
    """Classify vintage maturity relative to last **complete** sales month.

    Training always uses ``date < forecast_origin`` regardless of this status.
    Sales availability affects evaluation maturity only.

    Precedence:
    1. ``future_origin`` — origin is more than one month after ``as_of``
       (cannot train through ``origin - 1`` with complete sales).
    2. ``fully_matured`` — all ``horizon`` target months ``<= as_of``.
    3. ``partially_matured`` — at least the origin month ``<= as_of``.
    4. ``forecastable`` — can train (``origin <= as_of + 1``) but no target
       months yet have complete actuals.
    """
    origin = int(forecast_origin)
    as_of = int(as_of_complete_month)
    max_trainable_origin = shamsi_add_months(as_of, 1)
    if origin > max_trainable_origin:
        return "future_origin"

    last_target = shamsi_add_months(origin, int(horizon) - 1)
    if last_target <= as_of:
        return "fully_matured"
    if origin <= as_of:
        return "partially_matured"
    return "forecastable"


def resolve_as_of_complete_month(
    *,
    sales_max_month: Optional[int] = None,
    incomplete_months: Optional[Iterable[int]] = None,
) -> int:
    """Last complete Shamsi sales month for maturity status.

    Prefer an explicit ``sales_max_month`` (e.g. from frozen sales). Incomplete
    months listed in ``INCOMPLETE_SHAMSI_MONTHS`` are never treated as complete.
    """
    incomplete = frozenset(
        int(x) for x in (incomplete_months if incomplete_months is not None else INCOMPLETE_SHAMSI_MONTHS)
    )
    if sales_max_month is None:
        root = default_benchmark_root()
        sales_path = root / "raw" / "sales.parquet"
        if not sales_path.exists():
            raise FileNotFoundError(
                f"Cannot resolve as_of_complete_month: missing {sales_path}"
            )
        sales = pd.read_parquet(sales_path, columns=["date"])
        sales_max_month = int(pd.to_numeric(sales["date"], errors="coerce").max())

    as_of = int(sales_max_month)
    while as_of in incomplete:
        as_of = shamsi_add_months(as_of, -1)
    return as_of


def _notes_for_quarter(quarter: str) -> str:
    parts: list[str] = []
    if quarter in HISTORICAL_MODAL_ORIGIN_DEVIATIONS:
        parts.append(
            f"Historical TS CSV modal origin was "
            f"{HISTORICAL_MODAL_ORIGIN_DEVIATIONS[quarter]}; "
            f"canonical experiment origin is {origin_from_quarter(quarter)} "
            f"(first month of quarter)."
        )
    if quarter in EXCLUDED_EMPTY_FORECAST_QRTS:
        parts.append(
            "Listed in EXCLUDED_EMPTY_FORECAST_QRTS for the v1 freeze "
            "(not in ALLOWED_FORECAST_QRTS); included here for contiguous backfill."
        )
    elif quarter not in ALLOWED_FORECAST_QRTS:
        parts.append("Not in ALLOWED_FORECAST_QRTS freeze allow-list.")
    return " ".join(parts)


def build_vintage_specs(
    start_quarter: str = START_QUARTER,
    end_quarter: str = END_QUARTER,
    *,
    horizon: int = DEFAULT_HORIZON,
    as_of_complete_month: Optional[int] = None,
) -> list[VintageSpec]:
    """Build contiguous VintageSpec list; origins from ``origin_from_quarter`` only."""
    if int(horizon) != DEFAULT_HORIZON:
        raise VintageManifestError(
            f"this experiment locks horizon={DEFAULT_HORIZON}, got {horizon}"
        )
    as_of = (
        int(as_of_complete_month)
        if as_of_complete_month is not None
        else resolve_as_of_complete_month()
    )
    specs: list[VintageSpec] = []
    for qrt in iter_shamsi_quarters(start_quarter, end_quarter):
        origin = origin_from_quarter(qrt)
        # Guard: round-trip must hold (single mapping definition).
        if quarter_from_origin(origin) != qrt:
            raise VintageManifestError(
                f"quarter/origin round-trip failed: {qrt} -> {origin} -> "
                f"{quarter_from_origin(origin)}"
            )
        status = classify_vintage_status(
            origin, horizon=horizon, as_of_complete_month=as_of
        )
        specs.append(
            VintageSpec(
                quarter=qrt,
                forecast_origin=origin,
                horizon=int(horizon),
                status=status,
                notes=_notes_for_quarter(qrt),
            )
        )
    return specs


def specs_to_frame(specs: Sequence[VintageSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "quarter": spec.quarter,
                "forecast_origin": int(spec.forecast_origin),
                "horizon": int(spec.horizon),
                "status": spec.status,
                "training_cutoff_exclusive": int(spec.training_cutoff_exclusive),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def frame_to_specs(frame: pd.DataFrame) -> list[VintageSpec]:
    specs: list[VintageSpec] = []
    for _, row in frame.iterrows():
        specs.append(
            VintageSpec(
                quarter=str(row["quarter"]).strip(),
                forecast_origin=int(row["forecast_origin"]),
                horizon=int(row["horizon"]),
                status=str(row["status"]).strip(),  # type: ignore[arg-type]
                notes=str(row.get("notes", "") or ""),
            )
        )
    return specs


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    out = frame.loc[:, list(MANIFEST_COLUMNS)].copy()
    out["forecast_origin"] = out["forecast_origin"].astype(int)
    out["horizon"] = out["horizon"].astype(int)
    out["training_cutoff_exclusive"] = out["training_cutoff_exclusive"].astype(int)
    out["notes"] = out["notes"].fillna("").astype(str)
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8")


def build_manifest_meta(
    specs: Sequence[VintageSpec],
    *,
    as_of_complete_month: int,
    csv_sha256: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "manifest_name": MANIFEST_NAME,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "Canonical vintage list for historical TS V1/V2/V3 backfill experiments "
            f"from {START_QUARTER} through {END_QUARTER}. Forecast origins are "
            "derived only via origin_from_quarter (first month of quarter)."
        ),
        "start_quarter": START_QUARTER,
        "end_quarter": END_QUARTER,
        "n_quarters": len(specs),
        "horizon": DEFAULT_HORIZON,
        "as_of_complete_month": int(as_of_complete_month),
        "as_of_source": (
            "frozen src/data/benchmarks/v1/raw/sales.parquet max(date), "
            "walking back through INCOMPLETE_SHAMSI_MONTHS"
        ),
        "origin_mapping": (
            "pkg.benchmark.calendar.origin_from_quarter — inverse of "
            "quarter_from_origin; not hard-coded per quarter"
        ),
        "training_rule": "date < forecast_origin (training_cutoff_exclusive)",
        "status_policy": (
            "Evaluation maturity only. Sales availability never changes the "
            "training cutoff. Status precedence: future_origin, fully_matured, "
            "partially_matured, forecastable."
        ),
        "historical_modal_origin_deviations": {
            k: v for k, v in HISTORICAL_MODAL_ORIGIN_DEVIATIONS.items()
        },
        "quarter_to_forecast_origin": {
            s.quarter: int(s.forecast_origin) for s in specs
        },
        "csv_sha256": csv_sha256,
    }


def write_vintage_manifest(
    specs: Sequence[VintageSpec],
    *,
    as_of_complete_month: int,
    out_dir: Optional[Path] = None,
    force: bool = False,
) -> tuple[Path, Path]:
    directory = Path(out_dir) if out_dir is not None else vintages_dir()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{MANIFEST_NAME}.csv"
    meta_path = directory / f"{MANIFEST_NAME}.meta.json"
    if csv_path.exists() and not force:
        raise VintageManifestError(
            f"Refusing to overwrite {csv_path}. Pass force=True to regenerate."
        )
    frame = specs_to_frame(specs)
    csv_bytes = dataframe_to_csv_bytes(frame)
    meta = build_manifest_meta(
        specs,
        as_of_complete_month=as_of_complete_month,
        csv_sha256=_sha256_bytes(csv_bytes),
    )
    csv_path.write_bytes(csv_bytes)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, meta_path


def load_vintage_manifest(
    csv_path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> list[VintageSpec]:
    path = Path(csv_path) if csv_path is not None else default_manifest_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Vintage manifest missing: {path}. "
            "Run: python -m pkg.benchmark.vintages build"
        )
    frame = pd.read_csv(path, dtype={"quarter": str, "status": str, "notes": str})
    frame["notes"] = frame["notes"].fillna("")
    specs = frame_to_specs(frame)
    if validate:
        result = validate_vintage_manifest(specs, csv_path=path)
        result.raise_if_invalid()
    return specs


def load_vintage_manifest_by_name(
    name: str,
    *,
    validate: bool = True,
) -> list[VintageSpec]:
    """Load ``{name}.csv`` from the vintages package directory."""
    stem = str(name).strip()
    if stem.endswith(".csv"):
        stem = stem[:-4]
    path = vintages_dir() / f"{stem}.csv"
    return load_vintage_manifest(path, validate=validate)


@dataclass(frozen=True)
class VintageValidationResult:
    ok: bool
    n_quarters: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    mapping: tuple[tuple[str, int], ...]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise VintageManifestError("; ".join(self.errors))


def validate_vintage_manifest(
    specs: Optional[Sequence[VintageSpec]] = None,
    *,
    csv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    require_meta: bool = True,
    expected_start: str = START_QUARTER,
    expected_end: str = END_QUARTER,
    expected_n: int = EXPECTED_N_QUARTERS,
) -> VintageValidationResult:
    """Validate contiguous quarters, unique increasing origins, horizon, cutoffs."""
    errors: list[str] = []
    warnings: list[str] = []

    path = Path(csv_path) if csv_path is not None else default_manifest_csv_path()
    mpath = Path(meta_path) if meta_path is not None else default_manifest_meta_path()

    if specs is None:
        if not path.exists():
            return VintageValidationResult(
                False, 0, (f"CSV missing: {path}",), (), ()
            )
        frame = pd.read_csv(path, dtype={"quarter": str, "status": str, "notes": str})
        frame["notes"] = frame["notes"].fillna("")
        specs = frame_to_specs(frame)

    specs = list(specs)
    n = len(specs)
    mapping = tuple((s.quarter, int(s.forecast_origin)) for s in specs)

    if n != int(expected_n):
        errors.append(f"expected {expected_n} quarters, got {n}")

    expected_qrts = iter_shamsi_quarters(expected_start, expected_end)
    got_qrts = [s.quarter for s in specs]
    if got_qrts != expected_qrts:
        errors.append(
            "quarter sequence has gaps or unexpected labels: "
            f"got={got_qrts} expected={expected_qrts}"
        )

    origins = [int(s.forecast_origin) for s in specs]
    if len(set(origins)) != len(origins):
        errors.append("forecast_origin values are not unique")
    if origins != sorted(origins):
        errors.append("forecast_origin values do not strictly increase")
    for i in range(1, len(origins)):
        if not (origins[i] > origins[i - 1]):
            errors.append(
                f"origins not strictly increasing at index {i}: "
                f"{origins[i - 1]} -> {origins[i]}"
            )
            break

    for spec in specs:
        if int(spec.horizon) != DEFAULT_HORIZON:
            errors.append(
                f"{spec.quarter}: horizon={spec.horizon} != {DEFAULT_HORIZON}"
            )
        canonical = origin_from_quarter(spec.quarter)
        if int(spec.forecast_origin) != canonical:
            errors.append(
                f"{spec.quarter}: forecast_origin={spec.forecast_origin} != "
                f"origin_from_quarter={canonical}"
            )
        if int(spec.training_cutoff_exclusive) != int(spec.forecast_origin):
            errors.append(
                f"{spec.quarter}: training_cutoff_exclusive must equal "
                f"forecast_origin ({spec.forecast_origin})"
            )
        # Explicit training contract: cutoff is exclusive origin.
        if not (spec.training_cutoff_exclusive == spec.forecast_origin):
            errors.append(f"{spec.quarter}: training cutoff must be date < origin")
        if quarter_from_origin(spec.forecast_origin) != spec.quarter:
            errors.append(
                f"{spec.quarter}: quarter_from_origin("
                f"{spec.forecast_origin})="
                f"{quarter_from_origin(spec.forecast_origin)!r}"
            )
        if spec.status not in (
            "forecastable",
            "partially_matured",
            "fully_matured",
            "future_origin",
        ):
            errors.append(f"{spec.quarter}: invalid status {spec.status!r}")

    if require_meta:
        if not mpath.exists():
            errors.append(f"meta missing: {mpath}")
        elif path.exists():
            meta = json.loads(mpath.read_text(encoding="utf-8"))
            csv_hash = _sha256_bytes(path.read_bytes())
            if meta.get("csv_sha256") != csv_hash:
                errors.append("csv_sha256 mismatch vs meta.json")
            if meta.get("origin_mapping") is None:
                errors.append("meta missing origin_mapping description")
            stored = meta.get("quarter_to_forecast_origin") or {}
            for qrt, origin in mapping:
                if int(stored.get(qrt, -1)) != int(origin):
                    errors.append(
                        f"meta mapping mismatch for {qrt}: "
                        f"meta={stored.get(qrt)} csv={origin}"
                    )

    return VintageValidationResult(
        ok=not errors,
        n_quarters=n,
        errors=tuple(errors),
        warnings=tuple(warnings),
        mapping=mapping,
    )


def assert_training_cutoff_exclusive(specs: Sequence[VintageSpec]) -> None:
    """Assert every vintage trains strictly before its forecast origin."""
    for spec in specs:
        if int(spec.training_cutoff_exclusive) != int(spec.forecast_origin):
            raise VintageManifestError(
                f"{spec.quarter}: training_cutoff_exclusive "
                f"{spec.training_cutoff_exclusive} != origin {spec.forecast_origin}"
            )
        # Documented contract used by runners: date < forecast_origin
        assert spec.training_cutoff_exclusive == spec.forecast_origin


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Canonical TS backfill vintage manifest (1401Q1–1405Q2)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Write CSV+meta from origin_from_quarter")
    build_p.add_argument("--force", action="store_true")
    build_p.add_argument("--out-dir", type=Path, default=None)
    build_p.add_argument(
        "--as-of",
        type=int,
        default=None,
        help="Override last complete sales month (default: frozen sales)",
    )

    val_p = sub.add_parser("validate", help="Validate tracked vintage manifest")
    val_p.add_argument("--csv", type=Path, default=None)

    sub.add_parser("show", help="Print quarter → forecast_origin mapping")

    args = parser.parse_args(argv)

    if args.command == "build":
        as_of = (
            int(args.as_of)
            if args.as_of is not None
            else resolve_as_of_complete_month()
        )
        specs = build_vintage_specs(as_of_complete_month=as_of)
        csv_path, meta_path = write_vintage_manifest(
            specs, as_of_complete_month=as_of, out_dir=args.out_dir, force=args.force
        )
        result = validate_vintage_manifest(specs, csv_path=csv_path, meta_path=meta_path)
        print(f"wrote {csv_path}")
        print(f"wrote {meta_path}")
        print(f"n_quarters={result.n_quarters} as_of_complete_month={as_of}")
        for qrt, origin in result.mapping:
            print(f"  {qrt} -> {origin}")
        if not result.ok:
            print("VALIDATION FAILED:", "; ".join(result.errors))
            return 1
        return 0

    if args.command == "validate":
        result = validate_vintage_manifest(csv_path=args.csv)
        print(f"ok={result.ok} n_quarters={result.n_quarters}")
        for qrt, origin in result.mapping:
            print(f"  {qrt} -> {origin}")
        for w in result.warnings:
            print(f"warning: {w}")
        if not result.ok:
            print("ERRORS:", "; ".join(result.errors))
            return 1
        return 0

    if args.command == "show":
        specs = load_vintage_manifest(validate=True)
        print(f"n_quarters={len(specs)}")
        for spec in specs:
            print(
                f"  {spec.quarter} -> {spec.forecast_origin}  "
                f"[{spec.status}]  train date < {spec.training_cutoff_exclusive}"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
