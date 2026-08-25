"""Immutable MVP product universe for historical V1/V2/V3 backfill experiments.

The forecasting MVP research universe is the set of distinct ``product`` values
in the frozen benchmark ``matched_universe.parquet`` (Analysis B matched
Human/TS panel), **not** today's ``ProductBasket``.

Logical product key: ``ProductTitleEN`` (column ``product``). No inventing of
warehouse IDs — ``product_id`` is left blank until a validated shared SKU key
exists (see ``docs/ts_v2_product_identity.md``).

CLI::

    python -m pkg.benchmark.universes validate
    python -m pkg.benchmark.universes build [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.benchmark.config import (
    EXCLUDED_ODD_COVERAGE_PRODUCTS,
    default_benchmark_root,
)
from pkg.benchmark.dataset import file_sha256, load_manifest
from pkg.db.query.constants import TARGET_GENERIC_EN

UNIVERSE_NAME = "mvp_products"
UNIVERSE_VERSION = "1"

MANIFEST_COLUMNS = (
    "product",
    "product_title",
    "product_id",
    "generic",
    "field",
    "product_form",
    "provider",
)


class MvpUniverseError(Exception):
    """Raised when the MVP universe manifest fails validation."""


def universes_dir() -> Path:
    return Path(__file__).resolve().parent


def mvp_products_csv_path() -> Path:
    return universes_dir() / f"{UNIVERSE_NAME}.csv"


def mvp_products_meta_path() -> Path:
    return universes_dir() / f"{UNIVERSE_NAME}.meta.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_product_list(products: list[str]) -> list[str]:
    return sorted({str(p).strip() for p in products if str(p).strip()}, key=str.casefold)


def products_list_sha256(products: list[str]) -> str:
    """Content hash of the sorted logical product list (newline-joined)."""
    return _sha256_text("\n".join(_canonical_product_list(products)) + "\n")


def derive_mvp_frame(
    matched_universe: pd.DataFrame,
    *,
    product_attrs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the MVP manifest frame from frozen matched panel rows.

    Does not query ``Dim.Product``. Does not invent ``product_id``.
    """
    if matched_universe is None or matched_universe.empty:
        raise MvpUniverseError("matched_universe is empty; cannot derive MVP products")
    if "product" not in matched_universe.columns:
        raise MvpUniverseError("matched_universe missing required column 'product'")

    work = matched_universe.copy()
    work["product"] = work["product"].astype(str).str.strip()
    work = work.loc[work["product"].ne("") & work["product"].ne("nan")].copy()

    agg: dict[str, Any] = {}
    for col, out in (
        ("generic", "generic"),
        ("Field", "field"),
        ("ProductForm", "product_form"),
        ("Provider", "provider"),
    ):
        if col in work.columns:
            agg[out] = (col, "first")

    if agg:
        frame = work.groupby("product", as_index=False, sort=False).agg(**agg)
    else:
        frame = work[["product"]].drop_duplicates().copy()

    if product_attrs is not None and not product_attrs.empty:
        attrs = product_attrs.copy()
        attrs["product"] = attrs["product"].astype(str).str.strip()
        rename = {
            "generic": "generic",
            "Field": "field",
            "ProductForm": "product_form",
            "Provider": "provider",
        }
        keep = ["product"] + [c for c in rename if c in attrs.columns]
        attrs = attrs[keep].drop_duplicates(subset=["product"], keep="first")
        attrs = attrs.rename(columns={k: v for k, v in rename.items() if k in attrs.columns})
        frame = frame.merge(attrs, on="product", how="left", suffixes=("", "_attr"))
        for col in ("generic", "field", "product_form", "provider"):
            attr_col = f"{col}_attr"
            if attr_col in frame.columns:
                if col in frame.columns:
                    frame[col] = frame[col].where(frame[col].notna(), frame[attr_col])
                else:
                    frame[col] = frame[attr_col]
                frame = frame.drop(columns=[attr_col])

    for col in MANIFEST_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    frame["product_title"] = frame["product"].astype(str)
    # No inventing IDs — freeze has ProductTitleEN only.
    frame["product_id"] = pd.NA
    frame = frame.loc[:, list(MANIFEST_COLUMNS)]
    frame = frame.sort_values("product", key=lambda s: s.str.casefold(), kind="mergesort")
    return frame.reset_index(drop=True)


def build_mvp_universe_from_freeze(
    benchmark_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive MVP products + metadata from the frozen v1 benchmark."""
    root = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
    matched_path = root / "matched_universe.parquet"
    if not matched_path.exists():
        raise FileNotFoundError(
            f"Frozen matched universe missing: {matched_path}. "
            "Run: python -m pkg.benchmark.freeze"
        )

    matched = pd.read_parquet(matched_path)
    attrs_path = root / "raw" / "product_attrs.parquet"
    attrs = pd.read_parquet(attrs_path) if attrs_path.exists() else None
    frame = derive_mvp_frame(matched, product_attrs=attrs)

    tracked = load_manifest()
    matched_checksum = file_sha256(matched_path)
    products = frame["product"].astype(str).tolist()
    attrs_only: list[str] = []
    if attrs is not None and not attrs.empty and "product" in attrs.columns:
        attrs_only = sorted(
            set(attrs["product"].astype(str)) - set(products),
            key=str.casefold,
        )
    meta: dict[str, Any] = {
        "universe_name": UNIVERSE_NAME,
        "universe_version": UNIVERSE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "Immutable MVP product list for historical V1/V2/V3 backfill experiments. "
            "Derived from frozen benchmark matched_universe (Analysis B matched "
            "Human/TS panel), not from today's ProductBasket."
        ),
        "source": {
            "benchmark_version": tracked.get("version", "v1"),
            "panel": "matched_universe.parquet",
            "panel_path_relative": "src/data/benchmarks/v1/matched_universe.parquet",
            "matched_universe_sha256": matched_checksum,
            "tracked_manifest_matched_universe_sha256": tracked.get("checksums", {}).get(
                "matched_universe.parquet"
            ),
            "tracked_manifest_matched_products": tracked.get("row_counts", {}).get(
                "matched_products"
            ),
            "product_attrs_used": bool(attrs is not None),
            "logical_product_key": "ProductTitleEN (column product)",
            "product_id_policy": (
                "left blank — no inventing; Dim.Product.ID_INT not present in freeze"
            ),
            "excluded_odd_coverage_products": sorted(EXCLUDED_ODD_COVERAGE_PRODUCTS),
            "target_generic_en_products_in_attrs_not_in_matched": attrs_only,
            "note": (
                "Freeze construction already applied TARGET_GENERIC_EN filter and "
                "EXCLUDED_ODD_COVERAGE_PRODUCTS before building matched_universe. "
                "This manifest freezes the resulting matched product set. "
                "Titles present in freeze product_attrs but absent from matched_universe "
                "are listed under target_generic_en_products_in_attrs_not_in_matched "
                "(they failed the Human/TS matched intersection, not the generic filter)."
            ),
        },
        "n_products": int(len(frame)),
        "products_sha256": products_list_sha256(products),
        "csv_sha256": None,
        "target_generic_en_comparison": compare_to_target_generic_en(frame),
    }
    return frame, meta


def compare_to_target_generic_en(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare MVP products/generics against ``TARGET_GENERIC_EN``."""
    target = list(TARGET_GENERIC_EN)
    target_set = set(target)
    generics = (
        sorted(frame["generic"].dropna().astype(str).unique(), key=str.casefold)
        if "generic" in frame.columns
        else []
    )
    generic_set = {g for g in generics if g}
    return {
        "n_target_generics": len(target),
        "n_mvp_products": int(len(frame)),
        "n_mvp_generics": len(generic_set),
        "target_generics_missing_from_mvp": sorted(target_set - generic_set, key=str.casefold),
        "mvp_generics_not_in_target": sorted(generic_set - target_set, key=str.casefold),
        "note": (
            "TARGET_GENERIC_EN is a generic (brand-group) filter used when building "
            "the freeze. MVP products are SKUs (ProductTitleEN) that survived the "
            "matched Human/TS intersection. A live Dim.Product query filtered by "
            "TARGET_GENERIC_EN can return additional titles not in this freeze."
        ),
    }


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Deterministic UTF-8 CSV bytes for hashing."""
    out = frame.loc[:, list(MANIFEST_COLUMNS)].copy()
    out["product_id"] = out["product_id"].fillna("").astype(str)
    for col in out.columns:
        if col != "product_id":
            out[col] = out[col].fillna("").astype(str)
    text = out.to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def write_mvp_universe(
    frame: pd.DataFrame,
    meta: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write CSV + meta.json. Refuses to overwrite unless ``force``."""
    directory = Path(out_dir) if out_dir is not None else universes_dir()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{UNIVERSE_NAME}.csv"
    meta_path = directory / f"{UNIVERSE_NAME}.meta.json"

    if csv_path.exists() and not force:
        raise MvpUniverseError(
            f"Refusing to overwrite immutable universe at {csv_path}. "
            "Pass force=True only when intentionally regenerating."
        )

    csv_bytes = dataframe_to_csv_bytes(frame)
    meta = dict(meta)
    meta["csv_sha256"] = _sha256_bytes(csv_bytes)
    meta["products_sha256"] = products_list_sha256(frame["product"].astype(str).tolist())
    meta["n_products"] = int(len(frame))
    meta["target_generic_en_comparison"] = compare_to_target_generic_en(frame)

    csv_path.write_bytes(csv_bytes)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, meta_path


def load_mvp_universe(
    csv_path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the tracked MVP product manifest (not today's basket)."""
    path = Path(csv_path) if csv_path is not None else mvp_products_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"MVP universe missing: {path}. "
            "Run: python -m pkg.benchmark.universes build"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if validate:
        validate_mvp_universe(frame, csv_path=path)
    return frame


def load_mvp_product_names(
    csv_path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> list[str]:
    """Sorted logical product titles for backfill runners."""
    frame = load_mvp_universe(csv_path, validate=validate)
    return frame["product"].astype(str).tolist()


@dataclass(frozen=True)
class MvpUniverseValidationResult:
    ok: bool
    n_products: int
    products_sha256: str
    csv_sha256: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise MvpUniverseError("; ".join(self.errors))


def validate_mvp_universe(
    frame: Optional[pd.DataFrame] = None,
    *,
    csv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    require_meta: bool = True,
    check_freeze_checksum: bool = False,
    benchmark_root: Optional[Path] = None,
) -> MvpUniverseValidationResult:
    """Validate immutability and integrity of the MVP universe manifest."""
    errors: list[str] = []
    warnings: list[str] = []

    path = Path(csv_path) if csv_path is not None else mvp_products_csv_path()
    mpath = Path(meta_path) if meta_path is not None else mvp_products_meta_path()

    if frame is None:
        if not path.exists():
            errors.append(f"CSV missing: {path}")
            return MvpUniverseValidationResult(
                False, 0, "", "", tuple(errors), tuple(warnings)
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")

    products = (
        frame["product"].astype(str).str.strip()
        if "product" in frame.columns
        else pd.Series(dtype=str)
    )
    n = int(len(frame))
    if n == 0:
        errors.append("universe is empty")

    if products.eq("").any():
        errors.append("blank product values present")

    if products.duplicated().any():
        dups = sorted(products[products.duplicated()].unique().tolist(), key=str.casefold)
        errors.append(f"duplicate logical products: {dups[:10]}")

    if "product_title" in frame.columns:
        titles = frame["product_title"].astype(str).str.strip()
        if titles.duplicated().any():
            dups = sorted(titles[titles.duplicated()].unique().tolist(), key=str.casefold)
            errors.append(f"duplicate product_title values: {dups[:10]}")
        mismatch = products != titles
        if mismatch.any():
            errors.append(
                f"product != product_title for {int(mismatch.sum())} rows "
                "(logical key must match title metadata)"
            )

    if "product_id" in frame.columns:
        ids = frame["product_id"].astype(str).str.strip()
        nonempty = ids.ne("")
        if nonempty.any():
            filled = ids[nonempty]
            if filled.duplicated().any():
                dups = sorted(filled[filled.duplicated()].unique().tolist(), key=str.casefold)
                errors.append(f"duplicate product_id values: {dups[:10]}")
        else:
            warnings.append(
                "product_id column is blank for all rows "
                "(no invented IDs; ProductTitleEN is the logical key)"
            )

    expected_order = products.sort_values(key=lambda s: s.str.casefold(), kind="mergesort")
    if not products.reset_index(drop=True).equals(expected_order.reset_index(drop=True)):
        errors.append("products are not sorted deterministically by casefold")

    csv_hash = ""
    products_hash = products_list_sha256(products.tolist()) if n else ""
    if path.exists():
        csv_hash = file_sha256(path)

    if require_meta:
        if not mpath.exists():
            errors.append(f"meta missing: {mpath}")
        else:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
            if meta.get("products_sha256") != products_hash:
                errors.append(
                    "products_sha256 mismatch vs meta.json — "
                    "manifest content changed (or meta is stale)"
                )
            if path.exists() and meta.get("csv_sha256") != csv_hash:
                errors.append(
                    "csv_sha256 mismatch vs meta.json — "
                    "CSV was modified after generation"
                )
            if int(meta.get("n_products", -1)) != n:
                errors.append(f"n_products meta={meta.get('n_products')} != csv={n}")
            source = meta.get("source") or {}
            if source.get("panel") != "matched_universe.parquet":
                errors.append(
                    "meta.source.panel must be matched_universe.parquet "
                    "(universe must not be derived from live ProductBasket)"
                )

    if check_freeze_checksum:
        root = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
        matched_path = root / "matched_universe.parquet"
        if matched_path.exists():
            live_products = _canonical_product_list(
                pd.read_parquet(matched_path, columns=["product"])["product"]
                .astype(str)
                .tolist()
            )
            if live_products != _canonical_product_list(products.tolist()):
                errors.append(
                    "CSV product list does not match current matched_universe.parquet "
                    "(freeze changed or manifest drifted)"
                )
            tracked = load_manifest()
            expected = tracked.get("checksums", {}).get("matched_universe.parquet")
            actual = file_sha256(matched_path)
            if expected and expected != actual:
                warnings.append(
                    "tracked v1_manifest matched_universe checksum differs from on-disk "
                    "parquet (freeze may have been rebuilt without updating tracked manifest)"
                )
        else:
            warnings.append(
                f"freeze panel missing; skipped freeze cross-check: {matched_path}"
            )

    return MvpUniverseValidationResult(
        ok=not errors,
        n_products=n,
        products_sha256=products_hash,
        csv_sha256=csv_hash,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def assert_mvp_universe_immutable(
    *,
    csv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> MvpUniverseValidationResult:
    """Hard check used by tests and future backfill runners."""
    result = validate_mvp_universe(
        csv_path=csv_path,
        meta_path=meta_path,
        require_meta=True,
        check_freeze_checksum=False,
    )
    result.raise_if_invalid()
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Immutable MVP product universe for historical backfill experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Derive CSV+meta from frozen matched_universe")
    build_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tracked universe (use with care)",
    )
    build_p.add_argument("--benchmark-root", type=Path, default=None)
    build_p.add_argument("--out-dir", type=Path, default=None)

    val_p = sub.add_parser("validate", help="Validate tracked MVP universe")
    val_p.add_argument(
        "--check-freeze",
        action="store_true",
        help="Also assert product list matches on-disk matched_universe.parquet",
    )
    val_p.add_argument("--benchmark-root", type=Path, default=None)

    sub.add_parser("show", help="Print product count and TARGET_GENERIC_EN diff")

    args = parser.parse_args(argv)

    if args.command == "build":
        frame, meta = build_mvp_universe_from_freeze(args.benchmark_root)
        csv_path, meta_path = write_mvp_universe(
            frame, meta, out_dir=args.out_dir, force=args.force
        )
        result = validate_mvp_universe(csv_path=csv_path, meta_path=meta_path)
        print(f"wrote {csv_path}")
        print(f"wrote {meta_path}")
        print(f"n_products={result.n_products}")
        print(f"products_sha256={result.products_sha256}")
        cmp_ = meta["target_generic_en_comparison"]
        print(
            "TARGET_GENERIC_EN: "
            f"missing_from_mvp={cmp_['target_generics_missing_from_mvp']} "
            f"extra_in_mvp={cmp_['mvp_generics_not_in_target']}"
        )
        if not result.ok:
            print("VALIDATION FAILED:", "; ".join(result.errors))
            return 1
        for w in result.warnings:
            print(f"warning: {w}")
        return 0

    if args.command == "validate":
        result = validate_mvp_universe(
            require_meta=True,
            check_freeze_checksum=bool(args.check_freeze),
            benchmark_root=args.benchmark_root,
        )
        print(f"ok={result.ok} n_products={result.n_products}")
        print(f"products_sha256={result.products_sha256}")
        print(f"csv_sha256={result.csv_sha256}")
        for w in result.warnings:
            print(f"warning: {w}")
        if not result.ok:
            print("ERRORS:", "; ".join(result.errors))
            return 1
        return 0

    if args.command == "show":
        frame = load_mvp_universe(validate=True)
        cmp_ = compare_to_target_generic_en(frame)
        print(f"n_products={len(frame)}")
        print(json.dumps(cmp_, indent=2))
        return 0

    return 1
