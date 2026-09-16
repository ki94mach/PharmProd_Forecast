"""Versioned product-universe snapshot + V2.1/A0 reusable-job inventory.

Uses the existing ``pkg.db.query.dim_product`` interface only (no replacement SQL).
Does not run forecasting.

Writes immutable artifacts under::

    src/data/results/product_universe/backfill_inventory_v1/
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.universes import load_universe_product_names
from pkg.db.query.dim_product import (
    load_dim_product,
    select_basket_products,
)
from pkg.forecast_jobs.plan import decode_product_key
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import DEFAULT_CONFIG

REPO = Path(__file__).resolve().parents[1]
SALES_PATH = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
FORECAST_JOBS_ROOT = REPO / "src" / "data" / "forecast_jobs"
ARCHVAL25_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval25_export"
)
OUT_DIR = (
    REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v1"
)

SNAPSHOT_VERSION = "backfill_inventory_v1"
CUTOFF_CONTRACT = "production"
A0_SEEDS = (41, 42, 43)
ENGINES = ("v2.1", "a0")
A0_MODEL_VERSION = "legacy_adaptive_recursive_lstm"
HORIZON = 15
MIN_TRAIN = int(DEFAULT_CONFIG.min_train_months)
# Classic Archval / production panel origins (also used for focused coverage).
PANEL_ORIGINS = (140104, 140201, 140301, 140402)

# Archval55 / reuse-gate pinned sales identity (byte-level whole file).
PINNED_SALES_SHA = (
    "1d146d49261357f6b35d743a4c4064b22abe82047f6f5c36bbe4496d03f1ffc7"
)

SKIP_RUN_NAME_MARKERS = ("_bak", "_partial", "laptop_partial")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def series_history_sha256(dates: Iterable[int], values: Iterable[float]) -> str:
    """Stable hash of (date, sales) pairs through a cutoff (inclusive)."""
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        # Canonical text form; NaN → empty token.
        if v != v:  # NaN
            token = f"{int(d)}|\n"
        else:
            token = f"{int(d)}|{float(v):.10g}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _basket_candidates(dim: pd.DataFrame) -> pd.DataFrame:
    """Same filters as ``select_basket_products`` but keep identity columns + pre-dedupe."""
    work = dim.copy()
    titles = work["ProductTitleEN"].astype("string").str.strip()
    work = work.loc[titles.notna() & titles.ne("") & titles.ne("nan")].copy()
    work["ProductTitleEN"] = (
        work["ProductTitleEN"].astype("string").str.strip().astype(str)
    )
    work = work.loc[
        pd.to_numeric(work["ProductBasket"], errors="coerce") == 1
    ].copy()
    fields = work["Field"].astype("string").str.strip()
    work = work.loc[fields.notna() & fields.ne("") & fields.ne("-")].copy()
    if "StatusCode" not in work.columns:
        return work.iloc[0:0].copy()
    status = work["StatusCode"].astype("string").str.strip()
    work = work.loc[status == "Active"].copy()
    return work


def build_available_universe(dim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (available_universe, ambiguous_pre_dedupe_rows)."""
    candidates = _basket_candidates(dim)
    amb_mask = candidates.duplicated(subset=["ProductTitleEN"], keep=False)
    ambiguous = candidates.loc[amb_mask].copy()

    basket = select_basket_products(dim)
    # Attach stable warehouse IDs via first matching Active basket candidate.
    id_map = (
        candidates.sort_values("ProductTitleEN", key=lambda s: s.str.casefold())
        .drop_duplicates(subset=["ProductTitleEN"], keep="first")
        .loc[:, ["ProductTitleEN", "ID", "ID_INT", "Generic", "GenericEN", "BrandGroup",
                 "ProductForm", "StatusCode", "ProductBasket", "FinanceCode"]]
    )
    universe = basket.merge(id_map, on="ProductTitleEN", how="left")
    universe = universe.rename(columns={"ProductTitleEN": "product"})
    universe["product"] = universe["product"].astype(str)
    universe["logical_product_key"] = universe["product"]
    counts = candidates.groupby("ProductTitleEN").size().rename("n_dim_rows_same_title")
    universe = universe.merge(
        counts, left_on="product", right_index=True, how="left"
    )
    universe["n_dim_rows_same_title"] = (
        pd.to_numeric(universe["n_dim_rows_same_title"], errors="coerce")
        .fillna(1)
        .astype(int)
    )
    universe["ambiguous_identity"] = universe["n_dim_rows_same_title"] > 1
    universe = universe.sort_values(
        "product", key=lambda s: s.str.casefold(), kind="mergesort"
    ).reset_index(drop=True)
    return universe, ambiguous


def reconcile_sales(
    universe: pd.DataFrame,
    sales: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[int],
    dict[str, set[int]],
]:
    """Return eligible, excluded, hashes, matured origins, per-product full origins."""
    sales = sales.copy()
    sales["product"] = sales["product"].astype(str).str.strip()
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype("Int64")
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")

    sales_products = set(sales["product"].unique())
    # Casefold collisions inside sales.
    cf = (
        pd.DataFrame({"product": sorted(sales_products)})
        .assign(cf=lambda d: d["product"].str.casefold())
    )
    sales_cf_dups = set(cf.loc[cf.duplicated("cf", keep=False), "product"])

    excluded_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    product_full_origins: dict[str, set[int]] = {}

    for row in universe.itertuples(index=False):
        product = str(row.product)
        reasons: list[str] = []
        detail: dict[str, Any] = {
            "product": product,
            "id_int": getattr(row, "ID_INT", None),
            "provider": getattr(row, "Provider", None),
            "field": getattr(row, "Field", None),
        }

        if bool(getattr(row, "ambiguous_identity", False)):
            reasons.append("AMBIGUOUS_IDENTITY")

        if product in sales_cf_dups:
            reasons.append("SALES_CASEFOLD_COLLISION")

        if product not in sales_products:
            soft = [p for p in sales_products if p.casefold() == product.casefold()]
            if soft:
                reasons.append("SALES_TITLE_CASE_MISMATCH")
                detail["sales_case_variants"] = ",".join(soft)
            else:
                reasons.append("NO_SALES")
            excluded_rows.append(
                {**detail, "reason_codes": "|".join(sorted(set(reasons)))}
            )
            continue

        g = sales.loc[sales["product"] == product].sort_values("date")
        if g.empty or int(g["sales"].fillna(0).ne(0).sum()) == 0:
            reasons.append("NO_POSITIVE_SALES")

        series = pd.Series(
            pd.to_numeric(g["sales"], errors="coerce").to_numpy(),
            index=pd.Index([int(x) for x in g["date"].tolist()], dtype="int64"),
        )
        covers = discover_origins(series, config=DEFAULT_CONFIG)
        full = [c for c in covers if c.full_horizon_coverage]
        product_full_origins[product] = {int(c.origin.shamsi_yyyymm) for c in full}

        detail["n_months_observed"] = int(g["date"].nunique())
        detail["n_positive_months"] = int(g["sales"].fillna(0).ne(0).sum())
        detail["first_date"] = int(g["date"].min()) if len(g) else None
        detail["last_date"] = int(g["date"].max()) if len(g) else None
        detail["n_full_coverage_origins"] = len(full)
        detail["full_coverage_origins"] = ",".join(
            str(o) for o in sorted(product_full_origins[product])
        )

        if not full:
            reasons.append("INSUFFICIENT_HISTORY")

        # Per-origin history hashes through production last_complete (O−2).
        for origin in sorted(product_full_origins[product]) or []:
            last_complete = shamsi_add_months(int(origin), -2)
            hist = g.loc[g["date"] <= last_complete].sort_values("date")
            h = series_history_sha256(
                [int(x) for x in hist["date"].tolist()],
                [float(x) if pd.notna(x) else float("nan") for x in hist["sales"].tolist()],
            )
            hash_rows.append(
                {
                    "product": product,
                    "forecast_origin": int(origin),
                    "last_complete": int(last_complete),
                    "n_history_rows": int(len(hist)),
                    "history_sha256": h,
                }
            )

        if reasons:
            excluded_rows.append({**detail, "reason_codes": "|".join(sorted(set(reasons)))})
        else:
            eligible_rows.append(detail)

    # Also hash every eligible/available product for archval origins even if not full,
    # so reuse checks can compare SKU-origin series when whole-file hash drifts.
    archval_origins = PANEL_ORIGINS
    hashed_keys = {(r["product"], r["forecast_origin"]) for r in hash_rows}
    for product in universe["product"].astype(str):
        if product not in sales_products:
            continue
        g = sales.loc[sales["product"] == product].sort_values("date")
        for origin in archval_origins:
            if (product, origin) in hashed_keys:
                continue
            last_complete = shamsi_add_months(int(origin), -2)
            hist = g.loc[g["date"] <= last_complete].sort_values("date")
            if hist.empty:
                continue
            h = series_history_sha256(
                [int(x) for x in hist["date"].tolist()],
                [float(x) if pd.notna(x) else float("nan") for x in hist["sales"].tolist()],
            )
            hash_rows.append(
                {
                    "product": product,
                    "forecast_origin": int(origin),
                    "last_complete": int(last_complete),
                    "n_history_rows": int(len(hist)),
                    "history_sha256": h,
                }
            )

    eligible = pd.DataFrame(eligible_rows)
    excluded = pd.DataFrame(excluded_rows)
    hashes = pd.DataFrame(hash_rows)

    # Fully matured feasible origins: every eligible product has full H coverage.
    if eligible.empty:
        matured: list[int] = []
    else:
        origin_sets = [
            product_full_origins.get(str(p), set())
            for p in eligible["product"].astype(str)
        ]
        inter = set.intersection(*origin_sets) if origin_sets else set()
        matured = sorted(inter)

    # Attach full-origin map onto eligible for coverage matrix.
    if not eligible.empty:
        eligible["full_coverage_origin_set"] = eligible["product"].map(
            lambda p: sorted(product_full_origins.get(str(p), set()))
        )

    return eligible, excluded, hashes, matured, product_full_origins


@dataclass(frozen=True)
class LogicalJobKey:
    product: str
    forecast_origin: int
    engine: str
    seed: Optional[int]
    cutoff_contract: str
    model_version: str

    def as_tuple(self) -> tuple:
        seed_s = "" if self.seed is None else int(self.seed)
        return (
            self.product,
            int(self.forecast_origin),
            self.engine,
            seed_s,
            self.cutoff_contract,
            self.model_version,
        )


def normalize_model_version(engine: str, selected_model: str, seed: Optional[int]) -> str:
    """Canonical model_version for the logical job key.

    A0 stores ``legacy_adaptive_recursive_lstm:seed41`` in selected_model; seed is
    already a separate key dimension, so strip the ``:seedN`` suffix.
    """
    raw = str(selected_model or "").strip()
    if engine == "a0":
        if not raw:
            return A0_MODEL_VERSION
        if raw.startswith(A0_MODEL_VERSION):
            return A0_MODEL_VERSION
        return raw
    if not raw:
        return "v2.1_selected"
    return raw


def _skip_run_dir(name: str) -> bool:
    lower = name.lower()
    return any(m in lower for m in SKIP_RUN_NAME_MARKERS)


def iter_job_run_dirs() -> list[Path]:
    runs: list[Path] = []
    if FORECAST_JOBS_ROOT.is_dir():
        for d in sorted(FORECAST_JOBS_ROOT.iterdir()):
            if d.is_dir() and not _skip_run_dir(d.name) and (d / "state.sqlite").is_file():
                runs.append(d)
    if (ARCHVAL25_EXPORT / "state.sqlite").is_file():
        runs.append(ARCHVAL25_EXPORT)
    return runs


def load_run_cutoff_and_sales_sha(run_dir: Path) -> tuple[str, Optional[str]]:
    man_path = run_dir / "manifest.json"
    snap_path = run_dir / "input_snapshot.json"
    cutoff = ""
    sales_sha: Optional[str] = None
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sci = man.get("scientific_config") or {}
        cutoff = str(
            (sci.get("cutoff_policy") or {}).get("name")
            or man.get("cutoff_policy")
            or ""
        ).strip().lower()
        snap = man.get("input_snapshot") or {}
        if snap.get("content_sha256"):
            sales_sha = str(snap["content_sha256"]).lower()
    if sales_sha is None and snap_path.is_file():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        if snap.get("content_sha256"):
            sales_sha = str(snap["content_sha256"]).lower()
    return cutoff, sales_sha


def inventory_completed_jobs(
    *,
    current_sales_sha: str,
    history_hash_lookup: dict[tuple[str, int], str],
) -> pd.DataFrame:
    """Collect SUCCESS v2.1 / a0 jobs with logical keys + reuse flags."""
    rows: list[dict[str, Any]] = []
    whole_file_match = current_sales_sha.lower() == PINNED_SALES_SHA.lower()

    for run_dir in iter_job_run_dirs():
        cutoff, run_sales_sha = load_run_cutoff_and_sales_sha(run_dir)
        db = run_dir / "state.sqlite"
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            job_rows = conn.execute(
                """
                SELECT engine_version, forecast_origin, product_id, status,
                       selected_model, output_path, error_type, error_message
                FROM jobs
                WHERE lower(engine_version) IN ('v2.1', 'a0')
                """
            ).fetchall()
        finally:
            conn.close()

        for r in job_rows:
            engine = str(r["engine_version"]).lower()
            status = str(r["status"]).upper()
            product_id = str(r["product_id"])
            if engine == "a0":
                product, seed = decode_product_key(product_id)
            else:
                product, seed = product_id, None
            selected = str(r["selected_model"] or "").strip()
            model_version = normalize_model_version(engine, selected, seed)

            origin = int(r["forecast_origin"])
            last_complete = shamsi_add_months(origin, -2)
            hist_key = (str(product), origin)
            current_hist = history_hash_lookup.get(hist_key)

            run_sha_match = (
                run_sales_sha is not None
                and run_sales_sha.lower() == current_sales_sha.lower()
            )
            pinned_run_match = (
                run_sales_sha is not None
                and run_sales_sha.lower() == PINNED_SALES_SHA.lower()
            )

            # Reuse science: SUCCESS + production cutoff + A0 seed policy.
            science_ok = status == "SUCCESS" and cutoff == CUTOFF_CONTRACT
            if engine == "a0":
                science_ok = science_ok and seed in A0_SEEDS
                # Prefer exact adaptive LSTM; still inventory fallback SUCCESS separately.
            if engine == "v2.1":
                science_ok = science_ok and seed is None

            sales_reuse_ok = False
            sales_reuse_mode = "blocked"
            if science_ok:
                if run_sha_match or (whole_file_match and pinned_run_match):
                    sales_reuse_ok = True
                    sales_reuse_mode = "whole_file_sha_match"
                elif current_hist is not None and run_sha_match is False:
                    # Without a stored per-SKU hash from the run, whole-file mismatch
                    # cannot prove byte-equivalence. Record current hash for future
                    # comparisons; do not auto-reuse on mismatch.
                    sales_reuse_ok = False
                    sales_reuse_mode = "needs_sku_history_compare_no_baseline"
                else:
                    sales_reuse_mode = "sales_sha_unknown_or_mismatch"

            reusable = bool(science_ok and sales_reuse_ok)
            # Fallback / wrong model still listed but not scientifically reusable
            # for A0 adaptive contract.
            if engine == "a0" and model_version != A0_MODEL_VERSION:
                reusable = False
                if science_ok and sales_reuse_ok:
                    sales_reuse_mode = "model_version_not_a0_contract"

            rows.append(
                {
                    "product": str(product),
                    "forecast_origin": origin,
                    "engine": engine,
                    "seed": "" if seed is None else int(seed),
                    "cutoff_contract": cutoff or "",
                    "model_version": model_version,
                    "status": status,
                    "run_id": run_dir.name,
                    "run_path": run_dir.as_posix(),
                    "run_sales_sha256": run_sales_sha or "",
                    "output_path": r["output_path"],
                    "last_complete": last_complete,
                    "current_history_sha256": current_hist or "",
                    "science_ok": science_ok,
                    "sales_reuse_mode": sales_reuse_mode,
                    "reusable": reusable,
                    "logical_key": "|".join(
                        str(x)
                        for x in (
                            product,
                            origin,
                            engine,
                            "" if seed is None else seed,
                            cutoff or "",
                            model_version,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def expand_required_keys(
    product_full_origins: dict[str, set[int]],
    eligible_products: list[str],
) -> list[LogicalJobKey]:
    """Required scientifically complete keys for eligible products at their matured origins."""
    keys: list[LogicalJobKey] = []
    for product in eligible_products:
        for origin in sorted(product_full_origins.get(product, set())):
            keys.append(
                LogicalJobKey(
                    product=product,
                    forecast_origin=int(origin),
                    engine="v2.1",
                    seed=None,
                    cutoff_contract=CUTOFF_CONTRACT,
                    model_version="*",  # any selected v2.1 model
                )
            )
            for seed in A0_SEEDS:
                keys.append(
                    LogicalJobKey(
                        product=product,
                        forecast_origin=int(origin),
                        engine="a0",
                        seed=int(seed),
                        cutoff_contract=CUTOFF_CONTRACT,
                        model_version=A0_MODEL_VERSION,
                    )
                )
    return keys


def main() -> int:
    extracted_at = _utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dim = load_dim_product()
    available, ambiguous_pre = build_available_universe(dim)
    sales = pd.read_parquet(SALES_PATH)
    sales_sha = file_sha256(SALES_PATH)
    sales_matches_archval55 = sales_sha.lower() == PINNED_SALES_SHA.lower()

    eligible, excluded, hashes, matured_origins, product_full_origins = reconcile_sales(
        available, sales
    )

    mvp55 = set(load_universe_product_names("mvp_products", validate=True))
    available_set = set(available["product"].astype(str))
    eligible_set = (
        set(eligible["product"].astype(str)) if not eligible.empty else set()
    )

    hash_lookup = {
        (str(r.product), int(r.forecast_origin)): str(r.history_sha256)
        for r in hashes.itertuples(index=False)
    } if not hashes.empty else {}

    jobs = inventory_completed_jobs(
        current_sales_sha=sales_sha,
        history_hash_lookup=hash_lookup,
    )

    reusable = (
        jobs.loc[jobs["reusable"]].copy()
        if not jobs.empty
        else pd.DataFrame()
    )
    # Deduplicate reusable keys: keep one row per logical key (prefer newest run_id).
    if not reusable.empty:
        reusable = reusable.sort_values(["run_id"]).drop_duplicates(
            subset=[
                "product",
                "forecast_origin",
                "engine",
                "seed",
                "cutoff_contract",
                "model_version",
            ],
            keep="last",
        )

    required = expand_required_keys(product_full_origins, sorted(eligible_set, key=str.casefold))

    reusable_index: set[tuple] = set()
    if not reusable.empty:
        for r in reusable.itertuples(index=False):
            seed = None if r.seed == "" or r.seed is None else int(r.seed)
            if str(r.engine) == "v2.1":
                # Wildcard model_version for v2.1
                reusable_index.add(
                    (str(r.product), int(r.forecast_origin), "v2.1", "", CUTOFF_CONTRACT)
                )
            else:
                reusable_index.add(
                    (
                        str(r.product),
                        int(r.forecast_origin),
                        "a0",
                        int(seed) if seed is not None else "",
                        CUTOFF_CONTRACT,
                        str(r.model_version),
                    )
                )

    missing_rows: list[dict[str, Any]] = []
    for key in required:
        if key.engine == "v2.1":
            hit = (
                key.product,
                key.forecast_origin,
                "v2.1",
                "",
                CUTOFF_CONTRACT,
            ) in reusable_index
        else:
            hit = (
                key.product,
                key.forecast_origin,
                "a0",
                int(key.seed) if key.seed is not None else "",
                CUTOFF_CONTRACT,
                A0_MODEL_VERSION,
            ) in reusable_index
        if not hit:
            missing_rows.append(
                {
                    "product": key.product,
                    "forecast_origin": key.forecast_origin,
                    "engine": key.engine,
                    "seed": "" if key.seed is None else int(key.seed),
                    "cutoff_contract": key.cutoff_contract,
                    "model_version": key.model_version,
                    "in_original_55": key.product in mvp55,
                }
            )
    missing = pd.DataFrame(missing_rows)

    # Coverage matrix: product × origin × engine → reusable seed coverage.
    cov_rows: list[dict[str, Any]] = []
    all_origins = sorted({o for s in product_full_origins.values() for o in s})
    for product in sorted(eligible_set, key=str.casefold):
        for origin in sorted(product_full_origins.get(product, set())):
            v21 = (
                product,
                origin,
                "v2.1",
                "",
                CUTOFF_CONTRACT,
            ) in reusable_index
            a0_seeds_done = [
                s
                for s in A0_SEEDS
                if (
                    product,
                    origin,
                    "a0",
                    int(s),
                    CUTOFF_CONTRACT,
                    A0_MODEL_VERSION,
                )
                in reusable_index
            ]
            cov_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "in_original_55": product in mvp55,
                    "v21_reusable": bool(v21),
                    "a0_seeds_reusable": ",".join(str(s) for s in a0_seeds_done),
                    "a0_n_seeds_reusable": len(a0_seeds_done),
                    "a0_complete": len(a0_seeds_done) == len(A0_SEEDS),
                    "slot_complete": bool(v21) and len(a0_seeds_done) == len(A0_SEEDS),
                }
            )
    coverage = pd.DataFrame(cov_rows)

    # Panel-origin slice (classic 4 origins): required/missing only where the SKU
    # itself has full H coverage at that origin.
    panel_full_origins = {
        p: {o for o in product_full_origins.get(p, set()) if o in PANEL_ORIGINS}
        for p in eligible_set
    }
    required_panel = expand_required_keys(
        panel_full_origins, sorted(eligible_set, key=str.casefold)
    )
    missing_panel_rows: list[dict[str, Any]] = []
    for key in required_panel:
        if key.engine == "v2.1":
            hit = (
                key.product,
                key.forecast_origin,
                "v2.1",
                "",
                CUTOFF_CONTRACT,
            ) in reusable_index
        else:
            hit = (
                key.product,
                key.forecast_origin,
                "a0",
                int(key.seed) if key.seed is not None else "",
                CUTOFF_CONTRACT,
                A0_MODEL_VERSION,
            ) in reusable_index
        if not hit:
            missing_panel_rows.append(
                {
                    "product": key.product,
                    "forecast_origin": key.forecast_origin,
                    "engine": key.engine,
                    "seed": "" if key.seed is None else int(key.seed),
                    "cutoff_contract": key.cutoff_contract,
                    "model_version": key.model_version,
                    "in_original_55": key.product in mvp55,
                }
            )
    missing_panel = pd.DataFrame(missing_panel_rows)

    # Per-origin eligible coverage (how many eligible SKUs have full H at origin).
    origin_coverage_summary: list[dict[str, Any]] = []
    all_origins_union = sorted({o for s in product_full_origins.values() for o in s})
    for origin in all_origins_union:
        n_cov = sum(1 for p in eligible_set if origin in product_full_origins.get(p, set()))
        origin_coverage_summary.append(
            {
                "forecast_origin": origin,
                "n_eligible_with_full_coverage": n_cov,
                "n_eligible": len(eligible_set),
                "fraction_eligible": round(n_cov / max(len(eligible_set), 1), 4),
                "is_panel_origin": origin in PANEL_ORIGINS,
                "is_fully_matured_for_all_eligible": n_cov == len(eligible_set) and len(eligible_set) > 0,
            }
        )

    # Fully matured among original-55 eligible subset.
    orig_eligible_list = sorted(eligible_set & mvp55, key=str.casefold)
    if orig_eligible_list:
        matured_orig55 = sorted(
            set.intersection(
                *[product_full_origins.get(p, set()) for p in orig_eligible_list]
            )
        )
    else:
        matured_orig55 = []

    # Counts: original 55 vs previously unseen
    n_available = int(len(available))
    n_eligible = int(len(eligible_set))
    n_excluded = int(len(excluded)) if not excluded.empty else 0
    orig_in_available = sorted(available_set & mvp55, key=str.casefold)
    orig_in_eligible = orig_eligible_list
    new_in_available = sorted(available_set - mvp55, key=str.casefold)
    new_in_eligible = sorted(eligible_set - mvp55, key=str.casefold)
    orig_missing_from_available = sorted(mvp55 - available_set, key=str.casefold)

    # Missing breakdown (all matured origins + panel slice)
    missing_summary: dict[str, Any] = {}
    if not missing.empty:
        for engine, g in missing.groupby("engine"):
            missing_summary[str(engine)] = {
                "n_missing": int(len(g)),
                "by_origin": {
                    str(o): int(n)
                    for o, n in g.groupby("forecast_origin").size().items()
                },
                "by_seed": {
                    str(s): int(n) for s, n in g.groupby("seed").size().items()
                },
            }
    missing_panel_summary: dict[str, Any] = {}
    if not missing_panel.empty:
        for engine, g in missing_panel.groupby("engine"):
            missing_panel_summary[str(engine)] = {
                "n_missing": int(len(g)),
                "by_origin": {
                    str(o): int(n)
                    for o, n in g.groupby("forecast_origin").size().items()
                },
                "by_seed": {
                    str(s): int(n) for s, n in g.groupby("seed").size().items()
                },
            }

    meta = {
        "snapshot_version": SNAPSHOT_VERSION,
        "extracted_at_utc": extracted_at,
        "immutable": True,
        "source": {
            "dim_product_interface": "pkg.db.query.dim_product.load_dim_product / select_basket_products",
            "dim_table": "[Iris_DW].[Dim].[Product]",
            "sales_path": SALES_PATH.as_posix(),
            "sales_content_sha256": sales_sha,
            "sales_matches_archval55_pinned": sales_matches_archval55,
            "archval55_pinned_sales_sha256": PINNED_SALES_SHA,
            "cutoff_contract": CUTOFF_CONTRACT,
            "horizon": HORIZON,
            "min_train_months": MIN_TRAIN,
            "a0_seeds": list(A0_SEEDS),
            "a0_model_version": A0_MODEL_VERSION,
            "panel_origins": list(PANEL_ORIGINS),
            "logical_job_key": (
                "product_id × forecast_origin × engine × seed × "
                "cutoff_contract × model_version"
            ),
            "job_sources_scanned": [p.as_posix() for p in iter_job_run_dirs()],
        },
        "counts": {
            "dim_product_rows": int(len(dim)),
            "available_universe": n_available,
            "eligible_backfill_universe": n_eligible,
            "excluded": n_excluded,
            "ambiguous_pre_dedupe_rows": int(len(ambiguous_pre)),
            "original_55_mvp": len(mvp55),
            "original_55_in_available": len(orig_in_available),
            "original_55_in_eligible": len(orig_in_eligible),
            "original_55_missing_from_available": len(orig_missing_from_available),
            "previously_unseen_in_available": len(new_in_available),
            "previously_unseen_in_eligible": len(new_in_eligible),
            "fully_matured_feasible_origins_all_eligible": matured_origins,
            "n_fully_matured_feasible_origins_all_eligible": len(matured_origins),
            "fully_matured_feasible_origins_original55_eligible": matured_orig55,
            "n_fully_matured_feasible_origins_original55_eligible": len(matured_orig55),
            "n_completed_v21_a0_job_rows_scanned": int(len(jobs)) if not jobs.empty else 0,
            "n_reusable_completed_jobs": int(len(reusable)) if not reusable.empty else 0,
            "n_required_job_keys_all_matured_origins": len(required),
            "n_missing_job_keys_all_matured_origins": int(len(missing)) if not missing.empty else 0,
            "n_required_job_keys_panel_origins": len(required_panel),
            "n_missing_job_keys_panel_origins": int(len(missing_panel)) if not missing_panel.empty else 0,
        },
        "original_55_missing_from_available": orig_missing_from_available,
        "missing_jobs_summary_all_matured_origins": missing_summary,
        "missing_jobs_summary_panel_origins": missing_panel_summary,
        "exclusion_reason_counts": (
            excluded["reason_codes"]
            .value_counts()
            .astype(int)
            .to_dict()
            if not excluded.empty and "reason_codes" in excluded.columns
            else {}
        ),
    }

    # Persist artifacts
    write_csv(OUT_DIR / "universe_available.csv", available)
    available.to_parquet(OUT_DIR / "universe_available.parquet", index=False)
    write_csv(
        OUT_DIR / "universe_eligible.csv",
        eligible if not eligible.empty else pd.DataFrame(columns=["product"]),
    )
    write_csv(
        OUT_DIR / "universe_excluded.csv",
        excluded if not excluded.empty else pd.DataFrame(columns=["product", "reason_codes"]),
    )
    write_csv(
        OUT_DIR / "product_origin_history_hashes.csv",
        hashes if not hashes.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "completed_jobs_scanned.csv",
        jobs if not jobs.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "reusable_completed_jobs.csv",
        reusable if not reusable.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "missing_jobs.csv",
        missing if not missing.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "missing_jobs_panel_origins.csv",
        missing_panel if not missing_panel.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "coverage_matrix.csv",
        coverage if not coverage.empty else pd.DataFrame(),
    )
    write_csv(
        OUT_DIR / "origin_coverage_summary.csv",
        pd.DataFrame(origin_coverage_summary),
    )
    if not ambiguous_pre.empty:
        write_csv(
            OUT_DIR / "ambiguous_identity_pre_dedupe.csv",
            ambiguous_pre.loc[
                :,
                [
                    c
                    for c in (
                        "ID",
                        "ID_INT",
                        "ProductTitleEN",
                        "Title",
                        "Provider",
                        "Field",
                        "StatusCode",
                    )
                    if c in ambiguous_pre.columns
                ],
            ],
        )

    atomic_write_json(OUT_DIR / "meta.json", meta)
    atomic_write_json(
        OUT_DIR / "fully_matured_feasible_origins.json",
        {
            "cutoff_contract": CUTOFF_CONTRACT,
            "horizon": HORIZON,
            "definition_all_eligible": (
                "Origins where every eligible-backfill product has "
                "discover_origins full_horizon_coverage under production cutoff."
            ),
            "origins_all_eligible": matured_origins,
            "n_eligible_products": n_eligible,
            "definition_original55_eligible": (
                "Origins where every original-55 product that is still eligible "
                "has full_horizon_coverage."
            ),
            "origins_original55_eligible": matured_orig55,
            "n_original55_eligible": len(orig_in_eligible),
            "panel_origins": list(PANEL_ORIGINS),
            "origin_coverage_summary": origin_coverage_summary,
        },
    )
    atomic_write_json(
        OUT_DIR / "counts_original55_vs_unseen.json",
        {
            "original_55_mvp_universe": sorted(mvp55, key=str.casefold),
            "original_55_in_available": orig_in_available,
            "original_55_in_eligible": orig_in_eligible,
            "original_55_missing_from_available": orig_missing_from_available,
            "previously_unseen_in_available": new_in_available,
            "previously_unseen_in_eligible": new_in_eligible,
            "counts": meta["counts"],
        },
    )

    # Human-readable report
    report_lines = [
        f"# Product universe + V2.1/A0 backfill inventory (`{SNAPSHOT_VERSION}`)",
        "",
        f"- Extracted at (UTC): `{extracted_at}`",
        f"- Dim.Product via `load_dim_product` / `select_basket_products` (no replacement SQL)",
        f"- Sales: `{SALES_PATH.as_posix()}`",
        f"- Sales SHA256: `{sales_sha}`",
        f"- Matches archval55 pinned sales: **{sales_matches_archval55}**",
        "",
        "## Counts",
        "",
        f"| Metric | N |",
        f"| --- | ---: |",
        f"| Available universe | {n_available} |",
        f"| Eligible backfill universe | {n_eligible} |",
        f"| Excluded | {n_excluded} |",
        f"| Original 55 (MVP) in available | {len(orig_in_available)} |",
        f"| Original 55 in eligible | {len(orig_in_eligible)} |",
        f"| Previously unseen in available | {len(new_in_available)} |",
        f"| Previously unseen in eligible | {len(new_in_eligible)} |",
        f"| Fully matured origins (all eligible) | {len(matured_origins)} |",
        f"| Fully matured origins (original-55 eligible) | {len(matured_orig55)} |",
        f"| Reusable completed jobs | {meta['counts']['n_reusable_completed_jobs']} |",
        f"| Required keys (all matured origins) | {len(required)} |",
        f"| Missing keys (all matured origins) | {meta['counts']['n_missing_job_keys_all_matured_origins']} |",
        f"| Required keys (panel origins) | {len(required_panel)} |",
        f"| Missing keys (panel origins) | {meta['counts']['n_missing_job_keys_panel_origins']} |",
        "",
        "## Fully matured feasible origins",
        "",
        f"- All eligible intersection: "
        + (
            ", ".join(str(o) for o in matured_origins)
            if matured_origins
            else "_none_"
        ),
        f"- Original-55 eligible intersection: "
        + (
            ", ".join(str(o) for o in matured_orig55)
            if matured_orig55
            else "_none_"
        ),
        "",
        "## Exclusion reason codes",
        "",
    ]
    for code, n in sorted(meta["exclusion_reason_counts"].items(), key=lambda x: (-x[1], x[0])):
        report_lines.append(f"- `{code}`: {n}")
    report_lines += [
        "",
        "## Missing jobs by engine (panel origins)",
        "",
    ]
    if missing_panel_summary:
        for eng, info in missing_panel_summary.items():
            report_lines.append(
                f"- **{eng}**: {info['n_missing']} missing "
                f"(by origin {info['by_origin']}; by seed {info['by_seed']})"
            )
    else:
        report_lines.append("- none")
    report_lines += [
        "",
        "## Artifacts",
        "",
        f"Directory: `{OUT_DIR.as_posix()}`",
        "",
        "- `universe_available.csv` / `.parquet`",
        "- `universe_eligible.csv`",
        "- `universe_excluded.csv`",
        "- `reusable_completed_jobs.csv`",
        "- `missing_jobs.csv` (all per-SKU matured origins)",
        "- `missing_jobs_panel_origins.csv`",
        "- `coverage_matrix.csv`",
        "- `origin_coverage_summary.csv`",
        "- `product_origin_history_hashes.csv`",
        "- `meta.json`",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(meta["counts"], indent=2))
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
