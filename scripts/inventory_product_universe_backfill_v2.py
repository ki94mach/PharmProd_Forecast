"""Full-portfolio product universe + V2.1/A0 inventory (v2).

Does not mutate backfill_inventory_v1 or the archval55 sales parquet.
Does not run forecasting. Stops after inventory + workload estimate.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.universes import load_universe_product_names
from pkg.db.query.dim_product import load_dim_product, select_basket_products
from pkg.forecast_jobs.plan import decode_product_key
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import DEFAULT_CONFIG

REPO = Path(__file__).resolve().parents[1]
OLD_SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
NEW_SALES_DIR = (
    REPO / "src" / "data" / "results" / "product_universe" / "sales_full_portfolio_v1"
)
NEW_SALES = NEW_SALES_DIR / "sales.parquet"
BUDGET_UNIVERSE = REPO / "src" / "data" / "benchmarks" / "v1" / "budget_universe.parquet"
VINTAGES_CSV = (
    REPO / "src" / "pkg" / "benchmark" / "vintages" / "ts_backfill_1401Q1_1405Q2.csv"
)
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
    REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v2"
)
DOCS_PATH = REPO / "docs" / "product_universe_backfill_inventory_v2.md"

SNAPSHOT_VERSION = "backfill_inventory_v2"
CUTOFF_CONTRACT = "production"
A0_SEEDS = (41, 42, 43)
A0_MODEL_VERSION = "legacy_adaptive_recursive_lstm"
HORIZON = 15
MIN_TRAIN = int(DEFAULT_CONFIG.min_train_months)
ARCHVAL_ORIGINS = (140104, 140201, 140301, 140402)
PINNED_ARCHVAL55_SHA = (
    "1d146d49261357f6b35d743a4c4064b22abe82047f6f5c36bbe4496d03f1ffc7"
)
SKIP_RUN_NAME_MARKERS = ("_bak", "_partial", "laptop_partial")

# VPS timing heuristics from archval55-scale runs (seconds per SUCCESS job).
# V2.1 classical ensemble ~45s; A0 adaptive LSTM ~180s on Windows VPS.
SEC_PER_V21 = 45.0
SEC_PER_A0 = 180.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def series_history_sha256(dates: Iterable[int], values: Iterable[float]) -> str:
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        if v != v:
            token = f"{int(d)}|\n"
        else:
            token = f"{int(d)}|{float(v):.10g}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def history_hash_through(
    sales: pd.DataFrame, product: str, origin: int
) -> tuple[str, int]:
    last_complete = shamsi_add_months(int(origin), -2)
    g = sales.loc[
        (sales["product"].astype(str) == str(product))
        & (pd.to_numeric(sales["date"], errors="coerce") <= last_complete)
    ].sort_values("date")
    h = series_history_sha256(
        [int(x) for x in g["date"].tolist()],
        [float(x) if pd.notna(x) else float("nan") for x in g["sales"].tolist()],
    )
    return h, int(len(g))


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


def normalize_model_version(engine: str, selected_model: str) -> str:
    raw = str(selected_model or "").strip()
    if engine == "a0":
        if not raw:
            return A0_MODEL_VERSION
        if raw.startswith(A0_MODEL_VERSION):
            return A0_MODEL_VERSION
        return raw
    return raw or "v2.1_selected"


def _basket_candidates(dim: pd.DataFrame) -> pd.DataFrame:
    work = dim.copy()
    titles = work["ProductTitleEN"].astype("string").str.strip()
    work = work.loc[titles.notna() & titles.ne("") & titles.ne("nan")].copy()
    work["ProductTitleEN"] = (
        work["ProductTitleEN"].astype("string").str.strip().astype(str)
    )
    work = work.loc[pd.to_numeric(work["ProductBasket"], errors="coerce") == 1].copy()
    fields = work["Field"].astype("string").str.strip()
    work = work.loc[fields.notna() & fields.ne("") & fields.ne("-")].copy()
    if "StatusCode" not in work.columns:
        return work.iloc[0:0].copy()
    status = work["StatusCode"].astype("string").str.strip()
    return work.loc[status == "Active"].copy()


def build_available_universe(dim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _basket_candidates(dim)
    ambiguous = candidates.loc[
        candidates.duplicated(subset=["ProductTitleEN"], keep=False)
    ].copy()
    basket = select_basket_products(dim)
    id_cols = [
        c
        for c in (
            "ProductTitleEN",
            "ID",
            "ID_INT",
            "Generic",
            "GenericEN",
            "BrandGroup",
            "ProductForm",
            "StatusCode",
            "ProductBasket",
            "FinanceCode",
        )
        if c in candidates.columns
    ]
    id_map = (
        candidates.sort_values("ProductTitleEN", key=lambda s: s.str.casefold())
        .drop_duplicates(subset=["ProductTitleEN"], keep="first")
        .loc[:, id_cols]
    )
    universe = basket.merge(id_map, on="ProductTitleEN", how="left")
    universe = universe.rename(columns={"ProductTitleEN": "product"})
    universe["product"] = universe["product"].astype(str)
    counts = candidates.groupby("ProductTitleEN").size().rename("n_dim_rows_same_title")
    universe = universe.merge(counts, left_on="product", right_index=True, how="left")
    universe["n_dim_rows_same_title"] = (
        pd.to_numeric(universe["n_dim_rows_same_title"], errors="coerce")
        .fillna(1)
        .astype(int)
    )
    universe["identity_ambiguous"] = universe["n_dim_rows_same_title"] > 1
    universe = universe.sort_values(
        "product", key=lambda s: s.str.casefold(), kind="mergesort"
    ).reset_index(drop=True)
    return universe, ambiguous


def calendar_h15_matured(
    origin: int,
    max_date: int,
    observed_dates: Optional[set[int]] = None,
) -> bool:
    """True when all 15 target months exist in the snapshot calendar.

    ``max_date`` alone is insufficient when an intermediate incomplete month
    (e.g. 140505) was dropped from the Parquet.
    """
    targets = [shamsi_add_months(int(origin), h) for h in range(HORIZON)]
    if targets[-1] > int(max_date):
        return False
    if observed_dates is not None:
        return all(t in observed_dates for t in targets)
    return True


def train_months_through(sales_g: pd.DataFrame, origin: int) -> int:
    last_complete = shamsi_add_months(int(origin), -2)
    return int((pd.to_numeric(sales_g["date"], errors="coerce") <= last_complete).sum())


def product_full_at_origin(sales_g: pd.DataFrame, origin: int) -> bool:
    series = pd.Series(
        pd.to_numeric(sales_g["sales"], errors="coerce").to_numpy(),
        index=pd.Index(
            [int(x) for x in sales_g["date"].tolist()], dtype="int64"
        ),
    )
    covers = discover_origins(
        series, config=DEFAULT_CONFIG, explicit_origins=(int(origin),)
    )
    if not covers:
        return False
    return bool(covers[0].full_horizon_coverage)


def lock_proposed_origins(
    *,
    max_date: int,
    observed_dates: set[int],
    hb_origins: Sequence[int],
    v21_hist_origins: Sequence[int],
    quarterly_origins: Sequence[int],
    target_min: int = 12,
    target_max: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministic selection; returns (locked, excluded_notes, lock_note)."""
    hb_set = set(int(x) for x in hb_origins)
    v21_set = set(int(x) for x in v21_hist_origins)
    q_set = set(int(x) for x in quarterly_origins)

    def is_quarterly(o: int) -> bool:
        return int(o) % 100 in {1, 4, 7, 10}

    excluded: list[dict[str, Any]] = []
    matured = []
    for o in sorted(hb_set | v21_set | q_set):
        if calendar_h15_matured(o, max_date, observed_dates):
            matured.append(o)
        else:
            missing_targets = [
                shamsi_add_months(o, h)
                for h in range(HORIZON)
                if shamsi_add_months(o, h) not in observed_dates
            ]
            excluded.append(
                {
                    "forecast_origin": o,
                    "reason": "horizons_not_fully_matured_in_full_snapshot",
                    "hb_overlap": o in hb_set,
                    "v21_historical": o in v21_set,
                    "required_max_target": shamsi_add_months(o, HORIZON - 1),
                    "snapshot_max_date": int(max_date),
                    "missing_target_months_sample": missing_targets[:5],
                }
            )

    selected: list[int] = []
    # 1) HB matured
    for o in sorted(hb_set):
        if o in matured and o not in selected:
            selected.append(o)
    # 2) historical V2.1 / archval
    for o in sorted(v21_set):
        if o in matured and o not in selected:
            selected.append(o)
    # 3) fill quarterly gaps for temporal coverage
    remaining_q = [o for o in sorted(matured) if is_quarterly(o) and o not in selected]
    while len(selected) < target_max and remaining_q:
        if not selected:
            selected.append(remaining_q.pop(0))
            continue
        # pick origin maximizing min-distance to existing set (spread)
        best = None
        best_score = -1
        for o in remaining_q:
            dist = min(abs(o - s) for s in selected)
            # prefer ends of range slightly
            span_bonus = 1 if o in (remaining_q[0], remaining_q[-1]) else 0
            score = dist * 10 + span_bonus
            if score > best_score:
                best_score = score
                best = o
        selected.append(best)  # type: ignore[arg-type]
        remaining_q.remove(best)

    if len(selected) > target_max:
        selected = selected[:target_max]

    locked_rows: list[dict[str, Any]] = []
    for o in selected:
        locked_rows.append(
            {
                "forecast_origin": int(o),
                "hb_overlap": o in hb_set,
                "v21_historical": o in v21_set,
                "quarterly_canonical": is_quarterly(o),
                "full_h15_matured": True,
                "selection_reason": (
                    "human_budget"
                    if o in hb_set
                    else (
                        "historical_v21_archval"
                        if o in v21_set
                        else "quarterly_temporal_fill"
                    )
                ),
            }
        )
    note = {
        "n_locked": len(locked_rows),
        "target_min": target_min,
        "target_max": target_max,
        "shortfall": max(0, target_min - len(locked_rows)),
    }
    return locked_rows, excluded, note


def iter_job_run_dirs() -> list[Path]:
    runs: list[Path] = []
    if FORECAST_JOBS_ROOT.is_dir():
        for d in sorted(FORECAST_JOBS_ROOT.iterdir()):
            if (
                d.is_dir()
                and not any(m in d.name.lower() for m in SKIP_RUN_NAME_MARKERS)
                and (d / "state.sqlite").is_file()
            ):
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


def main() -> int:
    extracted_at = _utc_now()
    if not NEW_SALES.is_file():
        raise FileNotFoundError(
            f"Missing full-portfolio sales at {NEW_SALES}. "
            "Run scripts/extract_sales_full_portfolio_v1.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sales_meta = json.loads((NEW_SALES_DIR / "meta.json").read_text(encoding="utf-8"))
    new_sales = pd.read_parquet(NEW_SALES)
    new_sales["product"] = new_sales["product"].astype(str).str.strip()
    new_sales["date"] = pd.to_numeric(new_sales["date"], errors="coerce").astype(int)
    new_sales["sales"] = pd.to_numeric(new_sales["sales"], errors="coerce")
    max_date = int(new_sales["date"].max())
    min_date = int(new_sales["date"].min())
    new_products = set(new_sales["product"].unique())

    old_sales = pd.read_parquet(OLD_SALES)
    old_sales["product"] = old_sales["product"].astype(str).str.strip()
    old_sales["date"] = pd.to_numeric(old_sales["date"], errors="coerce").astype(int)
    old_sales["sales"] = pd.to_numeric(old_sales["sales"], errors="coerce")
    old_products = set(old_sales["product"].unique())
    old_sha = file_sha256(OLD_SALES)
    new_sha = file_sha256(NEW_SALES)

    dim = load_dim_product()
    available, ambiguous_pre = build_available_universe(dim)
    available_set = set(available["product"].astype(str))
    mvp55 = set(load_universe_product_names("mvp_products", validate=True))

    # Human Budget + vintage + historical V2.1 origins
    hb_origins: list[int] = []
    if BUDGET_UNIVERSE.is_file():
        bud = pd.read_parquet(BUDGET_UNIVERSE, columns=["budget_origin"])
        hb_origins = sorted(int(x) for x in bud["budget_origin"].dropna().unique())
    quarterly: list[int] = []
    if VINTAGES_CSV.is_file():
        vdf = pd.read_csv(VINTAGES_CSV)
        quarterly = sorted(int(x) for x in vdf["forecast_origin"].tolist())

    # Scan jobs early for historical V2.1 origins
    v21_hist: set[int] = set(ARCHVAL_ORIGINS)
    for run_dir in iter_job_run_dirs():
        db = run_dir / "state.sqlite"
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT DISTINCT forecast_origin FROM jobs "
                "WHERE lower(engine_version)='v2.1'"
            ).fetchall()
            for (o,) in rows:
                v21_hist.add(int(o))
        finally:
            conn.close()

    locked_rows, excluded_origins, lock_note = lock_proposed_origins(
        max_date=max_date,
        observed_dates=set(int(x) for x in new_sales["date"].unique()),
        hb_origins=hb_origins,
        v21_hist_origins=sorted(v21_hist),
        quarterly_origins=quarterly,
    )
    # unpack fix — function returns 3 values
    locked_origins = [int(r["forecast_origin"]) for r in locked_rows]

    # --- Eligibility ---
    # Precompute per product×locked origin feasibility
    sales_by_product = {
        str(p): g.sort_values("date") for p, g in new_sales.groupby("product")
    }
    product_origin_status: dict[tuple[str, int], str] = {}
    # ELIGIBLE | EXPECTED_MODEL_INELIGIBLE | INSUFFICIENT (no full cover)
    for product in available_set:
        g = sales_by_product.get(product)
        for origin in locked_origins:
            if g is None or g.empty:
                product_origin_status[(product, origin)] = "NO_SALES"
                continue
            n_train = train_months_through(g, origin)
            if n_train < MIN_TRAIN:
                product_origin_status[(product, origin)] = "EXPECTED_MODEL_INELIGIBLE"
                continue
            if product_full_at_origin(g, origin):
                product_origin_status[(product, origin)] = "ELIGIBLE"
            else:
                # has min train but not full H=15 evaluable actuals
                product_origin_status[(product, origin)] = "INSUFFICIENT_HISTORY"

    elig_rows: list[dict[str, Any]] = []
    excl_rows: list[dict[str, Any]] = []
    for row in available.itertuples(index=False):
        product = str(row.product)
        in_old = product in old_products
        not_in_old_flag = not in_old
        detail: dict[str, Any] = {
            "product": product,
            "id_int": getattr(row, "ID_INT", None),
            "provider": getattr(row, "Provider", None),
            "field": getattr(row, "Field", None),
            "not_in_old_snapshot": not_in_old_flag,
            "in_original_55": product in mvp55,
        }
        if bool(getattr(row, "identity_ambiguous", False)):
            excl_rows.append(
                {**detail, "status": "IDENTITY_AMBIGUOUS", "reason_codes": "IDENTITY_AMBIGUOUS"}
            )
            continue

        g = sales_by_product.get(product)
        if g is None or g.empty or int(g["sales"].fillna(0).ne(0).sum()) == 0:
            excl_rows.append(
                {
                    **detail,
                    "status": "NO_SALES_IN_FULL_SOURCE",
                    "reason_codes": "NO_SALES_IN_FULL_SOURCE",
                }
            )
            continue

        statuses = [product_origin_status.get((product, o), "INSUFFICIENT_HISTORY") for o in locked_origins]
        n_elig_origins = sum(1 for s in statuses if s == "ELIGIBLE")
        n_expected_inelig = sum(1 for s in statuses if s == "EXPECTED_MODEL_INELIGIBLE")
        detail["n_locked_origins_eligible"] = n_elig_origins
        detail["n_locked_origins_expected_ineligible"] = n_expected_inelig
        detail["first_date"] = int(g["date"].min())
        detail["last_date"] = int(g["date"].max())
        detail["n_months"] = int(g["date"].nunique())

        if n_elig_origins == 0:
            # If every origin is expected-ineligible, use that code; else insufficient.
            if n_expected_inelig == len(locked_origins) and locked_origins:
                code = "EXPECTED_MODEL_INELIGIBLE"
            else:
                code = "INSUFFICIENT_HISTORY"
            excl_rows.append({**detail, "status": code, "reason_codes": code})
            continue

        elig_rows.append({**detail, "status": "ELIGIBLE", "reason_codes": "ELIGIBLE"})

    # NOT_CURRENTLY_AVAILABLE: in old cohort/sales but not in basket
    not_available_rows: list[dict[str, Any]] = []
    for product in sorted(mvp55 | old_products, key=str.casefold):
        if product in available_set:
            continue
        dim_hits = dim.loc[
            dim["ProductTitleEN"].astype(str).str.strip() == product
        ] if "ProductTitleEN" in dim.columns else dim.iloc[0:0]
        not_available_rows.append(
            {
                "product": product,
                "status": "NOT_CURRENTLY_AVAILABLE",
                "reason_codes": "NOT_CURRENTLY_AVAILABLE",
                "in_original_55": product in mvp55,
                "in_old_sales": product in old_products,
                "n_dim_rows": int(len(dim_hits)),
                "status_codes": (
                    ",".join(sorted(dim_hits["StatusCode"].astype(str).unique()))
                    if len(dim_hits) and "StatusCode" in dim_hits.columns
                    else ""
                ),
                "product_basket_values": (
                    ",".join(
                        sorted(
                            dim_hits["ProductBasket"].astype(str).unique()
                        )
                    )
                    if len(dim_hits) and "ProductBasket" in dim_hits.columns
                    else ""
                ),
                "field_values": (
                    ",".join(sorted(dim_hits["Field"].astype(str).unique()))
                    if len(dim_hits) and "Field" in dim_hits.columns
                    else ""
                ),
            }
        )

    eligible = pd.DataFrame(elig_rows)
    excluded = pd.DataFrame(excl_rows)
    not_available = pd.DataFrame(not_available_rows)
    eligible_set = set(eligible["product"].astype(str)) if not eligible.empty else set()

    # Enrich locked origins with n_eligible products
    for r in locked_rows:
        o = int(r["forecast_origin"])
        r["n_eligible_products_at_origin"] = sum(
            1
            for p in eligible_set
            if product_origin_status.get((p, o)) == "ELIGIBLE"
        )

    # --- History hash compare for overlapping products ---
    compare_origins = sorted(set(locked_origins) | set(ARCHVAL_ORIGINS) | set(v21_hist))
    overlap_products = sorted(
        (available_set | mvp55) & old_products & new_products, key=str.casefold
    )
    compare_rows: list[dict[str, Any]] = []
    hash_match_lookup: dict[tuple[str, int], bool] = {}
    new_hash_lookup: dict[tuple[str, int], str] = {}
    for product in overlap_products:
        for origin in compare_origins:
            h_old, n_old = history_hash_through(old_sales, product, origin)
            h_new, n_new = history_hash_through(new_sales, product, origin)
            new_hash_lookup[(product, origin)] = h_new
            if n_old == 0 and n_new == 0:
                status = "both_empty"
                match = True
            elif n_old == 0:
                status = "missing_in_old"
                match = False
            elif n_new == 0:
                status = "missing_in_new"
                match = False
            elif h_old == h_new:
                status = "match"
                match = True
            else:
                status = "mismatch"
                match = False
            hash_match_lookup[(product, origin)] = match
            compare_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "old_history_sha256": h_old,
                    "new_history_sha256": h_new,
                    "n_old": n_old,
                    "n_new": n_new,
                    "status": status,
                    "match": match,
                }
            )
    # New-only products: store hashes (no old baseline → cannot reuse old jobs)
    for product in sorted(new_products - old_products, key=str.casefold):
        for origin in locked_origins:
            h_new, n_new = history_hash_through(new_sales, product, origin)
            new_hash_lookup[(product, origin)] = h_new
            hash_match_lookup[(product, origin)] = False  # no old job reusable path
            compare_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "old_history_sha256": "",
                    "new_history_sha256": h_new,
                    "n_old": 0,
                    "n_new": n_new,
                    "status": "not_in_old_snapshot",
                    "match": False,
                }
            )
    history_compare = pd.DataFrame(compare_rows)

    # --- Job inventory ---
    job_rows: list[dict[str, Any]] = []
    for run_dir in iter_job_run_dirs():
        cutoff, run_sales_sha = load_run_cutoff_and_sales_sha(run_dir)
        conn = sqlite3.connect(str(run_dir / "state.sqlite"))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT engine_version, forecast_origin, product_id, status,
                       selected_model, output_path
                FROM jobs
                WHERE lower(engine_version) IN ('v2.1', 'a0')
                """
            ).fetchall()
        finally:
            conn.close()
        for r in rows:
            engine = str(r["engine_version"]).lower()
            status = str(r["status"]).upper()
            product_id = str(r["product_id"])
            if engine == "a0":
                product, seed = decode_product_key(product_id)
            else:
                product, seed = product_id, None
            model_version = normalize_model_version(engine, str(r["selected_model"] or ""))
            origin = int(r["forecast_origin"])
            hist = new_hash_lookup.get((str(product), origin), "")
            match = hash_match_lookup.get((str(product), origin))
            # Products only in old+new overlap can match; if product not in old,
            # match is False unless we never need reuse from old runs for new SKUs.
            if str(product) in old_products and str(product) in new_products:
                if match is None:
                    h_old, _ = history_hash_through(old_sales, str(product), origin)
                    h_new, _ = history_hash_through(new_sales, str(product), origin)
                    match = h_old == h_new
                    hist = h_new
                    new_hash_lookup[(str(product), origin)] = h_new
                    hash_match_lookup[(str(product), origin)] = match
            else:
                match = False

            science_ok = status == "SUCCESS" and cutoff == CUTOFF_CONTRACT
            if engine == "a0":
                science_ok = science_ok and seed in A0_SEEDS
                science_ok = science_ok and model_version == A0_MODEL_VERSION
            else:
                science_ok = science_ok and seed is None

            reusable = bool(science_ok and match)
            job_rows.append(
                {
                    "product": str(product),
                    "forecast_origin": origin,
                    "engine": engine,
                    "seed": "" if seed is None else int(seed),
                    "cutoff_contract": cutoff or "",
                    "model_version": model_version,
                    "product_history_hash": hist,
                    "history_hash_match_old_new": bool(match),
                    "status": status,
                    "run_id": run_dir.name,
                    "run_sales_sha256": run_sales_sha or "",
                    "science_ok": science_ok,
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
                            hist,
                        )
                    ),
                }
            )
    jobs = pd.DataFrame(job_rows)
    reusable = (
        jobs.loc[jobs["reusable"]].copy() if not jobs.empty else pd.DataFrame()
    )
    if not reusable.empty:
        reusable = reusable.sort_values(["run_id"]).drop_duplicates(
            subset=[
                "product",
                "forecast_origin",
                "engine",
                "seed",
                "cutoff_contract",
                "model_version",
                "product_history_hash",
            ],
            keep="last",
        )

    reusable_index: set[tuple] = set()
    if not reusable.empty:
        for r in reusable.itertuples(index=False):
            seed = None if r.seed == "" or (isinstance(r.seed, float) and pd.isna(r.seed)) else int(r.seed)
            if str(r.engine) == "v2.1":
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
                        A0_MODEL_VERSION,
                    )
                )

    # Required slots: ELIGIBLE product×origin only
    slot_rows: list[dict[str, Any]] = []
    for product in sorted(eligible_set, key=str.casefold):
        for origin in locked_origins:
            po_status = product_origin_status.get((product, origin), "INSUFFICIENT_HISTORY")
            hist = new_hash_lookup.get((product, origin), "")
            if not hist:
                hist, _ = history_hash_through(new_sales, product, origin)
                new_hash_lookup[(product, origin)] = hist

            base = {
                "product": product,
                "forecast_origin": origin,
                "product_origin_status": po_status,
                "product_history_hash": hist,
                "in_original_55": product in mvp55,
                "not_in_old_snapshot": product not in old_products,
            }
            if po_status == "EXPECTED_MODEL_INELIGIBLE":
                for engine, seed in [("v2.1", "")] + [("a0", s) for s in A0_SEEDS]:
                    slot_rows.append(
                        {
                            **base,
                            "engine": engine,
                            "seed": seed if seed != "" else "",
                            "bucket": "expected_ineligible",
                        }
                    )
                continue
            if po_status != "ELIGIBLE":
                continue

            v21_hit = (product, origin, "v2.1", "", CUTOFF_CONTRACT) in reusable_index
            a0_hits = [
                s
                for s in A0_SEEDS
                if (product, origin, "a0", int(s), CUTOFF_CONTRACT, A0_MODEL_VERSION)
                in reusable_index
            ]
            # v2.1
            slot_rows.append(
                {
                    **base,
                    "engine": "v2.1",
                    "seed": "",
                    "bucket": "reusable" if v21_hit else "genuinely_missing",
                }
            )
            for s in A0_SEEDS:
                hit = s in a0_hits
                slot_rows.append(
                    {
                        **base,
                        "engine": "a0",
                        "seed": int(s),
                        "bucket": "reusable" if hit else "genuinely_missing",
                    }
                )
            # partial coverage marker at product-origin level (added once)
            if v21_hit or a0_hits:
                if not (v21_hit and len(a0_hits) == 3):
                    slot_rows.append(
                        {
                            **base,
                            "engine": "slot",
                            "seed": "",
                            "bucket": "partial_coverage",
                            "v21_reusable": v21_hit,
                            "a0_seeds_reusable": ",".join(str(x) for x in a0_hits),
                        }
                    )

    slots = pd.DataFrame(slot_rows)
    missing = (
        slots.loc[slots["bucket"] == "genuinely_missing"].copy()
        if not slots.empty
        else pd.DataFrame()
    )
    expected_inelig_slots = (
        slots.loc[slots["bucket"] == "expected_ineligible"].copy()
        if not slots.empty
        else pd.DataFrame()
    )

    def _count_engine(df: pd.DataFrame, engine: str) -> dict[str, Any]:
        if df.empty:
            return {"n": 0, "by_origin": {}, "by_seed": {}}
        g = df.loc[df["engine"] == engine]
        return {
            "n": int(len(g)),
            "by_origin": {
                str(o): int(n) for o, n in g.groupby("forecast_origin").size().items()
            },
            "by_seed": {
                str(s): int(n) for s, n in g.groupby("seed").size().items()
            },
        }

    counts_missing_v21 = _count_engine(missing, "v2.1")
    counts_missing_a0 = _count_engine(missing, "a0")
    counts_reusable_v21 = _count_engine(
        reusable if not reusable.empty else pd.DataFrame(columns=["engine"]), "v2.1"
    )
    counts_reusable_a0 = _count_engine(
        reusable if not reusable.empty else pd.DataFrame(columns=["engine"]), "a0"
    )

    n_miss_v21 = counts_missing_v21["n"]
    n_miss_a0 = counts_missing_a0["n"]
    workload = {
        "assumptions": {
            "sec_per_v21_job": SEC_PER_V21,
            "sec_per_a0_job": SEC_PER_A0,
            "note": (
                "Heuristics from archval55-scale Windows VPS runs; "
                "not measured in this inventory."
            ),
        },
        "genuinely_missing_v21": n_miss_v21,
        "genuinely_missing_a0": n_miss_a0,
        "estimated_v21_hours": round(n_miss_v21 * SEC_PER_V21 / 3600.0, 2),
        "estimated_a0_hours": round(n_miss_a0 * SEC_PER_A0 / 3600.0, 2),
        "estimated_total_hours_serial": round(
            (n_miss_v21 * SEC_PER_V21 + n_miss_a0 * SEC_PER_A0) / 3600.0, 2
        ),
    }

    # Special investigations
    berak = []
    for title in ("Beraksurf 4", "Beraksurf 8"):
        hits = dim.loc[dim["ProductTitleEN"].astype(str).str.strip() == title]
        berak.append(
            {
                "product": title,
                "in_mvp55": title in mvp55,
                "in_old_sales": title in old_products,
                "in_new_sales": title in new_products,
                "in_select_basket": title in available_set,
                "classification": "NOT_CURRENTLY_AVAILABLE",
                "n_dim_rows": int(len(hits)),
                "rows": hits.loc[
                    :,
                    [
                        c
                        for c in (
                            "ID",
                            "ID_INT",
                            "ProductTitleEN",
                            "StatusCode",
                            "ProductBasket",
                            "Field",
                            "Provider",
                        )
                        if c in hits.columns
                    ],
                ].to_dict(orient="records")
                if len(hits)
                else [],
            }
        )

    new_skus = []
    for title in ("Recigen AutoInjector", "Reditux 100"):
        g = sales_by_product.get(title)
        origins_ok = [
            o
            for o in locked_origins
            if product_origin_status.get((title, o)) == "ELIGIBLE"
        ]
        new_skus.append(
            {
                "product": title,
                "in_mvp55": title in mvp55,
                "in_available": title in available_set,
                "in_eligible": title in eligible_set,
                "in_old_sales": title in old_products,
                "in_new_sales": title in new_products,
                "n_months": int(g["date"].nunique()) if g is not None else 0,
                "first_date": int(g["date"].min()) if g is not None and len(g) else None,
                "last_date": int(g["date"].max()) if g is not None and len(g) else None,
                "locked_origins_eligible": origins_ok,
                "n_reusable_jobs": (
                    int(len(reusable.loc[reusable["product"] == title]))
                    if not reusable.empty
                    else 0
                ),
            }
        )

    orig_in_available = sorted(available_set & mvp55, key=str.casefold)
    orig_in_eligible = sorted(eligible_set & mvp55, key=str.casefold)
    new_in_available = sorted(available_set - mvp55, key=str.casefold)
    new_in_eligible = sorted(eligible_set - mvp55, key=str.casefold)
    orig_missing_available = sorted(mvp55 - available_set, key=str.casefold)

    status_counts = {}
    if not excluded.empty:
        status_counts = excluded["status"].value_counts().astype(int).to_dict()
    status_counts["ELIGIBLE"] = int(len(eligible))
    status_counts["NOT_CURRENTLY_AVAILABLE"] = int(len(not_available))

    meta = {
        "snapshot_version": SNAPSHOT_VERSION,
        "extracted_at_utc": extracted_at,
        "immutable": True,
        "awaiting_approval_before_forecasting": True,
        "source": {
            "full_portfolio_sales": NEW_SALES.as_posix(),
            "full_portfolio_sales_sha256": new_sha,
            "archval55_sales": OLD_SALES.as_posix(),
            "archval55_sales_sha256": old_sha,
            "archval55_matches_pinned": old_sha.lower() == PINNED_ARCHVAL55_SHA.lower(),
            "sales_meta": sales_meta,
            "dim_product_interface": "pkg.db.query.dim_product",
            "cutoff_contract": CUTOFF_CONTRACT,
            "a0_seeds": list(A0_SEEDS),
            "a0_model_version": A0_MODEL_VERSION,
            "logical_job_key": (
                "product × origin × engine × seed × cutoff × "
                "model_version × product_history_hash"
            ),
            "job_sources_scanned": [p.as_posix() for p in iter_job_run_dirs()],
        },
        "sales_range": {"min_date": min_date, "max_date": max_date},
        "origin_lock_note": lock_note,
        "counts": {
            "available_universe": int(len(available)),
            "eligible": int(len(eligible)),
            "excluded_from_available": int(len(excluded)),
            "not_currently_available": int(len(not_available)),
            "not_in_old_snapshot_flag_on_available": int(
                (~available["product"].isin(old_products)).sum()
            ),
            "original_55_in_available": len(orig_in_available),
            "original_55_in_eligible": len(orig_in_eligible),
            "original_55_missing_from_available": len(orig_missing_available),
            "previously_unseen_in_available": len(new_in_available),
            "previously_unseen_in_eligible": len(new_in_eligible),
            "n_locked_origins": len(locked_origins),
            "n_reusable_jobs": int(len(reusable)) if not reusable.empty else 0,
            "n_genuinely_missing_v21": n_miss_v21,
            "n_genuinely_missing_a0": n_miss_a0,
            "n_expected_ineligible_slots": int(len(expected_inelig_slots)),
            "history_compare_match_rows": int(
                (history_compare["status"] == "match").sum()
            )
            if not history_compare.empty
            else 0,
            "history_compare_mismatch_rows": int(
                (history_compare["status"] == "mismatch").sum()
            )
            if not history_compare.empty
            else 0,
            "history_compare_not_in_old_snapshot_rows": int(
                (history_compare["status"] == "not_in_old_snapshot").sum()
            )
            if not history_compare.empty
            else 0,
        },
        "status_counts": status_counts,
        "missing_v21": counts_missing_v21,
        "missing_a0": counts_missing_a0,
        "reusable_v21": counts_reusable_v21,
        "reusable_a0": counts_reusable_a0,
        "workload_estimate": workload,
        "original_55_missing_from_available": orig_missing_available,
        "investigations": {"beraksurf": berak, "new_skus": new_skus},
    }

    # Persist
    write_csv(OUT_DIR / "universe_available.csv", available)
    write_csv(OUT_DIR / "universe_eligible.csv", eligible)
    write_csv(OUT_DIR / "universe_excluded.csv", excluded)
    write_csv(OUT_DIR / "universe_not_currently_available.csv", not_available)
    write_csv(OUT_DIR / "history_hash_compare.csv", history_compare)
    write_csv(OUT_DIR / "completed_jobs_scanned.csv", jobs)
    write_csv(OUT_DIR / "reusable_completed_jobs.csv", reusable)
    write_csv(OUT_DIR / "job_slots.csv", slots)
    write_csv(OUT_DIR / "missing_jobs.csv", missing)
    write_csv(OUT_DIR / "expected_ineligible_slots.csv", expected_inelig_slots)
    if not ambiguous_pre.empty:
        write_csv(OUT_DIR / "ambiguous_identity_pre_dedupe.csv", ambiguous_pre)

    atomic_write_json(
        OUT_DIR / "proposed_origins_locked.json",
        {
            "locked_origins": locked_rows,
            "excluded_candidates": excluded_origins,
            "human_budget_origins_inventory": hb_origins,
            "historical_v21_origins_inventory": sorted(v21_hist),
            "quarterly_vintage_origins": quarterly,
            "lock_note": lock_note,
            "snapshot_max_date": max_date,
        },
    )
    atomic_write_json(OUT_DIR / "meta.json", meta)
    atomic_write_json(
        OUT_DIR / "counts_original55_vs_unseen.json",
        {
            "original_55_in_available": orig_in_available,
            "original_55_in_eligible": orig_in_eligible,
            "original_55_missing_from_available": orig_missing_available,
            "previously_unseen_in_available": new_in_available,
            "previously_unseen_in_eligible": new_in_eligible,
            "counts": meta["counts"],
        },
    )
    atomic_write_json(OUT_DIR / "workload_estimate.json", workload)
    atomic_write_json(
        OUT_DIR / "investigations.json", meta["investigations"]
    )

    # Coverage matrix
    cov_rows = []
    for product in sorted(eligible_set, key=str.casefold):
        for origin in locked_origins:
            if product_origin_status.get((product, origin)) != "ELIGIBLE":
                continue
            v21 = (product, origin, "v2.1", "", CUTOFF_CONTRACT) in reusable_index
            a0_done = [
                s
                for s in A0_SEEDS
                if (product, origin, "a0", int(s), CUTOFF_CONTRACT, A0_MODEL_VERSION)
                in reusable_index
            ]
            cov_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "in_original_55": product in mvp55,
                    "v21_reusable": v21,
                    "a0_n_seeds_reusable": len(a0_done),
                    "a0_complete": len(a0_done) == 3,
                    "slot_complete": v21 and len(a0_done) == 3,
                }
            )
    write_csv(OUT_DIR / "coverage_matrix.csv", pd.DataFrame(cov_rows))

    # Reports
    hb_overlap = [r for r in locked_rows if r.get("hb_overlap")]
    report = [
        f"# Product universe backfill inventory (`{SNAPSHOT_VERSION}`)",
        "",
        f"- Extracted at (UTC): `{extracted_at}`",
        f"- Full-portfolio sales: `{NEW_SALES.as_posix()}` (sha `{new_sha[:16]}…`)",
        f"- Archval55 sales retained: `{OLD_SALES.as_posix()}` (match pinned: "
        f"**{old_sha.lower() == PINNED_ARCHVAL55_SHA.lower()}**)",
        f"- Sales months: `{min_date}` … `{max_date}`",
        "- **Awaiting approval before preparing or running any forecasting job.**",
        "",
        "## Counts",
        "",
        "| Metric | N |",
        "| --- | ---: |",
        f"| Available universe | {len(available)} |",
        f"| ELIGIBLE | {len(eligible)} |",
        f"| Excluded (from available) | {len(excluded)} |",
        f"| NOT_CURRENTLY_AVAILABLE | {len(not_available)} |",
        f"| Available but NOT_IN_OLD_SNAPSHOT (flag) | {meta['counts']['not_in_old_snapshot_flag_on_available']} |",
        f"| Original 55 in available / eligible | {len(orig_in_available)} / {len(orig_in_eligible)} |",
        f"| Previously unseen eligible | {len(new_in_eligible)} |",
        f"| Locked origins | {len(locked_origins)} |",
        f"| Reusable jobs | {meta['counts']['n_reusable_jobs']} |",
        f"| Genuinely missing V2.1 | {n_miss_v21} |",
        f"| Genuinely missing A0 | {n_miss_a0} |",
        "",
        "## Exclusion / status codes",
        "",
    ]
    for k, v in sorted(status_counts.items(), key=lambda x: (-x[1], x[0])):
        report.append(f"- `{k}`: {v}")
    report += [
        "",
        "## Locked proposed origins",
        "",
        "| Origin | Why selected | HB | V2.1 hist | N eligible SKUs |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in locked_rows:
        report.append(
            f"| {r['forecast_origin']} | {r['selection_reason']} | "
            f"{r['hb_overlap']} | {r['v21_historical']} | "
            f"{r['n_eligible_products_at_origin']} |"
        )
    report += [
        "",
        f"HB origins overlapping lock: "
        + (", ".join(str(r['forecast_origin']) for r in hb_overlap) or "_none_"),
        "",
        f"HB / other candidates excluded (not fully matured): {len(excluded_origins)}",
        "",
        "## Workload estimate (serial Windows VPS)",
        "",
        f"- Missing V2.1: **{n_miss_v21}** ≈ **{workload['estimated_v21_hours']} h** "
        f"(@ {SEC_PER_V21}s/job)",
        f"- Missing A0: **{n_miss_a0}** ≈ **{workload['estimated_a0_hours']} h** "
        f"(@ {SEC_PER_A0}s/job)",
        f"- Total serial: **{workload['estimated_total_hours_serial']} h**",
        "",
        "## Investigations",
        "",
        "### Beraksurf 4 / 8",
        "",
        json.dumps(berak, indent=2, ensure_ascii=False, default=str),
        "",
        "### Recigen AutoInjector / Reditux 100",
        "",
        json.dumps(new_skus, indent=2, ensure_ascii=False, default=str),
        "",
        "## Stop",
        "",
        "Inventory complete. **Do not prepare or run forecasting until approved.**",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    docs = [
        "# Full-portfolio sales + backfill inventory v2",
        "",
        "Operator summary. Artifacts are immutable under "
        "`src/data/results/product_universe/`.",
        "",
        "## Sales",
        "",
        f"- New: `sales_full_portfolio_v1/sales.parquet`",
        f"- Old archval55 snapshot **not** overwritten.",
        "",
        "## Inventory",
        "",
        f"- `backfill_inventory_v2/` — see `report.md`",
        f"- Locked origins ({len(locked_origins)}): "
        + ", ".join(str(o) for o in locked_origins),
        f"- ELIGIBLE products: {len(eligible)}",
        f"- Genuinely missing: V2.1={n_miss_v21}, A0={n_miss_a0}",
        f"- Estimated serial VPS hours: {workload['estimated_total_hours_serial']}",
        "",
        "**Awaiting approval before any forecasting job prep or run.**",
        "",
    ]
    DOCS_PATH.write_text("\n".join(docs), encoding="utf-8")

    print(json.dumps(meta["counts"], indent=2))
    print(json.dumps(workload, indent=2))
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
