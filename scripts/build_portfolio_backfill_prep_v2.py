"""Build portfolio_backfill_prep_v2: dense eligibility, reconciliation, configs.

Does not mutate prep_v1. Does not execute forecasting.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.universes import load_universe_product_names
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import DEFAULT_CONFIG
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window

REPO = Path(__file__).resolve().parents[1]
INV_V2 = REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v2"
SALES_SPARSE = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)
SALES_META = SALES_SPARSE.parent / "meta.json"
OUT = REPO / "src" / "data" / "results" / "product_universe" / "portfolio_backfill_prep_v2"
CFG_DIR = REPO / "configs" / "forecast_jobs"

SOURCE_LAST_COMPLETE_MONTH = 140505
PARTIAL_MONTH = 140506
A0_SEEDS = (41, 42, 43)
A0_MODEL = "legacy_adaptive_recursive_lstm"
CUTOFF = "production"
MIN_TRAIN = int(DEFAULT_CONFIG.min_train_months)
HORIZON = 15
ACTIVITY_THRESHOLD = float(DEFAULT_CONFIG.activity_start_min_sales or 5.0)

HB_EVAL = (140304, 140307, 140310, 140312)
POST_HB = (140401, 140402)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def series_history_sha256(dates: Iterable[int], values: Iterable[float]) -> str:
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        if v != v:
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


def load_locked_origins() -> list[int]:
    payload = json.loads(
        (INV_V2 / "proposed_origins_locked.json").read_text(encoding="utf-8")
    )
    return [int(r["forecast_origin"]) for r in payload["locked_origins"]]


def month_range(start: int, end: int) -> list[int]:
    out: list[int] = []
    cur = int(start)
    end_i = int(end)
    while cur <= end_i:
        out.append(cur)
        cur = shamsi_add_months(cur, 1)
    return out


def build_dense_sales(
    sparse: pd.DataFrame, *, through_date: int, activity_threshold: float
) -> pd.DataFrame:
    """Contiguous monthly grid per product from first activity through through_date."""
    rows: list[dict[str, Any]] = []
    for product, g in sparse.groupby("product", sort=False):
        g = g.sort_values("date")
        sales = pd.to_numeric(g["sales"], errors="coerce")
        dates = pd.to_numeric(g["date"], errors="coerce").astype(int)
        observed = pd.Series(sales.to_numpy(), index=pd.Index(dates.tolist(), dtype="int64"))
        observed = observed.groupby(level=0).sum()
        active = observed.loc[observed > float(activity_threshold)]
        if active.empty:
            # Keep positive-any activity fallback
            pos = observed.loc[observed.fillna(0).ne(0)]
            if pos.empty:
                continue
            first_active = int(pos.index.min())
        else:
            first_active = int(active.index.min())
        if first_active > through_date:
            continue
        grid = month_range(first_active, through_date)
        for ym in grid:
            val = float(observed[ym]) if ym in observed.index and pd.notna(observed[ym]) else 0.0
            is_miss = ym not in observed.index
            rows.append(
                {
                    "product": str(product),
                    "date": int(ym),
                    "sales": val,
                    "is_missing_month": bool(is_miss),
                }
            )
    return pd.DataFrame(rows)


def train_months(g: pd.DataFrame, origin: int) -> int:
    lc = shamsi_add_months(int(origin), -2)
    return int((pd.to_numeric(g["date"], errors="coerce") <= lc).sum())


def full_at_origin(g: pd.DataFrame, origin: int) -> bool:
    series = pd.Series(
        pd.to_numeric(g["sales"], errors="coerce").to_numpy(),
        index=pd.Index([int(x) for x in g["date"].tolist()], dtype="int64"),
    )
    covers = discover_origins(
        series, config=DEFAULT_CONFIG, explicit_origins=(int(origin),)
    )
    return bool(covers) and bool(covers[0].full_horizon_coverage)


def history_hash(sales: pd.DataFrame, product: str, origin: int) -> str:
    lc = shamsi_add_months(int(origin), -2)
    g = sales.loc[
        (sales["product"] == product)
        & (pd.to_numeric(sales["date"], errors="coerce") <= lc)
    ].sort_values("date")
    return series_history_sha256(
        [int(x) for x in g["date"].tolist()],
        [float(x) if pd.notna(x) else float("nan") for x in g["sales"].tolist()],
    )


def po_classify(g: Optional[pd.DataFrame], origin: int) -> str:
    if g is None or g.empty or int(g["sales"].fillna(0).ne(0).sum()) == 0:
        return "NO_SALES"
    if train_months(g, origin) < MIN_TRAIN:
        return "EXPECTED_MODEL_INELIGIBLE"
    if full_at_origin(g, origin):
        return "ELIGIBLE"
    return "INSUFFICIENT_HISTORY"


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    extracted_at = _utc()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = load_locked_origins()
    panel_max_target = max(shamsi_add_months(o, HORIZON - 1) for o in locked)

    sales_meta = json.loads(SALES_META.read_text(encoding="utf-8"))
    sparse = pd.read_parquet(SALES_SPARSE)
    sparse["product"] = sparse["product"].astype(str).str.strip()
    sparse["date"] = pd.to_numeric(sparse["date"], errors="coerce").astype(int)
    sparse["sales"] = pd.to_numeric(sparse["sales"], errors="coerce")
    # Exclude partial month from inputs
    sparse = sparse.loc[sparse["date"] != PARTIAL_MONTH].copy()

    dense = build_dense_sales(
        sparse, through_date=panel_max_target, activity_threshold=ACTIVITY_THRESHOLD
    )
    dense_path = OUT / "sales_dense_for_eligibility.parquet"
    dense.to_parquet(dense_path, index=False)

    # Recigen audit: prepare_monthly_series zero-fill through training_end
    recigen = "Recigen AutoInjector"
    recigen_audit: dict[str, Any] = {"product": recigen}
    g_sp = sparse.loc[sparse["product"] == recigen]
    g_dn = dense.loc[dense["product"] == recigen]
    recigen_audit["sparse_last_date"] = int(g_sp["date"].max()) if len(g_sp) else None
    recigen_audit["dense_last_date"] = int(g_dn["date"].max()) if len(g_dn) else None
    recigen_audit["dense_n_zero_fill_months"] = (
        int(g_dn["is_missing_month"].sum()) if len(g_dn) else 0
    )
    # prepare_monthly_series at a late origin
    sample_origin = 140402
    if len(g_sp):
        prepared = prepare_monthly_series(sparse, recigen, sample_origin)
        window = make_forecast_window(sample_origin, config=DEFAULT_CONFIG)
        recigen_audit["prepare_monthly_series"] = {
            "origin": sample_origin,
            "training_end": int(window.training_end),
            "n_observations": int(prepared.n_observations),
            "series_max_date": (
                int(prepared.values.index.max()) if prepared.n_observations else None
            ),
            "extends_through_training_end": (
                bool(prepared.n_observations)
                and int(prepared.values.index.max()) == int(window.training_end)
            ),
            "missing_month_policy": prepared.missing_month_policy,
            "conclusion": (
                "Absent fact months after activity are treated as zero demand "
                "(missing_month_policy=zero); series reindexes through origin-2."
            ),
        }

    eligible_csv = pd.read_csv(INV_V2 / "universe_eligible.csv")
    # Use same 363 eligible product list as cohort base (titles); reclassify origins on dense
    products = sorted(eligible_csv["product"].astype(str).tolist(), key=str.casefold)
    mvp55 = set(load_universe_product_names("mvp_products", validate=True))

    dense_by = {str(p): g.sort_values("date") for p, g in dense.groupby("product")}
    sparse_by = {str(p): g.sort_values("date") for p, g in sparse.groupby("product")}

    # Before/after Recigen origin status
    recigen_audit["status_by_origin_sparse"] = {
        str(o): po_classify(sparse_by.get(recigen), o) for o in locked
    }
    recigen_audit["status_by_origin_dense"] = {
        str(o): po_classify(dense_by.get(recigen), o) for o in locked
    }

    po_status: dict[tuple[str, int], str] = {}
    for product in products:
        for origin in locked:
            po_status[(product, origin)] = po_classify(dense_by.get(product), origin)

    # Reusable from inventory (history hash on dense)
    reusable = pd.read_csv(INV_V2 / "reusable_completed_jobs.csv")
    reusable_index: set[tuple] = set()
    hash_rows = []
    for r in reusable.itertuples(index=False):
        product = str(r.product)
        origin = int(r.forecast_origin)
        engine = str(r.engine).lower()
        seed_raw = r.seed
        seed = (
            None
            if seed_raw == "" or (isinstance(seed_raw, float) and pd.isna(seed_raw))
            else int(seed_raw)
        )
        h = history_hash(dense, product, origin) if product in dense_by else ""
        inv_h = str(getattr(r, "product_history_hash", "") or "")
        # Dense fill after last_complete may change hash if last_complete >= first missing
        # Compare hash through last_complete: zeros after last sparse date change hash
        # only when last_complete extends past last sparse observation.
        match = h == inv_h
        # For reuse: require dense history through last_complete to match sparse-derived
        # inventory hash when sparse series already covered last_complete; else recompute
        # match against sparse hash (import still valid if sparse through O-2 unchanged).
        h_sparse = history_hash(sparse, product, origin) if product in sparse_by else ""
        reuse_ok = h_sparse == inv_h if inv_h else True
        hash_rows.append(
            {
                "product": product,
                "forecast_origin": origin,
                "engine": engine,
                "seed": "" if seed is None else seed,
                "inventory_hash": inv_h,
                "dense_hash": h,
                "sparse_hash": h_sparse,
                "reuse_ok": reuse_ok,
            }
        )
        if not reuse_ok:
            continue
        if engine == "v2.1":
            reusable_index.add((product, origin, "v2.1", "", CUTOFF))
        else:
            reusable_index.add(
                (product, origin, "a0", int(seed) if seed is not None else "", CUTOFF, A0_MODEL)
            )

    write_csv(OUT / "reuse_hash_audit.csv", pd.DataFrame(hash_rows))

    # Build slots / missing / reconciliation
    slot_rows = []
    for product in products:
        for origin in locked:
            st = po_status[(product, origin)]
            for engine, seeds in (("v2.1", [None]), ("a0", list(A0_SEEDS))):
                for seed in seeds:
                    if st == "ELIGIBLE":
                        if engine == "v2.1":
                            hit = (product, origin, "v2.1", "", CUTOFF) in reusable_index
                        else:
                            hit = (
                                product,
                                origin,
                                "a0",
                                int(seed),
                                CUTOFF,
                                A0_MODEL,
                            ) in reusable_index
                        bucket = "SUCCESS_IMPORTED" if hit else "PENDING_MISSING"
                    elif st == "EXPECTED_MODEL_INELIGIBLE":
                        bucket = "SKIPPED_EXPECTED_MODEL_INELIGIBLE"
                    elif st == "INSUFFICIENT_HISTORY":
                        bucket = "SKIPPED_INSUFFICIENT_HISTORY"
                    elif st == "NO_SALES":
                        bucket = "SKIPPED_NO_SALES"
                    else:
                        bucket = "SKIPPED_OTHER"
                    slot_rows.append(
                        {
                            "product": product,
                            "forecast_origin": origin,
                            "engine": engine,
                            "seed": "" if seed is None else int(seed),
                            "product_origin_status": st,
                            "bucket": bucket,
                            "in_original_55": product in mvp55,
                        }
                    )

    slots = pd.DataFrame(slot_rows)
    write_csv(OUT / "cartesian_reconciliation_v2.csv", slots)

    def _bucket_counts(engine: str) -> dict[str, int]:
        sub = slots.loc[slots["engine"] == engine]
        return sub["bucket"].value_counts().astype(int).to_dict()

    n_products = len(products)
    n_origins = len(locked)
    grid_v21 = n_products * n_origins
    grid_a0 = n_products * n_origins * len(A0_SEEDS)
    counts_v21 = _bucket_counts("v2.1")
    counts_a0 = _bucket_counts("a0")
    sum_v21 = int(sum(counts_v21.values()))
    sum_a0 = int(sum(counts_a0.values()))

    missing = slots.loc[slots["bucket"] == "PENDING_MISSING"].copy()
    n_miss_v21 = int(len(missing.loc[missing["engine"] == "v2.1"]))
    n_miss_a0 = int(len(missing.loc[missing["engine"] == "a0"]))
    n_imp_v21 = int(counts_v21.get("SUCCESS_IMPORTED", 0))
    n_imp_a0 = int(counts_a0.get("SUCCESS_IMPORTED", 0))

    recon = {
        "n_eligible_products_cohort": n_products,
        "n_locked_origins": n_origins,
        "grid_v21": grid_v21,
        "grid_a0": grid_a0,
        "sum_v21_buckets": sum_v21,
        "sum_a0_buckets": sum_a0,
        "grid_v21_ok": sum_v21 == grid_v21,
        "grid_a0_ok": sum_a0 == grid_a0,
        "buckets_v21": counts_v21,
        "buckets_a0": counts_a0,
        "skipped_v21": int(sum(v for k, v in counts_v21.items() if k.startswith("SKIPPED"))),
        "skipped_a0": int(sum(v for k, v in counts_a0.items() if k.startswith("SKIPPED"))),
        "pending_v21": n_miss_v21,
        "pending_a0": n_miss_a0,
        "imported_v21": n_imp_v21,
        "imported_a0": n_imp_a0,
        "v1_comparison": {
            "v1_pending_v21": 2557,
            "v1_pending_a0": 7668,
            "v1_expected_ineligible_slots": 10068,
            "v1_allowlist_skip_success": 10708,
            "note": (
                "v2 SKIPPED_* replaces allowlist_skip SUCCESS; "
                "dense eligibility may change PENDING vs SKIPPED_INSUFFICIENT_HISTORY."
            ),
        },
    }
    atomic_write_json(OUT / "cartesian_reconciliation_v2.json", recon)

    # Origin roles
    residual = [o for o in locked if o < 140304]
    origin_roles = {
        "residual_development": residual,
        "human_budget_final_evaluation": list(HB_EVAL),
        "post_hb_ts_only_validation": list(POST_HB),
        "locked_origins_ordered": locked,
        "rules": {
            "never_use_post_hb_to_develop_for_earlier_hb_eval": True,
            "residual_training_for_origin_O": (
                "Only historical forecast rows with target_date <= O-2 may be used; "
                "an earlier forecast origin alone does not release all 15 residual targets."
            ),
        },
        "calendar": {
            "source_last_complete_month": SOURCE_LAST_COMPLETE_MONTH,
            "partial_month": PARTIAL_MONTH,
            "locked_panel_max_target_date": panel_max_target,
        },
    }
    atomic_write_json(OUT / "origin_roles_v2.json", origin_roles)
    atomic_write_json(OUT / "recigen_zero_fill_audit.json", recigen_audit)

    # Allowlists + configs
    jobs21 = [
        {
            "product": str(r.product),
            "forecast_origin": int(r.forecast_origin),
            "seed": None,
        }
        for r in missing.loc[missing["engine"] == "v2.1"].itertuples(index=False)
    ]
    jobs_a0 = [
        {
            "product": str(r.product),
            "forecast_origin": int(r.forecast_origin),
            "seed": int(r.seed),
        }
        for r in missing.loc[missing["engine"] == "a0"].itertuples(index=False)
    ]
    (CFG_DIR / "portfolio_v21_backfill_v2.allowlist.json").write_text(
        json.dumps({"n": len(jobs21), "jobs": jobs21}, indent=2) + "\n", encoding="utf-8"
    )
    (CFG_DIR / "portfolio_a0_backfill_v2.allowlist.json").write_text(
        json.dumps({"n": len(jobs_a0), "jobs": jobs_a0}, indent=2) + "\n", encoding="utf-8"
    )

    expected = {
        "content_sha256": sales_meta["sales"]["content_sha256"],
        "n_rows": sales_meta["sales"]["n_rows"],
        "size_bytes": sales_meta["sales"]["size_bytes"],
        "mtime_ns": None,
    }
    sales_block = {
        "path": "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet",
        "read_only": True,
        "schema": {
            "columns": ["product", "date", "sales"],
            "dtypes": {"product": "string", "date": "int64", "sales": "float64"},
        },
        "expected": expected,
    }

    def base_cfg(run_id, arches, seeds, origins, prods, workers):
        return {
            "schema_version": "1",
            "run": {"run_id": run_id, "output_dir": "src/data/forecast_jobs"},
            "cohort": {"products": prods},
            "origins": origins,
            "architectures": arches,
            "seeds": seeds,
            "horizon": 15,
            "cutoff_policy": {"name": "production"},
            "fallback": {"policy": "naive_last_value", "version": "v1"},
            "sales": sales_block,
            "execution": {
                "resume": True,
                "retry_failed": False,
                "force_job": False,
                "workers": workers,
                "pool": "processes",
            },
        }

    # Cohort = all products appearing in slots (eligible list)
    _dump_yaml(
        CFG_DIR / "portfolio_v21_backfill_v2.win.yaml",
        base_cfg("portfolio_v21_backfill_v2", ["v2.1"], [], locked, products, 8),
    )
    _dump_yaml(
        CFG_DIR / "portfolio_a0_backfill_v2.win.yaml",
        base_cfg("portfolio_a0_backfill_v2", ["a0"], [41, 42, 43], locked, products, 4),
    )

    smoke_products = ["Recigen AutoInjector", "Alvinotec 100", "Ablyze"]
    smoke_origins = [140101, 140104]
    smoke21 = [
        j
        for j in jobs21
        if j["product"] in smoke_products and j["forecast_origin"] in smoke_origins
    ]
    smoke_a0 = [
        j
        for j in jobs_a0
        if j["product"] in smoke_products
        and j["forecast_origin"] in smoke_origins
        and int(j["seed"]) == 41
    ]
    (CFG_DIR / "portfolio_v21_backfill_v2.smoke.allowlist.json").write_text(
        json.dumps({"n": len(smoke21), "jobs": smoke21}, indent=2) + "\n", encoding="utf-8"
    )
    (CFG_DIR / "portfolio_a0_backfill_v2.smoke.allowlist.json").write_text(
        json.dumps({"n": len(smoke_a0), "jobs": smoke_a0}, indent=2) + "\n", encoding="utf-8"
    )
    _dump_yaml(
        CFG_DIR / "portfolio_v21_backfill_v2.smoke.win.yaml",
        base_cfg(
            "portfolio_v21_backfill_v2_smoke",
            ["v2.1"],
            [],
            smoke_origins,
            smoke_products,
            2,
        ),
    )
    _dump_yaml(
        CFG_DIR / "portfolio_a0_backfill_v2.smoke.win.yaml",
        base_cfg(
            "portfolio_a0_backfill_v2_smoke",
            ["a0"],
            [41],
            smoke_origins,
            smoke_products,
            2,
        ),
    )

    write_csv(OUT / "missing_jobs_v2.csv", missing)
    atomic_write_json(
        OUT / "meta.json",
        {
            "snapshot_version": "portfolio_backfill_prep_v2",
            "extracted_at_utc": extracted_at,
            "calendar": origin_roles["calendar"],
            "zero_fill_conclusion": (
                "Absent monthly fact rows after commercial activity mean zero demand. "
                "prepare_monthly_series already zero-fills through origin-2; "
                "eligibility now uses dense sales through locked_panel_max_target_date."
            ),
            "reconciliation": recon,
            "recigen_audit": recigen_audit,
            "dense_sales_path": dense_path.as_posix(),
            "n_dense_rows": int(len(dense)),
        },
    )
    atomic_write_json(OUT / "recigen_zero_fill_audit.json", recigen_audit)

    print(json.dumps(recon, indent=2))
    print("recigen dense statuses", recigen_audit["status_by_origin_dense"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
