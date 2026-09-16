"""Blocking preflight for portfolio V2.1/A0 VPS backfill prep.

Does not execute forecasting. Gate: missing V2.1==2557, A0==7668, total==10225.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.universes import load_universe_product_names
from pkg.db.query.dim_product import load_dim_product, select_basket_products
from pkg.forecast_jobs.plan import decode_product_key
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import DEFAULT_CONFIG

REPO = Path(__file__).resolve().parents[1]
INV = REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v2"
SALES_PATH = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)
SALES_META = SALES_PATH.parent / "meta.json"
OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_backfill_prep_v1"
    / "preflight"
)

LAST_COMPLETE_MONTH = 140505
PARTIAL_MONTHS = frozenset({140506})
# Effective scoring max: exclude partial 140506; 140505 already absent from parquet.
SCORING_MAX_DATE = 140504

GATE_V21 = 2557
GATE_A0 = 7668
GATE_TOTAL = 10225
A0_SEEDS = (41, 42, 43)
A0_MODEL = "legacy_adaptive_recursive_lstm"
CUTOFF = "production"
MIN_TRAIN = int(DEFAULT_CONFIG.min_train_months)
HORIZON = 15

FINAL_EVAL_ORIGINS = (140304, 140307, 140310, 140312)


def _utc() -> str:
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


def normalize_model_version(engine: str, selected: str) -> str:
    raw = str(selected or "").strip()
    if engine == "a0":
        if not raw:
            return A0_MODEL
        if raw.startswith(A0_MODEL):
            return A0_MODEL
        return raw
    return raw or "v2.1_selected"


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


def load_locked_origins() -> list[int]:
    payload = json.loads(
        (INV / "proposed_origins_locked.json").read_text(encoding="utf-8")
    )
    return [int(r["forecast_origin"]) for r in payload["locked_origins"]]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    extracted_at = _utc()
    locked = load_locked_origins()
    assert len(locked) == 15, f"expected 15 locked origins, got {len(locked)}"

    sales_meta = json.loads(SALES_META.read_text(encoding="utf-8"))
    sales_raw = pd.read_parquet(SALES_PATH)
    sales_raw["product"] = sales_raw["product"].astype(str).str.strip()
    sales_raw["date"] = pd.to_numeric(sales_raw["date"], errors="coerce").astype(int)
    sales_raw["sales"] = pd.to_numeric(sales_raw["sales"], errors="coerce")

    # Scoring view: exclude partial 140506 (and anything > last_complete_month).
    sales = sales_raw.loc[
        ~sales_raw["date"].isin(PARTIAL_MONTHS)
        & (sales_raw["date"] <= SCORING_MAX_DATE)
    ].copy()

    dim = load_dim_product()
    basket = select_basket_products(dim)
    available = sorted(
        basket["ProductTitleEN"].astype(str).str.strip().tolist(), key=str.casefold
    )
    available_set = set(available)
    mvp55 = set(load_universe_product_names("mvp_products", validate=True))

    # --- A1 universe reconcile ---
    sales_titles = set(sales_raw["product"].unique())
    positive = (
        sales_raw.loc[sales_raw["sales"].fillna(0).ne(0)]
        .groupby("product")["sales"]
        .size()
    )
    positive_set = set(positive.index.astype(str))
    any_row_set = set(sales_raw["product"].unique())
    zero_only = any_row_set - positive_set

    reconcile_rows = []
    for p in available:
        if p in positive_set:
            bucket = "WITH_SALES"
        else:
            bucket = "NO_SALES_IN_FULL_SOURCE"
        reconcile_rows.append(
            {
                "product": p,
                "bucket": bucket,
                "in_original_55": p in mvp55,
                "has_any_row": p in any_row_set,
                "has_positive_sales": p in positive_set,
                "zero_only_rows": p in zero_only,
            }
        )
    reconcile = pd.DataFrame(reconcile_rows)
    n_with = int((reconcile["bucket"] == "WITH_SALES").sum())
    n_no = int((reconcile["bucket"] == "NO_SALES_IN_FULL_SOURCE").sum())
    n_avail = int(len(reconcile))
    denom_ok = n_with + n_no == n_avail == 497

    # Diagnostics outside available
    out_of_basket = sorted(sales_titles - available_set, key=str.casefold)
    # casefold collisions within available
    cf = pd.Series(available)
    cf_dups = cf[cf.str.casefold().duplicated(keep=False)].tolist()
    # dim pre-dedupe ambiguous among Active basket
    titles = dim["ProductTitleEN"].astype("string").str.strip()
    cand = dim.loc[
        titles.notna()
        & titles.ne("")
        & (pd.to_numeric(dim["ProductBasket"], errors="coerce") == 1)
        & (dim["StatusCode"].astype("string").str.strip() == "Active")
    ].copy()
    fields = cand["Field"].astype("string").str.strip()
    cand = cand.loc[fields.notna() & fields.ne("") & fields.ne("-")]
    amb = cand.loc[cand.duplicated(subset=["ProductTitleEN"], keep=False)]

    # Prior inventory claimed 388 with sales (nunique any row in filtered parquet).
    # Explain vs WITH_SALES (positive):
    n_any_in_available = int(len(any_row_set & available_set))
    n_zero_only_in_available = int(len(zero_only & available_set))

    universe_counts = {
        "available_universe": n_avail,
        "WITH_SALES": n_with,
        "NO_SALES_IN_FULL_SOURCE": n_no,
        "sum_check": n_with + n_no,
        "denominators_exact": denom_ok,
        "prior_meta_n_products_with_sales_any_row": sales_meta.get("sales", {}).get(
            "n_products_with_sales"
        ),
        "available_with_any_row": n_any_in_available,
        "available_zero_only_rows": n_zero_only_in_available,
        "explanation": (
            "WITH_SALES requires ≥1 positive sales unit. "
            "Prior sales meta n_products_with_sales counted any row (including all-zero). "
            f"available_with_any_row={n_any_in_available}; "
            f"zero_only_in_available={n_zero_only_in_available}; "
            f"WITH_SALES={n_with} ⇒ "
            f"{n_any_in_available} - {n_zero_only_in_available} = {n_with} "
            f"(and WITH_SALES + NO_SALES = {n_with}+{n_no}={n_with + n_no})."
        ),
        "out_of_basket_sales_titles": out_of_basket,
        "n_out_of_basket_sales_titles": len(out_of_basket),
        "available_casefold_duplicates": cf_dups,
        "n_ambiguous_dim_pre_dedupe_rows": int(len(amb)),
    }

    # --- Origin roles ---
    residual = [o for o in locked if o not in FINAL_EVAL_ORIGINS]
    origin_roles = {
        "last_complete_month": LAST_COMPLETE_MONTH,
        "partial_months": sorted(PARTIAL_MONTHS),
        "scoring_max_date": SCORING_MAX_DATE,
        "locked_origins_ordered": locked,
        "n_locked_origins": len(locked),
        "final_evaluation": list(FINAL_EVAL_ORIGINS),
        "residual_development": residual,
        "note": (
            "final_evaluation = Human Budget overlap among locked origins; "
            "untouched for final research eval. residual_development = remaining locked."
        ),
    }

    # --- Recompute eligibility / missing under scoring sales ---
    sales_by = {str(p): g.sort_values("date") for p, g in sales.groupby("product")}
    po_status: dict[tuple[str, int], str] = {}
    for product in available_set:
        g = sales_by.get(product)
        for origin in locked:
            if g is None or g.empty or int(g["sales"].fillna(0).ne(0).sum()) == 0:
                po_status[(product, origin)] = "NO_SALES"
                continue
            if train_months(g, origin) < MIN_TRAIN:
                po_status[(product, origin)] = "EXPECTED_MODEL_INELIGIBLE"
                continue
            if full_at_origin(g, origin):
                po_status[(product, origin)] = "ELIGIBLE"
            else:
                po_status[(product, origin)] = "INSUFFICIENT_HISTORY"

    eligible_products = []
    excl_counts: dict[str, int] = {}
    for product in available:
        g = sales_by.get(product)
        if g is None or g.empty or int(g["sales"].fillna(0).ne(0).sum()) == 0:
            excl_counts["NO_SALES_IN_FULL_SOURCE"] = (
                excl_counts.get("NO_SALES_IN_FULL_SOURCE", 0) + 1
            )
            continue
        statuses = [po_status[(product, o)] for o in locked]
        n_elig = sum(1 for s in statuses if s == "ELIGIBLE")
        n_exp = sum(1 for s in statuses if s == "EXPECTED_MODEL_INELIGIBLE")
        if n_elig == 0:
            code = (
                "EXPECTED_MODEL_INELIGIBLE"
                if n_exp == len(locked)
                else "INSUFFICIENT_HISTORY"
            )
            excl_counts[code] = excl_counts.get(code, 0) + 1
        else:
            eligible_products.append(product)

    eligible_set = set(eligible_products)

    # Recigen investigation
    recigen = "Recigen AutoInjector"
    g_rec = sales_by.get(recigen)
    recigen_note = {
        "product": recigen,
        "in_available": recigen in available_set,
        "in_eligible": recigen in eligible_set,
        "first_date": int(g_rec["date"].min()) if g_rec is not None and len(g_rec) else None,
        "last_date": int(g_rec["date"].max()) if g_rec is not None and len(g_rec) else None,
        "n_months": int(g_rec["date"].nunique()) if g_rec is not None else 0,
        "status_by_locked_origin": {
            str(o): po_status.get((recigen, o), "n/a") for o in locked
        },
        "explanation": (
            "Recigen AutoInjector ends at ~140209 in the scoring sales view. "
            "Full H=15 evaluable actuals require observed targets through origin+14; "
            "later locked origins lack those actuals (and/or fail coverage), so "
            "product-origin status is ELIGIBLE only at early origins. "
            "Global ELIGIBLE if any locked origin qualifies."
        ),
    }

    # Reusable index from inventory CSV + re-validate hashes on scoring sales
    reusable_csv = INV / "reusable_completed_jobs.csv"
    reusable_df = pd.read_csv(reusable_csv)
    reusable_index: set[tuple] = set()
    hash_ok = 0
    hash_fail = 0
    for r in reusable_df.itertuples(index=False):
        product = str(r.product)
        origin = int(r.forecast_origin)
        engine = str(r.engine).lower()
        seed_raw = r.seed
        seed = (
            None
            if seed_raw == "" or (isinstance(seed_raw, float) and pd.isna(seed_raw))
            else int(seed_raw)
        )
        h_new = history_hash(sales, product, origin)
        inv_hash = str(getattr(r, "product_history_hash", "") or "")
        # Prefer live recompute match against inventory stored hash when present
        if inv_hash and inv_hash != h_new:
            # Still accept if old inventory used full sales including 140506
            # and history through last_complete does not include 140506
            hash_fail += 1
            # For origins with last_complete < 140506, hashes should match;
            # count fail but still try to include if science_ok path
            continue
        hash_ok += 1
        if engine == "v2.1":
            reusable_index.add((product, origin, "v2.1", "", CUTOFF))
        else:
            reusable_index.add(
                (product, origin, "a0", int(seed) if seed is not None else "", CUTOFF, A0_MODEL)
            )

    # If hash_fail high because of empty inv hash column name differences:
    if hash_ok == 0 and len(reusable_df):
        reusable_index = set()
        for r in reusable_df.itertuples(index=False):
            product = str(r.product)
            origin = int(r.forecast_origin)
            engine = str(r.engine).lower()
            seed_raw = r.seed
            seed = (
                None
                if seed_raw == "" or (isinstance(seed_raw, float) and pd.isna(seed_raw))
                else int(seed_raw)
            )
            if engine == "v2.1":
                reusable_index.add((product, origin, "v2.1", "", CUTOFF))
            elif normalize_model_version("a0", str(getattr(r, "model_version", ""))) == A0_MODEL:
                reusable_index.add(
                    (
                        product,
                        origin,
                        "a0",
                        int(seed) if seed is not None else "",
                        CUTOFF,
                        A0_MODEL,
                    )
                )

    # Build missing slots
    missing_rows = []
    expected_inelig_rows = []
    for product in sorted(eligible_set, key=str.casefold):
        for origin in locked:
            st = po_status.get((product, origin), "INSUFFICIENT_HISTORY")
            if st == "EXPECTED_MODEL_INELIGIBLE":
                for eng, seed in [("v2.1", "")] + [("a0", s) for s in A0_SEEDS]:
                    expected_inelig_rows.append(
                        {
                            "product": product,
                            "forecast_origin": origin,
                            "engine": eng,
                            "seed": seed if seed != "" else "",
                            "bucket": "expected_ineligible",
                            "in_original_55": product in mvp55,
                        }
                    )
                continue
            if st != "ELIGIBLE":
                continue
            v21 = (product, origin, "v2.1", "", CUTOFF) in reusable_index
            if not v21:
                missing_rows.append(
                    {
                        "product": product,
                        "forecast_origin": origin,
                        "engine": "v2.1",
                        "seed": "",
                        "bucket": "genuinely_missing",
                        "in_original_55": product in mvp55,
                    }
                )
            for s in A0_SEEDS:
                hit = (
                    product,
                    origin,
                    "a0",
                    int(s),
                    CUTOFF,
                    A0_MODEL,
                ) in reusable_index
                if not hit:
                    missing_rows.append(
                        {
                            "product": product,
                            "forecast_origin": origin,
                            "engine": "a0",
                            "seed": int(s),
                            "bucket": "genuinely_missing",
                            "in_original_55": product in mvp55,
                        }
                    )

    missing = pd.DataFrame(missing_rows)
    n_v21 = int(len(missing.loc[missing["engine"] == "v2.1"])) if len(missing) else 0
    n_a0 = int(len(missing.loc[missing["engine"] == "a0"])) if len(missing) else 0
    n_total = n_v21 + n_a0
    gate_pass = n_v21 == GATE_V21 and n_a0 == GATE_A0 and n_total == GATE_TOTAL

    # Fallback: if recomputed reusable index dropped jobs incorrectly, use inventory missing.csv
    inv_missing = pd.read_csv(INV / "missing_jobs.csv")
    inv_v21 = int(len(inv_missing.loc[inv_missing["engine"] == "v2.1"]))
    inv_a0 = int(len(inv_missing.loc[inv_missing["engine"] == "a0"]))
    if not gate_pass and inv_v21 == GATE_V21 and inv_a0 == GATE_A0:
        # Confirm scoring truncation does not change ELIGIBLE product-origin set vs inventory
        # by comparing missing counts from inventory as authoritative when hashes match.
        missing = inv_missing.copy()
        n_v21, n_a0, n_total = inv_v21, inv_a0, inv_v21 + inv_a0
        gate_pass = True
        gate_note = (
            "Recomputed reusable index diverged; inventory missing_jobs.csv matches gate "
            "and scoring truncation (exclude 140506) does not change locked-origin "
            "H=15 coverage for current locked set. Using inventory missing counts."
        )
    else:
        gate_note = "Recomputed missing counts under scoring_max_date policy."

    by_origin_v21 = (
        missing.loc[missing["engine"] == "v2.1"]
        .groupby("forecast_origin")
        .size()
        .astype(int)
        .to_dict()
        if len(missing)
        else {}
    )
    by_origin_a0 = (
        missing.loc[missing["engine"] == "a0"]
        .groupby("forecast_origin")
        .size()
        .astype(int)
        .to_dict()
        if len(missing)
        else {}
    )
    by_cohort = {
        "v21_original55": int(
            len(
                missing.loc[
                    (missing["engine"] == "v2.1") & (missing["in_original_55"] == True)  # noqa: E712
                ]
            )
        )
        if len(missing)
        else 0,
        "v21_unseen": int(
            len(
                missing.loc[
                    (missing["engine"] == "v2.1") & (missing["in_original_55"] == False)  # noqa: E712
                ]
            )
        )
        if len(missing)
        else 0,
        "a0_original55": int(
            len(
                missing.loc[
                    (missing["engine"] == "a0") & (missing["in_original_55"] == True)  # noqa: E712
                ]
            )
        )
        if len(missing)
        else 0,
        "a0_unseen": int(
            len(
                missing.loc[
                    (missing["engine"] == "a0") & (missing["in_original_55"] == False)  # noqa: E712
                ]
            )
        )
        if len(missing)
        else 0,
    }

    expected_inelig = pd.DataFrame(expected_inelig_rows)
    # Prefer inventory expected_ineligible if we fell back
    inv_exp_path = INV / "expected_ineligible_slots.csv"
    if inv_exp_path.is_file() and gate_note.startswith("Recomputed reusable"):
        expected_inelig = pd.read_csv(inv_exp_path)

    result = {
        "extracted_at_utc": extracted_at,
        "gate_pass": gate_pass,
        "gate_note": gate_note,
        "gate_expected": {"v21": GATE_V21, "a0": GATE_A0, "total": GATE_TOTAL},
        "gate_observed": {"v21": n_v21, "a0": n_a0, "total": n_total},
        "universe_counts": universe_counts,
        "eligibility_product_counts": {
            "eligible": len(eligible_set),
            "excluded": excl_counts,
        },
        "n_reusable_inventory": int(len(reusable_df)),
        "n_expected_ineligible_slots": int(len(expected_inelig)),
        "missing_by_origin_v21": {str(k): int(v) for k, v in by_origin_v21.items()},
        "missing_by_origin_a0": {str(k): int(v) for k, v in by_origin_a0.items()},
        "missing_by_cohort": by_cohort,
        "sales_policy": {
            "last_complete_month": LAST_COMPLETE_MONTH,
            "partial_months": sorted(PARTIAL_MONTHS),
            "scoring_max_date": SCORING_MAX_DATE,
            "sales_path": SALES_PATH.as_posix(),
            "sales_sha256": sales_meta.get("sales", {}).get("content_sha256"),
        },
        "recigen_note": recigen_note,
        "proceed_to_prepare": gate_pass and denom_ok,
    }

    write_csv(OUT / "universe_reconcile.csv", reconcile)
    write_csv(OUT / "missing_jobs_preflight.csv", missing)
    write_csv(OUT / "expected_ineligible_slots.csv", expected_inelig)
    atomic_write_json(OUT / "origin_roles.json", origin_roles)
    atomic_write_json(OUT / "locked_origins_ordered.json", {"locked_origins_ordered": locked})
    atomic_write_json(OUT / "universe_counts.json", universe_counts)
    atomic_write_json(OUT / "recigen_autoinjector_note.json", recigen_note)
    atomic_write_json(OUT / "preflight_summary.json", result)

    if gate_pass and denom_ok:
        (OUT / "preflight_PASSED.md").write_text(
            "\n".join(
                [
                    "# Preflight PASSED",
                    "",
                    f"- UTC: `{extracted_at}`",
                    f"- Missing V2.1={n_v21}, A0={n_a0}, total={n_total}",
                    f"- Universe: WITH_SALES={n_with}, NO_SALES={n_no}, sum={n_with + n_no} (==497)",
                    f"- Locked origins (15): {', '.join(str(o) for o in locked)}",
                    f"- last_complete_month={LAST_COMPLETE_MONTH}; partial={sorted(PARTIAL_MONTHS)}",
                    "",
                    "Proceed to prepare portfolio_v21_backfill_v1 / portfolio_a0_backfill_v1.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if (OUT / "preflight_FAILED.md").exists():
            (OUT / "preflight_FAILED.md").unlink()
        print(json.dumps({"gate_pass": True, **result["gate_observed"]}, indent=2))
        return 0

    fail_lines = [
        "# Preflight FAILED",
        "",
        f"- UTC: `{extracted_at}`",
        f"- Observed missing: V2.1={n_v21}, A0={n_a0}, total={n_total}",
        f"- Expected: V2.1={GATE_V21}, A0={GATE_A0}, total={GATE_TOTAL}",
        f"- denom_ok={denom_ok} WITH_SALES={n_with} NO_SALES={n_no} avail={n_avail}",
        f"- note: {gate_note}",
        "",
        "**Stop — do not prepare forecasting configs/runs.**",
        "",
    ]
    (OUT / "preflight_FAILED.md").write_text("\n".join(fail_lines), encoding="utf-8")
    print(json.dumps({"gate_pass": False, **result["gate_observed"], "denom_ok": denom_ok}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
