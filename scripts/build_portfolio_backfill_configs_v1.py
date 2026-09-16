"""Build portfolio V2.1/A0 allowlists and Windows YAML configs (no execute)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
INV = REPO / "src" / "data" / "results" / "product_universe" / "backfill_inventory_v2"
PRE = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_backfill_prep_v1"
    / "preflight"
)
SALES_META = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "meta.json"
)
CFG_DIR = REPO / "configs" / "forecast_jobs"

SALES_PATH = "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet"
SMOKE_PRODUCTS = ["Recigen AutoInjector", "Alvinotec 100", "Ablyze"]
SMOKE_ORIGINS = [140101, 140104]


def _load_origins() -> list[int]:
    return list(
        json.loads((PRE / "locked_origins_ordered.json").read_text(encoding="utf-8"))[
            "locked_origins_ordered"
        ]
    )


def _sales_expected() -> dict[str, Any]:
    meta = json.loads(SALES_META.read_text(encoding="utf-8"))["sales"]
    return {
        "content_sha256": meta["content_sha256"],
        "n_rows": meta["n_rows"],
        "size_bytes": meta["size_bytes"],
        "mtime_ns": None,
    }


def _write_allowlist(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n": len(jobs), "jobs": jobs}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _base_config(
    *,
    run_id: str,
    architectures: list[str],
    seeds: list[int],
    origins: list[int],
    products: list[str],
    workers: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run": {"run_id": run_id, "output_dir": "src/data/forecast_jobs"},
        "cohort": {"products": products},
        "origins": origins,
        "architectures": architectures,
        "seeds": seeds,
        "horizon": 15,
        "cutoff_policy": {"name": "production"},
        "fallback": {"policy": "naive_last_value", "version": "v1"},
        "sales": {
            "path": SALES_PATH,
            "read_only": True,
            "schema": {
                "columns": ["product", "date", "sales"],
                "dtypes": {"product": "string", "date": "int64", "sales": "float64"},
            },
            "expected": _sales_expected(),
        },
        "execution": {
            "resume": True,
            "retry_failed": False,
            "force_job": False,
            "workers": workers,
            "pool": "processes",
        },
    }


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    locked = _load_origins()
    missing = pd.read_csv(INV / "missing_jobs.csv")
    reusable = pd.read_csv(INV / "reusable_completed_jobs.csv")

    # V2.1 allowlist
    m21 = missing.loc[missing["engine"] == "v2.1"]
    jobs21 = [
        {
            "product": str(r["product"]),
            "forecast_origin": int(r["forecast_origin"]),
            "seed": None,
        }
        for _, r in m21.iterrows()
    ]
    _write_allowlist(CFG_DIR / "portfolio_v21_backfill_v1.allowlist.json", jobs21)

    # A0 allowlist
    ma0 = missing.loc[missing["engine"] == "a0"]
    jobs_a0 = [
        {
            "product": str(r["product"]),
            "forecast_origin": int(r["forecast_origin"]),
            "seed": int(r["seed"]),
        }
        for _, r in ma0.iterrows()
    ]
    _write_allowlist(CFG_DIR / "portfolio_a0_backfill_v1.allowlist.json", jobs_a0)

    # Cohort products = missing ∪ reusable for engine
    r21 = reusable.loc[reusable["engine"] == "v2.1", "product"].astype(str)
    ra0 = reusable.loc[reusable["engine"] == "a0", "product"].astype(str)
    products21 = sorted(
        set(m21["product"].astype(str)) | set(r21), key=str.casefold
    )
    products_a0 = sorted(
        set(ma0["product"].astype(str)) | set(ra0), key=str.casefold
    )

    _dump_yaml(
        CFG_DIR / "portfolio_v21_backfill_v1.win.yaml",
        _base_config(
            run_id="portfolio_v21_backfill_v1",
            architectures=["v2.1"],
            seeds=[],
            origins=locked,
            products=products21,
            workers=8,
        ),
    )
    _dump_yaml(
        CFG_DIR / "portfolio_a0_backfill_v1.win.yaml",
        _base_config(
            run_id="portfolio_a0_backfill_v1",
            architectures=["a0"],
            seeds=[41, 42, 43],
            origins=locked,
            products=products_a0,
            workers=4,
        ),
    )

    # Smoke: previously unseen SKUs at early origins
    smoke21_products = list(SMOKE_PRODUCTS)
    _dump_yaml(
        CFG_DIR / "portfolio_v21_backfill_v1.smoke.win.yaml",
        _base_config(
            run_id="portfolio_v21_backfill_v1_smoke",
            architectures=["v2.1"],
            seeds=[],
            origins=SMOKE_ORIGINS,
            products=smoke21_products,
            workers=2,
        ),
    )
    _dump_yaml(
        CFG_DIR / "portfolio_a0_backfill_v1.smoke.win.yaml",
        _base_config(
            run_id="portfolio_a0_backfill_v1_smoke",
            architectures=["a0"],
            seeds=[41],
            origins=SMOKE_ORIGINS,
            products=smoke21_products,
            workers=2,
        ),
    )

    # Smoke allowlists = intersection of smoke grid with missing
    smoke21_jobs = [
        j
        for j in jobs21
        if j["product"] in smoke21_products and int(j["forecast_origin"]) in SMOKE_ORIGINS
    ]
    smoke_a0_jobs = [
        j
        for j in jobs_a0
        if j["product"] in smoke21_products
        and int(j["forecast_origin"]) in SMOKE_ORIGINS
        and int(j["seed"]) == 41
    ]
    _write_allowlist(
        CFG_DIR / "portfolio_v21_backfill_v1.smoke.allowlist.json", smoke21_jobs
    )
    _write_allowlist(
        CFG_DIR / "portfolio_a0_backfill_v1.smoke.allowlist.json", smoke_a0_jobs
    )

    summary = {
        "n_v21_allowlist": len(jobs21),
        "n_a0_allowlist": len(jobs_a0),
        "n_v21_products": len(products21),
        "n_a0_products": len(products_a0),
        "n_smoke_v21": len(smoke21_jobs),
        "n_smoke_a0": len(smoke_a0_jobs),
        "locked_origins": locked,
    }
    out = PRE.parent / "config_build_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
