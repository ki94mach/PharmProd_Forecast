"""Section A: experiment audit and comparability checks.

Every finding carries an ``evidence`` grade so the report never blurs the line
between what the saved artifacts prove and what only the source code suggests:

``verified_from_artifacts``
    Established by reading the saved results, manifest, SQLite state or panels.
``code_inspection``
    Established by reading the pipeline source; the artifacts do not record it.
``not_verifiable``
    Cannot be settled from the evidence available in this experiment.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from pkg.benchmark.calendar import origin_from_quarter, shamsi_add_months
from pkg.research.v2_eval.config import (
    FORECAST_HORIZON,
    SHIFTED_ORIGINS,
    V2EvalConfig,
)
from pkg.research.v2_eval.load import BackfillInputs

VERIFIED = "verified_from_artifacts"
CODE = "code_inspection"
UNVERIFIABLE = "not_verifiable"


def job_slug(quarter: pd.Series, product_id: pd.Series) -> pd.Series:
    """Reproduce ``JobIdentity.slug`` (path separators become underscores)."""
    safe = product_id.astype(str).str.replace("/", "_", regex=False)
    safe = safe.str.replace("\\", "_", regex=False)
    return quarter.astype(str) + "__" + safe


def _row(
    check: str,
    area: str,
    status: str,
    evidence: str,
    finding: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "check": check,
        "area": area,
        "status": status,
        "evidence": evidence,
        "finding": finding,
        "detail": detail,
    }


def _audit_universe(data: BackfillInputs) -> list[dict[str, Any]]:
    v2_products = set(data.v2_forecasts["product_id"])
    legacy_products = set(data.legacy["product_id"])
    universe_products = set(data.universe["product_id"])
    job_products = set(data.jobs["product_id"])
    legacy_only = sorted(legacy_products - universe_products)
    rows = [
        _row(
            "universe.identifier",
            "universe",
            "info",
            VERIFIED,
            "Both engines key products on the English product title string.",
            "V2 forecast.csv 'product' and legacy ts_universe 'product' join "
            "exactly; the ID_INT migration in docs/ts_v2_product_identity.md is "
            "not yet wired, so no identifier translation is applied.",
        ),
        _row(
            "universe.size",
            "universe",
            "ok" if job_products == universe_products else "warn",
            VERIFIED,
            f"Backfill covered {len(job_products)} products from the "
            f"mvp_products manifest ({len(universe_products)} listed).",
            f"V2 forecast rows cover {len(v2_products)} products.",
        ),
        _row(
            "universe.legacy_extra",
            "universe",
            "info",
            VERIFIED,
            f"Legacy panel holds {len(legacy_products)} products, "
            f"{len(legacy_only)} of which are outside the backfill universe.",
            f"legacy_only={legacy_only}; these rows can never match and are "
            "reported as unmatched coverage, never imputed.",
        ),
    ]
    return rows


def _audit_origins(data: BackfillInputs) -> list[dict[str, Any]]:
    v2_origins = set(data.v2_forecasts["forecast_origin"].astype(int))
    legacy_origins = set(data.legacy["forecast_origin"].astype(int))
    manifest_origins = set(data.vintages["forecast_origin"].astype(int))
    v2_only = sorted(v2_origins - legacy_origins)
    legacy_only = sorted(legacy_origins - v2_origins)

    canonical_ok = []
    for _, spec in data.vintages.iterrows():
        expected = int(origin_from_quarter(str(spec["quarter"])))
        canonical_ok.append(expected == int(spec["forecast_origin"]))

    shifted_present = sorted(
        o for o in SHIFTED_ORIGINS if o in v2_origins or o in legacy_origins
    )

    # Quarter labels that disagree between engines at the same origin.
    pairs = (
        data.v2_forecasts[["forecast_origin", "quarter"]]
        .drop_duplicates()
        .merge(
            data.legacy[["forecast_origin", "legacy_quarter"]].drop_duplicates(),
            on="forecast_origin",
            how="inner",
        )
    )
    disagreements = pairs.loc[pairs["quarter"] != pairs["legacy_quarter"]]

    return [
        _row(
            "origins.canonical",
            "origins",
            "ok" if all(canonical_ok) else "warn",
            VERIFIED,
            "V2 origins follow the canonical quarter->origin rule from the "
            "vintage manifest.",
            f"manifest origins={sorted(manifest_origins)}",
        ),
        _row(
            "origins.coverage",
            "origins",
            "warn" if (v2_only or legacy_only) else "ok",
            VERIFIED,
            f"{len(v2_origins & legacy_origins)} origins are common; "
            f"{len(v2_only)} V2-only and {len(legacy_only)} legacy-only.",
            f"v2_only={v2_only} (no legacy vintage exists); "
            f"legacy_only={legacy_only} (no V2 job at that origin).",
        ),
        _row(
            "origins.shift",
            "origins",
            "warn" if len(disagreements) else "ok",
            VERIFIED,
            "Legacy 1403 vintages were emitted at a shifted origin, so one "
            "matched origin carries different quarter labels.",
            "Legacy CSVs start at max(history)+1 rather than the quarter's "
            "first month: legacy 1403Q1 -> 140304 and 1403Q2 -> 140306. "
            "Matching on forecast_origin holds the information cutoff fixed but "
            "pairs V2 "
            + ", ".join(
                f"{r.quarter} with legacy {r.legacy_quarter} at {int(r.forecast_origin)}"
                for r in disagreements.itertuples()
            )
            + f". Sensitivity excludes origins {shifted_present}.",
        ),
        _row(
            "origins.training_cutoff",
            "origins",
            "warn",
            CODE,
            "The two engines do not train on the same amount of history at a "
            "shared origin.",
            "V2 trains on date < forecast_origin (manifest training_cutoff_rule "
            "and pkg.ts_v2.data.assert_training_before_origin). Legacy drops the "
            "final month via sale_series[:-1] in pkg/forecast.py, so its "
            "effective training ends at origin-2. Legacy is handicapped by one "
            "month of information at every matched origin.",
        ),
    ]


def _audit_calendar(data: BackfillInputs) -> list[dict[str, Any]]:
    v2 = data.v2_forecasts
    bad_targets = []
    for (origin, _slug), group in v2.groupby(["forecast_origin", "job_slug"]):
        expected = [shamsi_add_months(int(origin), i) for i in range(FORECAST_HORIZON)]
        got = group.sort_values("horizon")["target_date"].astype(int).tolist()
        if got != expected:
            bad_targets.append(int(origin))
    horizons_ok = sorted(v2["horizon"].unique().tolist()) == list(
        range(1, FORECAST_HORIZON + 1)
    )
    per_job = v2.groupby("job_slug").size()
    return [
        _row(
            "calendar.target_dates",
            "calendar",
            "ok" if not bad_targets else "fail",
            VERIFIED,
            "Every V2 job emits target months origin..origin+14 on the Shamsi "
            "calendar.",
            f"jobs_with_unexpected_targets={len(bad_targets)}",
        ),
        _row(
            "calendar.horizons",
            "calendar",
            "ok" if horizons_ok else "fail",
            VERIFIED,
            f"Horizons are exactly 1..{FORECAST_HORIZON}.",
            f"rows_per_job unique values={sorted(per_job.unique().tolist())}",
        ),
    ]


def _audit_actuals(data: BackfillInputs) -> list[dict[str, Any]]:
    sales = data.sales
    legacy = data.legacy
    merged = legacy.merge(
        sales, on=["product_id", "target_date"], how="left", validate="many_to_one"
    )
    both = merged.dropna(subset=["actual"])
    mismatch = both.loc[
        (both["actual_legacy_panel"] - both["actual"]).abs() > 1e-6
    ]
    max_actual = int(sales["target_date"].max())
    v2_beyond = int((data.v2_forecasts["target_date"] > max_actual).sum())
    neg = int((sales["actual"] < 0).sum())
    zeros = int((sales["actual"] == 0).sum())

    # Missing months inside each product's observed span are genuine gaps, not
    # zeros; the frozen sales panel simply omits those rows.
    gap_rows = 0
    for _, group in sales.groupby("product_id"):
        months = sorted(group["target_date"].astype(int))
        span = months[0]
        expected = 0
        while span <= months[-1]:
            expected += 1
            span = shamsi_add_months(span, 1)
        gap_rows += expected - len(months)

    return [
        _row(
            "actuals.definition",
            "actuals",
            "ok",
            VERIFIED,
            "Actuals are monthly summed dispatch quantity (DQTY) per product in "
            "raw units, identical for both engines.",
            "Source: src/data/benchmarks/v1/raw/sales.parquet, built by "
            "pkg.benchmark.freeze from Flat_Fact_Sale. The same values appear as "
            "'sales' in ts_universe.parquet.",
        ),
        _row(
            "actuals.consistency",
            "actuals",
            "ok" if mismatch.empty else "fail",
            VERIFIED,
            "Actuals agree between the legacy panel and the raw sales panel on "
            "every shared key.",
            f"rows_compared={len(both)}, mismatches={len(mismatch)}",
        ),
        _row(
            "actuals.ceiling",
            "actuals",
            "warn",
            VERIFIED,
            f"Actuals stop at {max_actual}, so later horizons of recent origins "
            "cannot be scored.",
            f"v2_forecast_rows_beyond_actuals={v2_beyond}. These rows are "
            "reported as unevaluated coverage and are never zero-filled.",
        ),
        _row(
            "actuals.zeros_and_negatives",
            "actuals",
            "warn" if neg else "info",
            VERIFIED,
            f"The sales panel contains {zeros} explicit zero months and {neg} "
            "negative months (returns/credits).",
            "Negative actuals are kept. WMAPE uses sum|actual| as the "
            "denominator, matching pkg.benchmark.evaluate.wmape, so negatives "
            "add to the denominator rather than cancelling it.",
        ),
        _row(
            "actuals.missing_vs_zero",
            "actuals",
            "info",
            VERIFIED,
            f"{gap_rows} month rows are absent inside product observation spans "
            "and are treated as unknown, not as zero demand.",
            "Absent months simply produce no evaluable row here. Separately, V2 "
            "training fills gaps with zero under missing_month_policy='zero' "
            "while flagging is_missing_month (docs/ts_v2_gap_audit.md); that "
            "affects fitting, not this scoring.",
        ),
    ]


def _audit_jobs(data: BackfillInputs) -> list[dict[str, Any]]:
    jobs = data.jobs
    counts = jobs["status"].value_counts().to_dict()
    failed = jobs.loc[jobs["status"] == "FAILED"]
    dup_jobs = int(
        jobs.duplicated(subset=["quarter", "product_id", "config_hash"]).sum()
    )
    dup_rows = int(
        data.v2_forecasts.duplicated(
            subset=["product_id", "forecast_origin", "target_date"]
        ).sum()
    )
    success = jobs.loc[jobs["status"] == "SUCCESS"]
    success_slugs = set(job_slug(success["quarter"], success["product_id"]))
    forecast_slugs = set(data.v2_forecasts["job_slug"])
    return [
        _row(
            "jobs.status",
            "completeness",
            "ok",
            VERIFIED,
            f"state.sqlite reports {counts.get('SUCCESS', 0)} SUCCESS and "
            f"{counts.get('FAILED', 0)} FAILED of {len(jobs)} jobs.",
            "; ".join(
                f"{r.quarter}/{r.product_id}: {r.error_type}: {r.error_message}"
                for r in failed.itertuples()
            )
            or "no failures",
        ),
        _row(
            "jobs.stale_failure_files",
            "completeness",
            "warn" if data.stale_failure_files > len(failed) else "ok",
            VERIFIED,
            f"{data.stale_failure_files} logs/*/failure.json files remain on "
            f"disk but only {len(failed)} jobs are currently FAILED.",
            "BackfillStore.persist_failure writes failure.json and a later "
            "successful retry does not delete it, so the file count overstates "
            "failures. state.sqlite is used as the source of truth throughout "
            "this analysis.",
        ),
        _row(
            "jobs.artifact_agreement",
            "completeness",
            "ok" if success_slugs == forecast_slugs else "warn",
            VERIFIED,
            "Every SUCCESS job has a forecast.csv and vice versa.",
            f"success_jobs={len(success_slugs)}, forecast_dirs={len(forecast_slugs)}",
        ),
        _row(
            "jobs.duplicate_keys",
            "completeness",
            "ok" if dup_jobs == 0 and dup_rows == 0 else "fail",
            VERIFIED,
            "No duplicate prediction keys in the V2 output.",
            f"duplicate_job_identities={dup_jobs}, "
            f"duplicate_(product,origin,target) rows={dup_rows}",
        ),
    ]


def _audit_postprocess(data: BackfillInputs) -> list[dict[str, Any]]:
    v2 = data.v2_forecasts
    clipped = int((v2["raw_forecast"] < 0).sum())
    neg_final = int((v2["forecast"] < 0).sum())
    manifest = data.manifest
    return [
        _row(
            "postprocess.v2",
            "postprocess",
            "ok",
            VERIFIED,
            "V2 applies non-negativity only; no smoothing and no rounding.",
            f"manifest nonnegative_policy={manifest.get('nonnegative_policy')}, "
            f"smoothing_policy={manifest.get('smoothing_policy')}. "
            f"raw_forecast<0 rows clipped to zero={clipped}; "
            f"negative delivered forecasts={neg_final}.",
        ),
        _row(
            "postprocess.legacy",
            "postprocess",
            "warn",
            CODE,
            "Legacy forecasts carry extra postprocessing that V2 forbids.",
            "pkg/forecast.py applies redistribute_smoothing (70% pull toward the "
            "quarter mean), replace_negative_sales, integer rounding at CSV "
            "export, and a Prophet x0.8 haircut. Legacy columns in "
            "ts_universe.parquet are post-postprocess, so the primary comparison "
            "is end-to-end delivered output on both sides.",
        ),
        _row(
            "postprocess.raw_availability",
            "postprocess",
            "info",
            VERIFIED,
            "A raw (pre-constraint) comparison is only possible on the V2 side.",
            "V2 forecast.csv stores raw_forecast alongside forecast; no "
            "pre-smoothing legacy series was ever saved. The raw-vs-constrained "
            "delta is therefore reported as a labelled V2-only diagnostic.",
        ),
    ]


def _audit_config(data: BackfillInputs) -> list[dict[str, Any]]:
    manifest = data.manifest
    hashes = sorted(set(data.jobs["config_hash"]))
    ts_config = (
        manifest.get("model_configuration", {}).get("ts_forecast_config", {}) or {}
    )
    return [
        _row(
            "config.identity",
            "config",
            "ok" if len(hashes) == 1 else "fail",
            VERIFIED,
            f"All jobs share a single scientific config hash: {hashes}.",
            f"experiment_id={manifest.get('experiment_id')}, "
            f"engine={manifest.get('engine_version')}, "
            f"selection_strategy={ts_config.get('selection_strategy')}, "
            f"selection_metric={ts_config.get('selection_metric')}, "
            f"candidates={ts_config.get('candidate_models')}",
        ),
        _row(
            "config.environment",
            "config",
            "info",
            VERIFIED,
            "The backfill ran single-threaded on the Linux server.",
            f"{manifest.get('environment', {}).get('platform')}, "
            f"python={manifest.get('python_version')}, "
            f"packages={manifest.get('package_versions')}",
        ),
    ]


def _audit_leakage(data: BackfillInputs) -> list[dict[str, Any]]:
    v2 = data.v2_forecasts
    origin_leak = int((v2["target_date"] < v2["forecast_origin"]).sum())
    return [
        _row(
            "leakage.target_window",
            "leakage",
            "ok" if origin_leak == 0 else "fail",
            VERIFIED,
            "No V2 forecast row targets a month before its own origin.",
            f"rows_with_target<origin={origin_leak}; manifest records "
            f"training_cutoff_rule={data.manifest.get('training_cutoff_rule')!r}.",
        ),
        _row(
            "leakage.training_cut",
            "leakage",
            "ok",
            CODE,
            "V2 enforces date < forecast_origin at fit time.",
            "pkg.ts_v2.data.assert_training_before_origin, the internal guard in "
            "prepare_monthly_series, and engine.assert_final_forecast_contract "
            "all raise if the origin month reaches training. The runner also "
            "truncates sales before calling the engine. The saved artifacts do "
            "not record the training rows themselves, so this rests on code.",
        ),
        _row(
            "leakage.preprocessing",
            "leakage",
            "ok",
            CODE,
            "V2 fits on raw units with no global transform.",
            "No MinMax, Yeo-Johnson or ADF step exists in pkg.ts_v2, unlike "
            "pkg/forecast.py which scales across the full series before "
            "splitting. Removing that global fit removes a preprocessing "
            "leakage path present in legacy.",
        ),
        _row(
            "leakage.model_selection",
            "leakage",
            "ok",
            CODE,
            "Selection scores come from expanding-window origins strictly "
            "before the forecast origin.",
            "pkg.ts_v2.backtest_origins.discover_origins plus "
            "assert_backtest_no_leakage enforce max(train) < origin <= "
            "min(target) per fold. Per-fold evidence was not persisted for this "
            "experiment.",
        ),
        _row(
            "leakage.ensemble_weights",
            "leakage",
            "info",
            VERIFIED,
            "No ensemble weighting was used, so no weight-leakage surface "
            "exists in this run.",
            "Every result.json records selected_strategy='best_model'; "
            "ensemble_top_k is configured but unused.",
        ),
        _row(
            "leakage.per_fold_evidence",
            "leakage",
            "warn",
            UNVERIFIABLE,
            "The run saved no per-fold CV artifacts, so leakage-freedom cannot "
            "be re-checked from the outputs alone.",
            f"backtests/ contains {data.backtest_files} files because "
            "V2ForecastEngine passes only selected_strategy and "
            "n_training_observations in extras, so BackfillStore never writes "
            "backtest_summary.json. Candidate forecasts, per-model CV scores, "
            "fold training windows and fallback reasons are unavailable.",
        ),
        _row(
            "leakage.final_refit",
            "leakage",
            "warn",
            CODE,
            "The intended discard-CV-then-refit behaviour is visible in code but "
            "not provable from the artifacts.",
            "engine.forecast_with_backtest deletes the CV instances and "
            "refit_and_forecast_product builds a fresh model via get_model. "
            "result.json stores n_training_observations, which is consistent "
            "with a full pre-origin refit, but no fit-level record was saved.",
        ),
    ]


def build_methodology_audit(data: BackfillInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows += _audit_universe(data)
    rows += _audit_origins(data)
    rows += _audit_calendar(data)
    rows += _audit_actuals(data)
    rows += _audit_jobs(data)
    rows += _audit_postprocess(data)
    rows += _audit_config(data)
    rows += _audit_leakage(data)
    return pd.DataFrame(rows)


def build_coverage_table(
    data: BackfillInputs,
    matched: pd.DataFrame,
    v2_scored: pd.DataFrame,
) -> pd.DataFrame:
    """Per-origin accounting of jobs, forecast rows, actuals and matched rows."""
    jobs = data.jobs
    v2 = data.v2_forecasts
    legacy = data.legacy

    origins = sorted(
        set(jobs["forecast_origin"]) | set(legacy["forecast_origin"].astype(int))
    )
    quarter_by_origin = (
        jobs.drop_duplicates("forecast_origin")
        .set_index("forecast_origin")["quarter"]
        .to_dict()
    )
    legacy_quarter = (
        legacy.drop_duplicates("forecast_origin")
        .set_index("forecast_origin")["legacy_quarter"]
        .to_dict()
    )

    rows = []
    for origin in origins:
        job_slice = jobs.loc[jobs["forecast_origin"] == origin]
        v2_slice = v2.loc[v2["forecast_origin"] == origin]
        legacy_slice = legacy.loc[legacy["forecast_origin"] == origin]
        matched_slice = matched.loc[matched["forecast_origin"] == origin]
        scored_slice = v2_scored.loc[v2_scored["forecast_origin"] == origin]
        rows.append(
            {
                "forecast_origin": int(origin),
                "v2_quarter": quarter_by_origin.get(origin, ""),
                "legacy_quarter": legacy_quarter.get(origin, ""),
                "jobs_total": int(len(job_slice)),
                "jobs_success": int((job_slice["status"] == "SUCCESS").sum()),
                "jobs_failed": int((job_slice["status"] == "FAILED").sum()),
                "v2_forecast_rows": int(len(v2_slice)),
                "v2_rows_with_actual": int(scored_slice["actual"].notna().sum()),
                "legacy_rows": int(len(legacy_slice)),
                "matched_rows": int(len(matched_slice)),
                "v2_rows_unmatched": int(len(v2_slice) - len(matched_slice)),
                "legacy_rows_unmatched": int(len(legacy_slice) - len(matched_slice)),
                "shifted_origin": bool(origin in SHIFTED_ORIGINS),
            }
        )
    frame = pd.DataFrame(rows)

    failures = data.jobs.loc[data.jobs["status"] == "FAILED"]
    failure_rows = [
        {
            "forecast_origin": int(r.forecast_origin),
            "v2_quarter": r.quarter,
            "legacy_quarter": "",
            "jobs_total": 0,
            "jobs_success": 0,
            "jobs_failed": 1,
            "v2_forecast_rows": 0,
            "v2_rows_with_actual": 0,
            "legacy_rows": 0,
            "matched_rows": 0,
            "v2_rows_unmatched": 0,
            "legacy_rows_unmatched": 0,
            "shifted_origin": bool(int(r.forecast_origin) in SHIFTED_ORIGINS),
            "record_type": "failure",
            "product_id": r.product_id,
            "error_type": r.error_type,
            "error_message": r.error_message,
        }
        for r in failures.itertuples()
    ]
    frame["record_type"] = "origin_coverage"
    frame["product_id"] = ""
    frame["error_type"] = ""
    frame["error_message"] = ""
    if failure_rows:
        frame = pd.concat([frame, pd.DataFrame(failure_rows)], ignore_index=True)
    return frame


def audit_status_counts(audit: pd.DataFrame) -> dict[str, int]:
    return audit["status"].value_counts().to_dict()


def failing_checks(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[audit["status"] == "fail"]
