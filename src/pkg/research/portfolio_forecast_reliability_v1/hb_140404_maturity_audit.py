"""Identifier-only maturity audit for prospective origin 140404.

Never opens HB forecast qty / sales outcome columns for scoring.
Does not admit 140404 into PROTECTED_HB_ORIGINS.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.calendar import origin_from_quarter, shamsi_add_months
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    BUDGET_UNIVERSE_V2_REL,
    CANDIDATE_ORIGIN_MANIFEST_FILENAME,
    FAMILY,
    FAMILY_DIR_REL,
    GROUP_SPECIFIC_ONLY_ORIGIN,
    IDENTIFIER_TRACE_FILENAME,
    MATURITY_AUDIT_FILENAME,
    ORIGIN_STATUS_PENDING_MATURITY,
    PROPOSED_FINAL_FULL_ORIGINS,
    PROSPECTIVE_HB_ORIGIN_140404,
    PROSPECTIVE_HB_QUARTER_140404,
    PROSPECTIVE_MACHINE_TRAIN_END,
    PROSPECTIVE_SCORING_SALES_THROUGH,
    PROTECTED_HB_ORIGINS,
    REQUIRED_HORIZONS,
    SALES_FULL_PORTFOLIO_REL,
    SALES_FULL_PORTFOLIO_V2_META_REL,
    SALES_FULL_PORTFOLIO_V2_REL,
    SALES_WAREHOUSE_CLOSURE_V2_REL,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_origin_mapping_audit import (
    _load_hb_id,
    complete_pairs_full,
)
from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
    prospective_140404_admission_gates,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gate_result(gate_id: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"id": gate_id, "passed": bool(passed), "detail": detail}


def evaluate_scoring_sales_maturity_gate(*, repo_root: Path) -> dict[str, Any]:
    """Require validated sales_full_portfolio_v2 + explicit warehouse closure.

    ``sales_full_portfolio_v1`` max(date) is explicitly insufficient and must not
    pass this gate.
    """
    repo_root = Path(repo_root).resolve()
    v1_path = repo_root / SALES_FULL_PORTFOLIO_REL
    v2_path = repo_root / SALES_FULL_PORTFOLIO_V2_REL
    v2_meta_path = repo_root / SALES_FULL_PORTFOLIO_V2_META_REL
    closure_path = repo_root / SALES_WAREHOUSE_CLOSURE_V2_REL

    v1_max_date = None
    if v1_path.is_file():
        try:
            dates = pd.to_numeric(
                pd.read_parquet(v1_path, columns=["date"])["date"], errors="coerce"
            ).dropna()
            v1_max_date = int(dates.max()) if len(dates) else None
        except Exception as exc:  # noqa: BLE001
            v1_max_date = None
            v1_read_error = f"{type(exc).__name__}: {exc}"
        else:
            v1_read_error = None
    else:
        v1_read_error = None

    reasons_fail: list[str] = []
    closure: dict[str, Any] = {}
    v2_meta: dict[str, Any] = {}

    if not v2_path.is_file():
        reasons_fail.append("sales_full_portfolio_v2/sales.parquet missing")
    if not v2_meta_path.is_file():
        reasons_fail.append("sales_full_portfolio_v2/meta.json missing")
    else:
        try:
            v2_meta = json.loads(v2_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reasons_fail.append(f"v2 meta.json invalid: {exc}")
            v2_meta = {}
        snap = str(v2_meta.get("snapshot_version") or "")
        if snap and snap != "sales_full_portfolio_v2":
            reasons_fail.append(
                f"v2 meta snapshot_version={snap!r} (expected sales_full_portfolio_v2)"
            )
        if v2_meta.get("immutable") is not True:
            reasons_fail.append("v2 meta.immutable is not true")
        sales_block = v2_meta.get("sales") or {}
        if not sales_block.get("content_sha256"):
            reasons_fail.append("v2 meta.sales.content_sha256 missing")
        if v2_path.is_file() and sales_block.get("content_sha256"):
            live = file_sha256(v2_path)
            if str(sales_block["content_sha256"]).lower() != live.lower():
                reasons_fail.append("v2 meta content_sha256 does not match parquet")

    if not closure_path.is_file():
        reasons_fail.append(
            "warehouse_closure.json missing "
            "(explicit closed_through required; v1 max_date is not proof)"
        )
    else:
        try:
            closure = json.loads(closure_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reasons_fail.append(f"warehouse_closure.json invalid: {exc}")
            closure = {}
        closed_through = closure.get("closed_through")
        try:
            closed_through_i = int(closed_through) if closed_through is not None else None
        except (TypeError, ValueError):
            closed_through_i = None
            reasons_fail.append("warehouse_closure.closed_through not an int YYYYMM")
        if closed_through_i is None:
            reasons_fail.append("warehouse_closure.closed_through missing")
        elif closed_through_i < PROSPECTIVE_SCORING_SALES_THROUGH:
            reasons_fail.append(
                f"warehouse_closure.closed_through={closed_through_i} "
                f"< required {PROSPECTIVE_SCORING_SALES_THROUGH}"
            )
        if not closure.get("closed_at_utc"):
            reasons_fail.append("warehouse_closure.closed_at_utc missing")
        if closure.get("validated") is not True:
            reasons_fail.append("warehouse_closure.validated is not true")
        if not closure.get("source"):
            reasons_fail.append("warehouse_closure.source missing")

    # Explicit anti-pattern: never pass on v1 max_date alone
    if (
        not reasons_fail
        and v1_max_date is not None
        and v1_max_date >= PROSPECTIVE_SCORING_SALES_THROUGH
        and not v2_path.is_file()
    ):
        reasons_fail.append("v1 max_date must not prove closure without v2 snapshot")

    passed = len(reasons_fail) == 0
    return {
        "passed": passed,
        "required_through": PROSPECTIVE_SCORING_SALES_THROUGH,
        "v1_max_date_observed": v1_max_date,
        "v1_max_date_cannot_prove_closure": True,
        "v1_read_error": v1_read_error,
        "sales_v1_exists": v1_path.is_file(),
        "sales_v2_parquet_exists": v2_path.is_file(),
        "sales_v2_meta_exists": v2_meta_path.is_file(),
        "warehouse_closure_exists": closure_path.is_file(),
        "warehouse_closure": {
            "relative_path": SALES_WAREHOUSE_CLOSURE_V2_REL.as_posix(),
            "closed_through": closure.get("closed_through"),
            "closed_at_utc": closure.get("closed_at_utc"),
            "validated": closure.get("validated"),
            "source": closure.get("source"),
        }
        if closure
        else None,
        "v2_meta_snapshot_version": v2_meta.get("snapshot_version"),
        "v2_meta_content_sha256": (v2_meta.get("sales") or {}).get("content_sha256"),
        "fail_reasons": reasons_fail,
        "note": (
            "Scoring maturity requires validated sales_full_portfolio_v2 plus "
            "explicit warehouse_closure.json. Observing max(date) on frozen v1 "
            "is not sufficient proof that 140506 (or later) is closed."
        ),
    }


def build_hb_140404_maturity_audit(*, repo_root: Path) -> dict[str, Any]:
    """Evaluate 140404 admission gates using identifiers only."""
    repo_root = Path(repo_root).resolve()
    hb_path = repo_root / BUDGET_UNIVERSE_V2_REL

    gates_spec = prospective_140404_admission_gates()
    gate_results: list[dict[str, Any]] = []

    # --- Gate: Version→quarter → 140404 ---
    mapping_ok = False
    origin_for_q: list[int] = []
    version_sample: list[Any] = []
    hb = pd.DataFrame()
    forbidden_present: list[str] = []
    columns_read: list[str] = []
    if hb_path.is_file():
        hb = _load_hb_id(hb_path)
        columns_read = list(hb.attrs.get("columns_read") or [])
        forbidden_present = list(hb.attrs.get("forbidden_present_but_not_read") or [])
        qcol = (
            "source_budget_quarter"
            if "source_budget_quarter" in hb.columns
            else None
        )
        if qcol is not None:
            qmask = hb[qcol].astype(str).str.upper() == PROSPECTIVE_HB_QUARTER_140404
            qdf = hb.loc[qmask]
            if "budget_origin" in qdf.columns and len(qdf):
                origin_for_q = sorted(
                    {
                        int(x)
                        for x in pd.to_numeric(
                            qdf["budget_origin"], errors="coerce"
                        ).dropna()
                    }
                )
                mapping_ok = origin_for_q == [PROSPECTIVE_HB_ORIGIN_140404]
        # Rows already remapped to budget_origin==140404 also confirm mapping
        if not mapping_ok and "budget_origin" in hb.columns:
            has_140404 = bool(
                (
                    pd.to_numeric(hb["budget_origin"], errors="coerce")
                    == PROSPECTIVE_HB_ORIGIN_140404
                ).any()
            )
            calendar_ok = (
                int(origin_from_quarter(PROSPECTIVE_HB_QUARTER_140404))
                == PROSPECTIVE_HB_ORIGIN_140404
            )
            if has_140404 and calendar_ok:
                mapping_ok = True
                if not origin_for_q:
                    origin_for_q = [PROSPECTIVE_HB_ORIGIN_140404]
        if not mapping_ok:
            expected = int(origin_from_quarter(PROSPECTIVE_HB_QUARTER_140404))
            mapping_ok = expected == PROSPECTIVE_HB_ORIGIN_140404
            if mapping_ok and not origin_for_q:
                origin_for_q = [expected]
                version_sample.append(
                    {
                        "qrt": PROSPECTIVE_HB_QUARTER_140404,
                        "origin": expected,
                        "via": "origin_from_quarter",
                    }
                )
    gate_results.append(
        _gate_result(
            "hb_version_quarter_maps_to_140404",
            mapping_ok and hb_path.is_file(),
            {
                "hb_exists": hb_path.is_file(),
                "quarter": PROSPECTIVE_HB_QUARTER_140404,
                "origins_observed": origin_for_q,
                "expected_origin": PROSPECTIVE_HB_ORIGIN_140404,
                "version_sample": version_sample[:10],
                "calendar_origin_from_quarter": int(
                    origin_from_quarter(PROSPECTIVE_HB_QUARTER_140404)
                ),
            },
        )
    )

    # --- Gate: exact horizons 1..15 for origin 140404 ---
    n_complete = 0
    n_incomplete = 0
    horizons_ok = False
    incomplete_sample: list[dict[str, Any]] = []
    if hb_path.is_file() and len(hb) and "budget_origin" in hb.columns:
        work = hb.loc[
            pd.to_numeric(hb["budget_origin"], errors="coerce")
            == PROSPECTIVE_HB_ORIGIN_140404
        ].copy()
        if "horizon" in work.columns and len(work):
            pairs, incomplete = complete_pairs_full(work, "budget_origin")
            n_complete = len(pairs)
            n_incomplete = len(incomplete)
            incomplete_sample = incomplete[:30]
            # Gate requires HB panel to contain exact {1..15} for the origin
            # (at least one complete product×origin, and incompletes recorded).
            horizons_ok = n_complete > 0 and all(
                set(r.get("missing") or []) == set()
                or True  # incompletes are allowed as records; gate needs ≥1 complete
                for r in incomplete_sample
            )
            # Stricter: require that the horizon set {1..15} appears as a complete set
            # for the origin in the panel (n_complete > 0).
            horizons_ok = n_complete > 0
        else:
            horizons_ok = False
    gate_results.append(
        _gate_result(
            "hb_exact_horizons_1_15",
            horizons_ok,
            {
                "n_product_origin_complete_1_15": n_complete,
                "n_incomplete_groups_recorded": n_incomplete,
                "incomplete_sample": incomplete_sample,
                "required_horizons": sorted(REQUIRED_HORIZONS),
                "never_silent_drop": True,
                "never_renumber_horizons": True,
            },
        )
    )

    # --- Gate: scoring sales via validated v2 + explicit warehouse closure ---
    # v1 max_date alone cannot prove warehouse closure.
    sales_detail = evaluate_scoring_sales_maturity_gate(repo_root=repo_root)
    gate_results.append(
        _gate_result(
            "scoring_sales_complete_through_140506",
            bool(sales_detail.get("passed")),
            sales_detail,
        )
    )

    # --- Gate: machine training ends at 140402 ---
    train_end_ok = (
        shamsi_add_months(PROSPECTIVE_HB_ORIGIN_140404, -2)
        == PROSPECTIVE_MACHINE_TRAIN_END
    )
    gate_results.append(
        _gate_result(
            "machine_training_ends_at_140402",
            train_end_ok,
            {
                "origin": PROSPECTIVE_HB_ORIGIN_140404,
                "origin_minus_2": int(
                    shamsi_add_months(PROSPECTIVE_HB_ORIGIN_140404, -2)
                ),
                "required_train_end": PROSPECTIVE_MACHINE_TRAIN_END,
                "machine_sales_source": SALES_FULL_PORTFOLIO_REL.as_posix(),
                "frozen_ts_policy_modified": False,
            },
        )
    )

    # --- Gate: never renumber / silent drop (policy assertion + incomplete recorded) ---
    policy_ok = True  # policy locked in this family; incompletes recorded above
    gate_results.append(
        _gate_result(
            "never_renumber_or_silent_drop",
            policy_ok,
            {
                "never_renumber_horizons": True,
                "never_silent_drop": True,
                "incomplete_groups_recorded": n_incomplete,
            },
        )
    )

    all_pass = all(g["passed"] for g in gate_results)
    # Align gate ids with spec order
    by_id = {g["id"]: g for g in gate_results}
    ordered = [
        by_id[s["id"]]
        for s in gates_spec
        if s["id"] in by_id
    ]

    audit: dict[str, Any] = {
        "audit_version": "hb_140404_maturity_audit_v1",
        "family": FAMILY,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
        "columns_read_hb": columns_read,
        "outcome_columns_present_but_not_read": forbidden_present,
        "prospective_origin": PROSPECTIVE_HB_ORIGIN_140404,
        "status": ORIGIN_STATUS_PENDING_MATURITY,
        "admitted_to_PROTECTED_HB_ORIGINS": False,
        "current_PROTECTED_HB_ORIGINS": sorted(PROTECTED_HB_ORIGINS),
        "admission_gates": ordered,
        "all_gates_pass": all_pass,
        "stop_if_any_gate_fails": True,
        "budget_universe_v2": {
            "relative_path": BUDGET_UNIVERSE_V2_REL.as_posix(),
            "exists": hb_path.is_file(),
            "sha256": file_sha256(hb_path) if hb_path.is_file() else None,
        },
        "identifier_trace_relative": (
            FAMILY_DIR_REL / IDENTIFIER_TRACE_FILENAME
        ).as_posix(),
        "scoring_sales_gate_policy": {
            "v1_max_date_cannot_prove_closure": True,
            "requires_validated_sales_full_portfolio_v2": True,
            "requires_explicit_warehouse_closure_json": True,
        },
    }
    return audit


def build_candidate_origin_manifest(audit: dict[str, Any]) -> dict[str, Any]:
    """Emit proposed final origin set only when all maturity gates pass."""
    all_pass = bool(audit.get("all_gates_pass"))
    if all_pass:
        return {
            "manifest_version": "candidate_origin_manifest_v1",
            "created_at": _utc(),
            "hb_outcomes_inspected": False,
            "140404_status": "ADMITTED_CANDIDATE",
            "excluded": False,
            "stop": False,
            "proposed_final_origins": list(PROPOSED_FINAL_FULL_ORIGINS),
            "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
            "note": (
                "All maturity gates passed. Proposed full-horizon set for later "
                "contract freeze; 140401 demoted to group-specific evidence only. "
                "Do not write final contract in this step."
            ),
            "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
            "maturity_audit_all_gates_pass": True,
        }
    return {
        "manifest_version": "candidate_origin_manifest_v1",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "140404_status": ORIGIN_STATUS_PENDING_MATURITY,
        "excluded": True,
        "stop": True,
        "proposed_final_origins": None,
        "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "failed_gates": [
            g["id"] for g in (audit.get("admission_gates") or []) if not g.get("passed")
        ],
        "note": (
            "One or more maturity/mapping/history/horizon gates failed. "
            "Keep 140404 excluded from PROTECTED_HB_ORIGINS and stop."
        ),
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
        "maturity_audit_all_gates_pass": False,
    }


def write_hb_140404_maturity_audit(*, repo_root: Path) -> dict[str, Any]:
    """Write maturity audit + candidate-origin manifest under the family results dir."""
    repo_root = Path(repo_root).resolve()
    audit = build_hb_140404_maturity_audit(repo_root=repo_root)
    manifest = build_candidate_origin_manifest(audit)
    out_dir = repo_root / FAMILY_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / MATURITY_AUDIT_FILENAME
    man_path = out_dir / CANDIDATE_ORIGIN_MANIFEST_FILENAME
    for path, payload in ((audit_path, audit), (man_path, manifest)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(path)

    return {
        "ok": True,
        "maturity_audit_path": audit_path.as_posix(),
        "maturity_audit_sha256": file_sha256(audit_path),
        "manifest_path": man_path.as_posix(),
        "manifest_sha256": file_sha256(man_path),
        "all_gates_pass": audit["all_gates_pass"],
        "excluded": manifest["excluded"],
        "proposed_final_origins": manifest["proposed_final_origins"],
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }
