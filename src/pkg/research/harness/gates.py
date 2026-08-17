"""Freeze checksums, canonical F0 reproduction, WMAPE gates."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from pkg.benchmark import backtest
from pkg.benchmark.config import EXPECTED_ANALYSIS_B_PRIMARY, PANEL_FILES, PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult

F0_WMAPE_TOL = 0.05
LOCKED_F0_WMAPE = {
    "ts_xgb": EXPECTED_ANALYSIS_B_PRIMARY["ts_xgb"],
    "human_xgb": EXPECTED_ANALYSIS_B_PRIMARY["human_xgb"],
    "n": EXPECTED_ANALYSIS_B_PRIMARY["n"],
    "n_origins": EXPECTED_ANALYSIS_B_PRIMARY["n_origins"],
}


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def freeze_checksums(ds: BenchmarkDataset) -> dict[str, str]:
    out = {}
    for name in PANEL_FILES:
        p = ds.root / name
        if p.exists():
            out[name] = _file_sha256(p)
    man = ds.root / "manifest.json"
    if man.exists():
        out["manifest.json"] = _file_sha256(man)
    return out


def assert_freeze_unchanged(ds: BenchmarkDataset, before: dict[str, str]) -> None:
    after = freeze_checksums(ds)
    if after != before:
        raise AssertionError(
            "evaluation modified frozen benchmark files "
            f"(before={before} after={after})"
        )


def run_frozen_f0(ds: BenchmarkDataset, anchor: str) -> BacktestResult:
    name = "ts_xgb" if anchor == "ts" else "human_xgb"
    return backtest(name, dataset=ds, universe="matched", eligibility="primary")


def confirm_canonical_f0(ds: BenchmarkDataset) -> dict:
    """Reproduce frozen F0; return canonical metrics for research families.

    Does not rewrite locked EXPECTED_ANALYSIS_B_PRIMARY. Refuses to run if
    n / origins do not match the contract.
    """
    rows = []
    f0_results: dict[str, BacktestResult] = {}
    for anchor, key in (("ts", "ts_xgb"), ("human", "human_xgb")):
        res = run_frozen_f0(ds, anchor)
        f0_results[anchor] = res
        got = float(res.overall["wmape"].iloc[0])
        n = int(res.overall["n"].iloc[0])
        n_origins = len(res.origins)
        locked = LOCKED_F0_WMAPE[key]
        rows.append(
            {
                "anchor": anchor,
                "frozen_name": key,
                "wmape_reproduced": got,
                "wmape_locked_contract": locked,
                "wmape_gap": got - locked,
                "n": n,
                "n_locked": LOCKED_F0_WMAPE["n"],
                "n_origins": n_origins,
                "n_origins_locked": LOCKED_F0_WMAPE["n_origins"],
                "matches_locked_wmape": abs(got - locked) <= F0_WMAPE_TOL,
            }
        )
        if n != LOCKED_F0_WMAPE["n"]:
            raise AssertionError(
                f"F0 {anchor} n={n} != locked contract n={LOCKED_F0_WMAPE['n']}"
            )
        if n_origins != LOCKED_F0_WMAPE["n_origins"]:
            raise AssertionError(
                f"F0 {anchor} n_origins={n_origins} != "
                f"{LOCKED_F0_WMAPE['n_origins']}"
            )
        if sorted(int(o) for o in res.origins) != list(PRIMARY_ORIGINS):
            raise AssertionError(
                f"F0 {anchor} origins {res.origins} != PRIMARY {PRIMARY_ORIGINS}"
            )

    summary = pd.DataFrame(rows)
    canonical = {
        "ts": float(summary.loc[summary["anchor"] == "ts", "wmape_reproduced"].iloc[0]),
        "human": float(
            summary.loc[summary["anchor"] == "human", "wmape_reproduced"].iloc[0]
        ),
        "n": LOCKED_F0_WMAPE["n"],
        "n_origins": LOCKED_F0_WMAPE["n_origins"],
        "source": "current frozen backtest(ts_xgb/human_xgb) on pkg.benchmark v1",
        "locked_contract_matches": bool(summary["matches_locked_wmape"].all()),
    }
    return {"summary": summary, "canonical": canonical, "results": f0_results}


def wmape_gate_row(
    label: str,
    got: float,
    expected: float,
    n: int,
    n_origins: int,
    *,
    n_expected: int = LOCKED_F0_WMAPE["n"],
    n_origins_expected: int = LOCKED_F0_WMAPE["n_origins"],
    tol: float = F0_WMAPE_TOL,
) -> dict:
    ok = abs(got - expected) <= tol and n == n_expected and n_origins == n_origins_expected
    return {
        "label": label,
        "wmape_got": got,
        "wmape_expected": expected,
        "wmape_gap": got - expected,
        "n": n,
        "n_origins": n_origins,
        "ok": ok,
    }


def assert_wmape_gate(row: dict) -> None:
    if not row["ok"]:
        raise AssertionError(
            f"Reproduction gate FAILED for {row['label']}: "
            f"WMAPE={row['wmape_got']} expected={row['wmape_expected']} "
            f"gap={row['wmape_gap']} n={row['n']} n_origins={row['n_origins']}"
        )
