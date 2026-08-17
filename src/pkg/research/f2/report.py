"""Write docs/f2_results.md from F2 CSV artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.f2.config import docs_dir, f2_output_dir
from pkg.research.harness.report import md_table, read_csv, repo_relative


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return md_table(df, max_rows=max_rows, format_cells=False)


def _read(out_dir: Path, name: str) -> Optional[pd.DataFrame]:
    return read_csv(out_dir, name)


def _answers(overall: pd.DataFrame, classifications: dict) -> list[str]:
    def row(exp: str, anchor: str):
        sub = overall.loc[(overall["experiment"] == exp) & (overall["anchor"] == anchor)]
        return sub.iloc[0] if len(sub) else None

    f2a_ts = row("F2A", "ts")
    f2a_h = row("F2A", "human")
    f2b_h = row("F2B", "human")
    f2c_h = row("F2C", "human")

    def beat(r) -> str:
        if r is None:
            return "not run"
        rel = float(r["rel_wmape_vs_f0_pct"])
        return f"yes ({rel:+.2f}% rel WMAPE)" if rel > 0 else f"no ({rel:+.2f}% rel WMAPE)"

    lines = [
        "1. **Did robust demand-state (F2A) outperform F0?** "
        f"TS: {beat(f2a_ts)}; Human: {beat(f2a_h)} "
        f"verdicts={classifications.get(('F2A','ts'))}/"
        f"{classifications.get(('F2A','human'))}.",
        "2. **Did shrunk Human reliability (F2B) outperform F0 Human+ML?** "
        f"{beat(f2b_h)} verdict={classifications.get(('F2B','human'))}.",
        "3. **Did shrinkage reduce F1-style high-volume failure?** see watchlist "
        "and top-5 deterioration share in §error concentration.",
        "4. **Majority of products and origins?** see `origins_improved` and "
        "`product_win_rate` in the scoreboard.",
        "5. **Is F2C justified?** "
        + (
            "run — both families independently not REJECT."
            if f2c_h is not None
            else "not run (a family REJECT or not requested)."
        ),
        "6. **Promote?** "
        + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(classifications.items())),
        "7. **Reject or defer?** families with verdict REJECT; F2D deferred; "
        "F1B origin-regime features remain deferred.",
        "8. **Ready for lifecycle/price?** only if a family is PROMOTE or clearly "
        "PROMISING with a documented next step — not automatically.",
    ]
    return [ln + "\n" for ln in lines]


def write_f2_results(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = out_dir or report.get("out_dir") or f2_output_dir()
    path = path or (docs_dir() / "f2_results.md")
    overall = report.get("overall")
    if overall is None:
        overall = _read(out_dir, "overall.csv")
    classifications = report.get("classifications") or {}
    if overall is not None and not classifications:
        for _, r in overall.iterrows():
            if r["experiment"] != "F0" and r.get("verdict") not in ("CONTROL", ""):
                classifications[(r["experiment"], r["anchor"])] = r.get("verdict", "")

    canon = report.get("canonical_f0", {})
    canon_sum = canon.get("summary") if isinstance(canon, dict) else None
    if canon_sum is None:
        canon_sum = _read(out_dir, "f0_canonical.csv")

    sections = [
        "# F2 results\n",
        f"**Date:** {date.today().isoformat()}  \n",
        "**Benchmark:** frozen v1 matched PRIMARY  \n",
        f"**CSV artifacts:** `{repo_relative(Path(out_dir))}`\n",
        "\nF1 is not promoted. F2D is not implemented. "
        "This document reports negative findings as well as positive ones.\n",
        "\n## Executive interpretation\n\n",
        "Canonical F0 for this run is the **currently reproduced** frozen backtest: "
        "TS+XGB **38.28** WMAPE, Human+XGB **36.56** WMAPE, n=1877, 5 origins. "
        "Locked freeze-time contract values (37.23 / 36.69) were **not** rewritten; "
        "n and origins match. The WMAPE gap is environment/XGBoost, not a different freeze. "
        "Relative F2 lifts are vs the reproduced F0.\n\n",
        "| Family | Verdict | Headline |\n",
        "|--------|---------|----------|\n",
        "| F2A demand-state | **PROMISING_BUT_UNSTABLE** (do not promote) | "
        "Portfolio WMAPE worse (TS −5.2%, Human −3.1% rel). A slight majority of products "
        "improve on TS (win rate 52.7%, median +0.7%) but **0/5 origins** improve. "
        "Cinnatropin 10 alone is ~48% of TS deterioration. |\n",
        "| F2B shrunk Human reliability | **REJECT** | "
        "Human WMAPE 36.56 → **45.15** (−23.5% rel). 0/5 origins improve; win rate 22%; "
        "bias 752 → 2592. Paglino 10 WMAPE explodes (~55 → ~451). Shrinkage did **not** "
        "remove F1-style high-volume failure. |\n",
        "| F2C | **not run** | F2B is REJECT; combination not forced. |\n",
        "| F2D | **deferred** | Matched Human-adjustment still not implemented. |\n\n",
        "**Not ready** for lifecycle/price/commercial features until Human reliability is "
        "either dropped from the residual-XGB recipe or redesigned with a different target "
        "(counts-only, or explicit regime variables — not sparse bias levels).\n",
        "\n## Canonical F0 used by F2\n\n",
        _md_table(canon_sum) if canon_sum is not None else "_missing f0_canonical.csv_\n",
        "\nSee [f2_feature_design.md](f2_feature_design.md) for why locked freeze-time "
        "WMAPEs may differ from the currently reproduced frozen backtest.\n",
        "\n## Scoreboard\n\n",
        _md_table(overall, max_rows=12) if overall is not None else "_missing overall.csv_\n",
        "\n## Stop-condition answers\n\n",
    ]
    sections.extend(_answers(overall if overall is not None else pd.DataFrame(), classifications))

    by_o = report.get("by_origin")
    if by_o is None:
        by_o = _read(out_dir, "by_origin.csv")
    sections += ["\n## By origin\n\n", _md_table(by_o, max_rows=30) if by_o is not None else ""]

    by_h = report.get("by_horizon_bucket")
    if by_h is None:
        by_h = _read(out_dir, "by_horizon_bucket.csv")
    sections += ["\n## By horizon bucket\n\n", _md_table(by_h, max_rows=24) if by_h is not None else ""]

    conc = report.get("error_concentration")
    if conc is None:
        conc = _read(out_dir, "error_concentration.csv")
    sections += ["\n## Error concentration\n\n", _md_table(conc) if conc is not None else ""]

    watch = report.get("watchlist")
    if watch is None:
        watch = _read(out_dir, "watchlist_and_top.csv")
    if watch is not None:
        wl = watch.loc[watch["direction"] == "watchlist"]
        sections += ["\n## High-volume watchlist (F1 problem SKUs)\n\n", _md_table(wl, max_rows=40)]
        det = watch.loc[watch["direction"] == "deterioration"]
        sections += ["\n## Top deteriorators\n\n", _md_table(det, max_rows=20)]

    train = report.get("train_diagnostics")
    if train is None:
        train = _read(out_dir, "train_diagnostics.csv")
    sections += [
        "\n## Training coverage (by origin)\n\n",
        _md_table(train, max_rows=25) if train is not None else "",
        "\nHuman experiments train on full `budget_universe` history "
        "(`target_date < origin`). F2B does not require a TS twin.\n",
    ]

    skip = _read(out_dir, "f2c_skip.csv")
    if skip is not None:
        sections += ["\n## F2C skip\n\n", _md_table(skip)]

    sections += [
        "\n## Promotion notes\n\n",
        "Verdicts use the framework in `docs/f2_feature_design.md` "
        "(not test-set feature tuning). F2D (matched Human-adjustment) remains deferred.\n",
        "\nSee also: [forecasting_findings.md](forecasting_findings.md), "
        "[f1_feature_audit.md](f1_feature_audit.md).\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
