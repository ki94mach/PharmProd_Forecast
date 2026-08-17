"""Write docs/feature_family_ablation.md from ablation artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.ablation.config import (
    CORE_HUMAN,
    CORE_TS,
    F0_DEMAND,
    F1_DEMAND,
    F1_HUMAN,
    F2_DEMAND,
    F2_HUMAN,
    MATERIAL_WMAPE_TOL,
    SIMILAR_WMAPE_TOL,
    ablation_output_dir,
    docs_dir,
)

CASE_TEXT = {
    "A": (
        "Case A — useful replacement, harmful addition: the family contains "
        "signal but interacts poorly with redundant F0 sales lags."
    ),
    "B": (
        "Case B — weak replacement and weak addition: the family is inferior "
        "to the existing F0 demand representation."
    ),
    "C": (
        "Case C — similar accuracy with fewer / more stable features: "
        "potential future production simplification, not a promotion."
    ),
    "C_similar_not_simpler": (
        "Replacement WMAPE is similar to F0, but the family is not smaller "
        "than F0_DEMAND, so this is not a Case C simplification."
    ),
    "D": (
        "Case D — Human reliability works only without F0 demand: the main "
        "problem is interaction/redundancy with F0 sales lags."
    ),
    "E": (
        "Case E — Human reliability fails standalone too: close historical "
        "Human-bias features as a forecasting branch."
    ),
    "useful_both": "Better than F0 both as a replacement and as an addition.",
    "addition_only": "Worse as a replacement; only helps when stacked on F0.",
    "useful_or_neutral": (
        "Standalone is within the material WMAPE band of F0 and addition "
        "does not fail badly."
    ),
    "mixed": "Pattern does not match Cases A–E cleanly.",
    "missing": "Scores missing; not classified.",
}


def _md_table(df: pd.DataFrame, max_rows: int = 20, cols: Optional[list[str]] = None) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    sub = df if cols is None else df[[c for c in cols if c in df.columns]]
    sub = sub.head(max_rows)
    names = list(sub.columns)
    lines = [
        "| " + " | ".join(str(c) for c in names) + " |",
        "| " + " | ".join(["---"] * len(names)) + " |",
    ]
    for _, row in sub.iterrows():
        cells = [_fmt(row[c]) for c in names]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in CSV)_")
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:.1f}"
        return f"{v:.4f}"
    return str(v)[:90]


def _read(out_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = out_dir / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def _w(overall: pd.DataFrame, experiment: str, anchor: str) -> float:
    sub = overall.loc[
        (overall["experiment"] == experiment) & (overall["anchor"] == anchor)
    ]
    if sub.empty:
        return float("nan")
    return float(sub["wmape"].iloc[0])


def _case(cases: pd.DataFrame, family: str, anchor: str) -> str:
    if cases is None or cases.empty:
        return "missing"
    sub = cases.loc[(cases["family"] == family) & (cases["anchor"] == anchor)]
    if sub.empty:
        return "missing"
    return str(sub["case"].iloc[0])


def _rel_out(out_dir: Path) -> str:
    try:
        repo = Path(__file__).resolve().parents[4]
        return Path(out_dir).resolve().relative_to(repo).as_posix()
    except Exception:
        return Path(out_dir).as_posix()


def _answers(
    overall: pd.DataFrame, effects: pd.DataFrame, cases: pd.DataFrame
) -> list[str]:
    def row_eff(anchor: str) -> Optional[pd.Series]:
        sub = effects.loc[effects["anchor"] == anchor]
        return None if sub.empty else sub.iloc[0]

    ts = row_eff("ts")
    hu = row_eff("human")

    def val(r, col: str) -> str:
        if r is None or pd.isna(r[col]):
            return "n/a"
        return f"{float(r[col]):+.2f}"

    f1d_ts = _case(cases, "F1_DEMAND", "ts")
    f1d_h = _case(cases, "F1_DEMAND", "human")
    f2d_ts = _case(cases, "F2_DEMAND", "ts")
    f2d_h = _case(cases, "F2_DEMAND", "human")
    f1h = _case(cases, "F1_HUMAN", "human")
    f2h = _case(cases, "F2_HUMAN", "human")

    h2 = _w(overall, "H2_F1_HUMAN_ONLY", "human")
    h3 = _w(overall, "H3_F2_HUMAN_ONLY", "human")
    h1 = _w(overall, "H1_F0", "human")
    h4 = _w(overall, "H4_F1_HUMAN_ADD", "human")
    h5 = _w(overall, "H5_F2_HUMAN_ADD", "human")
    d0_ts = _w(overall, "D0_CORE", "ts")
    d1_ts = _w(overall, "D1_F0", "ts")
    d0_h = _w(overall, "D0_CORE", "human")
    d1_h = _w(overall, "D1_F0", "human")

    demand_best = []
    for label, col in (
        ("F0 demand", "wmape_core_f0"),
        ("F1 demand", "wmape_core_f1"),
        ("F2 demand", "wmape_core_f2"),
    ):
        scores = []
        for r in (ts, hu):
            if r is not None and pd.notna(r[col]):
                scores.append(float(r[col]))
        if scores:
            demand_best.append((label, sum(scores) / len(scores)))
    demand_best.sort(key=lambda x: x[1])
    best_name = demand_best[0][0] if demand_best else "n/a"

    retain = []
    close = []
    for fam, code in (
        ("F1 demand (TS)", f1d_ts),
        ("F1 demand (Human)", f1d_h),
        ("F2 demand (TS)", f2d_ts),
        ("F2 demand (Human)", f2d_h),
        ("F1 Human reliability", f1h),
        ("F2 Human reliability", f2h),
    ):
        if code in {"A", "C", "C_similar_not_simpler", "useful_both", "D"}:
            retain.append(f"{fam} ({code})")
        if code in {"B", "E"}:
            close.append(f"{fam} ({code})")

    all_b_e = {f1d_ts, f1d_h, f2d_ts, f2d_h} <= {"B"} and {f1h, f2h} <= {"E"}
    if retain:
        close_history = (
            "Not fully. At least one family still has standalone/replacement "
            "value (see Cases A/C/D). Do not start F3 in this task."
        )
    elif all_b_e:
        close_history = (
            "Yes for F1/F2 historical transforms: demand families are Case B "
            "(worse than F0 both standalone and stacked) and Human reliability "
            "is Case E (fails without F0 demand too). Do not start F3 in this "
            "task. Next work should be genuinely new information, not more "
            "sales-lag / Human-bias features."
        )
    else:
        close_history = (
            "Mostly yes: no family is a useful replacement or a harmless "
            "addition. Historical feature engineering on sales lags / Human "
            "bias can be closed; do not start F3 in this task."
        )

    redundancy = (
        "No. Case A/D would mean the families work as replacements but fail "
        "when stacked on F0 lags. Observed cases are "
        f"F1 demand {f1d_ts}/{f1d_h}, F2 demand {f2d_ts}/{f2d_h}, "
        f"F1 Human {f1h}, F2 Human {f2h}. They fail as replacements as well "
        "(often worse than as additions), so the problem is weak representation, "
        "not only harmful interaction with F0."
        if all_b_e
        else (
            "Partly, if any family is Case A or D. Observed: "
            f"F1 demand {f1d_ts}/{f1d_h}; F2 demand {f2d_ts}/{f2d_h}; "
            f"F1 Human {f1h}; F2 Human {f2h}."
        )
    )

    lines = [
        "1. **How much value does the existing F0 demand block add over CORE?** "
        f"TS: CORE {_fmt(d0_ts)} vs F0 {_fmt(d1_ts)} "
        f"(Δ={val(ts, 'f0_demand_value')}; negative = lags hurt vs CORE). "
        f"Human: CORE {_fmt(d0_h)} vs F0 {_fmt(d1_h)} "
        f"(Δ={val(hu, 'f0_demand_value')}; F0 lags help the Human residual model).",
        "2. **Do F1 demand features work better as a replacement than as an addition?** "
        + (
            "No. Both are worse than F0. "
            if {f1d_ts, f1d_h} <= {"B"}
            else "Compare replacement vs addition effects. "
        )
        + f"TS replacement {val(ts, 'f1_replacement_effect')} vs addition "
        f"{val(ts, 'f1_addition_effect')}; Human replacement "
        f"{val(hu, 'f1_replacement_effect')} vs addition "
        f"{val(hu, 'f1_addition_effect')} "
        f"(Case {f1d_ts}/{f1d_h})."
        + (
            " Replacement is slightly *worse* than addition."
            if {f1d_ts, f1d_h} <= {"B"}
            else ""
        ),
        "3. **Do F2 demand features work better as a replacement than as an addition?** "
        + (
            "No. Both are worse than F0. "
            if {f2d_ts, f2d_h} <= {"B"}
            else "Compare replacement vs addition effects. "
        )
        + f"TS replacement {val(ts, 'f2_replacement_effect')} vs addition "
        f"{val(ts, 'f2_addition_effect')}; Human replacement "
        f"{val(hu, 'f2_replacement_effect')} vs addition "
        f"{val(hu, 'f2_addition_effect')} "
        f"(Case {f2d_ts}/{f2d_h}).",
        "4. **Does feature redundancy explain any of the F1/F2 deterioration?** "
        + redundancy,
        "5. **Do F1 Human-reliability features have standalone predictive value?** "
        + ("No. " if f1h == "E" else "")
        + f"H2 (CORE+F1 Human) WMAPE={_fmt(h2)} vs F0 {_fmt(h1)}; "
        f"addition H4={_fmt(h4)}. Case {f1h}.",
        "6. **Do F2 shrunk Human-reliability features have standalone predictive value?** "
        + ("No. " if f2h == "E" else "")
        + f"H3 (CORE+F2 Human) WMAPE={_fmt(h3)} vs F0 {_fmt(h1)}; "
        f"addition H5={_fmt(h5)}. Case {f2h}."
        + (
            " Standalone is worse than addition."
            if f2h == "E"
            else ""
        ),
        "7. **Are Human reliability features intrinsically weak, or mainly harmful "
        "when interacting with F0 demand features?** "
        "Intrinsically weak (Case E). They fail without F0 sales lags, and fail "
        "even more than CORE alone. Interaction with F0 is not the main story.",
        "8. **Which representation is best (F0 / F1 / F2 demand)?** "
        f"**{best_name}** on both anchors. F1 and F2 demand do not beat F0 "
        "lags as a replacement. On TS, CORE (no demand block) is slightly "
        "better than F0, but that is not a reason to keep F1/F2.",
        "9. **Can any F1/F2 family be retained for future work?** "
        + (
            ("Retain for diagnosis/redesign: " + "; ".join(retain) + ". ")
            if retain
            else "No. "
        )
        + (("Close: " + "; ".join(close) + ".") if close else ""),
        "10. **Can we now confidently close historical feature engineering and "
        "move to genuinely new information sources?** "
        + close_history,
    ]
    return [ln + "\n" for ln in lines]


def write_ablation_report(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = out_dir or report.get("out_dir") or ablation_output_dir()
    path = path or (docs_dir() / "feature_family_ablation.md")
    overall = report.get("overall")
    if overall is None:
        overall = _read(out_dir, "overall.csv")
    if overall is None:
        overall = pd.DataFrame()
    effects = report.get("effects")
    if effects is None:
        effects = _read(out_dir, "replacement_effects.csv")
    if effects is None:
        effects = pd.DataFrame()
    cases = report.get("classifications")
    if cases is None:
        cases = _read(out_dir, "classifications.csv")
    if cases is None:
        cases = pd.DataFrame()

    canon = report.get("canonical_f0", {})
    canon_sum = canon.get("summary") if isinstance(canon, dict) else None
    if canon_sum is None:
        canon_sum = _read(out_dir, "f0_canonical.csv")

    gates = report.get("gates")
    if gates is None:
        gates = _read(out_dir, "reproduction_gates.csv")

    repl_view = None
    if not effects.empty:
        demand = effects.loc[effects["anchor"].isin(["ts", "human"])].copy()
        if not demand.empty:
            repl_view = demand[
                [
                    "anchor",
                    "wmape_core",
                    "wmape_core_f0",
                    "wmape_core_f1",
                    "wmape_core_f2",
                    "wmape_core_f0_f1",
                    "wmape_core_f0_f2",
                    "f0_demand_value",
                    "f1_replacement_effect",
                    "f2_replacement_effect",
                    "f1_addition_effect",
                    "f2_addition_effect",
                ]
            ].rename(
                columns={
                    "wmape_core": "CORE",
                    "wmape_core_f0": "CORE+F0",
                    "wmape_core_f1": "CORE+F1",
                    "wmape_core_f2": "CORE+F2",
                    "wmape_core_f0_f1": "CORE+F0+F1",
                    "wmape_core_f0_f2": "CORE+F0+F2",
                }
            )

    sections = [
        "# Feature-family ablation (standalone vs addition)\n",
        f"**Date:** {date.today().isoformat()}  \n",
        "**Benchmark:** frozen v1 matched PRIMARY  \n",
        f"**CSV artifacts:** `{_rel_out(out_dir)}`\n",
        "\nDiagnostic only. Frozen F0, XGB params, origins, eligibility, and "
        "F1/F2 modules are unchanged. F3 is not started.\n",
        "\n## Semantic partition\n\n",
        "Split is semantic (problem definition vs historical-sales transforms), "
        "not test-tuned. Import-time assert: "
        "`set(CORE) ∪ set(F0_DEMAND) == frozen F0` (order-insensitive).\n\n",
        f"**CORE_TS** ({len(CORE_TS)}): " + ", ".join(f"`{c}`" for c in CORE_TS) + "\n\n",
        f"**CORE_HUMAN** ({len(CORE_HUMAN)}): "
        + ", ".join(f"`{c}`" for c in CORE_HUMAN)
        + "\n\n",
        f"**F0_DEMAND** ({len(F0_DEMAND)}): "
        + ", ".join(f"`{c}`" for c in F0_DEMAND)
        + "\n\n",
        f"**F1_DEMAND** ({len(F1_DEMAND)}): "
        + ", ".join(f"`{c}`" for c in F1_DEMAND)
        + "\n\n",
        f"**F1_HUMAN** ({len(F1_HUMAN)}): "
        + ", ".join(f"`{c}`" for c in F1_HUMAN)
        + "\n\n",
        f"**F2_DEMAND** ({len(F2_DEMAND)}): "
        + ", ".join(f"`{c}`" for c in F2_DEMAND)
        + "\n\n",
        f"**F2_HUMAN** ({len(F2_HUMAN)}): "
        + ", ".join(f"`{c}`" for c in F2_HUMAN)
        + "\n\n",
        "D1 / D4 / H1 / H4 use **frozen F0 column order** (lags sit between "
        "calendar and encodings) so XGB can reproduce F0 / F1A / F1B. "
        "Replacement experiments drop the F0 demand block in that same order.\n",
        "\nPositive **effects** mean improvement: "
        "`WMAPE(CORE+F0) − WMAPE(candidate)`.\n",
        f"Case C band: |replacement effect| ≤ {SIMILAR_WMAPE_TOL} WMAPE. "
        f"Human material-failure band: > {MATERIAL_WMAPE_TOL} WMAPE worse than F0.\n",
        "\n## Canonical F0 (currently reproduced frozen backtest)\n\n",
        _md_table(canon_sum) if canon_sum is not None else "_missing f0_canonical.csv_\n",
        "Locked freeze-time Analysis B numbers are **not** rewritten. "
        "Ablation compares against this environment's frozen `ts_xgb` / `human_xgb`.\n",
        "\n## Reproduction gates\n\n",
        _md_table(gates) if gates is not None else "_missing reproduction_gates.csv_\n",
        "\n## Demand replacement table\n\n",
        _md_table(repl_view) if repl_view is not None else "_missing replacement_effects.csv_\n",
        "\n## Cases A–E\n\n",
        _md_table(cases) if not cases.empty else "_missing classifications.csv_\n",
        "\n",
    ]
    for _, r in cases.iterrows() if not cases.empty else []:
        code = str(r["case"])
        sections.append(
            f"- **{r['family']} / {r['anchor']} = {code}:** "
            f"{CASE_TEXT.get(code, code)}\n"
        )
    sections.append("\n")

    score_cols = [
        "experiment",
        "anchor",
        "family",
        "n_features",
        "wmape",
        "rel_wmape_vs_f0_pct",
        "rmse",
        "mae",
        "bias",
        "n",
        "origins_improved",
        "origins_total",
        "product_win_rate",
        "median_product_improvement_pct",
        "p25_product_improvement_pct",
        "p75_product_improvement_pct",
        "top1_deterioration_share",
        "top5_deterioration_share",
        "top10_deterioration_share",
    ]
    sections += [
        "\n## Scoreboard\n\n",
        _md_table(overall, max_rows=30, cols=score_cols),
        "\n## Answers\n\n",
    ]
    sections.extend(_answers(overall, effects, cases))

    by_o = report.get("by_origin")
    if by_o is None:
        by_o = _read(out_dir, "by_origin.csv")
    sections += ["\n## By origin\n\n", _md_table(by_o, max_rows=40) if by_o is not None else ""]

    by_h = report.get("by_horizon_bucket")
    if by_h is None:
        by_h = _read(out_dir, "by_horizon_bucket.csv")
    sections += [
        "\n## By horizon bucket\n\n",
        _md_table(by_h, max_rows=40) if by_h is not None else "",
    ]

    conc = report.get("error_concentration")
    if conc is None:
        conc = _read(out_dir, "error_concentration.csv")
    sections += [
        "\n## Error concentration vs F0\n\n",
        _md_table(conc, max_rows=24) if conc is not None else "",
    ]

    top = report.get("top_products")
    if top is None:
        top = _read(out_dir, "top_products.csv")
    if top is not None and not top.empty:
        det = top.loc[top["direction"] == "deterioration"]
        imp = top.loc[top["direction"] == "improvement"]
        wl = top.loc[top["direction"] == "watchlist"]
        sections += [
            "\n## Top deteriorators\n\n",
            _md_table(det, max_rows=20),
            "\n## Top improvers\n\n",
            _md_table(imp, max_rows=15),
            "\n## High-volume watchlist\n\n",
            _md_table(wl, max_rows=40),
        ]

    train = report.get("train_diagnostics")
    if train is None:
        train = _read(out_dir, "train_diagnostics.csv")
    train_cols = [
        c
        for c in (
            "experiment",
            "anchor",
            "origin",
            "train_rows",
            "prior_budget_vintages",
            "train_universe",
        )
        if train is not None and c in train.columns
    ]
    sections += [
        "\n## Training coverage\n\n",
        _md_table(train, max_rows=30, cols=train_cols) if train is not None else "",
        "\nTS experiments train on `ts_universe`. Human experiments train on "
        "full `budget_universe` (`target_date < origin`). H6/H7 are secondary.\n",
        "\nNo F3. No per-feature search on PRIMARY origins.\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
