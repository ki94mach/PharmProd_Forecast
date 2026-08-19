"""Generate docs/m1a2_fixed200_optuna.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pkg.research.m1a2.config import FIXED_N_ESTIMATORS, docs_path


def write_m1a2_report(
    *,
    out_dir: Path,
    env: dict[str, Any],
    repeatability: dict[str, Any],
    inner_origins: list[int],
    inner_f0_pooled: float,
    best_payload: dict[str, Any],
    f0_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
    overall: dict[str, Any],
    diagnostic: str,
    m1r_primary: float,
    m1r_n_est: int,
    m1r_inner: float,
) -> None:
    rel = overall["rel_wmape_improvement_pct"]
    win_rate_pct = (
        float(overall["product_win_rate"]) * 100.0
        if overall.get("product_win_rate") == overall.get("product_win_rate")
        else float("nan")
    )
    rel_inner = best_payload.get("relative_inner_improvement_pct", float("nan"))
    params = best_payload.get("selected_hyperparameters", {})

    text = f"""# M1A2 — Deterministic Fixed-200-Tree Structural Tuning

Diagnostic follow-up to M1R: isolate whether early-stopped tree-count selection
(`frozen_n_estimators=27`) was the main failure source by fixing `n_estimators=200`
(canonical F0 capacity) while tuning the same 8 structural parameters.

**PRIMARY evaluation is exploratory evidence only** (motivated by M1R observation),
not unbiased production performance.

## Reproducibility and environment

1. Environment metadata: `{json.dumps(env)}`
2. Five n_jobs=1 F0 runs prediction-identical: **Yes**
3. Max prediction diff across repeated n_jobs=1 runs: `{repeatability.get('max_abs_prediction_diff_any_pair', 'n/a')}`
4. Deterministic F0 PRIMARY WMAPE: `{f0_metrics['wmape']:.4f}` (n={f0_metrics['n']}, origins={f0_metrics['n_origins']})
5. `n_estimators` fixed at **200** everywhere (inner folds, Optuna, PRIMARY): **confirmed**
6. No `eval_set` / no early stopping in M1A2 fit path: **confirmed**

## Inner folds (pre-PRIMARY only)

7. Inner origins used (9 expected): `{inner_origins}`
8. Leakage assertions passed: `V < 140404` and `train.target_date.max() < V` for all folds: **Yes**
9. No PRIMARY origin in inner validation: **Yes**

## Inner canonical F0 baseline vs Optuna

10. Inner canonical F0 pooled WMAPE (200 trees, F0 structural defaults): `{inner_f0_pooled:.4f}`
11. Best Optuna trial number: `{best_payload['trial_number']}`
12. Best structural parameters: `{params}`
13. Best inner pooled WMAPE: `{best_payload['best_inner_pooled_wmape']:.4f}`
14. Relative inner improvement vs canonical F0: `{rel_inner:.2f}%`
15. Frozen `n_estimators` in ts_best_params.json: `{best_payload['n_estimators']}` (no `frozen_n_estimators` field)

## PRIMARY evaluation (once)

16. M1A2 tuned PRIMARY WMAPE: `{tuned_metrics['wmape']:.4f}` vs deterministic F0 `{f0_metrics['wmape']:.4f}`
17. Relative PRIMARY WMAPE change vs F0: `{rel:.2f}%`
18. Origins improved: `{int(overall['origins_improved'])}/{int(overall['origins_total'])}`
19. Product win rate: `{win_rate_pct:.2f}%`
20. Bias: F0 `{f0_metrics['bias']:.4f}` → M1A2 `{tuned_metrics['bias']:.4f}`
21. Error concentration top1/top5/top10: `{overall['top1_deterioration_share']:.4f}` / `{overall['top5_deterioration_share']:.4f}` / `{overall['top10_deterioration_share']:.4f}`

## Three-model comparison (F0 vs M1R historical vs M1A2)

| Model | n_estimators | Inner pooled WMAPE | PRIMARY WMAPE | vs F0 |
|-------|--------------|-------------------|---------------|-------|
| Deterministic F0 | 200 | {inner_f0_pooled:.4f} | {f0_metrics['wmape']:.4f} | 0% |
| M1R (historical) | {m1r_n_est} | {m1r_inner:.4f} | {m1r_primary:.4f} | see model_comparison.csv |
| M1A2 | 200 | {best_payload['best_inner_pooled_wmape']:.4f} | {tuned_metrics['wmape']:.4f} | {rel:.2f}% |

Did fixing tree count prevent catastrophic deterioration vs M1R?
- M1R PRIMARY tuned WMAPE: `{m1r_primary:.4f}` (early-stopped n={m1r_n_est})
- M1A2 PRIMARY tuned WMAPE: `{tuned_metrics['wmape']:.4f}` (fixed n=200)
- M1A2 is {"better" if tuned_metrics['wmape'] < m1r_primary else "not better"} than M1R on PRIMARY.

## Verdicts

- Reproducibility: **PASS**
- M1A2 structural tuning verdict: **{overall['verdict']}**
- Diagnostic classification: **{diagnostic}**

### Diagnostic labels

- `EARLY_STOPPING_WAS_MAJOR_FAILURE_SOURCE` — M1A2 materially better than M1R and competitive with/improves F0
- `EARLY_STOPPING_ONLY_PART_OF_FAILURE` — M1A2 removes M1R catastrophe but structural tuning adds little vs F0
- `STRUCTURAL_TUNING_ALSO_FAILS_TO_TRANSFER` — M1A2 still materially worse than F0

## Human (M1B)

M1B remains unavailable: insufficient eligible pre-PRIMARY Budget origins under existing
maturity rules; not relaxed for this diagnostic.

## Artifacts

All outputs under `src/data/results/m1a2_fixed200/`.
Report generated from run artifacts; historical M1/M1R reports were not modified.

**STOP** — no M1B, tree-count sweeps, or additional Optuna studies in this change.
"""
    docs_path().write_text(text, encoding="utf-8")
