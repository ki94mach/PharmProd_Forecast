"""Run Optuna studies for M1A (TS) and M1B (Human).

SQLite storage → resume-safe. Runs to a TOTAL of N_TRIALS per study (not
N_TRIALS additional trials after a restart).

No PRIMARY data is involved. Call evaluate_tuned after this returns.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd

from pkg.benchmark.dataset import load_benchmark, prep_lags
from pkg.research.tuning.config import (
    EXPECTED_N,
    EXPECTED_N_ORIGINS,
    F0_FEATURES,
    N_JOBS_OPTUNA,
    N_TRIALS,
    OPTUNA_SEED,
    STUDY_NAME_HUMAN,
    STUDY_NAME_TS,
    TRAIN_UNIVERSE,
    m1_output_dir,
    optuna_db_url,
)
from pkg.research.tuning.folds import InsufficientFoldsError, build_inner_folds
from pkg.research.tuning.objective import make_objective

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


def _remaining_trials(study: optuna.Study) -> int:
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    return max(0, N_TRIALS - len(completed))


def run_anchor_study(
    anchor: str,
    study_name: str,
    universe: pd.DataFrame,
    *,
    db_url: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Run (or resume) one Optuna study for ``anchor``.

    Returns dict with best_params, frozen_n_estimators, and trial DataFrame.
    Raises InsufficientFoldsError if pre-PRIMARY history is too short.
    """
    features = F0_FEATURES[anchor]
    train_univ_key = TRAIN_UNIVERSE[anchor]

    # Build inner folds — raises if < MIN_INNER_FOLDS
    print(f"  [{anchor}] Building inner folds from pre-PRIMARY origins …")
    folds = build_inner_folds(universe, anchor)
    inner_origins = [f.origin for f in folds]
    print(f"  [{anchor}] {len(folds)} eligible inner fold(s): {inner_origins}")

    objective_fn = make_objective(anchor, features, folds)

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    storage = optuna.storages.RDBStorage(db_url)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    remaining = _remaining_trials(study)
    if remaining > 0:
        print(f"  [{anchor}] Running {remaining} trial(s) (total target: {N_TRIALS}) …")
        study.optimize(objective_fn, n_trials=remaining, n_jobs=N_JOBS_OPTUNA, show_progress_bar=False)
    else:
        print(f"  [{anchor}] Study already has {N_TRIALS} completed trials — skipping.")

    best = study.best_trial
    best_search_params = best.params

    # frozen_n_estimators: median best_iteration+1 from that trial's inner folds
    import json as _json
    best_iter_by_origin_raw = best.user_attrs.get("best_iteration_by_origin", "{}")
    best_iter_by_origin: dict = _json.loads(best_iter_by_origin_raw)
    best_iters = list(best_iter_by_origin.values())
    frozen_n_estimators = int(round(float(np.median(best_iters)))) if best_iters else 200

    frozen_params = {
        **best_search_params,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "frozen_n_estimators": frozen_n_estimators,
    }

    output_meta = {
        "study_name": study_name,
        "anchor": anchor,
        "n_trials": N_TRIALS,
        "best_trial_number": int(best.number),
        "best_inner_pooled_wmape": float(best.value),
        "inner_origins": inner_origins,
        "frozen_n_estimators": frozen_n_estimators,
        "seed": OPTUNA_SEED,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **best_search_params,
    }

    fname = f"{'ts' if anchor == 'ts' else 'human'}_best_params.json"
    (out_dir / fname).write_text(json.dumps(output_meta, indent=2), encoding="utf-8")
    print(f"  [{anchor}] Best params written to {out_dir / fname}")

    # Full trials CSV
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row: dict = {"trial_number": t.number, "value": t.value}
        row.update(t.params)
        row.update({f"ua_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    csv_name = f"{'ts' if anchor == 'ts' else 'human'}_trials.csv"
    trials_df.to_csv(out_dir / csv_name, index=False)
    print(f"  [{anchor}] Trials CSV written to {out_dir / csv_name}")

    return {
        "anchor": anchor,
        "study": study,
        "best_params": frozen_params,
        "frozen_n_estimators": frozen_n_estimators,
        "best_inner_wmape": float(best.value),
        "inner_origins": inner_origins,
        "n_inner_folds": len(folds),
        "trials_df": trials_df,
    }


def run_all_optuna_studies(
    *,
    out_dir: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> dict[str, Any]:
    """Load benchmark, build inner folds, run both studies.

    Returns {anchor: result_dict} for each completed study.
    """
    out_dir = out_dir or m1_output_dir()
    db_url = db_url or optuna_db_url()

    print("Loading frozen benchmark …")
    ds = load_benchmark()
    ts_univ = prep_lags(ds.ts_universe)
    budget_univ = prep_lags(ds.budget_universe)

    results: dict[str, Any] = {}

    for anchor, universe, study_name in [
        ("ts", ts_univ, STUDY_NAME_TS),
        ("human", budget_univ, STUDY_NAME_HUMAN),
    ]:
        print(f"\n=== M1 Optuna: anchor={anchor}, study={study_name} ===")
        try:
            res = run_anchor_study(
                anchor,
                study_name,
                universe,
                db_url=db_url,
                out_dir=out_dir,
            )
            results[anchor] = res
            print(
                f"  [{anchor}] Best inner pooled WMAPE = {res['best_inner_wmape']:.4f} "
                f"(trial #{res['best_params']['frozen_n_estimators']} trees)"
            )
        except InsufficientFoldsError as exc:
            print(f"  [{anchor}] STOP — {exc}")
            results[anchor] = {"error": str(exc), "anchor": anchor}

    return results
