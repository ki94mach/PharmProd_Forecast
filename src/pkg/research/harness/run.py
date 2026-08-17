"""Template Method: load freeze, run specs, score vs control, assert freeze."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import BacktestResult
from pkg.research.harness.gates import (
    assert_freeze_unchanged,
    confirm_canonical_f0,
    freeze_checksums,
)
from pkg.research.harness.metrics import assert_same_eval_rows
from pkg.research.harness.residual import make_residual_model
from pkg.research.harness.spec import ExperimentSpec, FamilyConfig


class FamilySession:
    """Shared freeze/F0/backtest session. Family evaluate modules orchestrate specs."""

    def __init__(
        self,
        family: str,
        out_dir: Path,
        *,
        dataset: Optional[BenchmarkDataset] = None,
        verify_checksums: bool = False,
        fillna_extra: tuple[str, ...] = (),
        never_fillna: frozenset[str] = frozenset(),
        enrichers: Optional[dict] = None,
        model_name_prefix: str = "xgb",
    ) -> None:
        self.family = family
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ds = dataset or load_benchmark(verify_checksums=verify_checksums)
        self.freeze_before = freeze_checksums(self.ds)
        self.canon = confirm_canonical_f0(self.ds)
        self.f0_results: dict[str, BacktestResult] = self.canon["results"]
        self.fillna_extra = fillna_extra
        self.never_fillna = never_fillna
        self.enrichers = enrichers or {}
        self.model_name_prefix = model_name_prefix
        self.canon["summary"].to_csv(self.out_dir / "f0_canonical.csv", index=False)

    def f0(self, anchor: str) -> BacktestResult:
        return self.f0_results[anchor]

    def run(self, spec: ExperimentSpec) -> BacktestResult:
        f0 = self.f0(spec.anchor)
        if spec.use_frozen_adapter:
            frozen_name = "ts_xgb" if spec.anchor == "ts" else "human_xgb"
            result = backtest(
                frozen_name,
                dataset=self.ds,
                universe="matched",
                eligibility="primary",
            )
            assert_same_eval_rows(f0, result)
            return result

        ds = self.ds
        if spec.enrich:
            if spec.enrich not in self.enrichers:
                raise KeyError(
                    f"enricher {spec.enrich!r} not in {sorted(self.enrichers)}"
                )
            ds = self.enrichers[spec.enrich](self.ds)

        model = make_residual_model(
            spec.anchor,
            spec.features,
            fillna_extra=self.fillna_extra,
            never_fillna=self.never_fillna,
            name=f"{spec.anchor}_{self.model_name_prefix}_{self.family}",
        )
        result = backtest(
            model,
            dataset=ds,
            universe="matched",
            eligibility="primary",
            train_universe=spec.train_universe,
        )
        assert_same_eval_rows(f0, result)
        return result

    def finish(self) -> None:
        assert_freeze_unchanged(self.ds, self.freeze_before)


def run_family(config: FamilyConfig, *, verify_checksums: bool = False) -> dict:
    """Run every spec in ``config.experiments`` and return results keyed by name."""
    session = FamilySession(
        config.family,
        config.out_dir,
        verify_checksums=verify_checksums,
        fillna_extra=config.fillna_extra,
        never_fillna=config.never_fillna,
        enrichers=config.enrichers,
        model_name_prefix=config.model_name_prefix,
    )
    if config.pre_model is not None:
        config.pre_model(session)
    results: dict[str, BacktestResult] = {}
    for spec in config.experiments:
        results[spec.name] = session.run(spec)
    if config.post_score is not None:
        config.post_score(session, results)
    session.finish()
    return {
        "session": session,
        "results": results,
        "canonical_f0": session.canon,
        "out_dir": session.out_dir,
    }
