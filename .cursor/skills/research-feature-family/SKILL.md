---
name: research-feature-family
description: >-
  Add a new frozen-benchmark research feature family (F3B, F4, price,
  inventory, commercial) using pkg.research.harness, following F2/F3A.
  Use when starting a new research feature family, F3B/F4, price/inventory/
  commercial experiments, or anything “like F2/F3A”.
---

# Research feature family

New families reuse the F2/F3A residual-backtest loop via Template Method
(`pkg.research.harness`) and Strategy (family enricher, fillna, audit, verdict,
report). F3A is the smallest template.

## Frozen contracts (do not change)

- Never mutate freeze files under `src/data/benchmarks/`.
- Never change `XGB_PARAMS`, F0 feature lists, or locked Analysis B contracts.
- Do not tune XGB on PRIMARY. Do not “fix” WMAPE drift by changing features or params.
- Never overwrite freeze / F0 / F1 / F2 / F3A artifacts.
- Unique outputs only: `src/data/results/{family}/` and `docs/{family}_*.md`.
- Stop after that family. Do not start the next family in the same change.

## Checklist

1. New `src/pkg/research/features/{family}.py` — point-in-time, frozen sales (or the family’s frozen source) only, no SQL.
2. New `src/pkg/research/{family}/` — `config.py` (feature tuples, fillna / never-fillna), enricher, optional audit, `evaluate.py`, `report.py`.
3. Wire through `FamilySession` / `harness.run_family` with `ExperimentSpec` rows. Do not copy the F2/F3A loop.
4. CLI `python -m pkg.research.evaluate_{family}` staying on the frozen matched PRIMARY panel.
5. Reproduction: canonical F0 WMAPEs (TS 38.2848, Human 36.5602, n=1877, 5 origins) plus family WMAPE gates within `0.05` of the just-written report.
6. Unit tests for PIT, fillna policy, and `ExperimentSpec` feature lists.

## Harness API

- `pkg.research.harness.spec` — `ExperimentSpec`, `FamilyConfig`
- `pkg.research.harness.dataset` — `copy_dataset`, `resolve_origin_col`, `enrich_dataset` (in-memory; never writes parquet)
- `pkg.research.harness.residual` — `make_residual_model(..., fillna_extra, never_fillna)`
- `pkg.research.harness.gates` — freeze checksums, canonical F0, WMAPE gates
- `pkg.research.harness.metrics` — pair tables vs control, `assert_same_eval_rows`
- `pkg.research.harness.run` — `FamilySession`, `run_family`
- `pkg.research.harness.report` — `md_table`, `read_csv`

Family-specific pieces stay in the family package: enricher, fillna policy, pre-model audit, verdict, narrative report. Keep F2C-style skip/orchestration in the family evaluate module when the default `run_family` loop is too blunt.

Ablation/F1 keep their own experiment logic; only shared helpers live in the harness.
