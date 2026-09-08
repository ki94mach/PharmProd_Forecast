---
name: data-layout
description: >-
  Canonical Forecast repo data directory map under src/data/. Use when looking
  for backfill artifacts, V3A screening runs, research results, frozen
  benchmarks, external sources, results_v2, or deciding where to write new
  experiment outputs. Also use when unifying paths, fixing dual data roots, or
  answering “where is the X file?”. Also covers results_v2_1 / backfill v2.1.
---

# Data layout (`src/data/`)

**Canonical root:** `src/data/` (gitignored). Do **not** write new artifacts under
repo-root `data/` — that tree is legacy/stray only.

When locating or writing data files, prefer this map over guessing. Package
code defaults should resolve under `src/data/` via helpers such as
`default_screening_root()`, `default_backfill_root()`, `default_results_v2_root()`,
`default_results_v21_root()`.

## Quick map

| Purpose | Path | Written by |
|---------|------|------------|
| Frozen research benchmark | `src/data/benchmarks/v1/` | `python -m pkg.benchmark.freeze` |
| Historical TS backfill jobs | `src/data/backfill/{experiment_id}/{engine}/` | `python -m pkg.benchmark.backfill_runner` |
| V3A neural screening | `src/data/ts_v3a/screening/{experiment_id}/` | `python -m pkg.ts_v3a.screen` |
| V3A incomplete / resume | `src/data/ts_v3a/screening/.incomplete/{experiment_id}/` | same (checkpoint) |
| Research family outputs | `src/data/results/{family}/` | `pkg.research.*` / harness |
| V3A analysis tables / report | `src/data/results/ts_v3a_architecture_validation/` | `python -m pkg.ts_v3a.analysis report` |
| V2-vs-legacy eval | `src/data/results/ts_v2_backfill_eval/` | `python -m pkg.research.evaluate_v2_backfill` |
| V1 quarterly forecast CSVs | `src/data/results/{quarter}/` | production / legacy export |
| Standalone V2 run persistence | `src/data/results_v2/{quarter}/{run_id}/` | `pkg.ts_v2.persistence` |
| Standalone V2.1 run persistence | `src/data/results_v2_1/{quarter}/{run_id}/` | `pkg.ts_v2.persistence` (`ts_version=v2.1`) |
| External workbooks / maps | `src/data/external/{family}/` | manual / prepare steps |
| Pipeline scratch | `src/data/pipeline/` | pipeline helpers |
| Sales scratch | `src/data/sales/` | ad-hoc |

Narrative docs for families live in `docs/{family}_*.md` (often local-only;
`docs/*` is gitignored).

## Rules

1. **One root.** New code and agents write only under `src/data/…`.
2. **Never mutate** `src/data/benchmarks/` after freeze (see `AGENTS.md` and
   `research-feature-family` skill).
3. **Research isolation.** Each family writes `src/data/results/{family}/` only;
   do not overwrite F0/F1/F2/F3A artifacts.
4. **Screening vs analysis.** Raw immutable OOF lives under
   `src/data/ts_v3a/screening/`; derived tables/report go to
   `src/data/results/ts_v3a_architecture_validation/`. Both are intentional.
5. **Backfill naming.** Canonical directory is singular `backfill` (not
   `backfills`). Loaders may still accept legacy `src/data/backfills/` if present.
6. **Secrets.** `src/.env` and any `credentials.json` under `src/data/` stay
   local; never commit.

## Common lookups

**V2 MVP backfill forecasts (completed):**

`src/data/backfill/ts_mvp_backfill_1401Q1_1405Q2/v2/forecasts/{quarter}__{product}/forecast.csv`

Join keys vs V3A: `product`, `forecast_origin` (= V3A `origin`), `horizon`,
`target_date`. Columns include `forecast`, `raw_forecast`, `model`, `engine=v2`.

**V2.1 backfill (intermittency-gated Croston/TSB):**

`src/data/backfill/{experiment_id}/v2.1/forecasts/{quarter}__{product}/forecast.csv`

Engine name `v2.1`; standalone analysis runs also use
`src/data/results_v2_1/{quarter}/{run_id}/` (adds `candidate_eligibility.csv`,
`selection.csv`). Do not overwrite historical `…/v2/` artifacts.

**V3A architecture validation experiment:**

`src/data/ts_v3a/screening/20260908T103000Z_archval/`

**Frozen sales for MVP / V3A panel:**

`src/data/benchmarks/v1/raw/sales.parquet`

**Panel product list:**

`src/pkg/benchmark/universes/v3a_architecture_validation_products.csv`

## Legacy / do not use for new work

| Path | Status |
|------|--------|
| Repo-root `data/ts_v3a/` | Migrated → `src/data/ts_v3a/` |
| Repo-root `data/backfills/` | Unused; code default is `src/data/backfill/` |
| Repo-root `data/results/` | Stray (e.g. old Optuna DB); prefer `src/data/results/` |
| `src/data/backfills/` (plural) | Legacy empty alias; do not recreate |

If you find a path outside this map, update this skill when you add a durable
new artifact family.
