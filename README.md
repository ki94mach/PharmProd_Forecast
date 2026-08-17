# Forecast

Pharmaceutical sales time-series forecasting: loads historical sales from SQL Server, runs model selection (ARIMA, ETS, Prophet, LSTM), and writes department Excel workbooks under `src/data/results/`.

## Prerequisites

| Requirement             | Notes                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Windows**             | Windows auth (`SQL_AUTH=windows`) uses integrated security; SQL login also supported.                             |
| **Python 3.10**         | Matches `environment.yml`.                                                                                        |
| **Conda**               | Required — installs Python 3.10, Prophet/CmdStan, and the scientific stack.                                       |
| **ODBC**                | Windows: `{SQL Server}`. Linux: **FreeTDS** (`sudo dnf install freetds-libs`) or `ODBC Driver 18 for SQL Server`. |
| **Network / DB access** | Server `op-db1-srv`, database `DWOrchid`, read on `Flat_Fact_Sale`.                                               |
| **DB credentials**      | Windows: domain user with DB access. SQL login: set `SQL_AUTH=sql`, `SQL_USER`, `SQL_PASSWORD`.                   |

## Setup

### Conda (required)

Uses Python **3.10** from conda-forge (not system Python). Scientific packages come from conda; TensorFlow / Google Sheets extras from pip.

```bash
cd path/to/Forecast
conda env create -f environment.yml
conda activate forecast
python -m pip install -r requirements.txt   # TensorFlow / gspread extras
python -m pip install -e .                  # install pkg for imports (CLI + notebooks)
```

If the env already exists:

```bash
conda env update -f environment.yml --prune
conda activate forecast
python -m pip install -r requirements.txt
python -m pip install -e .
```

`conda env create` already installs the `pip:` block in `environment.yml`; re-running `requirements.txt` is safe and keeps extras explicit. `pip install -e .` makes `import pkg...` work from anywhere (including `notebooks/`).

### Environment variables

```powershell
copy .env.example .env
```

Edit `.env` — see [.env.example](.env.example). Set `ZERO_FORECAST_PRODUCTS` to a comma-separated list of English product names that should get zero forecasts.

### Data files (not in git)

Create or copy `src/data/`:

```
src/data/
  results/<quarter>/          # Output CSV + Excel (created by the tool)
  pipeline/<quarter>/           # Optional: <quarter>_pipeline.xlsx
  benchmarks/v1/                # Frozen research panels (python -m pkg.benchmark.freeze)
  credentials.json              # Optional: Google service account (Sheets upload)
```

Share sample quarter folders and templates with your colleague outside the repo.

### Google Sheets (optional)

1. Place a service account JSON at `src/data/credentials.json`.
2. Grant the service account access to the **Target 1** spreadsheet.
3. Use `pkg.google_sheet.GoogleSheet` from code (not wired in `main.py` today).

## Usage

From the `src` directory:

```powershell
conda activate forecast
cd src

# Full forecast
python main.py --qrt 1405Q1 --start-date 140501

# Excel template only (zero forecasts, no model run)
python main.py --qrt 1405Q1 --start-date 140501 --template
```

| Argument       | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `--qrt`        | Quarter label (e.g. `1405Q1`); used in paths and file names. |
| `--start-date` | Shamsi start month `YYYYMM` (e.g. `140501`).                 |
| `--template`   | Skip forecasting; write outputs with zeros only.             |

Outputs go to `src/data/results/<quarter>/`.

## Benchmark v1 (research baseline)

Phase 1 freezes the matched Human/TS rolling-origin evaluation from
`notebooks/residual_prediction.ipynb` so headline WMAPEs stop moving when the
warehouse or vintage CSVs change.

**Locked Analysis B PRIMARY** (identical matched rows, n=1877, origins
`140404, 140407, 140410, 140501, 140504`):

| Model       | Role                    |      WMAPE |
| ----------- | ----------------------- | ---------: |
| TS          | Quantitative baseline   | **43.88%** |
| Human       | Current Sales judgment  | **40.04%** |
| TS + XGB    | Automated candidate     | **37.23%** |
| Human + XGB | Human–machine candidate | **36.69%** |
| Integrated  | Experimental            |     40.14% |

These supersede the older “BEFORE expanded TS vintages” numbers in
`docs/forecasting_findings.md` (43.38 / 40.75 / 36.43, n=1657).

**API** (offline after freeze; no SQL):

```python
from pkg.benchmark import backtest, scoreboard

backtest("ts")       # ~43.88 WMAPE on frozen PRIMARY matched rows
scoreboard()         # TS / Human / TS+XGB / Human+XGB / Integrated
```

**CLI:**

```powershell
python -m pkg.benchmark.freeze    # once: DB + src/data/results -> src/data/benchmarks/v1/
python -m pkg.benchmark.verify    # offline: checksums + locked WMAPEs
```

- Panels: `src/data/benchmarks/v1/` (gitignored, under `src/data/`)
- Contract: [`src/pkg/benchmark/v1_manifest.json`](src/pkg/benchmark/v1_manifest.json)
  (checksums, schemas, expected WMAPEs — tracked in git)

Custom models: pass a callable `(train_df, test_df) -> forecasts` to `backtest`.

### Feature research (F0–F1C)

[`pkg/research`](src/pkg/research/) adds point-in-time feature experiments **on top of**
the freeze without changing locked WMAPEs or XGB hyperparameters.

| Experiment | Features |
|------------|----------|
| **F0** | Frozen benchmark feature set |
| **F1A** | F0 + demand dynamics |
| **F1B** | F0 + historical Human reliability |
| **F1C** | F0 + demand + Human reliability |

```python
from pkg.research import compare_feature_experiments

report = compare_feature_experiments()  # matched PRIMARY; TS+XGB and Human+XGB
report["overall"]   # WMAPE, rel vs F0, origins improved, product win rate, …
```

```powershell
python -m pkg.research.evaluate_features
python -m pkg.research.audit_f1          # F1 diagnostic audit → docs/f1_feature_audit.md
python -m pkg.research.evaluate_f2       # F2A/F2B/F2C on frozen v1 → docs/f2_results.md
```

F1 is not promoted. F2 redesigns a compact demand-state and shrunk Human-reliability set;
see [`docs/f2_feature_design.md`](docs/f2_feature_design.md).

Price / lifecycle / commercial modules exist as placeholders only.

## Project layout

```
Forecast/
  environment.yml      # Conda env (Python 3.10 + scientific stack, Prophet, …)
  requirements.txt     # Pip extras only (TensorFlow, gspread, pyarrow)
  pyproject.toml         # Editable install of pkg (pip install -e .)
  .env.example
  src/
    main.py              # CLI entry point
    pkg/
      db/                    # SQL client + queries
      benchmark/             # Frozen v1 dataset API + backtest()
      research/              # Feature experiments F0–F1C on frozen panels
      sales_forecasting.py   # Orchestration, Excel export
      forecast.py              # Per-product models
      excelmanager.py          # Excel formatting & protection
      utils.py                 # Paths, pivot, department mapping
      google_sheet.py          # Optional Sheets upload
    data/                  # Gitignored — results, pipeline, benchmarks/v1, credentials
```

## Handoff checklist

- [ ] Conda env created and `pip install -r requirements.txt` succeeded
- [ ] `.env` created from `.env.example`
- [ ] VPN / corporate network and SQL Server access verified
- [ ] `src/data/` structure and any pipeline Excel files provided
- [ ] Optional: `credentials.json` for Google Sheets
- [ ] Test run: `python main.py --qrt <quarter> --start-date <YYYYMM> --template`

## License

See [LICENSE](LICENSE).
