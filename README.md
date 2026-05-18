# Forecast

Pharmaceutical sales time-series forecasting: loads historical sales from SQL Server, runs model selection (ARIMA, ETS, Prophet, LSTM), and writes department Excel workbooks under `src/data/results/`.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Windows** | SQL Server uses Windows integrated auth; `pywin32` is used on Windows. |
| **Python 3.10** | Matches `environment.yml`. |
| **Conda** (recommended) or **pip** | Conda installs Prophet/CmdStan and scientific stack reliably. |
| **ODBC** | `{SQL Server}` driver (name used in code). |
| **Network / DB access** | Server `op-db1-srv`, database `DWOrchid`, read on `Flat_Fact_Sale`. |
| **Domain login** | `Trusted_Connection=yes` — run as a user with DB permissions. |

## Setup

### Option A — Conda (recommended)

```powershell
cd path\to\Forecast
conda env create -f environment.yml
conda activate Forecast
pip install -r requirements.txt
```

If `conda env create` fails because the env already exists:

```powershell
conda env update -f environment.yml --prune
conda activate Forecast
pip install -r requirements.txt
```

### Option B — Pip only

Use Python 3.10 on Windows, then:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Prophet may require [CmdStan](https://mc-stan.org/cmdstanpy/) separately when not using Conda.

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
conda activate Forecast
cd src

# Full forecast
python main.py --qrt 1405Q1 --start-date 140501

# Excel template only (zero forecasts, no model run)
python main.py --qrt 1405Q1 --start-date 140501 --template
```

| Argument | Description |
|----------|-------------|
| `--qrt` | Quarter label (e.g. `1405Q1`); used in paths and file names. |
| `--start-date` | Shamsi start month `YYYYMM` (e.g. `140501`). |
| `--template` | Skip forecasting; write outputs with zeros only. |

Outputs go to `src/data/results/<quarter>/`.

## Project layout

```
Forecast/
  environment.yml      # Conda environment (Python 3.10 + Prophet, pyodbc, …)
  requirements.txt     # Pip packages (install after conda env)
  .env.example
  src/
    main.py              # CLI entry point
    pkg/
      sales_forecasting.py   # DB load, orchestration, Excel export
      forecast.py              # Per-product models
      excelmanager.py          # Excel formatting & protection
      utils.py                 # Paths, pivot, department mapping
      google_sheet.py          # Optional Sheets upload
    data/                  # Gitignored — results, pipeline, credentials
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
