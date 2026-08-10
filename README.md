# Forecast

Pharmaceutical sales time-series forecasting: loads historical sales from SQL Server, runs model selection (ARIMA, ETS, Prophet, LSTM), and writes department Excel workbooks under `src/data/results/`.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Windows** | SQL Server uses Windows integrated auth; `pywin32` is used on Windows. |
| **Python 3.10** | Matches `environment.yml`. |
| **Conda** | Required — installs Python 3.10, Prophet/CmdStan, and the scientific stack. |
| **ODBC** | `{SQL Server}` driver (name used in code). |
| **Network / DB access** | Server `op-db1-srv`, database `DWOrchid`, read on `Flat_Fact_Sale`. |
| **Domain login** | `Trusted_Connection=yes` — run as a user with DB permissions. |

## Setup

### Conda (required)

Uses Python **3.10** from conda-forge (not system Python). Scientific packages come from conda; TensorFlow / Google Sheets extras from pip.

```bash
cd path/to/Forecast
conda env create -f environment.yml
conda activate Forecast
python -m pip install -r requirements.txt   # use the env's pip, not system pip
```

If the env already exists:

```bash
conda env update -f environment.yml --prune
conda activate Forecast
python -m pip install -r requirements.txt
```

`conda env create` already installs the `pip:` block in `environment.yml`; re-running `requirements.txt` is safe and keeps extras explicit.

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
  environment.yml      # Conda env (Python 3.10 + scientific stack, Prophet, …)
  requirements.txt     # Pip extras only (TensorFlow, gspread)
  .env.example
  src/
    main.py              # CLI entry point
    pkg/
      db/                    # SQL client + queries
      sales_forecasting.py   # Orchestration, Excel export
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
