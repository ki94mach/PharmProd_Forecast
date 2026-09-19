# Forecast on Windows Server (RDP)

Use this when the VPS has **no SSH** and you work over **Remote Desktop**.
Do **not** run long jobs from an RDP-redirected laptop drive — copy the tree to a local disk first.

## Target layout

```
C:\Forecast\                 # repo root (local VPS disk)
  .conda-hint.txt            # optional notes
  src\.env                   # secrets (never commit)
  src\data\benchmarks\...    # frozen sales parquet + universes
  src\data\forecast_jobs\    # run outputs
```

Override the root anytime: `$env:FORECAST_ROOT = 'D:\Forecast'`.

## One-time setup (on the VPS)

### 1. Copy code + data onto the VPS disk

Selective copy (recommended) — only install files, `src\pkg`, benchmarks, job configs, and Windows scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\deploy\windows\Copy-FromRedirectedDrive.ps1' `
  -Source '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast' `
  -Destination 'C:\Forecast'
```

**Included:** `environment.yml`, `requirements.txt`, `pyproject.toml`, `src\pkg`, `src\.env`, `src\data\benchmarks`, `configs\forecast_jobs`, `deploy\windows`, job launcher/verifier scripts.

**Skipped:** `.git`, `.venv`, notebooks, docs, `.cursor`, `src\data\results*`, backfill, ts_v3a, prior job outputs.

If the laptop `sales.parquet` is missing, pull benchmarks from minos (`/opt/app/Forecast/src/data/benchmarks`, ~5.4MB).

### 2. Install Miniconda (if needed)

- Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
- Install for **All Users** or current user; tick **Add to PATH** (or use the Anaconda Prompt).
- Open a **new** PowerShell after install.

### 3. Create the env + verify

```powershell
cd C:\Forecast
Set-ExecutionPolicy -Scope Process Bypass
.\deploy\windows\Setup-Forecast.ps1
```

This creates/updates the `forecast` conda env, installs the package editable, and imports TensorFlow / CmdStanPy / Prophet.

### 4. Smoke validate (no model fit)

```powershell
cd C:\Forecast
.\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -Validate
.\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -DryRun
```

### 5. Smoke execute (survives RDP disconnect)

Prefer Task Scheduler (not an interactive RDP window):

```powershell
cd C:\Forecast
.\deploy\windows\Register-ForecastJobsTask.ps1 `
  -ConfigStem smoke_v21_a0_a1.win `
  -StartNow
```

Or run interactively (dies if the session is logged off):

```powershell
.\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml
```

### 6. Verify smoke, then archval25

```powershell
conda activate forecast
$env:PYTHONPATH = 'C:\Forecast\src'
$env:FORECAST_ROOT = 'C:\Forecast'
python scripts\verify_forecast_jobs_smoke.py
```

When smoke is green:

```powershell
.\deploy\windows\Register-ForecastJobsTask.ps1 `
  -ConfigStem archval25_v21_a0_a1.win `
  -StartNow
```

## Status / logs

```powershell
# Job counts
conda activate forecast
$env:PYTHONPATH = 'C:\Forecast\src'
python -m pkg.forecast_jobs --config configs\forecast_jobs\archval25_v21_a0_a1.win.yaml --status

# Task Scheduler
Get-ScheduledTask -TaskName 'ForecastJobs-*' | Get-ScheduledTaskInfo
Get-Content C:\Forecast\src\data\forecast_jobs\logs\*.log -Tail 40
```

## Archval55 A0 empty-fix refit (90 jobs)

Laptop stopped; do **not** copy the laptop partial tree. On the VPS:

```powershell
cd C:\Forecast
Set-ExecutionPolicy -Scope Process Bypass

# 1) Refresh code + allow-list from the redirected laptop drive
& '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\deploy\windows\Copy-FromRedirectedDrive.ps1' `
  -Source '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast' `
  -Destination 'C:\Forecast'

# 2) Ensure no stale same run_id from a prior attempt
if (Test-Path 'C:\Forecast\src\data\forecast_jobs\archval55_a0_empty_fix_v1') {
  Rename-Item 'C:\Forecast\src\data\forecast_jobs\archval55_a0_empty_fix_v1' `
    ("archval55_a0_empty_fix_v1_bak_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}

# 3) Register + start (survives RDP disconnect); workers=12
.\deploy\windows\Register-A0EmptyFixRefit.ps1 -StartNow

# 4) Monitor
Get-Content C:\Forecast\src\data\forecast_jobs\logs\archval55_a0_empty_fix_v1.log -Wait -Tail 40
```

Allow-list: `configs\forecast_jobs\archval55_a0_empty_fix_v1.allowlist.json`  
Run id: `archval55_a0_empty_fix_v1`

When finished, copy `C:\Forecast\src\data\forecast_jobs\archval55_a0_empty_fix_v1` back to the laptop for merge/eval.

## Notes

- Keep **minos** `archval25` running unless you deliberately stop it — do not run the same `run_id` on two hosts writing different trees and then merge blindly.
- Windows configs use **relative** `src/data/...` paths (see `*.win.yaml`).
- Thread env vars (`OMP_NUM_THREADS=1`, etc.) are set by `run_forecast_jobs.ps1`.
- SQL / ODBC is optional for frozen-parquet jobs; still copy `src\.env` for consistency.
