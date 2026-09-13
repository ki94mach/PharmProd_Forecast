#Requires -Version 5.1
<#
.SYNOPSIS
  Create/update the Forecast conda env on Windows Server and verify imports.

.EXAMPLE
  cd C:\Forecast
  .\deploy\windows\Setup-Forecast.ps1
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot = $(if ($env:FORECAST_ROOT) { $env:FORECAST_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }),
  [string]$EnvName = 'forecast',
  [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

function Find-Conda {
  $candidates = @(
    (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    'C:\ProgramData\miniconda3\Scripts\conda.exe',
    'C:\ProgramData\Miniconda3\Scripts\conda.exe',
    'C:\Miniconda3\Scripts\conda.exe'
  ) | Where-Object { $_ -and (Test-Path $_) }
  if (-not $candidates) {
    throw @"
conda.exe not found. Install Miniconda first:
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
Then open a new PowerShell and re-run this script.
"@
  }
  return $candidates[0]
}

Write-Host "ForecastRoot=$ForecastRoot"
Set-Location $ForecastRoot

foreach ($required in @(
  'environment.yml',
  'requirements.txt',
  'pyproject.toml',
  'src\pkg\forecast_jobs\__main__.py'
)) {
  if (-not (Test-Path (Join-Path $ForecastRoot $required))) {
    throw "Missing $required under $ForecastRoot - copy the full repo to local disk first."
  }
}

$conda = Find-Conda
Write-Host "Using conda: $conda"

$envList = & $conda env list 2>&1 | Out-String
if ($envList -match "(?m)^\s*$([regex]::Escape($EnvName))\s") {
  Write-Host "Updating existing env '$EnvName'..."
  & $conda env update -f (Join-Path $ForecastRoot 'environment.yml') --prune
  if ($LASTEXITCODE -ne 0) { throw "conda env update failed ($LASTEXITCODE)" }
} else {
  Write-Host "Creating env '$EnvName'..."
  & $conda env create -f (Join-Path $ForecastRoot 'environment.yml')
  if ($LASTEXITCODE -ne 0) { throw "conda env create failed ($LASTEXITCODE)" }
}

# Resolve env python
$py = & $conda run -n $EnvName python -c "import sys; print(sys.executable)"
if (-not $py -or -not (Test-Path $py.Trim())) {
  throw "Could not resolve python for env '$EnvName'"
}
$py = $py.Trim()
Write-Host "Python=$py"

Write-Host "Installing pip extras + editable package..."
& $py -m pip install -r (Join-Path $ForecastRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
& $py -m pip install -e $ForecastRoot
if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }

New-Item -ItemType Directory -Force -Path (Join-Path $ForecastRoot 'src\data\forecast_jobs') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ForecastRoot 'src\data\forecast_jobs\logs') | Out-Null

if (-not $SkipVerify) {
  Write-Host "Verifying imports..."
  $code = @'
import sys
print("python", sys.version)
import numpy, pandas, pyarrow, yaml, xgboost
print("numpy", numpy.__version__, "pandas", pandas.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)
import cmdstanpy, prophet
print("cmdstanpy", cmdstanpy.__version__)
from pkg.forecast_jobs import __main__ as _
print("pkg.forecast_jobs OK")
'@
  $env:PYTHONPATH = Join-Path $ForecastRoot 'src'
  & $py -c $code
  if ($LASTEXITCODE -ne 0) { throw "Import verification failed" }
}

$sales = Join-Path $ForecastRoot 'src\data\benchmarks\v1\raw\sales.parquet'
if (Test-Path $sales) {
  $len = (Get-Item $sales).Length
  Write-Host "sales.parquet present ($len bytes)"
} else {
  Write-Warning "Missing $sales - copy frozen benchmarks before running jobs."
}

if (-not (Test-Path (Join-Path $ForecastRoot 'src\.env'))) {
  Write-Warning "Missing src\.env - copy from laptop/minos (optional for parquet-only jobs)."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next:"
Write-Host "  .\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -Validate"
Write-Host "  .\deploy\windows\Register-ForecastJobsTask.ps1 -ConfigStem smoke_v21_a0_a1.win -StartNow"
