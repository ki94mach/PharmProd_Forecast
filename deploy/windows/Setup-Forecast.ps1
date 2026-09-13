#Requires -Version 5.1
<#
.SYNOPSIS
  Create/update the Forecast conda env on Windows Server and verify imports.

.EXAMPLE
  cd C:\Forecast
  .\deploy\windows\Setup-Forecast.ps1 -ForecastRoot C:\Forecast
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$EnvName = 'forecast',
  [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

if (-not $ForecastRoot) {
  if ($env:FORECAST_ROOT) {
    $ForecastRoot = $env:FORECAST_ROOT
  } elseif ($PSScriptRoot) {
    $ForecastRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
  } else {
    $ForecastRoot = (Get-Location).Path
  }
}
$ForecastRoot = $ForecastRoot.TrimEnd('\')
if (-not (Test-Path -LiteralPath $ForecastRoot)) {
  throw "ForecastRoot not found: $ForecastRoot"
}

function Find-Conda {
  $candidates = @(
    "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    'C:\ProgramData\miniconda3\Scripts\conda.exe',
    'C:\ProgramData\Miniconda3\Scripts\conda.exe',
    'C:\ProgramData\anaconda3\Scripts\conda.exe',
    'C:\Miniconda3\Scripts\conda.exe',
    'C:\Anaconda3\Scripts\conda.exe'
  )
  $cmd = Get-Command conda -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) {
    $candidates = @($cmd.Source) + $candidates
  }
  foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) {
      return $c
    }
  }
  throw @"
conda.exe not found. Install Miniconda/Anaconda first, open a new PowerShell, re-run.
"@
}

function Get-CondaEnvPython {
  param([string]$CondaExe, [string]$Name)

  # Prefer stable path: <conda-base>\envs\<name>\python.exe
  $base = $null
  try {
    $baseLines = & $CondaExe info --base 2>$null
    foreach ($line in @($baseLines)) {
      $t = "$line".Trim()
      if ($t -and (Test-Path -LiteralPath $t)) {
        $base = $t
        break
      }
    }
  } catch {}

  if ($base) {
    $candidate = Join-Path $base "envs\$Name\python.exe"
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
  }

  # Fallback: last existing path-like line from conda run
  $lines = & $CondaExe run -n $Name python -c "import sys; print(sys.executable)" 2>&1
  $found = $null
  foreach ($line in @($lines)) {
    $t = "$line".Trim()
    if ($t -and (Test-Path -LiteralPath $t)) {
      $found = $t
    }
  }
  if ($found) { return $found }
  throw "Could not resolve python for env '$Name'"
}

Write-Host "ForecastRoot=$ForecastRoot"
Set-Location -LiteralPath $ForecastRoot

foreach ($required in @(
  'environment.yml',
  'requirements.txt',
  'pyproject.toml',
  'src\pkg\forecast_jobs\__main__.py'
)) {
  $p = Join-Path $ForecastRoot $required
  if (-not (Test-Path -LiteralPath $p)) {
    throw "Missing $required under $ForecastRoot - copy the repo to local disk first."
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

$py = Get-CondaEnvPython -CondaExe $conda -Name $EnvName
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
  $verifyPy = Join-Path $env:TEMP 'forecast_setup_verify.py'
  @'
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
'@ | Set-Content -Path $verifyPy -Encoding UTF8
  $env:PYTHONPATH = Join-Path $ForecastRoot 'src'
  & $py $verifyPy
  $verifyCode = $LASTEXITCODE
  Remove-Item -Force $verifyPy -ErrorAction SilentlyContinue
  if ($verifyCode -ne 0) { throw "Import verification failed" }
}

$sales = Join-Path $ForecastRoot 'src\data\benchmarks\v1\raw\sales.parquet'
if (Test-Path -LiteralPath $sales) {
  Write-Host ("sales.parquet present ({0} bytes)" -f (Get-Item -LiteralPath $sales).Length)
} else {
  Write-Warning "Missing $sales - copy frozen benchmarks before running jobs."
}

if (-not (Test-Path -LiteralPath (Join-Path $ForecastRoot 'src\.env'))) {
  Write-Warning "Missing src\.env - copy from laptop/minos (optional for parquet-only jobs)."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next:"
Write-Host "  .\scripts\run_forecast_jobs.ps1 -ForecastRoot C:\Forecast -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -Validate"
Write-Host "  .\deploy\windows\Register-ForecastJobsTask.ps1 -ForecastRoot C:\Forecast -ConfigStem smoke_v21_a0_a1.win -StartNow"
