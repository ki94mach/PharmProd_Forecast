#Requires -Version 5.1
<#
.SYNOPSIS
  Run the Archval55 A0 empty-fix refit (90 jobs) on the Windows VPS.

.EXAMPLE
  cd C:\Forecast
  .\scripts\run_a0_empty_fix_refit.ps1
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$Python,
  [int]$Workers = 12
)

$ErrorActionPreference = 'Stop'

if (-not $ForecastRoot) {
  if ($env:FORECAST_ROOT) {
    $ForecastRoot = $env:FORECAST_ROOT
  } else {
    $here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $ForecastRoot = (Resolve-Path (Join-Path $here '..')).Path
  }
}
Set-Location $ForecastRoot
$env:FORECAST_ROOT = $ForecastRoot
$env:PYTHONPATH = Join-Path $ForecastRoot 'src'

function Resolve-ForecastPython {
  param([string]$Explicit)
  if ($Explicit -and (Test-Path -LiteralPath $Explicit)) {
    return (Resolve-Path -LiteralPath $Explicit).Path
  }
  if ($env:FORECAST_PYTHON -and (Test-Path -LiteralPath $env:FORECAST_PYTHON)) {
    return (Resolve-Path -LiteralPath $env:FORECAST_PYTHON).Path
  }
  $direct = @(
    "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\anaconda3\envs\forecast\python.exe",
    "$env:LOCALAPPDATA\miniconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\miniconda3\envs\forecast\python.exe",
    'C:\ProgramData\anaconda3\envs\forecast\python.exe',
    'C:\ProgramData\miniconda3\envs\forecast\python.exe'
  )
  foreach ($candidate in $direct) {
    if (Test-Path -LiteralPath $candidate) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  throw "forecast python not found. Set FORECAST_PYTHON."
}

$py = Resolve-ForecastPython -Explicit $Python
$env:FORECAST_PYTHON = $py
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:TF_NUM_INTRAOP_THREADS = '1'
$env:TF_NUM_INTEROP_THREADS = '1'

$logDir = Join-Path $ForecastRoot 'src\data\forecast_jobs\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'archval55_a0_empty_fix_v1.log'

Write-Host "ForecastRoot=$ForecastRoot"
Write-Host "Python=$py"
Write-Host "Workers=$Workers"
Write-Host "Log=$log"

& $py (Join-Path $ForecastRoot 'scripts\refit_archval55_a0_empty_fix.py') --workers $Workers *>> $log
$code = $LASTEXITCODE
Write-Host "exit_code=$code"
exit $code
