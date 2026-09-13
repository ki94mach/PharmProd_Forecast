#Requires -Version 5.1
<#
.SYNOPSIS
  Windows launcher for pkg.forecast_jobs (mirrors scripts/run_forecast_jobs.sh).

.EXAMPLE
  .\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml
  .\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -Status
  .\scripts\run_forecast_jobs.ps1 -Config configs\forecast_jobs\smoke_v21_a0_a1.win.yaml -Validate
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Config,

  [string]$RunId,

  [string]$ForecastRoot,

  [string]$Python,

  [switch]$Validate,
  [switch]$DryRun,
  [switch]$Execute,
  [switch]$Status,
  [switch]$Resume,
  [switch]$RetryFailed,

  [string[]]$ExtraArgs = @()
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

function Resolve-ForecastPython {
  param([string]$Explicit)
  if ($Explicit) {
    if (-not (Test-Path -LiteralPath $Explicit)) { throw "FORECAST_PYTHON not found: $Explicit" }
    return (Resolve-Path -LiteralPath $Explicit).Path
  }
  if ($env:FORECAST_PYTHON -and (Test-Path -LiteralPath $env:FORECAST_PYTHON)) {
    return (Resolve-Path -LiteralPath $env:FORECAST_PYTHON).Path
  }

  # Prefer direct env python.exe (avoid `conda run` — its stderr trips ErrorActionPreference=Stop).
  $direct = @(
    "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\anaconda3\envs\forecast\python.exe",
    "$env:LOCALAPPDATA\miniconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\miniconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\Miniconda3\envs\forecast\python.exe",
    'C:\ProgramData\anaconda3\envs\forecast\python.exe',
    'C:\ProgramData\miniconda3\envs\forecast\python.exe'
  )
  foreach ($candidate in $direct) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }

  foreach ($rel in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
    $candidate = Join-Path $ForecastRoot $rel
    if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
  }

  # Last resort: PATH python, but skip Windows PyManager if possible
  $all = @(Get-Command python -All -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
  foreach ($p in $all) {
    if ($p -and ($p -notmatch 'PyManager') -and (Test-Path -LiteralPath $p)) {
      return $p
    }
  }
  throw 'No forecast Python found. Set FORECAST_PYTHON to ...\envs\forecast\python.exe'
}

# Absolute config path
if (-not [System.IO.Path]::IsPathRooted($Config)) {
  $Config = Join-Path $ForecastRoot $Config
}
if (-not (Test-Path $Config)) { throw "config not found: $Config" }
$Config = (Resolve-Path $Config).Path

if (-not $RunId) {
  if ($env:FORECAST_JOBS_RUN_ID) {
    $RunId = $env:FORECAST_JOBS_RUN_ID
  } else {
    $RunId = [System.IO.Path]::GetFileNameWithoutExtension($Config)
    # smoke_v21_a0_a1.win -> smoke_v21_a0_a1
    if ($RunId -match '^(.*)\.win$') { $RunId = $Matches[1] }
  }
}

$PythonBin = Resolve-ForecastPython -Explicit $Python

$env:FORECAST_ROOT = $ForecastRoot
$env:PYTHONPATH = Join-Path $ForecastRoot 'src'
$env:PYTHONHASHSEED = $(if ($env:PYTHONHASHSEED) { $env:PYTHONHASHSEED } else { '0' })
foreach ($k in @(
  'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'
)) {
  if (-not (Get-Item "env:$k" -ErrorAction SilentlyContinue)) {
    Set-Item -Path "env:$k" -Value '1'
  }
}

$modeArgs = @()
$hasMode = $Validate -or $DryRun -or $Execute -or $Status -or $Resume -or $RetryFailed
if (-not $hasMode) {
  $modeArgs += @('--execute', '--resume')
} else {
  if ($Validate) { $modeArgs += '--validate' }
  if ($DryRun) { $modeArgs += '--dry-run' }
  if ($Status) { $modeArgs += '--status' }
  if ($Execute -or $Resume -or $RetryFailed) {
    if (-not ($modeArgs -contains '--execute') -and -not $Validate -and -not $DryRun -and -not $Status) {
      $modeArgs += '--execute'
    } elseif ($Execute) {
      $modeArgs += '--execute'
    }
    if ($Resume) { $modeArgs += '--resume' }
    if ($RetryFailed) { $modeArgs += '--retry-failed' }
  }
}

$cmd = @(
  '-m', 'pkg.forecast_jobs',
  '--config', $Config,
  '--run-id', $RunId
) + $modeArgs + $ExtraArgs

Write-Host "forecast_root=$ForecastRoot"
Write-Host "python=$PythonBin"
Write-Host "config=$Config"
Write-Host "run_id=$RunId"
Write-Host "cmd=$PythonBin $($cmd -join ' ')"

& $PythonBin @cmd
exit $LASTEXITCODE
