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

  [string]$ForecastRoot = $(if ($env:FORECAST_ROOT) { $env:FORECAST_ROOT } else {
    (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  }),

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
Set-Location $ForecastRoot

function Resolve-ForecastPython {
  param([string]$Explicit)
  if ($Explicit) {
    if (-not (Test-Path $Explicit)) { throw "FORECAST_PYTHON not found: $Explicit" }
    return (Resolve-Path $Explicit).Path
  }
  if ($env:FORECAST_PYTHON -and (Test-Path $env:FORECAST_PYTHON)) {
    return (Resolve-Path $env:FORECAST_PYTHON).Path
  }
  $condaCandidates = @(
    (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
    'C:\ProgramData\miniconda3\Scripts\conda.exe',
    'C:\ProgramData\Miniconda3\Scripts\conda.exe'
  ) | Where-Object { $_ -and (Test-Path $_) }
  foreach ($conda in $condaCandidates) {
    $out = & $conda run -n forecast python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
      $p = ($out | Select-Object -Last 1).Trim()
      if (Test-Path $p) { return $p }
    }
  }
  foreach ($rel in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
    $candidate = Join-Path $ForecastRoot $rel
    if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
  }
  $pathPy = Get-Command python -ErrorAction SilentlyContinue
  if ($pathPy) { return $pathPy.Source }
  throw 'No Python found. Run deploy\windows\Setup-Forecast.ps1 or set FORECAST_PYTHON.'
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
