#Requires -Version 5.1
<#
.SYNOPSIS
  Prepare-only validation for portfolio residual-XGBoost v1 (no training, no HB).

.DESCRIPTION
  Copies study assets when -Source/-Destination are set, then validates the
  residual prepare config and optionally dry-runs forecast_jobs planning.
  Never Register-s Task Scheduler execute jobs. Never opens HB outcomes.
  HUMAN_XGB remains BLOCKED_NO_DEVELOPMENT_VINTAGES.

.EXAMPLE
  .\deploy\windows\Prepare-PortfolioResidualXgbV1.ps1 -Validate

.EXAMPLE
  .\deploy\windows\Prepare-PortfolioResidualXgbV1.ps1 -Validate -DryRun
#>
[CmdletBinding()]
param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Python = 'python',
  [switch]$Validate,
  [switch]$DryRun,
  [string]$Source,
  [string]$Destination
)

$ErrorActionPreference = 'Stop'

function Copy-OneFile {
  param([string]$RelPath)
  $from = Join-Path $Source $RelPath
  $to = Join-Path $Destination $RelPath
  if (-not (Test-Path -LiteralPath $from)) {
    Write-Warning "Skip missing file: $RelPath"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
  Copy-Item -LiteralPath $from -Destination $to -Force
  Write-Host "file  $RelPath"
}

if ($Source -and $Destination) {
  Write-Host "Copying residual-xgb prepare assets..."
  @(
    'scripts\prepare_portfolio_residual_xgb_v1.py',
    'scripts\evaluate_portfolio_ts_architecture_postlock_v2.py',
    'configs\forecast_jobs\portfolio_residual_xgb_v1.win.yaml',
    'configs\forecast_jobs\portfolio_residual_xgb_v1.allowlist.json',
    'docs\portfolio_residual_xgb_v1.md'
  ) | ForEach-Object { Copy-OneFile $_ }

  $pkgDirs = @(
    'src\pkg\research\portfolio_residual_xgb_v1',
    'src\pkg\portfolio_ts_lock',
    'tests\portfolio_residual_xgb_v1',
    'tests\portfolio_ts_lock'
  )
  foreach ($d in $pkgDirs) {
    $from = Join-Path $Source $d
    $to = Join-Path $Destination $d
    if (-not (Test-Path -LiteralPath $from)) { continue }
    New-Item -ItemType Directory -Force -Path $to | Out-Null
    robocopy $from $to /E /XD __pycache__ .pytest_cache /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
    Write-Host "dir   $d"
  }
  $RepoRoot = $Destination
}

Push-Location $RepoRoot
try {
  $env:PYTHONPATH = 'src'

  if (-not $Validate -and -not $DryRun) {
    $Validate = $true
  }

  Write-Host "Running residual prepare (no training)..."
  & $Python scripts\prepare_portfolio_residual_xgb_v1.py
  if ($LASTEXITCODE -ne 0) { throw "prepare_portfolio_residual_xgb_v1 failed: $LASTEXITCODE" }

  Write-Host "pytest leakage + postlock classifier..."
  & $Python -m pytest tests\portfolio_ts_lock tests\portfolio_residual_xgb_v1 -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }

  if ($Validate) {
    Write-Host "forecast_jobs --validate..."
    & $Python -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_residual_xgb_v1.win.yaml --validate
    if ($LASTEXITCODE -ne 0) { throw "forecast_jobs --validate failed: $LASTEXITCODE" }
  }

  if ($DryRun) {
    Write-Host "forecast_jobs --dry-run (planning only; no execute)..."
    & $Python -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_residual_xgb_v1.win.yaml --dry-run
    if ($LASTEXITCODE -ne 0) { throw "forecast_jobs --dry-run failed: $LASTEXITCODE" }
  }

  Write-Host "HUMAN_XGB: BLOCKED_NO_DEVELOPMENT_VINTAGES (zero executable jobs)."
  Write-Host "Prepare-only complete. Do not --execute residual XGB or open HB outcomes."
}
finally {
  Pop-Location
}
