#Requires -Version 5.1
<#
.SYNOPSIS
  Validate / dry-run / execute / resume / status / evaluate residual-XGBoost v1.1.

.DESCRIPTION
  Smoke profile: origin 140107 only.
  Full profile: seven EVALUABLE origins.
  outer_workers=1. HUMAN_XGB blocked. Never opens HB outcomes.
  Cursor should use Validate or DryRun only — do not Execute locally.

.EXAMPLE
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Validate -Profile Smoke
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action DryRun -Profile Full
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Execute -Profile Smoke
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Status -Profile Full
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Evaluate -Profile Full
#>
[CmdletBinding()]
param(
  [ValidateSet('Validate', 'DryRun', 'Execute', 'Status', 'Evaluate')]
  [string]$Action = 'Validate',

  [ValidateSet('Smoke', 'Full')]
  [string]$Profile = 'Smoke',

  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
Push-Location $RepoRoot
try {
  $env:PYTHONPATH = 'src'

  $cfg = if ($Profile -eq 'Smoke') {
    'configs\forecast_jobs\portfolio_residual_xgb_v1_1.smoke.win.yaml'
  } else {
    'configs\forecast_jobs\portfolio_residual_xgb_v1_1.full.win.yaml'
  }

  $actionMap = @{
    Validate = 'validate'
    DryRun   = 'dry-run'
    Execute  = 'execute'
    Status   = 'status'
    Evaluate = 'evaluate'
  }
  $act = $actionMap[$Action]

  if ($Action -eq 'Validate') {
    Write-Host "Preparing v1.1 artifacts (no training)..."
    & $Python scripts\prepare_portfolio_residual_xgb_v1_1.py
    if ($LASTEXITCODE -ne 0) { throw "prepare_v1_1 failed: $LASTEXITCODE" }
    & $Python -m pytest tests\portfolio_residual_xgb_v1 -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }
  }

  Write-Host "Action=$Action Profile=$Profile Config=$cfg"
  & $Python scripts\run_portfolio_residual_xgb_v1_1.py --config $cfg --action $act
  if ($LASTEXITCODE -ne 0) { throw "run_portfolio_residual_xgb_v1_1 failed: $LASTEXITCODE" }

  Write-Host "HUMAN_XGB=BLOCKED_NO_DEVELOPMENT_VINTAGES hb_outcomes_inspected=false"
}
finally {
  Pop-Location
}
