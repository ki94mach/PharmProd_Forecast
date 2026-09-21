#Requires -Version 5.1
<#
.SYNOPSIS
  Validate / dry-run / execute / accept-smoke / status / evaluate residual-XGBoost v1.1.

.DESCRIPTION
  Smoke profile: origin 140107 only.
  Full profile: seven EVALUABLE origins — blocked until smoke acceptance passes.
  outer_workers=1. HUMAN_XGB blocked. Never opens HB outcomes.
  Cursor should use Validate / DryRun / AcceptSmoke checks only — do not Execute training locally.

.EXAMPLE
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Validate -Profile Smoke
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Execute -Profile Smoke
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action AcceptSmoke
  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action Execute -Profile Full
#>
[CmdletBinding()]
param(
  [ValidateSet('Validate', 'DryRun', 'Execute', 'AcceptSmoke', 'Status', 'Evaluate')]
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

  if ($Action -eq 'Validate') {
    Write-Host "Preparing v1.1 artifacts (no training)..."
    & $Python scripts\prepare_portfolio_residual_xgb_v1_1.py
    if ($LASTEXITCODE -ne 0) { throw "prepare_v1_1 failed: $LASTEXITCODE" }
    & $Python -m pytest tests\portfolio_residual_xgb_v1 -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }
  }

  if ($Action -eq 'AcceptSmoke') {
    Write-Host "Smoke acceptance (blocks Full until pass)..."
    & $Python scripts\run_portfolio_residual_xgb_v1_1.py `
      --config configs\forecast_jobs\portfolio_residual_xgb_v1_1.smoke.win.yaml `
      --action accept-smoke
    if ($LASTEXITCODE -ne 0) { throw "smoke acceptance FAILED — do not run Full" }
    Write-Host "Smoke acceptance PASSED."
    return
  }

  if ($Action -eq 'Execute' -and $Profile -eq 'Full') {
    $acc = Join-Path $RepoRoot 'src\data\results\portfolio_residual_xgb_v1\smoke_acceptance_v1.json'
    if (-not (Test-Path -LiteralPath $acc)) {
      throw "Full Execute blocked: missing smoke_acceptance_v1.json. Run Execute Smoke then AcceptSmoke first."
    }
  }

  $actionMap = @{
    Validate = 'validate'
    DryRun   = 'dry-run'
    Execute  = 'execute'
    Status   = 'status'
    Evaluate = 'evaluate'
  }
  $act = $actionMap[$Action]

  Write-Host "Action=$Action Profile=$Profile Config=$cfg"
  & $Python scripts\run_portfolio_residual_xgb_v1_1.py --config $cfg --action $act
  if ($LASTEXITCODE -ne 0) { throw "run_portfolio_residual_xgb_v1_1 failed: $LASTEXITCODE" }

  if ($Action -eq 'Execute' -and $Profile -eq 'Smoke') {
    Write-Host "NEXT: run AcceptSmoke before Full Execute"
    Write-Host "  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action AcceptSmoke"
  }

  Write-Host "HUMAN_XGB=BLOCKED_NO_DEVELOPMENT_VINTAGES hb_outcomes_inspected=false"
}
finally {
  Pop-Location
}
