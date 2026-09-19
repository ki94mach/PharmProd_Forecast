#Requires -Version 5.1
<#
.SYNOPSIS
  On the Windows VPS: copy slug-fix + smoke configs from redirected laptop drive,
  wipe old smoke trees, prepare allowlists, execute V2.1 + A0 smoke with explicit run-ids.

.EXAMPLE
  cd C:\Forecast
  & '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\deploy\windows\Deploy-And-Run-PortfolioSmokeV2.ps1'
#>
[CmdletBinding()]
param(
  [string]$Source = '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast',
  [string]$Destination = 'C:\Forecast',
  [switch]$SkipExecute
)

$ErrorActionPreference = 'Stop'

function Copy-OneFile {
  param([string]$RelPath)
  $from = Join-Path $Source $RelPath
  $to = Join-Path $Destination $RelPath
  if (-not (Test-Path -LiteralPath $from)) {
    Write-Warning "Skip missing: $RelPath"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
  Copy-Item -LiteralPath $from -Destination $to -Force
  Write-Host "file  $RelPath"
}

if (-not (Test-Path -LiteralPath $Source)) {
  throw "Source not found: $Source - enable Local Resources -> Drives in RDP."
}
if (-not (Test-Path -LiteralPath $Destination)) {
  throw "Destination not found: $Destination"
}

Set-Location -LiteralPath $Destination
Write-Host "=== 1. Copy fixed code + smoke configs ==="

@(
  'src\pkg\benchmark\backfill_runner\state.py',
  'src\pkg\benchmark\backfill_runner\types.py',
  'src\pkg\benchmark\backfill_runner\store.py',
  'src\pkg\forecast_jobs\plan.py',
  'src\pkg\forecast_jobs\runner.py',
  'scripts\export_portfolio_backfill_panel.py',
  'scripts\verify_portfolio_backfill_smoke_v2.py',
  'scripts\prepare_portfolio_backfill_allowlist_v2.py',
  'scripts\run_portfolio_a0_backfill_v2.ps1',
  'scripts\run_forecast_jobs.ps1',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.allowlist.json',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.allowlist.json',
  'docs\portfolio_backfill_v2_prep.md',
  'deploy\windows\Copy-PortfolioSmokeV2.ps1',
  'deploy\windows\Deploy-And-Run-PortfolioSmokeV2.ps1'
) | ForEach-Object { Copy-OneFile $_ }

$salesRel = 'src\data\results\product_universe\sales_full_portfolio_v1'
$salesFrom = Join-Path $Source $salesRel
$salesTo = Join-Path $Destination $salesRel
if (Test-Path -LiteralPath $salesFrom) {
  New-Item -ItemType Directory -Force -Path $salesTo | Out-Null
  Copy-Item -Force (Join-Path $salesFrom '*') $salesTo
  Write-Host "dir   $salesRel"
} else {
  Write-Warning "Sales dir missing on source: $salesRel"
}

Write-Host "=== 2. Wipe old smoke trees ==="
@(
  'src\data\forecast_jobs\portfolio_v21_backfill_v2_smoke',
  'src\data\forecast_jobs\portfolio_a0_backfill_v2_smoke',
  'src\data\forecast_jobs\portfolio_a0_backfill_v2.smoke',
  'src\data\forecast_jobs\portfolio_v21_backfill_v2.smoke'
) | ForEach-Object {
  $p = Join-Path $Destination $_
  if (Test-Path -LiteralPath $p) {
    Remove-Item -Recurse -Force -LiteralPath $p
    Write-Host "removed $_"
  }
}

$env:FORECAST_ROOT = $Destination
$env:PYTHONPATH = Join-Path $Destination 'src'
$env:PYTHONIOENCODING = 'utf-8'
$env:TF_NUM_INTRAOP_THREADS = '1'
$env:TF_NUM_INTEROP_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$py = "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
  throw "forecast python not found: $py"
}

Write-Host "=== 3. Prepare allowlists ==="
& $py scripts\prepare_portfolio_backfill_allowlist_v2.py `
  --config configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml `
  --allowlist configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.allowlist.json
if ($LASTEXITCODE -ne 0) { throw "V2.1 allowlist prepare failed" }

& $py scripts\prepare_portfolio_backfill_allowlist_v2.py `
  --config configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml `
  --allowlist configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.allowlist.json
if ($LASTEXITCODE -ne 0) { throw "A0 allowlist prepare failed" }

Write-Host "=== 4. Status before execute ==="
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml `
  --run-id portfolio_v21_backfill_v2_smoke --status
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml `
  --run-id portfolio_a0_backfill_v2_smoke --status

if ($SkipExecute) {
  Write-Host "SkipExecute set - stop before forecasting."
  exit 0
}

Write-Host "=== 5. Execute V2.1 smoke ==="
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml `
  --run-id portfolio_v21_backfill_v2_smoke --execute
if ($LASTEXITCODE -ne 0) { throw "V2.1 smoke execute failed exit=$LASTEXITCODE" }

Write-Host "=== 6. Execute A0 smoke ==="
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml `
  --run-id portfolio_a0_backfill_v2_smoke --execute
if ($LASTEXITCODE -ne 0) { throw "A0 smoke execute failed exit=$LASTEXITCODE" }

Write-Host "=== 7. Status + export + verify ==="
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml `
  --run-id portfolio_v21_backfill_v2_smoke --status
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml `
  --run-id portfolio_a0_backfill_v2_smoke --status

& $py scripts\export_portfolio_backfill_panel.py --run-id portfolio_v21_backfill_v2_smoke --engine v2.1 `
  --out-dir src\data\results\product_universe\portfolio_backfill_prep_v2\export_v21_smoke
& $py scripts\export_portfolio_backfill_panel.py --run-id portfolio_a0_backfill_v2_smoke --engine a0 `
  --out-dir src\data\results\product_universe\portfolio_backfill_prep_v2\export_a0_smoke

& $py scripts\verify_portfolio_backfill_smoke_v2.py --run-id portfolio_v21_backfill_v2_smoke --engine v2.1
& $py scripts\verify_portfolio_backfill_smoke_v2.py --run-id portfolio_a0_backfill_v2_smoke --engine a0

Write-Host ""
Write-Host "Done. Spot-check origin-qualified paths, e.g.:"
Write-Host "  Get-ChildItem src\data\forecast_jobs\portfolio_a0_backfill_v2_smoke\jobs\a0 -Directory | Select-Object Name"
