#Requires -Version 5.1
<#
.SYNOPSIS
  Copy portfolio_backfill_prep_v2 smoke-gate essentials to a Windows VPS.

.DESCRIPTION
  Incremental sync after the base Copy-FromRedirectedDrive.ps1 install.
  Copies updated pkg code, smoke configs/allowlists, sales parquet used by
  smoke YAML, prepared smoke run trees, and smoke scripts.
  Does NOT copy full portfolio run trees or execute forecasting.

.EXAMPLE
  .\deploy\windows\Copy-PortfolioSmokeV2.ps1 `
    -Source '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast' `
    -Destination 'C:\Forecast'
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Source,

  [string]$Destination = 'C:\Forecast',

  [switch]$IncludeFullRunTrees,
  [switch]$IncludePrepArtifacts
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

function Copy-DirWithProgress {
  param(
    [string]$RelDir,
    [string[]]$ExcludeDirNames = @('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache')
  )
  $from = Join-Path $Source $RelDir
  $to = Join-Path $Destination $RelDir
  if (-not (Test-Path -LiteralPath $from)) {
    Write-Warning "Skip missing dir: $RelDir"
    return
  }

  Write-Host "dir   $RelDir ..."
  New-Item -ItemType Directory -Force -Path $to | Out-Null

  $files = Get-ChildItem -LiteralPath $from -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
      $p = $_.FullName
      foreach ($name in $ExcludeDirNames) {
        if ($p -like "*\$name\*") { return $false }
      }
      if ($_.Extension -in @('.pyc', '.pyo')) { return $false }
      return $true
    }

  $total = @($files).Count
  $i = 0
  foreach ($f in $files) {
    $i++
    $rel = $f.FullName.Substring($from.Length).TrimStart('\')
    $destPath = Join-Path $to $rel
    $parent = Split-Path $destPath -Parent
    if (-not (Test-Path -LiteralPath $parent)) {
      New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $f.FullName -Destination $destPath -Force
    if (($i % 25 -eq 0) -or ($i -eq $total)) {
      Write-Host ("      {0}/{1}  {2}" -f $i, $total, $rel)
    }
  }
  Write-Host ("      done {0} files" -f $total)
}

if (-not (Test-Path -LiteralPath $Source)) {
  throw "Source not found: $Source - enable Local Resources -> Drives in the RDP client."
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Write-Host "Portfolio smoke v2 copy: $Source -> $Destination"
Write-Host "(Progress every 25 files. \\tsclient is slow.)"
Write-Host ""

# --- Updated package code (SKIPPED status, resume, plan/runner) ---
@(
  'src\pkg\benchmark\backfill_runner\state.py',
  'src\pkg\forecast_jobs\plan.py',
  'src\pkg\forecast_jobs\runner.py'
) | ForEach-Object { Copy-OneFile $_ }

# Safety: refresh forecast_jobs + ts_v2 if base copy is stale
Copy-DirWithProgress -RelDir 'src\pkg\forecast_jobs'
Copy-DirWithProgress -RelDir 'src\pkg\ts_v2'
Copy-DirWithProgress -RelDir 'src\pkg\benchmark\backfill_runner'

# --- Smoke + full allowlists/configs ---
@(
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.allowlist.json',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.allowlist.json',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.allowlist.json',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.allowlist.json'
) | ForEach-Object { Copy-OneFile $_ }

# --- Scripts for smoke execute / export / verify ---
@(
  'scripts\run_forecast_jobs.ps1',
  'scripts\run_portfolio_a0_backfill_v2.ps1',
  'scripts\prepare_portfolio_backfill_allowlist_v2.py',
  'scripts\export_portfolio_backfill_panel.py',
  'scripts\verify_portfolio_backfill_smoke_v2.py',
  'scripts\audit_portfolio_reuse_effective_hash_v2.py',
  'scripts\import_portfolio_reusable_jobs.py',
  'deploy\windows\Copy-PortfolioSmokeV2.ps1'
) | ForEach-Object { Copy-OneFile $_ }

# --- Sales parquet referenced by smoke YAML ---
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\sales_full_portfolio_v1'

# --- Prepared smoke run trees (PENDING=5 / 15) ---
Copy-DirWithProgress -RelDir 'src\data\forecast_jobs\portfolio_v21_backfill_v2_smoke'
Copy-DirWithProgress -RelDir 'src\data\forecast_jobs\portfolio_a0_backfill_v2_smoke'

# --- Docs (optional but small) ---
@(
  'docs\portfolio_backfill_v2_prep.md'
) | ForEach-Object { Copy-OneFile $_ }

if ($IncludePrepArtifacts) {
  Copy-DirWithProgress -RelDir 'src\data\results\product_universe\portfolio_backfill_prep_v2'
}

if ($IncludeFullRunTrees) {
  Write-Host "Including full v2 run trees (large) ..."
  Copy-DirWithProgress -RelDir 'src\data\forecast_jobs\portfolio_v21_backfill_v2'
  Copy-DirWithProgress -RelDir 'src\data\forecast_jobs\portfolio_a0_backfill_v2'
}

New-Item -ItemType Directory -Force -Path (Join-Path $Destination 'src\data\forecast_jobs\logs') | Out-Null

$required = @(
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.allowlist.json',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.allowlist.json',
  'src\data\results\product_universe\sales_full_portfolio_v1\sales.parquet',
  'src\data\forecast_jobs\portfolio_v21_backfill_v2_smoke\state.sqlite',
  'src\data\forecast_jobs\portfolio_a0_backfill_v2_smoke\state.sqlite',
  'scripts\run_portfolio_a0_backfill_v2.ps1',
  'scripts\export_portfolio_backfill_panel.py',
  'scripts\verify_portfolio_backfill_smoke_v2.py',
  'src\pkg\benchmark\backfill_runner\state.py'
)
$missing = @()
foreach ($rel in $required) {
  if (-not (Test-Path -LiteralPath (Join-Path $Destination $rel))) {
    $missing += $rel
  }
}
if ($missing.Count -gt 0) {
  throw ("Copy incomplete; missing:`n  - " + ($missing -join "`n  - "))
}

Write-Host ""
Write-Host "Done (portfolio smoke v2)."
Write-Host "Next on VPS:"
Write-Host "  cd $Destination"
Write-Host "  `$env:PYTHONPATH='src'"
Write-Host "  & `$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.smoke.win.yaml --status"
Write-Host "  .\scripts\run_portfolio_a0_backfill_v2.ps1 -Config configs\forecast_jobs\portfolio_a0_backfill_v2.smoke.win.yaml -Status"
Write-Host "Then -Execute when ready. Do not start full portfolio runs until smoke is reviewed."
