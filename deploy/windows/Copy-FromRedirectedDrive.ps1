#Requires -Version 5.1
<#
.SYNOPSIS
  Copy only what is needed to run forecast_jobs on a Windows VPS.

.EXAMPLE
  .\deploy\windows\Copy-FromRedirectedDrive.ps1 `
    -Source '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast' `
    -Destination 'C:\Forecast'
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Source,

  [string]$Destination = 'C:\Forecast',

  [switch]$IncludeTests,
  [switch]$IncludeResearch
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

  # Enumerate files only (faster / clearer than robocopy over \\tsclient).
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
Write-Host "Selective copy: $Source -> $Destination"
Write-Host "Only install + forecast_jobs essentials (not full repo / results)."
Write-Host "(Progress prints every 25 files. \\tsclient is slow - this can take a few minutes.)"
Write-Host ""

@(
  'pyproject.toml',
  'environment.yml',
  'requirements.txt',
  'README.md',
  '.env.example'
) | ForEach-Object { Copy-OneFile $_ }

# Minimal pkg tree for V2.1 / A0 / A1 jobs (skip research by default - half the files).
$pkgDirs = @(
  'src\pkg\forecast_jobs',
  'src\pkg\ts_v2',
  'src\pkg\ts_v3a',
  'src\pkg\benchmark',
  'src\pkg\db'
)
if ($IncludeResearch) {
  $pkgDirs += 'src\pkg\research'
}
foreach ($d in $pkgDirs) {
  Copy-DirWithProgress -RelDir $d
}

@(
  'src\pkg\__init__.py',
  'src\pkg\env.py',
  'src\pkg\utils.py',
  'src\pkg\forecast.py',
  'src\pkg\product_status.py',
  'src\main.py',
  'src\__init__.py',
  'src\.env'
) | ForEach-Object { Copy-OneFile $_ }

Copy-DirWithProgress -RelDir 'src\data\benchmarks'
Copy-DirWithProgress -RelDir 'configs\forecast_jobs'
Copy-DirWithProgress -RelDir 'deploy\windows'

@(
  'scripts\run_forecast_jobs.ps1',
  'scripts\verify_forecast_jobs_smoke.py',
  'scripts\refit_archval55_a0_empty_fix.py',
  'scripts\prepare_archval55_a0_empty_fix_allowlist.py',
  'scripts\run_a0_empty_fix_refit.ps1',
  'scripts\run_archval55_a2_a5_late_prod.ps1'
) | ForEach-Object { Copy-OneFile $_ }

if ($IncludeTests) {
  Copy-DirWithProgress -RelDir 'tests'
}

New-Item -ItemType Directory -Force -Path (Join-Path $Destination 'src\data\forecast_jobs\logs') | Out-Null

$required = @(
  'environment.yml',
  'requirements.txt',
  'pyproject.toml',
  'src\pkg\forecast_jobs\__main__.py',
  'src\pkg\ts_v2\engine.py',
  'src\pkg\ts_v3a\backtest.py',
  'src\data\benchmarks\v1\raw\sales.parquet',
  'configs\forecast_jobs\smoke_v21_a0_a1.win.yaml',
  'deploy\windows\Setup-Forecast.ps1',
  'scripts\run_forecast_jobs.ps1'
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

$sales = Join-Path $Destination 'src\data\benchmarks\v1\raw\sales.parquet'
Write-Host ("OK    sales.parquet bytes={0}" -f (Get-Item $sales).Length)
if (-not (Test-Path -LiteralPath (Join-Path $Destination 'src\.env'))) {
  Write-Warning "src\.env missing - optional for parquet-only jobs"
}

Write-Host ""
Write-Host "Done (selective)."
Write-Host "Skipped: .git, .venv, notebooks, docs, .cursor, research (default), results/backfill/ts_v3a outputs"
Write-Host "Next: cd $Destination ; .\deploy\windows\Setup-Forecast.ps1"
