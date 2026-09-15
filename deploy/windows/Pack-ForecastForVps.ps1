#Requires -Version 5.1
<#
.SYNOPSIS
  Build Forecast-vps-bundle.zip on the LAPTOP (local disk). Copy that one zip to the VPS.

.EXAMPLE
  cd C:\Users\Mohajeri.K\MyProjects\Forecast
  .\deploy\windows\Pack-ForecastForVps.ps1
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$ZipPath
)

$ErrorActionPreference = 'Stop'

if (-not $ForecastRoot) {
  $here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
  $ForecastRoot = (Resolve-Path (Join-Path $here '..\..')).Path
}
if (-not $ZipPath) {
  $ZipPath = Join-Path $ForecastRoot 'Forecast-vps-bundle.zip'
}

$stage = Join-Path $env:TEMP ("Forecast-vps-stage-" + [guid]::NewGuid().ToString('N'))
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Stage-File([string]$Rel) {
  $src = Join-Path $ForecastRoot $Rel
  if (-not (Test-Path -LiteralPath $src)) {
    Write-Warning "Skip missing: $Rel"
    return
  }
  $dst = Join-Path $stage $Rel
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  Copy-Item -LiteralPath $src -Destination $dst -Force
  Write-Host "file  $Rel"
}

function Stage-Dir([string]$Rel) {
  $src = Join-Path $ForecastRoot $Rel
  if (-not (Test-Path -LiteralPath $src)) {
    Write-Warning "Skip missing dir: $Rel"
    return
  }
  $dst = Join-Path $stage $Rel
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Write-Host "dir   $Rel ..."
  & robocopy $src $dst /E /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache /XF *.pyc *.pyo `
    /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $Rel (exit $LASTEXITCODE)" }
  $n = @(Get-ChildItem -LiteralPath $dst -Recurse -File -ErrorAction SilentlyContinue).Count
  Write-Host "      $n files"
}

Write-Host "ForecastRoot=$ForecastRoot"
Write-Host "Staging to $stage"

@(
  'pyproject.toml',
  'environment.yml',
  'requirements.txt',
  'README.md',
  '.env.example',
  'src\pkg\__init__.py',
  'src\pkg\env.py',
  'src\pkg\utils.py',
  'src\pkg\forecast.py',
  'src\pkg\product_status.py',
  'src\main.py',
  'src\.env',
  'scripts\run_forecast_jobs.ps1',
  'scripts\verify_forecast_jobs_smoke.py',
  'scripts\refit_archval55_a0_empty_fix.py',
  'scripts\prepare_archval55_a0_empty_fix_allowlist.py',
  'scripts\run_a0_empty_fix_refit.ps1'
) | ForEach-Object { Stage-File $_ }

@(
  'src\pkg\forecast_jobs',
  'src\pkg\ts_v2',
  'src\pkg\ts_v3a',
  'src\pkg\benchmark',
  'src\pkg\db',
  'src\data\benchmarks',
  'configs\forecast_jobs',
  'deploy\windows'
) | ForEach-Object { Stage-Dir $_ }

New-Item -ItemType Directory -Force -Path (Join-Path $stage 'src\data\forecast_jobs\logs') | Out-Null

if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Write-Host "Compressing..."
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $ZipPath -Force
Remove-Item -Recurse -Force $stage

$item = Get-Item $ZipPath
Write-Host ("OK  {0}" -f $item.FullName)
Write-Host ("    {0} bytes ({1} MB)" -f $item.Length, [math]::Round($item.Length / 1MB, 2))
Write-Host ""
Write-Host "On the VPS:"
Write-Host "  Expand-Archive -Path '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\Forecast-vps-bundle.zip' -DestinationPath 'C:\Forecast' -Force"
Write-Host "  cd C:\Forecast"
Write-Host "  .\deploy\windows\Setup-Forecast.ps1"
