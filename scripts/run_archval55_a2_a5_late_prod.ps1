#Requires -Version 5.1
<#
.SYNOPSIS
  Run Archval55 A2-A5 late-origin production-contract jobs on Windows VPS.

.EXAMPLE
  cd C:\Forecast
  .\scripts\run_archval55_a2_a5_late_prod.ps1 -Workers 4
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$Python,
  [int]$Workers = 4
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
$env:PYTHONUNBUFFERED = '1'

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
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  throw "forecast python not found. Pass -Python or set FORECAST_PYTHON."
}

$py = Resolve-ForecastPython -Explicit $Python
$env:FORECAST_PYTHON = $py

$logDir = Join-Path $ForecastRoot 'src\data\forecast_jobs\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'archval55_a2_a5_late_prod_v1.log'

function Write-Log([string]$Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
  Write-Host $line
}

$config = Join-Path $ForecastRoot 'configs\forecast_jobs\archval55_a2_a5_late_prod_v1.win.yaml'
$launcher = Join-Path $ForecastRoot 'scripts\run_forecast_jobs.ps1'

if (-not (Test-Path -LiteralPath $config)) { throw "missing config: $config" }
if (-not (Test-Path -LiteralPath $launcher)) { throw "missing launcher: $launcher" }

# Patch workers into the yaml for this run.
$raw = Get-Content -LiteralPath $config -Raw
if ($raw -match '(?m)^(\s*workers:\s*)\d+') {
  $raw = [regex]::Replace($raw, '(?m)^(\s*workers:\s*)\d+', "`${1}$Workers")
  Set-Content -LiteralPath $config -Value $raw -Encoding UTF8
}

Write-Log "start ForecastRoot=$ForecastRoot Workers=$Workers (a2-a5 late prod)"
Write-Log "Python=$py"
$ErrorActionPreference = 'Continue'
& $launcher -ForecastRoot $ForecastRoot -Config $config -Python $py -Execute -Resume *>> $log
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 0 }
Write-Log ("exit_code={0}" -f $code)
exit $code
