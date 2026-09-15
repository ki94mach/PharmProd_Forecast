#Requires -Version 5.1
<#
.SYNOPSIS
  Register Task Scheduler entry for Archval55 A2-A5 late-origin prod run.

.EXAMPLE
  cd C:\Forecast
  .\deploy\windows\Register-A2A5LateProd.ps1 -StartNow
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$PythonExe,
  [int]$Workers = 4,
  [string]$TaskName = 'ForecastJobs-archval55_a2_a5_late_prod_v1',
  [switch]$StartNow,
  [switch]$Remove
)

$ErrorActionPreference = 'Stop'

if (-not $ForecastRoot) {
  if ($env:FORECAST_ROOT) {
    $ForecastRoot = $env:FORECAST_ROOT
  } else {
    $here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $ForecastRoot = (Resolve-Path (Join-Path $here '..\..')).Path
  }
}

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "Removed task $TaskName (if it existed)."
  return
}

$launcher = Join-Path $ForecastRoot 'scripts\run_archval55_a2_a5_late_prod.ps1'
if (-not (Test-Path -LiteralPath $launcher)) {
  throw ("Missing launcher: {0}. Copy latest tree from the laptop first." -f $launcher)
}

if (-not $PythonExe) { $PythonExe = $env:FORECAST_PYTHON }
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
  $PythonExe = "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  $PythonExe = "$env:USERPROFILE\anaconda3\envs\forecast\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  $PythonExe = 'C:\ProgramData\anaconda3\envs\forecast\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  throw "forecast python not found. Pass -PythonExe or set FORECAST_PYTHON."
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" -ForecastRoot `"$ForecastRoot`" -Python `"$PythonExe`" -Workers $Workers"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $TaskName"
Write-Host "  launcher=$launcher"
Write-Host "  python=$PythonExe"
Write-Host "  workers=$Workers"

if ($StartNow) {
  Start-ScheduledTask -TaskName $TaskName
  Start-Sleep -Seconds 3
  Write-Host "Started $TaskName"
  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List
  $log = Join-Path $ForecastRoot 'src\data\forecast_jobs\logs\archval55_a2_a5_late_prod_v1.log'
  if (Test-Path -LiteralPath $log) {
    Write-Host "--- log tail ---"
    Get-Content -LiteralPath $log -Tail 30
  } else {
    Write-Warning "Log not created yet: $log"
  }
}
