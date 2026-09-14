#Requires -Version 5.1
<#
.SYNOPSIS
  Register Task Scheduler entry for A0 empty-fix refit (survives RDP disconnect).

.EXAMPLE
  cd C:\Forecast
  .\deploy\windows\Register-A0EmptyFixRefit.ps1 -StartNow
#>
[CmdletBinding()]
param(
  [string]$ForecastRoot,
  [string]$TaskName = 'ForecastJobs-archval55_a0_empty_fix_v1',
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

$launcher = Join-Path $ForecastRoot 'scripts\run_a0_empty_fix_refit.ps1'
if (-not (Test-Path -LiteralPath $launcher)) {
  throw ("Missing launcher: {0}. Copy latest tree from the laptop first." -f $launcher)
}

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" -ForecastRoot `"$ForecastRoot`""
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $TaskName"
Write-Host "  launcher=$launcher"

if ($StartNow) {
  Start-ScheduledTask -TaskName $TaskName
  Write-Host "Started $TaskName"
  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List
}
