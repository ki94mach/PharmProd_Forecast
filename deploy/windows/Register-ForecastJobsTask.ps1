#Requires -Version 5.1
<#
.SYNOPSIS
  Register a Task Scheduler job so forecast_jobs survives RDP disconnect.

.EXAMPLE
  .\deploy\windows\Register-ForecastJobsTask.ps1 -ConfigStem smoke_v21_a0_a1.win -StartNow
  .\deploy\windows\Register-ForecastJobsTask.ps1 -ConfigStem archval25_v21_a0_a1.win -StartNow
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ConfigStem,

  [string]$ForecastRoot = $(if ($env:FORECAST_ROOT) { $env:FORECAST_ROOT } else {
    (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
  }),

  [string]$TaskName,

  [switch]$StartNow,
  [switch]$Remove
)

$ErrorActionPreference = 'Stop'

if (-not $TaskName) {
  $safe = $ConfigStem -replace '[^A-Za-z0-9_.-]', '_'
  $TaskName = "ForecastJobs-$safe"
}

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "Removed task $TaskName (if it existed)."
  return
}

$configRel = "configs\forecast_jobs\$ConfigStem.yaml"
$configAbs = Join-Path $ForecastRoot $configRel
if (-not (Test-Path $configAbs)) {
  throw "Config not found: $configAbs"
}

$launcher = Join-Path $ForecastRoot 'scripts\run_forecast_jobs.ps1'
if (-not (Test-Path $launcher)) {
  throw "Missing launcher: $launcher"
}

$logDir = Join-Path $ForecastRoot 'src\data\forecast_jobs\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "$ConfigStem.log"
$wrapper = Join-Path $logDir "$ConfigStem.wrapper.ps1"

@'
param(
  [string]$ForecastRoot,
  [string]$Launcher,
  [string]$Config,
  [string]$LogFile
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null
try {
  & $Launcher -ForecastRoot $ForecastRoot -Config $Config
  $code = $LASTEXITCODE
} finally {
  Stop-Transcript | Out-Null
}
exit $code
'@ | Set-Content -Path $wrapper -Encoding UTF8

$arg = @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', "`"$wrapper`"",
  '-ForecastRoot', "`"$ForecastRoot`"",
  '-Launcher', "`"$launcher`"",
  '-Config', "`"$configAbs`"",
  '-LogFile', "`"$logFile`""
) -join ' '

$action = New-ScheduledTaskAction `
  -Execute 'powershell.exe' `
  -Argument $arg `
  -WorkingDirectory $ForecastRoot

# Interactive: needs a logged-on session (typical for RDP VPS while you stay connected
# or leave the session disconnected but not logged off). For fully headless, re-register
# in Task Scheduler GUI with "Run whether user is logged on or not" + password.
$principal = New-ScheduledTaskPrincipal `
  -UserId $env:USERNAME `
  -LogonType Interactive `
  -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 5) `
  -MultipleInstances IgnoreNew

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Principal $principal `
  -Settings $settings `
  -Description "Forecast jobs: $ConfigStem; log=$logFile" | Out-Null

Write-Host "Registered task: $TaskName"
Write-Host "Config: $configAbs"
Write-Host "Log:    $logFile"
Write-Host "Tip: leave the RDP session disconnected (not signed out), or set the task to run whether user is logged on."

if ($StartNow) {
  Start-ScheduledTask -TaskName $TaskName
  Write-Host "Started $TaskName"
  Start-Sleep -Seconds 2
  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List
}

Write-Host "Status: Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "Tail:   Get-Content '$logFile' -Tail 40 -Wait"
Write-Host "Stop:   Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "Remove: .\deploy\windows\Register-ForecastJobsTask.ps1 -ConfigStem $ConfigStem -Remove"
