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

  [string]$ForecastRoot,

  [string]$PythonExe,

  [string]$TaskName,

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
if (-not (Test-Path -LiteralPath $configAbs)) {
  throw "Config not found: $configAbs"
}

$launcher = Join-Path $ForecastRoot 'scripts\run_forecast_jobs.ps1'
if (-not (Test-Path -LiteralPath $launcher)) {
  throw "Missing launcher: $launcher"
}

# Bake Python into the task — scheduled tasks do not inherit the shell env.
if (-not $PythonExe) { $PythonExe = $env:FORECAST_PYTHON }
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
  $PythonExe = "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  $PythonExe = "$env:USERPROFILE\anaconda3\envs\forecast\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  throw "forecast python not found. Pass -PythonExe or set FORECAST_PYTHON."
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

$logDir = Join-Path $ForecastRoot 'src\data\forecast_jobs\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "$ConfigStem.log"
$wrapper = Join-Path $logDir "$ConfigStem.wrapper.ps1"

@"
param(
  [string]`$ForecastRoot,
  [string]`$Launcher,
  [string]`$Config,
  [string]`$LogFile,
  [string]`$PythonExe
)
`$ErrorActionPreference = 'Continue'
`$env:FORECAST_ROOT = `$ForecastRoot
`$env:FORECAST_PYTHON = `$PythonExe
`$env:PYTHONPATH = Join-Path `$ForecastRoot 'src'
New-Item -ItemType Directory -Force -Path (Split-Path `$LogFile) | Out-Null
Start-Transcript -Path `$LogFile -Append | Out-Null
try {
  & `$Launcher -ForecastRoot `$ForecastRoot -Config `$Config -Python `$PythonExe
  `$code = `$LASTEXITCODE
} catch {
  Write-Host `$_
  `$code = 1
} finally {
  Stop-Transcript | Out-Null
}
exit `$code
"@ | Set-Content -Path $wrapper -Encoding UTF8

$arg = @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', "`"$wrapper`"",
  '-ForecastRoot', "`"$ForecastRoot`"",
  '-Launcher', "`"$launcher`"",
  '-Config', "`"$configAbs`"",
  '-LogFile', "`"$logFile`"",
  '-PythonExe', "`"$PythonExe`""
) -join ' '

$action = New-ScheduledTaskAction `
  -Execute 'powershell.exe' `
  -Argument $arg `
  -WorkingDirectory $ForecastRoot

# Admin PowerShell: Highest is fine. Non-admin: use Limited.
$runLevel = if ([bool](([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))) {
  'Highest'
} else {
  'Limited'
}

$principal = New-ScheduledTaskPrincipal `
  -UserId $env:USERNAME `
  -LogonType Interactive `
  -RunLevel $runLevel

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 5) `
  -MultipleInstances IgnoreNew

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
try {
  Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Description "Forecast jobs: $ConfigStem; python=$PythonExe; log=$logFile" | Out-Null
} catch {
  Write-Warning "ScheduledTask register failed: $($_.Exception.Message)"
  Write-Warning "Run PowerShell as Administrator, or start interactively:"
  Write-Warning "  .\scripts\run_forecast_jobs.ps1 -ForecastRoot `"$ForecastRoot`" -Config `"$configAbs`" -Python `"$PythonExe`""
  throw
}

Write-Host "Registered task: $TaskName"
Write-Host "Python: $PythonExe"
Write-Host "Config: $configAbs"
Write-Host "Log:    $logFile"
Write-Host "Tip: leave the RDP session disconnected (not signed out)."

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
