#Requires -Version 5.1
<#
.SYNOPSIS
  Deploy portfolio residual-XGBoost v1.1 to a Windows VPS: copy code/data, prepare artifacts, print start commands.

.DESCRIPTION
  Mirrors Deploy-And-Run-PortfolioFullV2.ps1 for the residual study.

  Copies:
    - residual-xgb + portfolio_ts_lock packages, scripts, configs, deploy helpers
    - sales parquet, V2.1/A0 export panels, TS lock/contract, budget_universe (origins only)
    - existing residual_study_contract_v1.json (required reference; never overwritten)

  On the VPS then:
    - runs prepare_portfolio_residual_xgb_v1_1.py (writes v1_1 contract + promotion gates if absent)
    - validate + dry-run smoke/full
    - prints PowerShell commands for Smoke → AcceptSmoke → Full

  Does NOT open HB outcomes. HUMAN_XGB stays blocked.
  Default is prepare-only (no training). Pass -ExecuteSmoke to register/start smoke only.
  Full execute is never auto-started here - requires AcceptSmoke first.

.EXAMPLE
  # On VPS after RDP with drive redirect:
  cd C:\Forecast
  Copy-Item -Force '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1' `
    'C:\Forecast\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1'
  .\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1

  # Refresh packages + prepare (still no training):
  .\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1 -ForceCopyPkgs

  # Start smoke training via Task Scheduler (survive RDP disconnect):
  .\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1 -ExecuteSmoke

  # Status only:
  .\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1 -StatusOnly
#>
[CmdletBinding()]
param(
  [string]$Source = '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast',
  [string]$Destination = 'C:\Forecast',
  [switch]$ForceCopyPkgs,
  [switch]$ForceCopyData,
  [switch]$SkipPrepare,
  [switch]$SkipTests,
  [switch]$StatusOnly,
  [switch]$ExecuteSmoke,
  [switch]$SkipRegister
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

function Copy-DirWithProgress {
  param(
    [string]$RelDir,
    [string]$ProbeRel = ''
  )
  $from = Join-Path $Source $RelDir
  $to = Join-Path $Destination $RelDir
  if (-not (Test-Path -LiteralPath $from)) {
    Write-Warning "Skip missing dir: $RelDir"
    return
  }

  if (-not $ForceCopyPkgs -and -not $ForceCopyData -and $ProbeRel) {
    $probe = Join-Path $Destination $ProbeRel
    if (Test-Path -LiteralPath $probe) {
      Write-Host "dir   $RelDir ... skip (already on VPS; pass -ForceCopyPkgs/-ForceCopyData to refresh)"
      return
    }
  }

  Write-Host "dir   $RelDir ..."
  New-Item -ItemType Directory -Force -Path $to | Out-Null

  $rc = Start-Process -FilePath 'robocopy.exe' -ArgumentList @(
    $from, $to, '/E', '/XO', '/R:2', '/W:2', '/NFL', '/NDL', '/NJH', '/NJS',
    '/XD', '__pycache__', '.pytest_cache',
    '/XF', '*.pyc', '*.pyo'
  ) -Wait -PassThru -NoNewWindow
  if ($rc.ExitCode -ge 8) {
    Write-Warning "robocopy failed for $RelDir (exit=$($rc.ExitCode)); falling back to file loop"
    $files = @(Get-ChildItem -LiteralPath $from -Recurse -File -Force -ErrorAction SilentlyContinue |
      Where-Object {
        $p = $_.FullName
        if ($p -like '*\__pycache__\*') { return $false }
        if ($_.Extension -in @('.pyc', '.pyo')) { return $false }
        return $true
      })
    $total = $files.Count
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
      if (($i % 50 -eq 0) -or ($i -eq $total)) {
        Write-Host ("      {0}/{1}  {2}" -f $i, $total, $rel)
      }
    }
    Write-Host ("      done {0} files (fallback)" -f $total)
  } else {
    Write-Host ("      done via robocopy (exit={0})" -f $rc.ExitCode)
  }
}

function Resolve-ForecastPython {
  $candidates = @(
    "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\anaconda3\envs\forecast\python.exe",
    "$env:USERPROFILE\miniconda3\envs\forecast\python.exe",
    'python'
  )
  foreach ($c in $candidates) {
    if ($c -eq 'python') { return $c }
    if (Test-Path -LiteralPath $c) { return $c }
  }
  throw 'forecast python not found'
}

function Register-ResidualSmokeTask {
  param(
    [string]$TaskName,
    [switch]$StartNow
  )
  $py = Resolve-ForecastPython
  $logDir = Join-Path $Destination 'src\data\results\portfolio_residual_xgb_v1\logs'
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $logFile = Join-Path $logDir "$TaskName.log"
  $wrapper = Join-Path $logDir "$TaskName.wrapper.ps1"
  $runner = Join-Path $Destination 'deploy\windows\PortfolioResidualXgbV1_1.ps1'

  @"
`$ErrorActionPreference = 'Continue'
`$env:FORECAST_ROOT = '$Destination'
`$env:PYTHONPATH = Join-Path `$env:FORECAST_ROOT 'src'
`$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path (Split-Path '$logFile') | Out-Null
Start-Transcript -Path '$logFile' -Append | Out-Null
try {
  Set-Location -LiteralPath '$Destination'
  & '$runner' -Action Execute -Profile Smoke -Python '$py' -RepoRoot '$Destination'
  `$code = `$LASTEXITCODE
  if (`$code -eq 0) {
    Write-Host 'Smoke execute finished. NEXT: AcceptSmoke then Full (not auto-started).'
    Write-Host '  .\deploy\windows\PortfolioResidualXgbV1_1.ps1 -Action AcceptSmoke'
  }
} catch {
  Write-Host `$_
  `$code = 1
}
try { Stop-Transcript | Out-Null } catch {}
exit `$code
"@ | Set-Content -Path $wrapper -Encoding ASCII

  $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`""
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $Destination
  $runLevel = if ([bool](([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))) {
    'Highest'
  } else {
    'Limited'
  }
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel $runLevel
  $settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings `
    -Description 'Residual XGB v1.1 smoke (140107 only); AcceptSmoke required before Full' | Out-Null
  Write-Host "Registered $TaskName  log=$logFile"
  if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started $TaskName"
  }
}

# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $Source)) {
  throw "Source not found: $Source - enable Local Resources -> Drives in RDP."
}
if (-not (Test-Path -LiteralPath $Destination)) {
  throw "Destination not found: $Destination"
}

Set-Location -LiteralPath $Destination
$env:FORECAST_ROOT = $Destination
$env:PYTHONPATH = Join-Path $Destination 'src'
$env:PYTHONIOENCODING = 'utf-8'
$py = Resolve-ForecastPython

$smokeCfg = 'configs\forecast_jobs\portfolio_residual_xgb_v1_1.smoke.win.yaml'
$fullCfg = 'configs\forecast_jobs\portfolio_residual_xgb_v1_1.full.win.yaml'
$runnerPs1 = 'deploy\windows\PortfolioResidualXgbV1_1.ps1'
$smokeTask = 'ForecastJobs-portfolio_residual_xgb_v1_1_smoke'

if ($StatusOnly) {
  & $py scripts\run_portfolio_residual_xgb_v1_1.py --config $smokeCfg --action status
  & $py scripts\run_portfolio_residual_xgb_v1_1.py --config $fullCfg --action status
  $acc = Join-Path $Destination 'src\data\results\portfolio_residual_xgb_v1\smoke_acceptance_v1.json'
  Write-Host ("smoke_acceptance_present=" + (Test-Path -LiteralPath $acc))
  Get-ScheduledTask -TaskName $smokeTask -ErrorAction SilentlyContinue |
    Get-ScheduledTaskInfo |
    Format-List TaskName, State, LastRunTime, NextRunTime
  exit 0
}

Write-Host "=== 1. Copy residual-XGBoost code, scripts, configs ==="
@(
  'scripts\prepare_portfolio_residual_xgb_v1_1.py',
  'scripts\run_portfolio_residual_xgb_v1_1.py',
  'scripts\evaluate_portfolio_ts_architecture_postlock_v2.py',
  'configs\forecast_jobs\portfolio_residual_xgb_v1_1.smoke.win.yaml',
  'configs\forecast_jobs\portfolio_residual_xgb_v1_1.full.win.yaml',
  'configs\forecast_jobs\portfolio_residual_xgb_v1.win.yaml',
  'configs\forecast_jobs\portfolio_residual_xgb_v1.allowlist.json',
  'deploy\windows\PortfolioResidualXgbV1_1.ps1',
  'deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1',
  'docs\portfolio_residual_xgb_v1.md',
  'docs\portfolio_ts_architecture_lock_v1.md'
) | ForEach-Object { Copy-OneFile $_ }

# Packages (ForceCopyPkgs refreshes even if probe exists)
$script:ForceCopyPkgs = [bool]$ForceCopyPkgs
Copy-DirWithProgress -RelDir 'src\pkg\research\portfolio_residual_xgb_v1' `
  -ProbeRel 'src\pkg\research\portfolio_residual_xgb_v1\prepare_v1_1.py'
Copy-DirWithProgress -RelDir 'src\pkg\portfolio_ts_lock' `
  -ProbeRel 'src\pkg\portfolio_ts_lock\constants.py'
Copy-DirWithProgress -RelDir 'src\pkg\benchmark' `
  -ProbeRel 'src\pkg\benchmark\config.py'
Copy-DirWithProgress -RelDir 'tests\portfolio_residual_xgb_v1' `
  -ProbeRel 'tests\portfolio_residual_xgb_v1\test_smoke_acceptance.py'
Copy-DirWithProgress -RelDir 'tests\portfolio_ts_lock' `
  -ProbeRel 'tests\portfolio_ts_lock\test_postlock_stability_v2.py'

Write-Host "=== 2. Copy required data panels (sales, exports, lock, budget) ==="
$script:ForceCopyData = [bool]$ForceCopyData
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\sales_full_portfolio_v1' `
  -ProbeRel 'src\data\results\product_universe\sales_full_portfolio_v1\sales.parquet'
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\portfolio_v21_backfill_v2_export' `
  -ProbeRel 'src\data\results\product_universe\portfolio_v21_backfill_v2_export\forecasts_panel.parquet'
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\portfolio_a0_backfill_v2_export' `
  -ProbeRel 'src\data\results\product_universe\portfolio_a0_backfill_v2_export\forecasts_panel.parquet'
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\portfolio_ts_architecture_lock_v1' `
  -ProbeRel 'src\data\results\product_universe\portfolio_ts_architecture_lock_v1\ts_architecture_lock.json'

# budget_universe: origins inventory only (prepare HB audit); do not open outcomes
$budRel = 'src\data\benchmarks\v1\budget_universe.parquet'
Copy-OneFile $budRel

# residual results: must include immutable v1 contract; copy tree but never wipe VPS-only run outputs if present
$residRel = 'src\data\results\portfolio_residual_xgb_v1'
$residFrom = Join-Path $Source $residRel
$residTo = Join-Path $Destination $residRel
if (Test-Path -LiteralPath $residFrom) {
  New-Item -ItemType Directory -Force -Path $residTo | Out-Null
  # Copy contract/prep artifacts file-by-file; skip runs/ to avoid clobbering VPS smoke outputs
  Get-ChildItem -LiteralPath $residFrom -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $destFile = Join-Path $residTo $_.Name
    # Never overwrite immutable contracts/gates if already present on VPS
    if ($_.Name -in @(
        'residual_study_contract_v1.json',
        'residual_study_contract_v1_1.json',
        'residual_study_contract_v1_1.json.sha256',
        'residual_promotion_gates_v1.json',
        'residual_promotion_gates_v1.json.sha256',
        'smoke_acceptance_v1.json'
      ) -and (Test-Path -LiteralPath $destFile)) {
      Write-Host "keep  $residRel\$($_.Name) (immutable/already on VPS)"
      return
    }
    Copy-Item -LiteralPath $_.FullName -Destination $destFile -Force
    Write-Host "file  $residRel\$($_.Name)"
  }
} else {
  Write-Warning "Missing laptop residual results dir: $residRel"
}

$v1Contract = Join-Path $Destination 'src\data\results\portfolio_residual_xgb_v1\residual_study_contract_v1.json'
if (-not (Test-Path -LiteralPath $v1Contract)) {
  throw "Required residual_study_contract_v1.json missing at $v1Contract - copy from laptop first."
}

Write-Host "=== 3. Prepare v1.1 artifacts on VPS (no training) ==="
Write-Host "  (Immutable contracts/gates are reused if already present.)"
if (-not $SkipPrepare) {
  & $py scripts\prepare_portfolio_residual_xgb_v1_1.py
  if ($LASTEXITCODE -ne 0) { throw "prepare_portfolio_residual_xgb_v1_1 failed: $LASTEXITCODE" }
} else {
  Write-Host "SkipPrepare set."
}

Write-Host "=== 4. Validate + dry-run (no execute) ==="
& $py scripts\run_portfolio_residual_xgb_v1_1.py --config $smokeCfg --action validate
if ($LASTEXITCODE -ne 0) { throw "smoke validate failed" }
& $py scripts\run_portfolio_residual_xgb_v1_1.py --config $smokeCfg --action dry-run
if ($LASTEXITCODE -ne 0) { throw "smoke dry-run failed" }
& $py scripts\run_portfolio_residual_xgb_v1_1.py --config $fullCfg --action dry-run
if ($LASTEXITCODE -ne 0) { throw "full dry-run failed" }

if (-not $SkipTests) {
  Write-Host "=== 5. Unit tests ==="
  & $py -m pytest tests\portfolio_residual_xgb_v1 -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }
} else {
  Write-Host "SkipTests set."
}

Write-Host ""
Write-Host "=== Deploy prepare complete (training not started) ==="
Write-Host "HUMAN_XGB=BLOCKED_NO_DEVELOPMENT_VINTAGES  hb_outcomes_inspected=false"
Write-Host ""
Write-Host "Start commands (run on VPS in order):"
Write-Host ""
Write-Host "  cd $Destination"
Write-Host "  `$env:PYTHONPATH = 'src'"
Write-Host ""
Write-Host "  # 1) Smoke execute (origin 140107 only)"
Write-Host "  .\$runnerPs1 -Action Execute -Profile Smoke"
Write-Host ""
Write-Host "  # 2) Smoke acceptance (REQUIRED before Full)"
Write-Host "  .\$runnerPs1 -Action AcceptSmoke"
Write-Host ""
Write-Host "  # 3) Optional smoke diagnostic eval (does not touch promotion gates)"
Write-Host "  .\$runnerPs1 -Action Evaluate -Profile Smoke"
Write-Host ""
Write-Host "  # 4) Full execute (7 origins) - blocked until AcceptSmoke passes"
Write-Host "  .\$runnerPs1 -Action Execute -Profile Full"
Write-Host ""
Write-Host "  # 5) Full evaluation against locked promotion gates"
Write-Host "  .\$runnerPs1 -Action Evaluate -Profile Full"
Write-Host ""
Write-Host "  # Status"
Write-Host "  .\deploy\windows\Deploy-And-Run-PortfolioResidualXgbV1_1.ps1 -StatusOnly"
Write-Host ""

if ($SkipRegister) {
  Write-Host "SkipRegister set. Not registering Task Scheduler jobs."
  exit 0
}

if (-not $ExecuteSmoke) {
  Write-Host "Prepared only (no -ExecuteSmoke). Re-run with -ExecuteSmoke to start smoke via Task Scheduler."
  exit 0
}

Write-Host "=== Register + start smoke Task Scheduler job ==="
Register-ResidualSmokeTask -TaskName $smokeTask -StartNow
Write-Host "Do not Sign out. Disconnect RDP is OK."
Write-Host "Monitor:"
Write-Host "  Get-Content $Destination\src\data\results\portfolio_residual_xgb_v1\logs\$smokeTask.log -Tail 40 -Wait"
Write-Host "After smoke SUCCESS:"
Write-Host "  .\$runnerPs1 -Action AcceptSmoke"
Write-Host "  .\$runnerPs1 -Action Execute -Profile Full"
