#Requires -Version 5.1
<#
.SYNOPSIS
  Deploy portfolio v2 FULL runs to Windows VPS and register/start Task Scheduler jobs.

.DESCRIPTION
  Copies code, configs, allowlists, sales parquet, and inventory from the laptop.
  Does NOT copy large portfolio_v21 / portfolio_a0 trees over tsclient.

  Builds those run trees ON THE VPS from existing archval* folders under
  src\data\forecast_jobs (import reusable SUCCESS, then allowlist SKIPPED marks).

  Default sequence: start V2.1 now; when V2.1 finishes, automatically start A0.

.EXAMPLE
  # On VPS: stop any stuck laptop tree-copy (Ctrl+C), then:
  cd C:\Forecast
  Copy-Item -Force '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1' `
    'C:\Forecast\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1'
  .\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1 -RebuildFromArchval -Execute

  # Start both at once (no wait; heavy):
  .\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1 -RebuildFromArchval -Execute -ParallelA0

  # Legacy slow laptop->VPS tree sync (avoid):
  .\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1 -ForceCopyTrees -Execute
#>
[CmdletBinding()]
param(
  [string]$Source = '\\tsclient\C\Users\Mohajeri.K\MyProjects\Forecast',
  [string]$Destination = 'C:\Forecast',
  [switch]$ForceCopyTrees,
  [switch]$ForceCopyPkgs,
  [switch]$RebuildFromArchval,
  [switch]$SkipRegister,
  [switch]$ParallelA0,
  [switch]$NoChainA0,
  [switch]$StatusOnly,
  [switch]$Execute
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

  # Prefer local VPS copy over slow/fragile tsclient recurse (smoke already deployed these).
  if (-not $ForceCopyPkgs -and $ProbeRel) {
    $probe = Join-Path $Destination $ProbeRel
    if (Test-Path -LiteralPath $probe) {
      Write-Host "dir   $RelDir ... skip (already on VPS; pass -ForceCopyPkgs to refresh)"
      return
    }
  }

  Write-Host "dir   $RelDir ..."
  New-Item -ItemType Directory -Force -Path $to | Out-Null

  # robocopy is more reliable than Get-ChildItem over \\tsclient\
  $robolog = Join-Path $env:TEMP ("forecast_robocopy_" + [Guid]::NewGuid().ToString('N') + ".log")
  $rc = Start-Process -FilePath 'robocopy.exe' -ArgumentList @(
    $from, $to, '/E', '/XO', '/R:2', '/W:2', '/NFL', '/NDL', '/NJH', '/NJS',
    '/XD', '__pycache__', '/XF', '*.pyc', '*.pyo'
  ) -Wait -PassThru -NoNewWindow
  # robocopy exit 0-7 = success/partial; >=8 = failure
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
  Remove-Item -LiteralPath $robolog -Force -ErrorAction SilentlyContinue
}

function Register-PortfolioTask {
  param(
    [string]$TaskName,
    [string]$ConfigRel,
    [string]$RunId,
    [switch]$WithTfCaps,
    [switch]$StartNow,
    [string]$ChainStartTaskName = ''
  )
  $py = "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe"
  if (-not (Test-Path -LiteralPath $py)) {
    throw "forecast python not found: $py"
  }
  $logDir = Join-Path $Destination 'src\data\forecast_jobs\logs'
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $logFile = Join-Path $logDir "$TaskName.log"
  $wrapper = Join-Path $logDir "$TaskName.wrapper.ps1"
  $configAbs = Join-Path $Destination $ConfigRel

  $tfBlock = ''
  if ($WithTfCaps) {
    $tfBlock = @"
`$env:TF_NUM_INTRAOP_THREADS = '1'
`$env:TF_NUM_INTEROP_THREADS = '1'
`$env:OMP_NUM_THREADS = '1'
`$env:OPENBLAS_NUM_THREADS = '1'
`$env:MKL_NUM_THREADS = '1'
`$env:NUMEXPR_NUM_THREADS = '1'
"@
  }

  $chainBlock = ''
  if ($ChainStartTaskName) {
    $chainBlock = @"
Write-Host "V2.1 finished (exit=`$code). Starting chained task: $ChainStartTaskName"
try {
  Start-ScheduledTask -TaskName '$ChainStartTaskName'
  Write-Host "Started $ChainStartTaskName"
} catch {
  Write-Host "WARN: failed to start $ChainStartTaskName : `$_"
}
"@
  }

  @"
`$ErrorActionPreference = 'Continue'
`$env:FORECAST_ROOT = '$Destination'
`$env:PYTHONPATH = Join-Path `$env:FORECAST_ROOT 'src'
`$env:PYTHONIOENCODING = 'utf-8'
$tfBlock
New-Item -ItemType Directory -Force -Path (Split-Path '$logFile') | Out-Null
Start-Transcript -Path '$logFile' -Append | Out-Null
try {
  & '$py' -m pkg.forecast_jobs --config '$configAbs' --run-id '$RunId' --execute --resume
  `$code = `$LASTEXITCODE
} catch {
  Write-Host `$_
  `$code = 1
}
$chainBlock
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
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings `
    -Description "Portfolio full $RunId" | Out-Null
  Write-Host "Registered $TaskName  log=$logFile"
  if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started $TaskName"
  }
}

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
$py = "$env:LOCALAPPDATA\anaconda3\envs\forecast\python.exe"

if ($StatusOnly) {
  & $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml --run-id portfolio_v21_backfill_v2 --status
  & $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml --run-id portfolio_a0_backfill_v2 --status
  Get-ScheduledTask -TaskName 'ForecastJobs-portfolio_*' -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo | Format-List TaskName, State, LastRunTime, NextRunTime
  exit 0
}

Write-Host "=== 1. Copy fixed code + configs + inventory ==="
@(
  'src\pkg\benchmark\backfill_runner\state.py',
  'src\pkg\benchmark\backfill_runner\types.py',
  'src\pkg\benchmark\backfill_runner\store.py',
  'src\pkg\forecast_jobs\runner.py',
  'src\pkg\forecast_jobs\plan.py',
  'scripts\prepare_portfolio_backfill_allowlist_v2.py',
  'scripts\import_portfolio_reusable_jobs.py',
  'scripts\export_portfolio_backfill_panel.py',
  'scripts\verify_portfolio_backfill_smoke_v2.py',
  'scripts\run_forecast_jobs.ps1',
  'scripts\run_portfolio_a0_backfill_v2.ps1',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml',
  'configs\forecast_jobs\portfolio_v21_backfill_v2.allowlist.json',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml',
  'configs\forecast_jobs\portfolio_a0_backfill_v2.allowlist.json',
  'src\data\results\product_universe\backfill_inventory_v2\reusable_completed_jobs.csv',
  'deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1'
) | ForEach-Object { Copy-OneFile $_ }

Copy-DirWithProgress -RelDir 'src\pkg\forecast_jobs' -ProbeRel 'src\pkg\forecast_jobs\runner.py'
Copy-DirWithProgress -RelDir 'src\pkg\ts_v2' -ProbeRel 'src\pkg\ts_v2\data.py'
Copy-DirWithProgress -RelDir 'src\pkg\ts_v3a' -ProbeRel 'src\pkg\ts_v3a\__init__.py'
Copy-DirWithProgress -RelDir 'src\data\results\product_universe\sales_full_portfolio_v1' `
  -ProbeRel 'src\data\results\product_universe\sales_full_portfolio_v1\sales.parquet'

$v21Rel = 'src\data\forecast_jobs\portfolio_v21_backfill_v2'
$a0Rel = 'src\data\forecast_jobs\portfolio_a0_backfill_v2'
$v21Root = Join-Path $Destination $v21Rel
$a0Root = Join-Path $Destination $a0Rel

function Test-ArchvalSourcesPresent {
  $jobs = Join-Path $Destination 'src\data\forecast_jobs'
  $hits = @(Get-ChildItem -LiteralPath $jobs -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'archval*' })
  if ($hits.Count -lt 1) {
    throw "No archval* folders under $jobs - cannot build portfolio from archval."
  }
  Write-Host ("  archval sources: " + (($hits | ForEach-Object { $_.Name }) -join ', '))
}

function Test-PortfolioTreeEmpty {
  param([string]$Root)
  $db = Join-Path $Root 'state.sqlite'
  if (-not (Test-Path -LiteralPath $db)) { return $true }
  $n = & $py -c @"
import sqlite3, sys
c=sqlite3.connect(sys.argv[1])
print(c.execute("select count(*) from jobs where status='SUCCESS'").fetchone()[0])
"@ $db
  return ([int]$n -eq 0)
}

if ($ForceCopyTrees) {
  Write-Host "=== 2. ForceCopyTrees: syncing full run trees from laptop (slow; avoid) ==="
  Copy-DirWithProgress -RelDir $v21Rel
  Copy-DirWithProgress -RelDir $a0Rel
} else {
  Write-Host "=== 2. Build portfolio trees on VPS from archval (no laptop tree copy) ==="
  Test-ArchvalSourcesPresent

  $needRebuild = [bool]$RebuildFromArchval
  if (-not $needRebuild) {
    if ((Test-PortfolioTreeEmpty $v21Root) -or (Test-PortfolioTreeEmpty $a0Root)) {
      Write-Host "  incomplete/missing portfolio trees -> rebuild from archval"
      $needRebuild = $true
    }
  }

  if ($needRebuild) {
    foreach ($root in @($v21Root, $a0Root)) {
      if (Test-Path -LiteralPath $root) {
        Write-Host "  remove incomplete $root"
        Remove-Item -LiteralPath $root -Recurse -Force
      }
    }

    Write-Host "  import V2.1 reusable from archval..."
    & $py scripts\import_portfolio_reusable_jobs.py `
      --config configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml `
      --engine v2.1 `
      --skip-hash-gate
    if ($LASTEXITCODE -ne 0) { throw "V2.1 archval import failed" }

    Write-Host "  import A0 reusable from archval..."
    & $py scripts\import_portfolio_reusable_jobs.py `
      --config configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml `
      --engine a0 `
      --skip-hash-gate
    if ($LASTEXITCODE -ne 0) { throw "A0 archval import failed" }
  } else {
    Write-Host "  keeping existing portfolio trees (pass -RebuildFromArchval to wipe+reimport)"
  }
}

Write-Host "=== 3. Apply allowlist SKIPPED marks ==="
& $py scripts\prepare_portfolio_backfill_allowlist_v2.py `
  --config configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml `
  --allowlist configs\forecast_jobs\portfolio_v21_backfill_v2.allowlist.json
if ($LASTEXITCODE -ne 0) { throw "V2.1 allowlist prepare failed" }

& $py scripts\prepare_portfolio_backfill_allowlist_v2.py `
  --config configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml `
  --allowlist configs\forecast_jobs\portfolio_a0_backfill_v2.allowlist.json
if ($LASTEXITCODE -ne 0) { throw "A0 allowlist prepare failed" }

Write-Host "=== 4. Status (expect V2.1 PENDING~2667 SUCCESS~156; A0 PENDING~7998 SUCCESS~471) ==="
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml --run-id portfolio_v21_backfill_v2 --status
& $py -m pkg.forecast_jobs --config configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml --run-id portfolio_a0_backfill_v2 --status

if ($SkipRegister) {
  Write-Host "SkipRegister set. Start manually with --execute --resume and explicit --run-id."
  exit 0
}

if (-not $Execute) {
  Write-Host ""
  Write-Host "Prepared only (no -Execute). Re-run with -RebuildFromArchval -Execute to start tasks,"
  Write-Host "or -Execute alone if trees already look correct above."
  exit 0
}

$a0Task = 'ForecastJobs-portfolio_a0_backfill_v2'
$v21Task = 'ForecastJobs-portfolio_v21_backfill_v2'

Write-Host "=== 5. Register scheduled tasks (survive RDP disconnect) ==="
Write-Host "ETA rough: V2.1 @ 12 workers then A0 @ 12 workers + TF caps (do not sign out)."

Register-PortfolioTask -TaskName $a0Task `
  -ConfigRel 'configs\forecast_jobs\portfolio_a0_backfill_v2.win.yaml' `
  -RunId 'portfolio_a0_backfill_v2' `
  -WithTfCaps

$chainName = ''
if (-not $NoChainA0 -and -not $ParallelA0) {
  $chainName = $a0Task
}

Register-PortfolioTask -TaskName $v21Task `
  -ConfigRel 'configs\forecast_jobs\portfolio_v21_backfill_v2.win.yaml' `
  -RunId 'portfolio_v21_backfill_v2' `
  -ChainStartTaskName $chainName `
  -StartNow

if ($ParallelA0) {
  Start-ScheduledTask -TaskName $a0Task
  Write-Host "ParallelA0: started $a0Task immediately (alongside V2.1)."
}

Write-Host ""
if ($ParallelA0) {
  Write-Host "Both tasks started in parallel."
} elseif ($NoChainA0) {
  Write-Host "V2.1 started. A0 registered only (NoChainA0)."
} else {
  Write-Host "V2.1 started. When it finishes, it will auto-start A0 ($a0Task)."
}
Write-Host "Do not Sign out. Disconnect RDP is OK."
Write-Host "Monitor:"
Write-Host "  Get-Content C:\Forecast\src\data\forecast_jobs\logs\$v21Task.log -Tail 40 -Wait"
Write-Host "  .\deploy\windows\Deploy-And-Run-PortfolioFullV2.ps1 -StatusOnly"
