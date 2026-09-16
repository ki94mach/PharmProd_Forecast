# Portfolio A0 backfill v2 launcher — TF thread caps then delegates.
# Does NOT execute unless -Execute is passed.
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [switch]$Validate,
    [switch]$DryRun,
    [switch]$Status,
    [switch]$Execute,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:TF_NUM_INTRAOP_THREADS = "1"
$env:TF_NUM_INTEROP_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $here "run_forecast_jobs.ps1"

$argsList = @("-Config", $Config)
if ($Validate) { $argsList += "-Validate" }
if ($DryRun) { $argsList += "-DryRun" }
if ($Status) { $argsList += "-Status" }
if ($Execute) { $argsList += "-Execute" }
if ($Resume) { $argsList += "-Resume" }

& $launcher @argsList
exit $LASTEXITCODE
