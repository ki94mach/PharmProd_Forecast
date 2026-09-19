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

# Hashtable splat so -Execute binds as a switch, not as ForecastRoot.
$params = @{ Config = $Config }
if ($Validate) { $params['Validate'] = $true }
if ($DryRun) { $params['DryRun'] = $true }
if ($Status) { $params['Status'] = $true }
if ($Execute) { $params['Execute'] = $true }
if ($Resume) { $params['Resume'] = $true }

& $launcher @params
exit $LASTEXITCODE
