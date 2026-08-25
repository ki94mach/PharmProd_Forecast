#!/usr/bin/env bash
# Server-friendly launcher for the TS historical backfill.
#
# Does NOT daemonize — run under systemd (preferred) or tmux/screen so the
# process survives SSH disconnect. Credentials stay in the project's .env
# (loaded by python-dotenv inside the CLI); do not put secrets in this script.
#
# Usage:
#   ./scripts/run_ts_backfill.sh                  # --resume (default)
#   ./scripts/run_ts_backfill.sh --status
#   ./scripts/run_ts_backfill.sh --retry-failed
#   ./scripts/run_ts_backfill.sh --dry-run
#   ./scripts/run_ts_backfill.sh --workers 4 --quarter 1405Q1
#
# Environment overrides (optional):
#   FORECAST_ROOT          repo root (default: parent of scripts/)
#   FORECAST_CONDA_ENV     conda env name (default: forecast)
#   FORECAST_PYTHON        absolute path to python (skips conda activate)
#   BACKFILL_ENGINE        default: v2
#   BACKFILL_VINTAGES      default: ts_backfill_1401Q1_1405Q2
#   BACKFILL_UNIVERSE      default: mvp_products
#   BACKFILL_WORKERS       default: 1
#   BACKFILL_EXPERIMENT_ID optional experiment id override
#   BACKFILL_EXTRA_ARGS    extra CLI args (word-split; prefer passing flags)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORECAST_ROOT="$(cd "${FORECAST_ROOT:-$SCRIPT_DIR/..}" && pwd)"
cd "$FORECAST_ROOT"

LOG_DIR="${BACKFILL_LOG_DIR:-$FORECAST_ROOT/data/backfills/logs}"
mkdir -p "$LOG_DIR"

# Deterministic / anti-nested-parallelism defaults (operators may override).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

activate_conda_env() {
  local env_name="${FORECAST_CONDA_ENV:-forecast}"
  if [[ -n "${FORECAST_PYTHON:-}" ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH; set FORECAST_PYTHON to the env python" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$env_name"
}

activate_conda_env

if [[ -n "${FORECAST_PYTHON:-}" ]]; then
  PYTHON_BIN="$FORECAST_PYTHON"
else
  PYTHON_BIN="$(command -v python)"
fi

ENGINE="${BACKFILL_ENGINE:-v2}"
VINTAGES="${BACKFILL_VINTAGES:-ts_backfill_1401Q1_1405Q2}"
UNIVERSE="${BACKFILL_UNIVERSE:-mvp_products}"
WORKERS="${BACKFILL_WORKERS:-1}"

# Default to resume unless the caller already passes a mode flag.
MODE_FLAGS=()
HAS_MODE=0
for arg in "$@"; do
  case "$arg" in
    --resume|--retry-failed|--status|--dry-run|--force-job)
      HAS_MODE=1
      ;;
  esac
done
if [[ "$HAS_MODE" -eq 0 ]]; then
  MODE_FLAGS+=(--resume)
fi

CMD=(
  "$PYTHON_BIN" -m pkg.benchmark.backfill_runner
  --engine "$ENGINE"
  --vintages "$VINTAGES"
  --universe "$UNIVERSE"
  --workers "$WORKERS"
)
if [[ -n "${BACKFILL_EXPERIMENT_ID:-}" ]]; then
  CMD+=(--experiment-id "$BACKFILL_EXPERIMENT_ID")
fi
CMD+=("${MODE_FLAGS[@]}")
CMD+=("$@")

# Optional space-separated extras from env (avoid putting secrets here).
if [[ -n "${BACKFILL_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( $BACKFILL_EXTRA_ARGS )
  CMD+=("${EXTRA[@]}")
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/backfill_${ENGINE}_${STAMP}.log"
LATEST_LOG="$LOG_DIR/backfill_latest.log"

echo "forecast_root=$FORECAST_ROOT"
echo "python=$PYTHON_BIN"
echo "log=$RUN_LOG"
echo "cmd=${CMD[*]}"

set +e
"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

ln -sfn "$(basename "$RUN_LOG")" "$LATEST_LOG" 2>/dev/null || cp -f "$RUN_LOG" "$LATEST_LOG"

# Pass through CLI exit codes:
#   0 = clean success / dry-run / status ok
#   1 = orchestration finished; one or more SKU-vintage jobs failed
#   2 = bad args / engine unavailable
#   3 = run lock held
#   4 = experiment manifest / config-hash conflict
# Fatal orchestration errors are non-zero (2+). Exit 1 is not "unexpected crash".
exit "$rc"
