#!/usr/bin/env bash
# Server-friendly launcher for pkg.forecast_jobs (V2.1 / A0 / A1).
#
# Does NOT daemonize — run under systemd (pharma-ts@.service) so the process
# survives SSH disconnect. Secrets stay in /etc/pharma-ts/secrets.env and/or
# src/.env; do not put passwords in this script.
#
# Usage (from repo root, or via systemd Environment=):
#   FORECAST_JOBS_CONFIG=/opt/app/Forecast/configs/forecast_jobs/v21_a0_a1.yaml \
#   FORECAST_JOBS_RUN_ID=v21_a0_a1 \
#     ./scripts/run_forecast_jobs.sh                 # --execute --resume
#   ./scripts/run_forecast_jobs.sh --status
#   ./scripts/run_forecast_jobs.sh --validate
#   ./scripts/run_forecast_jobs.sh --dry-run
#   ./scripts/run_forecast_jobs.sh --retry-failed
#
# Environment:
#   FORECAST_ROOT           repo root (default: parent of scripts/)
#   FORECAST_PYTHON         absolute path to python (required on pip-only server)
#   FORECAST_JOBS_CONFIG    absolute path to instance YAML (required)
#   FORECAST_JOBS_RUN_ID    run id / output dir name (default: YAML stem)
#   FORECAST_JOBS_EXTRA_ARGS  optional extra CLI args (word-split)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORECAST_ROOT="$(cd "${FORECAST_ROOT:-$SCRIPT_DIR/..}" && pwd)"
cd "$FORECAST_ROOT"

# Deterministic / anti-nested-parallelism defaults (operators may override).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

resolve_python() {
  if [[ -n "${FORECAST_PYTHON:-}" ]]; then
    if [[ ! -x "$FORECAST_PYTHON" ]]; then
      echo "error: FORECAST_PYTHON is not executable: $FORECAST_PYTHON" >&2
      exit 2
    fi
    echo "$FORECAST_PYTHON"
    return 0
  fi

  local candidate
  for candidate in \
    "$FORECAST_ROOT/.venv/bin/python" \
    "$FORECAST_ROOT/venv/bin/python" \
    "$FORECAST_ROOT/.venv/bin/python3" \
    "$FORECAST_ROOT/venv/bin/python3"
  do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    echo "warning: using PATH python3; prefer FORECAST_PYTHON or a repo .venv" >&2
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "warning: using PATH python; prefer FORECAST_PYTHON or a repo .venv" >&2
    command -v python
    return 0
  fi

  echo "error: no Python found. Set FORECAST_PYTHON=/path/to/python" >&2
  exit 2
}

PYTHON_BIN="$(resolve_python)"
export PYTHONPATH="${FORECAST_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${FORECAST_JOBS_CONFIG:-}" ]]; then
  echo "error: FORECAST_JOBS_CONFIG is required (absolute path to instance YAML)" >&2
  exit 2
fi
if [[ ! -f "$FORECAST_JOBS_CONFIG" ]]; then
  echo "error: config not found: $FORECAST_JOBS_CONFIG" >&2
  exit 2
fi

CONFIG_PATH="$FORECAST_JOBS_CONFIG"
RUN_ID="${FORECAST_JOBS_RUN_ID:-}"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(basename "$CONFIG_PATH")"
  RUN_ID="${RUN_ID%.yaml}"
  RUN_ID="${RUN_ID%.yml}"
  RUN_ID="${RUN_ID%.json}"
fi

# Default to execute+resume unless the caller already passes a mode flag.
MODE_FLAGS=()
HAS_MODE=0
NEED_EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    --execute|--status|--dry-run|--validate)
      HAS_MODE=1
      ;;
    --resume|--retry-failed|--force-job)
      HAS_MODE=1
      NEED_EXECUTE=1
      ;;
  esac
done
if [[ "$HAS_MODE" -eq 0 ]]; then
  MODE_FLAGS+=(--execute --resume)
elif [[ "$NEED_EXECUTE" -eq 1 ]]; then
  # --resume / --retry-failed require --execute to actually fit models.
  HAS_EXECUTE=0
  for arg in "$@"; do
    if [[ "$arg" == "--execute" ]]; then
      HAS_EXECUTE=1
    fi
  done
  if [[ "$HAS_EXECUTE" -eq 0 ]]; then
    MODE_FLAGS+=(--execute)
  fi
fi

CMD=(
  "$PYTHON_BIN" -m pkg.forecast_jobs
  --config "$CONFIG_PATH"
  --run-id "$RUN_ID"
)
CMD+=("${MODE_FLAGS[@]}")
CMD+=("$@")

if [[ -n "${FORECAST_JOBS_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( $FORECAST_JOBS_EXTRA_ARGS )
  CMD+=("${EXTRA[@]}")
fi

echo "forecast_root=$FORECAST_ROOT"
echo "python=$PYTHON_BIN"
echo "config=$CONFIG_PATH"
echo "run_id=$RUN_ID"
echo "cmd=${CMD[*]}"

# Journald (systemd) is the primary log sink; no file tee by default.
exec "${CMD[@]}"
