#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${1:?usage: scripts/run_native_checkpoint_scope.sh <generated-runner.py>}"

if [[ ! -f "$RUNNER" ]]; then
  echo "[model-forge] missing native checkpoint runner: $RUNNER" >&2
  exit 2
fi

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

CPU_QUOTA="${MODEL_FORGE_NATIVE_CHECKPOINT_CPU_QUOTA:-80%}"
MEMORY_MAX="${MODEL_FORGE_NATIVE_CHECKPOINT_MEMORY_MAX:-85%}"
IO_WEIGHT="${MODEL_FORGE_NATIVE_CHECKPOINT_IO_WEIGHT:-100}"
NICE_LEVEL="${MODEL_FORGE_NATIVE_CHECKPOINT_NICE:-10}"
RESERVE_CORES="${MODEL_FORGE_RESERVE_CORES:-1}"
export RESERVE_CORES

USABLE_CORES="$("$PYTHON" - <<'PY'
import os
reserve = int(os.environ.get("RESERVE_CORES", "1"))
print(max(1, (os.cpu_count() or 2) - reserve))
PY
)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$USABLE_CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$USABLE_CORES}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$USABLE_CORES}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$USABLE_CORES}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.05}"
export MODEL_FORGE_MIN_FREE_DISK_FRACTION="${MODEL_FORGE_MIN_FREE_DISK_FRACTION:-0.15}"

echo "[model-forge] runner: $RUNNER"
echo "[model-forge] CPUQuota: $CPU_QUOTA"
echo "[model-forge] MemoryMax: $MEMORY_MAX"
echo "[model-forge] IOWeight: $IO_WEIGHT"
echo "[model-forge] OMP_NUM_THREADS: $OMP_NUM_THREADS"
df -h /

cd "$REPO_DIR"

run_limited() {
  if command -v systemd-run >/dev/null 2>&1 && [[ ! -f /.dockerenv ]] && [[ "${MODEL_FORGE_DISABLE_SYSTEMD_SCOPE:-0}" != "1" ]]; then
    systemd-run --scope \
      -p "CPUQuota=$CPU_QUOTA" \
      -p "MemoryMax=$MEMORY_MAX" \
      -p "IOWeight=$IO_WEIGHT" \
      nice -n "$NICE_LEVEL" "$@"
  else
    nice -n "$NICE_LEVEL" "$@"
  fi
}

run_limited "$PYTHON" "$RUNNER"
