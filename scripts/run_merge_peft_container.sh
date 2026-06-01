#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MODEL_FORGE_MERGE_IMAGE:-model-forge-posttrain-tf5:latest}"
MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-$HOME/models}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
NICE_LEVEL="${MODEL_FORGE_NICE:-10}"

TOTAL_MEM_GB="$(awk '/MemTotal/ {printf "%d", ($2 / 1024 / 1024) * 0.85}' /proc/meminfo)"
TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN)"
CPU_LIMIT="$(python3 - <<'PY'
import os
cores = os.cpu_count() or 2
print(max(1, int(cores * 0.8)))
PY
)"
RESERVE_CORES="$(( TOTAL_CPUS - CPU_LIMIT ))"
if [[ "$RESERVE_CORES" -lt 1 ]]; then
  RESERVE_CORES=1
fi

GPU_ARGS=()
case "${MODEL_FORGE_MERGE_GPUS:-all}" in
  none|None|NONE|0)
    ;;
  *)
    GPU_ARGS=(--gpus "${MODEL_FORGE_MERGE_GPUS:-all}")
    ;;
esac

mkdir -p "$MODELS_DIR" "$HF_CACHE"

echo "[model-forge] image: $IMAGE"
echo "[model-forge] repo: $REPO_DIR"
echo "[model-forge] models: $MODELS_DIR"
echo "[model-forge] docker CPU limit: $CPU_LIMIT"
echo "[model-forge] docker memory limit: ${TOTAL_MEM_GB}g"
echo "[model-forge] reserve cores inside runner: $RESERVE_CORES"
df -h "$MODELS_DIR"

docker run --rm \
  "${GPU_ARGS[@]}" \
  --cpus="$CPU_LIMIT" \
  --memory="${TOTAL_MEM_GB}g" \
  --memory-swap="${TOTAL_MEM_GB}g" \
  --shm-size="${MODEL_FORGE_SHM_SIZE:-16g}" \
  --pids-limit="${MODEL_FORGE_PIDS_LIMIT:-4096}" \
  --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS="$CPU_LIMIT" \
  -e MKL_NUM_THREADS="$CPU_LIMIT" \
  -e OPENBLAS_NUM_THREADS="$CPU_LIMIT" \
  -e NUMEXPR_NUM_THREADS="$CPU_LIMIT" \
  -e MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.05}" \
  -e MODEL_FORGE_MIN_FREE_DISK_FRACTION="${MODEL_FORGE_MIN_FREE_DISK_FRACTION:-0.15}" \
  -e HF_HOME="$HF_CACHE" \
  -v "$REPO_DIR:$REPO_DIR" \
  -v "$MODELS_DIR:$MODELS_DIR" \
  -v "$HF_CACHE:$HF_CACHE" \
  -w "$REPO_DIR" \
  --entrypoint nice \
  "$IMAGE" \
  -n "$NICE_LEVEL" python3 scripts/merge_peft_adapter.py "$@"
