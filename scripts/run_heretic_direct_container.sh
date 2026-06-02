#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${1:?usage: scripts/run_heretic_direct_container.sh <generated-run_heretic.py>}"
IMAGE="${MODEL_FORGE_HERETIC_IMAGE:-model-forge-heretic-tf5:latest}"
MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-$HOME/models}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ ! -f "$RUNNER" ]]; then
  echo "[model-forge] missing Heretic runner: $RUNNER" >&2
  exit 2
fi

TOTAL_MEM_GB="${MODEL_FORGE_HERETIC_DOCKER_MEMORY_GB:-$(awk '/MemTotal/ {printf "%d", ($2 / 1024 / 1024) * 0.85}' /proc/meminfo)}"
CPU_LIMIT="${MODEL_FORGE_HERETIC_DOCKER_CPUS:-$(python3 - <<'PY'
import os
cores = os.cpu_count() or 2
print(max(1, int(cores * 0.8)))
PY
)}"

mkdir -p "$MODELS_DIR" "$HF_CACHE"

echo "[model-forge] image: $IMAGE"
echo "[model-forge] runner: $RUNNER"
echo "[model-forge] docker CPU limit: $CPU_LIMIT"
echo "[model-forge] docker memory limit: ${TOTAL_MEM_GB}g"
df -h "$MODELS_DIR"

docker run --rm --gpus all \
  --network host \
  --ipc host \
  --cpus="$CPU_LIMIT" \
  --memory="${TOTAL_MEM_GB}g" \
  --memory-swap="${TOTAL_MEM_GB}g" \
  --shm-size="${MODEL_FORGE_HERETIC_SHM_SIZE:-32g}" \
  --pids-limit="${MODEL_FORGE_HERETIC_PIDS_LIMIT:-4096}" \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="$REPO_DIR/src" \
  -e HF_HOME="$HF_CACHE" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  -e MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.05}" \
  -e MODEL_FORGE_MIN_FREE_DISK_FRACTION="${MODEL_FORGE_MIN_FREE_DISK_FRACTION:-0.15}" \
  -v "$REPO_DIR:$REPO_DIR" \
  -v "$MODELS_DIR:$MODELS_DIR" \
  -v "$HF_CACHE:$HF_CACHE" \
  -w "$REPO_DIR" \
  --entrypoint python3 \
  "$IMAGE" \
  "$RUNNER"
