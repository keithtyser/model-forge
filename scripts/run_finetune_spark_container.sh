#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${MODEL_FORGE_FINETUNE_RUN_DIR:-$REPO_DIR/runs/finetune/gemma4_26b_a4b_local_ft_v0}"
IMAGE="${MODEL_FORGE_TRAIN_IMAGE:-nemotron-runner:latest}"
MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-$HOME/models}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON_OVERLAY="${MODEL_FORGE_PYTHON_OVERLAY:-$RUN_DIR/python_overlay}"
CONTAINER_PYTHONPATH="$REPO_DIR/src"
if [[ -d "$PYTHON_OVERLAY" ]]; then
  CONTAINER_PYTHONPATH="$PYTHON_OVERLAY:$CONTAINER_PYTHONPATH"
fi

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

mkdir -p "$MODELS_DIR" "$HF_CACHE"

echo "[model-forge] image: $IMAGE"
echo "[model-forge] run dir: $RUN_DIR"
echo "[model-forge] docker CPU limit: ${CPU_LIMIT}"
echo "[model-forge] docker memory limit: ${TOTAL_MEM_GB}g"
echo "[model-forge] reserve cores inside runner: ${RESERVE_CORES}"
echo "[model-forge] python path: ${CONTAINER_PYTHONPATH}"
df -h /

docker run --rm --gpus all \
  --cpus="$CPU_LIMIT" \
  --memory="${TOTAL_MEM_GB}g" \
  --memory-swap="${TOTAL_MEM_GB}g" \
  --shm-size="${MODEL_FORGE_SHM_SIZE:-64g}" \
  --pids-limit="${MODEL_FORGE_PIDS_LIMIT:-4096}" \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="$CONTAINER_PYTHONPATH" \
  -e MODEL_FORGE_DISABLE_SYSTEMD_SCOPE=1 \
  -e MODEL_FORGE_SKIP_PREPARE="${MODEL_FORGE_SKIP_PREPARE:-0}" \
  -e MODEL_FORGE_RESERVE_CORES="$RESERVE_CORES" \
  -e HF_HOME="$HF_CACHE" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  -v "$REPO_DIR:$REPO_DIR" \
  -v "$MODELS_DIR:$MODELS_DIR" \
  -v "$HF_CACHE:$HF_CACHE" \
  -w "$REPO_DIR" \
  --entrypoint bash \
  "$IMAGE" \
  -lc "PYTHON=/usr/bin/python3 '$RUN_DIR/run.sh'"
