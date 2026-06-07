#!/usr/bin/env bash
# Canonical GPU container runner for model-forge post-training stages.
# GPU + host user, writable caches, repo/models/HF mounts, PYTHONPATH=src, NO systemd-run
# (works over SSH). Overlays writable dirs at the container ~/.cache /.config /.local so
# libs that ignore XDG (flashinfer/vllm) do not hit the root-owned dotdirs.
#   scripts/run_in_container.sh <executable> [args...]
# Overridable: MODEL_FORGE_POSTTRAIN_IMAGE, MODEL_FORGE_MODELS_DIR, MODEL_FORGE_HF_HOME, MODEL_FORGE_GPUS.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MODEL_FORGE_POSTTRAIN_IMAGE:-model-forge-posttrain-tf5:latest}"
MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-$HOME/models}"
HF_CACHE="${MODEL_FORGE_HF_HOME:-$HOME/.forgewright/hf_home}"
GPUS="${MODEL_FORGE_GPUS:-all}"
if [[ $# -lt 1 ]]; then echo "usage: run_in_container.sh <executable> [args...]" >&2; exit 2; fi
mkdir -p "$HF_CACHE" "$MODELS_DIR" "$HF_CACHE/dotcache" "$HF_CACHE/dotconfig" "$HF_CACHE/dotlocal"
exec docker run --rm --gpus "$GPUS" \
  --user "$(id -u):$(id -g)" -e HOME="$HOME" --shm-size="${MODEL_FORGE_SHM_SIZE:-16g}" \
  -e HF_HOME="$HF_CACHE" -e HF_DATASETS_CACHE="$HF_CACHE/datasets" -e HF_HUB_DISABLE_XET=1 \
  -e XDG_CACHE_HOME="$HF_CACHE/cache" -e TRITON_CACHE_DIR="$HF_CACHE/triton" \
  -e TORCHINDUCTOR_CACHE_DIR="$HF_CACHE/inductor" -e VLLM_CACHE_ROOT="$HF_CACHE/vllm" \
  -e PYTHONPATH="$REPO_DIR/src" \
  -e MODEL_FORGE_MIN_FREE_DISK_FRACTION="${MODEL_FORGE_MIN_FREE_DISK_FRACTION:-0.05}" \
  -e MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.03}" \
  -v "$REPO_DIR:$REPO_DIR" -v "$MODELS_DIR:$MODELS_DIR" -v "$HF_CACHE:$HF_CACHE" \
  -v "$HF_CACHE/dotcache:$HOME/.cache" -v "$HF_CACHE/dotconfig:$HOME/.config" -v "$HF_CACHE/dotlocal:$HOME/.local" \
  -w "$REPO_DIR" --entrypoint "$1" "$IMAGE" "${@:2}"
