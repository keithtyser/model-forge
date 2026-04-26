#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-$HOME/models}
HF_HOME=${HF_HOME:-$MODELS_DIR/.hf-cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
HF_XET_CACHE=${HF_XET_CACHE:-$HF_HOME/xet}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-32}

export HF_HOME
export HF_HUB_CACHE
export HF_XET_CACHE
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
unset HF_HUB_ENABLE_HF_TRANSFER

mkdir -p "$MODELS_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  PYTHON=${PYTHON:-python3}
fi

"$PYTHON" -m pip install -U "huggingface_hub[hf_xet]"

if [[ -x "$REPO_DIR/.venv/bin/hf" ]]; then
  HF="$REPO_DIR/.venv/bin/hf"
else
  HF=$(command -v hf)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  read -r -s -p "HF token: " HF_TOKEN
  echo
  export HF_TOKEN
fi

echo "[model-forge] cache: $HF_HOME"
echo "[model-forge] models: $MODELS_DIR"
echo "[model-forge] workers: $HF_MAX_WORKERS"
echo "[model-forge] hf: $HF"

"$HF" auth login --token "$HF_TOKEN" --force
"$HF" auth whoami

download_model() {
  local repo_id=$1
  local local_dir=$2
  echo
  echo "[model-forge] downloading $repo_id -> $local_dir"
  "$HF" download "$repo_id" \
    --local-dir "$local_dir" \
    --max-workers "$HF_MAX_WORKERS" \
    --token "$HF_TOKEN"
}

download_model \
  "google/gemma-4-26B-A4B-it" \
  "$MODELS_DIR/gemma-4-26B-A4B-it"

download_model \
  "Jackrong/Gemopus-4-26B-A4B-it" \
  "$MODELS_DIR/Gemopus-4-26B-A4B-it"

download_model \
  "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated" \
  "$MODELS_DIR/Huihui-gemma-4-26B-A4B-it-abliterated"

echo
echo "[model-forge] downloads complete"
