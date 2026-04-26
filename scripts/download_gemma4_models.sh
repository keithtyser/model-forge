#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-$HOME/models}
HF_HOME=${HF_HOME:-$MODELS_DIR/.hf-cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
HF_XET_CACHE=${HF_XET_CACHE:-$HF_HOME/xet}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-32}
HF_XET_NUM_CONCURRENT_RANGE_GETS=${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}
MODEL_SET=${1:-all}

export HF_HOME
export HF_HUB_CACHE
export HF_XET_CACHE
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export HF_XET_NUM_CONCURRENT_RANGE_GETS
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}
unset HF_HUB_ENABLE_HF_TRANSFER

mkdir -p "$MODELS_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  PYTHON=${PYTHON:-python3}
fi

if [[ "${MODEL_FORGE_SKIP_HF_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -U "huggingface_hub[hf_xet]"
fi

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
echo "[model-forge] xet range gets: $HF_XET_NUM_CONCURRENT_RANGE_GETS"
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

case "$MODEL_SET" in
  all)
    download_model \
      "google/gemma-4-26B-A4B-it" \
      "$MODELS_DIR/gemma-4-26B-A4B-it"

    download_model \
      "Jackrong/Gemopus-4-26B-A4B-it" \
      "$MODELS_DIR/Gemopus-4-26B-A4B-it"

    download_model \
      "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated" \
      "$MODELS_DIR/Huihui-gemma-4-26B-A4B-it-abliterated"
    ;;
  base)
    download_model \
      "google/gemma-4-26B-A4B-it" \
      "$MODELS_DIR/gemma-4-26B-A4B-it"
    ;;
  ft)
    download_model \
      "Jackrong/Gemopus-4-26B-A4B-it" \
      "$MODELS_DIR/Gemopus-4-26B-A4B-it"
    ;;
  abli)
    download_model \
      "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated" \
      "$MODELS_DIR/Huihui-gemma-4-26B-A4B-it-abliterated"
    ;;
  *)
    echo "usage: $0 [all|base|ft|abli]" >&2
    exit 2
    ;;
esac

echo
echo "[model-forge] downloads complete"
