#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SPARK_VLLM_DIR:-}" ]]; then
  if [[ -d "$HOME/projects/spark-vllm-docker" ]]; then
    SPARK_VLLM_DIR="$HOME/projects/spark-vllm-docker"
  else
    SPARK_VLLM_DIR="$HOME/spark-vllm-docker"
  fi
fi

MODEL=${MODEL_FORGE_MODEL:-google/gemma-4-26B-A4B-it}
SERVED_MODEL_NAME=${MODEL_FORGE_SERVED_MODEL_NAME:-$MODEL}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
PORT=${PORT:-8000}

EXTRA_VLLM_ARGS=(
  --served-model-name "$SERVED_MODEL_NAME"
  --language-model-only
  --enable-prefix-caching
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--quantization "$VLLM_QUANTIZATION")
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
fi
if [[ -n "${VLLM_MOE_BACKEND:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--moe-backend "$VLLM_MOE_BACKEND")
fi
if [[ "${VLLM_TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  EXTRA_VLLM_ARGS+=(--trust-remote-code)
fi

cd "$SPARK_VLLM_DIR"
./launch-cluster.sh --solo exec \
  vllm serve \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${EXTRA_VLLM_ARGS[@]}"
