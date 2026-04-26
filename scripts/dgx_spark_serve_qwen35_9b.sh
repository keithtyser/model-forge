#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SPARK_VLLM_DIR:-}" ]]; then
  if [[ -d "$HOME/projects/spark-vllm-docker" ]]; then
    SPARK_VLLM_DIR="$HOME/projects/spark-vllm-docker"
  else
    SPARK_VLLM_DIR="$HOME/spark-vllm-docker"
  fi
fi
MODEL=${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}
SERVED_MODEL_NAME=${MODEL_FORGE_SERVED_MODEL_NAME:-Qwen/Qwen3.5-9B}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
PORT=${PORT:-8000}
QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-false}

EXTRA_VLLM_ARGS=(
  --served-model-name "$SERVED_MODEL_NAME"
  --reasoning-parser qwen3
  --default-chat-template-kwargs "{\"enable_thinking\": $QWEN_ENABLE_THINKING}"
  --language-model-only
  --enable-prefix-caching
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

cd "$SPARK_VLLM_DIR"
./launch-cluster.sh --solo exec \
  vllm serve \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${EXTRA_VLLM_ARGS[@]}"
