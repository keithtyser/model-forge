#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR=${SPARK_VLLM_DIR:-$HOME/spark-vllm-docker}
MODEL=${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
PORT=${PORT:-8000}

cd "$SPARK_VLLM_DIR"
./launch-cluster.sh --solo exec \
  vllm serve \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN"
