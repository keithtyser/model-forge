#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-Qwen/Qwen3.5-9B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Install it first." >&2
  exit 1
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --enable-chunked-prefill
