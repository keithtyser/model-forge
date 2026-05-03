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

EXTRA_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" || "${VLLM_ENABLE_PREFIX_CACHING:-true}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [[ "${VLLM_ENABLE_CHUNKED_PREFILL:-1}" == "1" || "${VLLM_ENABLE_CHUNKED_PREFILL:-true}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
fi
if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  EXTRA_ARGS+=(--quantization "$VLLM_QUANTIZATION")
fi
if [[ -n "${VLLM_DTYPE:-}" ]]; then
  EXTRA_ARGS+=(--dtype "$VLLM_DTYPE")
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
fi
if [[ "${VLLM_TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi
if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-false}" == "true" || "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-auto-tool-choice)
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
  EXTRA_ARGS+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  EXTRA_ARGS+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [[ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--speculative-config "$VLLM_SPECULATIVE_CONFIG")
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${EXTRA_ARGS[@]}"
