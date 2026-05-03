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
MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-$HOME/models}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
PORT=${PORT:-8000}

EXTRA_VLLM_ARGS=(
  --served-model-name "$SERVED_MODEL_NAME"
  --language-model-only
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" || "${VLLM_ENABLE_PREFIX_CACHING:-true}" == "true" ]]; then
  EXTRA_VLLM_ARGS+=(--enable-prefix-caching)
fi
if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--quantization "$VLLM_QUANTIZATION")
fi
if [[ -n "${VLLM_DTYPE:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--dtype "$VLLM_DTYPE")
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
fi
if [[ -n "${VLLM_MOE_BACKEND:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--moe-backend "$VLLM_MOE_BACKEND")
fi
if [[ "${VLLM_ENABLE_CHUNKED_PREFILL:-0}" == "1" || "${VLLM_ENABLE_CHUNKED_PREFILL:-false}" == "true" ]]; then
  EXTRA_VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB")
fi
if [[ -n "${VLLM_SWAP_SPACE:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--swap-space "$VLLM_SWAP_SPACE")
fi
if [[ -n "${VLLM_MAX_NUM_SEQS:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if [[ "${VLLM_TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  EXTRA_VLLM_ARGS+=(--trust-remote-code)
fi
if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-false}" == "true" || "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  EXTRA_VLLM_ARGS+=(--enable-auto-tool-choice)
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [[ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--speculative-config "$VLLM_SPECULATIVE_CONFIG")
fi

if [[ "$MODEL" = /* && -d "$MODELS_DIR" ]]; then
  MODEL_MOUNT="-v $MODELS_DIR:$MODELS_DIR:ro"
  if [[ -n "${VLLM_SPARK_EXTRA_DOCKER_ARGS:-}" ]]; then
    case " $VLLM_SPARK_EXTRA_DOCKER_ARGS " in
      *" $MODEL_MOUNT "*) ;;
      *) export VLLM_SPARK_EXTRA_DOCKER_ARGS="$VLLM_SPARK_EXTRA_DOCKER_ARGS $MODEL_MOUNT" ;;
    esac
  else
    export VLLM_SPARK_EXTRA_DOCKER_ARGS="$MODEL_MOUNT"
  fi
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
