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
SERVED_MODEL_NAME=${MODEL_FORGE_SERVED_MODEL_NAME:-$MODEL}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-false}
DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-"{\"enable_thinking\": ${QWEN_ENABLE_THINKING}}"}
SPARK_VLLM_IMAGE=${MODEL_FORGE_SPARK_VLLM_IMAGE:-vllm-node}
SPARK_CONTAINER_NAME=${MODEL_FORGE_SPARK_CONTAINER_NAME:-vllm_node}
SPARK_MASTER_PORT=${MODEL_FORGE_SPARK_MASTER_PORT:-29501}
TENSOR_PARALLEL_SIZE=${MODEL_FORGE_TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL_SIZE:-}}
MODEL_FORGE_SPARK_CLUSTER=${MODEL_FORGE_SPARK_CLUSTER:-}
MODEL_FORGE_SPARK_CLUSTER_NODES=${MODEL_FORGE_SPARK_CLUSTER_NODES:-}
MODEL_FORGE_MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-$HOME/models}

if [[ ! -d "$SPARK_VLLM_DIR" ]]; then
  echo "[model-forge] spark-vllm-docker dir not found: $SPARK_VLLM_DIR" >&2
  exit 1
fi

if [[ -d "$MODEL_FORGE_MODELS_DIR" ]]; then
  mount_arg="-v ${MODEL_FORGE_MODELS_DIR}:${MODEL_FORGE_MODELS_DIR}:ro"
  if [[ "${VLLM_SPARK_EXTRA_DOCKER_ARGS:-}" != *"$mount_arg"* ]]; then
    export VLLM_SPARK_EXTRA_DOCKER_ARGS="${VLLM_SPARK_EXTRA_DOCKER_ARGS:-} $mount_arg"
  fi
fi

VLLM_ARGS=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}"
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  --language-model-only
  --enable-prefix-caching
)

if [[ "${VLLM_ENABLE_CHUNKED_PREFILL:-0}" == "1" || "${VLLM_ENABLE_CHUNKED_PREFILL:-false}" == "true" ]]; then
  VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
  VLLM_ARGS+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
fi
if [[ -n "${VLLM_DTYPE:-}" ]]; then
  VLLM_ARGS+=(--dtype "$VLLM_DTYPE")
fi
if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  VLLM_ARGS+=(--quantization "$VLLM_QUANTIZATION")
fi
if [[ -n "${VLLM_LOAD_FORMAT:-}" ]]; then
  VLLM_ARGS+=(--load-format "$VLLM_LOAD_FORMAT")
fi
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  VLLM_ARGS+=(--attention-backend "$VLLM_ATTENTION_BACKEND")
fi
if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
  VLLM_ARGS+=(--cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB")
fi
if [[ -n "${VLLM_SWAP_SPACE:-}" ]]; then
  VLLM_ARGS+=(--swap-space "$VLLM_SWAP_SPACE")
fi
if [[ -n "$VLLM_MAX_NUM_SEQS" ]]; then
  VLLM_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if [[ "${VLLM_TRUST_REMOTE_CODE:-false}" == "true" || "${VLLM_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi
if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-false}" == "true" || "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice)
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
  VLLM_ARGS+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]]; then
  VLLM_ARGS+=(--chat-template "$VLLM_CHAT_TEMPLATE")
fi
if [[ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]]; then
  VLLM_ARGS+=(--speculative-config "$VLLM_SPECULATIVE_CONFIG")
fi
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_vllm_args <<< "$VLLM_EXTRA_ARGS"
  VLLM_ARGS+=("${extra_vllm_args[@]}")
fi
if [[ "${VLLM_ENABLE_LORA:-0}" == "1" || "${VLLM_ENABLE_LORA:-false}" == "true" ]]; then
  VLLM_ARGS+=(--enable-lora)
  if [[ -n "${MODEL_FORGE_LORA_MODULES:-}" ]]; then
    VLLM_ARGS+=(--lora-modules "$MODEL_FORGE_LORA_MODULES")
  fi
  if [[ -n "${VLLM_MAX_LORAS:-}" ]]; then
    VLLM_ARGS+=(--max-loras "$VLLM_MAX_LORAS")
  fi
  if [[ -n "${VLLM_MAX_LORA_RANK:-}" ]]; then
    VLLM_ARGS+=(--max-lora-rank "$VLLM_MAX_LORA_RANK")
  fi
fi

CLUSTER_ARGS=(-t "$SPARK_VLLM_IMAGE" --name "$SPARK_CONTAINER_NAME" --master-port "$SPARK_MASTER_PORT")

if [[ -n "$MODEL_FORGE_SPARK_CLUSTER_NODES" ]]; then
  CLUSTER_ARGS+=(-n "$MODEL_FORGE_SPARK_CLUSTER_NODES")
fi
if [[ -n "${MODEL_FORGE_SPARK_ETH_IF:-}" ]]; then
  CLUSTER_ARGS+=(--eth-if "$MODEL_FORGE_SPARK_ETH_IF")
fi
if [[ -n "${MODEL_FORGE_SPARK_IB_IF:-}" ]]; then
  CLUSTER_ARGS+=(--ib-if "$MODEL_FORGE_SPARK_IB_IF")
fi
if [[ -n "${MODEL_FORGE_SPARK_CONFIG:-}" ]]; then
  CLUSTER_ARGS+=(--config "$MODEL_FORGE_SPARK_CONFIG")
fi
if [[ "${MODEL_FORGE_SPARK_NON_PRIVILEGED:-0}" == "1" || "${MODEL_FORGE_SPARK_NON_PRIVILEGED:-false}" == "true" ]]; then
  CLUSTER_ARGS+=(--non-privileged --mem-limit-gb "${MODEL_FORGE_SPARK_MEM_LIMIT_GB:-110}" --pids-limit "${MODEL_FORGE_SPARK_PIDS_LIMIT:-4096}" --shm-size-gb "${MODEL_FORGE_SPARK_SHM_SIZE_GB:-64}")
fi
if [[ -n "${MODEL_FORGE_SPARK_APPLY_MODS:-}" ]]; then
  IFS=',' read -r -a mods <<< "$MODEL_FORGE_SPARK_APPLY_MODS"
  for mod in "${mods[@]}"; do
    [[ -n "$mod" ]] && CLUSTER_ARGS+=(--apply-mod "$mod")
  done
fi

if [[ "$MODEL_FORGE_SPARK_CLUSTER" == "1" || "$MODEL_FORGE_SPARK_CLUSTER" == "true" || -n "$MODEL_FORGE_SPARK_CLUSTER_NODES" ]]; then
  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
  VLLM_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --distributed-executor-backend ray)
else
  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
  CLUSTER_ARGS+=(--solo)
  VLLM_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

cd "$SPARK_VLLM_DIR"
if [[ "${MODEL_FORGE_DRY_RUN:-0}" == "1" || "${MODEL_FORGE_DRY_RUN:-false}" == "true" ]]; then
  printf '[model-forge] dry-run command:'
  printf ' %q' ./launch-cluster.sh "${CLUSTER_ARGS[@]}" exec vllm "${VLLM_ARGS[@]}"
  printf '\n'
  exit 0
fi
exec ./launch-cluster.sh "${CLUSTER_ARGS[@]}" exec vllm "${VLLM_ARGS[@]}"
