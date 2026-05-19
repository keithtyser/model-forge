#!/usr/bin/env bash
set -euo pipefail

if docker ps --format '{{.Names}} {{.Image}} {{.Command}}' | grep -Eiq 'vllm|vllm_node'; then
  echo "[model-forge] refusing to start teacher server: a vLLM container appears to be running" >&2
  docker ps --format '  {{.Names}} {{.Image}} {{.Status}}'
  exit 2
fi

MODEL_FORGE_MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-"$HOME/models"}
SPARK_VLLM_DOCKER_DIR=${SPARK_VLLM_DOCKER_DIR:-"$HOME/projects/spark-vllm-docker"}
MODEL_FORGE_TEACHER_MODEL=${MODEL_FORGE_TEACHER_MODEL:-"$MODEL_FORGE_MODELS_DIR/Qwen3.5-9B"}
MODEL_FORGE_TEACHER_NAME=${MODEL_FORGE_TEACHER_NAME:-local/qwen35-9b-teacher}
PORT=${PORT:-8011}

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1}
VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-fp8_e4m3}
VLLM_MEM_LIMIT_GB=${VLLM_MEM_LIMIT_GB:-90}
VLLM_SHM_SIZE_GB=${VLLM_SHM_SIZE_GB:-32}
VLLM_PIDS_LIMIT=${VLLM_PIDS_LIMIT:-4096}
QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-false}

if [[ ! -d "$SPARK_VLLM_DOCKER_DIR" ]]; then
  echo "[model-forge] spark-vllm-docker dir not found: $SPARK_VLLM_DOCKER_DIR" >&2
  exit 2
fi
if [[ ! -e "$MODEL_FORGE_TEACHER_MODEL" ]]; then
  echo "[model-forge] teacher model not found: $MODEL_FORGE_TEACHER_MODEL" >&2
  exit 2
fi

echo "[model-forge] teacher model: $MODEL_FORGE_TEACHER_MODEL"
echo "[model-forge] served name:   $MODEL_FORGE_TEACHER_NAME"
echo "[model-forge] port:          $PORT"
echo "[model-forge] memory limit:  ${VLLM_MEM_LIMIT_GB}GiB"

export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v ${MODEL_FORGE_MODELS_DIR}:${MODEL_FORGE_MODELS_DIR}:ro"
cd "$SPARK_VLLM_DOCKER_DIR"
exec ./launch-cluster.sh \
  --solo \
  --non-privileged \
  --mem-limit-gb "$VLLM_MEM_LIMIT_GB" \
  --mem-swap-limit-gb "$VLLM_MEM_LIMIT_GB" \
  --pids-limit "$VLLM_PIDS_LIMIT" \
  --shm-size-gb "$VLLM_SHM_SIZE_GB" \
  exec \
  vllm serve "$MODEL_FORGE_TEACHER_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$MODEL_FORGE_TEACHER_NAME" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs "{\"enable_thinking\": $QWEN_ENABLE_THINKING}" \
    --language-model-only \
    --enable-prefix-caching \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
