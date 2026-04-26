#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

source .venv/bin/activate
export MODEL_FORGE_BASE_URL=${MODEL_FORGE_BASE_URL:-http://127.0.0.1:8000/v1}
export MODEL_FORGE_MODEL=${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}
export MODEL_FORGE_HARDWARE_LABEL=${MODEL_FORGE_HARDWARE_LABEL:-DGX Spark}
export MODEL_FORGE_QUANT=${MODEL_FORGE_QUANT:-fp16}
export MODEL_FORGE_CONTEXT_LENGTH=${MODEL_FORGE_CONTEXT_LENGTH:-32768}
export MODEL_FORGE_MAX_CASES=${MODEL_FORGE_MAX_CASES:-4}

./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml smoke
