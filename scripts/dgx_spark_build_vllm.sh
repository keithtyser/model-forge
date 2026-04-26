#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR=${SPARK_VLLM_DIR:-$HOME/spark-vllm-docker}
cd "$SPARK_VLLM_DIR"
./build-and-copy.sh
