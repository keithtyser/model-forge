#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiments/qwen35_9b_v0.yaml}
python -m model_forge.evals.run_eval --config "$CONFIG" --dry-run
