#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi
cd "$REPO_DIR"

CONFIG=${1:-configs/experiments/qwen35_9b_v0.yaml}
shift || true

"$PYTHON" -m model_forge.evals.run_eval --config "$CONFIG" "$@"
