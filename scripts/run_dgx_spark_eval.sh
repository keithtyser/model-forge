#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi
cd "$REPO_DIR"

CONFIG=${1:-configs/experiments/qwen35_9b_v0.yaml}
RUN_NAME=${2:-dgx-spark}
shift $(( $# >= 2 ? 2 : $# ))
BASE_URL=${MODEL_FORGE_BASE_URL:-http://127.0.0.1:8000/v1}
MODEL_ALIAS=${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}
MAX_CASES=${MODEL_FORGE_MAX_CASES:-}
TRIALS=${MODEL_FORGE_TRIALS:-}
EXTRA_ARGS=()

if [[ -n "$MAX_CASES" ]]; then
  EXTRA_ARGS+=(--max-cases "$MAX_CASES")
fi
if [[ -n "$TRIALS" ]]; then
  EXTRA_ARGS+=(--trials "$TRIALS")
fi

export MODEL_FORGE_BASE_URL="$BASE_URL"
export MODEL_FORGE_MODEL="$MODEL_ALIAS"
export MODEL_FORGE_HARDWARE_LABEL="${MODEL_FORGE_HARDWARE_LABEL:-DGX Spark}"
export MODEL_FORGE_TIMEOUT_SECONDS="${MODEL_FORGE_TIMEOUT_SECONDS:-180}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  CYAN=$'\033[36m'
  GREEN=$'\033[32m'
  RESET=$'\033[0m'
else
  CYAN=""
  GREEN=""
  RESET=""
fi

echo "${CYAN}==>${RESET} Eval target"
echo "  endpoint: $MODEL_FORGE_BASE_URL"
echo "  model:    $MODEL_FORGE_MODEL"
echo "  run:      $RUN_NAME"

"$PYTHON" - <<'PY'
import json, os, sys, urllib.request
url = os.environ['MODEL_FORGE_BASE_URL'].rstrip('/') + '/models'
req = urllib.request.Request(url, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read().decode('utf-8'))
models = data.get('data', [])
ids = [m.get('id', '?') for m in models]
print('OK  /models reachable')
print('    advertised models:', ', '.join(ids[:10]))
if os.environ['MODEL_FORGE_MODEL'] not in ids:
    print(
        f"[model-forge] ERROR: requested model {os.environ['MODEL_FORGE_MODEL']!r} "
        f"is not advertised by the server",
        file=sys.stderr,
    )
    sys.exit(2)
PY

"$PYTHON" -m model_forge.evals.run_eval \
  --config "$CONFIG" \
  --output-suffix "$RUN_NAME" \
  "${EXTRA_ARGS[@]}" \
  "$@"
