#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

ACTION=${1:-help}
VARIANT=${2:-base}
EXTERNAL_TASKS=${3:-${MODEL_FORGE_EXTERNAL_TASKS:-ifeval}}

MODELS_DIR=${MODEL_FORGE_MODELS_DIR:-$HOME/models}
PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/gemma4_dgx.sh serve   [base|ft|abli]
  ./scripts/gemma4_dgx.sh smoke   [base|ft|abli]
  ./scripts/gemma4_dgx.sh full    [base|ft|abli]
  ./scripts/gemma4_dgx.sh artifact [base|ft|abli]
  ./scripts/gemma4_dgx.sh compare
  ./scripts/gemma4_dgx.sh external [base|ft|abli] [lm-eval-tasks]
  ./scripts/gemma4_dgx.sh external-dry-run [base|ft|abli] [lm-eval-tasks]
  ./scripts/gemma4_dgx.sh external-install

Examples:
  ./scripts/gemma4_dgx.sh serve base
  ./scripts/gemma4_dgx.sh smoke base
  ./scripts/gemma4_dgx.sh full ft
  ./scripts/gemma4_dgx.sh external base ifeval
  ./scripts/gemma4_dgx.sh external ft ifeval,gsm8k
  ./scripts/gemma4_dgx.sh compare

Environment overrides:
  MODEL_FORGE_MODELS_DIR     default: $HOME/models
  GPU_MEMORY_UTILIZATION     default from serve wrapper: 0.85
  MAX_MODEL_LEN              default from serve wrapper: 32768
  MODEL_FORGE_BASE_URL       default: http://127.0.0.1:8000/v1
  MODEL_FORGE_EXTERNAL_TASKS default: ifeval
  MODEL_FORGE_EXTERNAL_LIMIT optional lm-eval --limit value for quick checks
  MODEL_FORGE_EXTERNAL_CONCURRENCY default: 1
USAGE
}

variant_model_path() {
  case "$1" in
    base) printf '%s\n' "$MODELS_DIR/gemma-4-26B-A4B-it" ;;
    ft) printf '%s\n' "$MODELS_DIR/Gemopus-4-26B-A4B-it" ;;
    abli) printf '%s\n' "$MODELS_DIR/Huihui-gemma-4-26B-A4B-it-abliterated" ;;
    *) echo "unknown variant: $1" >&2; usage >&2; exit 2 ;;
  esac
}

variant_served_name() {
  case "$1" in
    base) printf '%s\n' "google/gemma-4-26B-A4B-it" ;;
    ft) printf '%s\n' "Jackrong/Gemopus-4-26B-A4B-it" ;;
    abli) printf '%s\n' "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated" ;;
    *) echo "unknown variant: $1" >&2; usage >&2; exit 2 ;;
  esac
}

run_eval() {
  local script=$1
  local variant=$2
  local served_name
  served_name=$(variant_served_name "$variant")

  MODEL_FORGE_MODEL="$served_name" \
  MODEL_FORGE_VARIANT="$variant" \
  "$script"
}

slugify() {
  printf '%s\n' "$1" | tr ',/:' '___' | tr -cd 'A-Za-z0-9_.-'
}

run_lm_eval_external() {
  local variant=$1
  local tasks=$2
  local dry_run=$3
  local served_name
  local base_url
  local output_dir
  local task_slug
  local concurrency
  served_name=$(variant_served_name "$variant")
  base_url=${MODEL_FORGE_BASE_URL:-http://127.0.0.1:8000/v1}
  task_slug=$(slugify "$tasks")
  concurrency=${MODEL_FORGE_EXTERNAL_CONCURRENCY:-1}
  output_dir="reports/generated/gemma4_26b_a4b_external/${variant}/lm-eval_${task_slug}"

  export OPENAI_API_KEY=${OPENAI_API_KEY:-local}

  if [[ "$dry_run" != "true" ]]; then
    MODEL_FORGE_BASE_URL="$base_url" MODEL_FORGE_MODEL="$served_name" "$PYTHON" - <<'PY'
import json
import os
import sys
import urllib.request

url = os.environ["MODEL_FORGE_BASE_URL"].rstrip("/") + "/models"
requested = os.environ["MODEL_FORGE_MODEL"]
try:
    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    print(f"[model-forge] ERROR: could not reach {url}: {exc}", file=sys.stderr)
    sys.exit(2)
ids = [item.get("id", "") for item in data.get("data", [])]
if requested not in ids:
    print(f"[model-forge] ERROR: requested external model {requested!r} is not advertised by server", file=sys.stderr)
    print(f"[model-forge] advertised models: {', '.join(ids) or '<none>'}", file=sys.stderr)
    sys.exit(2)
PY
  fi

  cmd=(
    "$PYTHON" -m model_forge.evals.external lm-eval
    --output-dir "$output_dir"
  )
  if [[ "$dry_run" == "true" ]]; then
    cmd+=(--dry-run)
  fi
  cmd+=(--
    --model local-chat-completions
    --model_args "model=${served_name},base_url=${base_url%/}/chat/completions,num_concurrent=${concurrency},max_retries=3,tokenized_requests=False"
    --tasks "$tasks"
    --apply_chat_template
    --fewshot_as_multiturn
    --output_path "$output_dir/lm_eval"
    --log_samples
    --confirm_run_unsafe_code
  )
  if [[ -n "${MODEL_FORGE_EXTERNAL_LIMIT:-}" ]]; then
    cmd+=(--limit "$MODEL_FORGE_EXTERNAL_LIMIT")
  fi

  echo "[model-forge] external lm-eval variant: $variant"
  echo "[model-forge] external lm-eval model:   $served_name"
  echo "[model-forge] external lm-eval tasks:   $tasks"
  echo "[model-forge] external lm-eval output:  $output_dir"
  "${cmd[@]}"
}

case "$ACTION" in
  serve)
    model_path=$(variant_model_path "$VARIANT")
    served_name=$(variant_served_name "$VARIANT")
    if [[ ! -d "$model_path" ]]; then
      echo "model path does not exist: $model_path" >&2
      exit 1
    fi
    MODEL_FORGE_MODEL="$model_path" \
    MODEL_FORGE_SERVED_MODEL_NAME="$served_name" \
    ./scripts/dgx_spark_serve_gemma4_26b_a4b.sh
    ;;
  smoke)
    run_eval ./scripts/dgx_spark_smoke_eval_gemma4_26b_a4b.sh "$VARIANT"
    ;;
  full)
    run_eval ./scripts/dgx_spark_full_eval_gemma4_26b_a4b.sh "$VARIANT"
    ;;
  artifact)
    run_eval ./scripts/dgx_spark_artifact_eval_gemma4_26b_a4b.sh "$VARIANT"
    ;;
  compare)
    "$PYTHON" -m model_forge.evals.compare_runs \
      --base results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_base_dgx_spark \
      --ft results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_ft_dgx_spark \
      --abli results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_abli_dgx_spark \
      --output-dir reports/generated/gemma4_26b_a4b_comparison
    ;;
  external)
    run_lm_eval_external "$VARIANT" "$EXTERNAL_TASKS" false
    ;;
  external-dry-run)
    run_lm_eval_external "$VARIANT" "$EXTERNAL_TASKS" true
    ;;
  external-install)
    if command -v uv >/dev/null 2>&1; then
      uv pip install -e ".[external]"
    else
      "$PYTHON" -m pip install -e ".[external]"
    fi
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
