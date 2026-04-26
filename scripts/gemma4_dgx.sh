#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

ACTION=${1:-help}
VARIANT=${2:-base}

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

Examples:
  ./scripts/gemma4_dgx.sh serve base
  ./scripts/gemma4_dgx.sh smoke base
  ./scripts/gemma4_dgx.sh full ft
  ./scripts/gemma4_dgx.sh compare

Environment overrides:
  MODEL_FORGE_MODELS_DIR     default: $HOME/models
  GPU_MEMORY_UTILIZATION     default from serve wrapper: 0.85
  MAX_MODEL_LEN              default from serve wrapper: 32768
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
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
