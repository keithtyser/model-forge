#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

ACTION=${1:-help}
if [[ "$ACTION" == "help" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/gemma4_dgx.sh serve   [base|ft|abli]
  ./scripts/gemma4_dgx.sh suite   [base|ft|abli] [lm-eval-tasks]
  ./scripts/gemma4_dgx.sh smoke   [base|ft|abli]
  ./scripts/gemma4_dgx.sh full    [base|ft|abli]
  ./scripts/gemma4_dgx.sh artifact [base|ft|abli]
  ./scripts/gemma4_dgx.sh compare
  ./scripts/gemma4_dgx.sh external [base|ft|abli] [lm-eval-tasks]
  ./scripts/gemma4_dgx.sh external-dry-run [base|ft|abli] [lm-eval-tasks]
  ./scripts/gemma4_dgx.sh external-install
USAGE
  exit 0
fi

PYTHON=${PYTHON:-$REPO_DIR/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi

"$PYTHON" "$REPO_DIR/scripts/model_forge_dgx.py" gemma4_26b_a4b "$@"
