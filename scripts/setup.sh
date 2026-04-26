#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

PROFILE=${1:-base}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is required for the recommended setup path.

Install uv, then rerun this script:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Or see:
  https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv --python "$PYTHON_VERSION" .venv
fi

case "$PROFILE" in
  base)
    uv pip install -e .
    ;;
  artifacts)
    uv pip install -e ".[artifacts]"
    .venv/bin/python -m playwright install chromium
    ;;
  external)
    uv pip install -e ".[external]"
    ;;
  all)
    uv pip install -e ".[artifacts,external]"
    .venv/bin/python -m playwright install chromium
    ;;
  *)
    echo "usage: $0 [base|artifacts|external|all]" >&2
    exit 2
    ;;
esac

echo "[model-forge] setup complete: $PROFILE"
