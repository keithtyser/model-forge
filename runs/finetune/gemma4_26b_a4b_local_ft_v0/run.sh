#!/usr/bin/env bash
set -euo pipefail
cd /home/ktyser/projects/model-forge

export MODEL_FORGE_PARALLELISM=32
export MODEL_FORGE_HIGH_PARALLELISM=192
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ktyser/projects/model-forge/.venv/bin/python}
CPU_QUOTA=${MODEL_FORGE_CPU_QUOTA:-80%}
MEMORY_MAX=${MODEL_FORGE_MEMORY_MAX:-85%}
IO_WEIGHT=${MODEL_FORGE_IO_WEIGHT:-100}
NICE_LEVEL=${MODEL_FORGE_NICE:-10}
RESERVE_CORES=${MODEL_FORGE_RESERVE_CORES:-1}
export RESERVE_CORES
USABLE_CORES=$("$PYTHON" - <<'PY'
import os
reserve = int(os.environ.get("RESERVE_CORES", "1"))
print(max(1, (os.cpu_count() or 2) - reserve))
PY
)
export OMP_NUM_THREADS="$USABLE_CORES"
export MKL_NUM_THREADS="$USABLE_CORES"
export NUMEXPR_NUM_THREADS="$USABLE_CORES"
export OPENBLAS_NUM_THREADS="$USABLE_CORES"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

df -h /

run_limited() {
  if command -v systemd-run >/dev/null 2>&1 && [[ ! -f /.dockerenv ]] && [[ "${MODEL_FORGE_DISABLE_SYSTEMD_SCOPE:-0}" != "1" ]]; then
    systemd-run --scope \
      -p "CPUQuota=$CPU_QUOTA" \
      -p "MemoryMax=$MEMORY_MAX" \
      -p "IOWeight=$IO_WEIGHT" \
      nice -n "$NICE_LEVEL" "$@"
  else
    nice -n "$NICE_LEVEL" "$@"
  fi
}

run_limited "$PYTHON" /home/ktyser/projects/model-forge/runs/finetune/gemma4_26b_a4b_local_ft_v0/train_trl_sft.py --plan /home/ktyser/projects/model-forge/runs/finetune/gemma4_26b_a4b_local_ft_v0/plan.json --prepare-data
run_limited "$PYTHON" /home/ktyser/projects/model-forge/runs/finetune/gemma4_26b_a4b_local_ft_v0/train_trl_sft.py --plan /home/ktyser/projects/model-forge/runs/finetune/gemma4_26b_a4b_local_ft_v0/plan.json --train
