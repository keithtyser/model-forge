#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${1:?usage: scripts/run_apostate_container.sh <generated-run_apostate.py>}"
IMAGE="${MODEL_FORGE_APOSTATE_IMAGE:-model-forge-apostate:latest}"
MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-$HOME/models}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ ! -f "$RUNNER" ]]; then
  echo "[model-forge] missing Apostate runner: $RUNNER" >&2
  exit 2
fi

TOTAL_MEM_GB="${MODEL_FORGE_APOSTATE_DOCKER_MEMORY_GB:-$(awk '/MemTotal/ {printf "%d", ($2 / 1024 / 1024) * 0.85}' /proc/meminfo)}"
CPU_LIMIT="${MODEL_FORGE_APOSTATE_DOCKER_CPUS:-$(python3 - <<'PY'
import os
cores = os.cpu_count() or 2
print(max(1, int(cores * 0.8)))
PY
)}"
RAM_FLOOR="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.05}"
WATCHDOG_INTERVAL="${MODEL_FORGE_APOSTATE_HOST_WATCHDOG_INTERVAL_SECONDS:-10}"
CONTAINER_NAME="${MODEL_FORGE_APOSTATE_CONTAINER_NAME:-model-forge-apostate-$(id -u)-$$}"
WATCHDOG_FLAG="$(mktemp "${TMPDIR:-/tmp}/model-forge-apostate-watchdog.XXXXXX")"

cleanup() {
  rm -f "$WATCHDOG_FLAG"
}
trap cleanup EXIT

python3 - "$RAM_FLOOR" <<'PY'
import sys

floor = float(sys.argv[1])
values = {}
with open("/proc/meminfo", "r", encoding="utf-8") as handle:
    for line in handle:
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0])
total = values.get("MemTotal", 0)
available = values.get("MemAvailable", 0)
if total <= 0 or available / total < floor:
    raise SystemExit(
        f"[model-forge] refusing to start Apostate: host MemAvailable "
        f"{available / 1024 / 1024:.1f} GiB is below floor {floor:.2%}"
    )
PY

host_memory_watchdog() {
  python3 - "$CONTAINER_NAME" "$RAM_FLOOR" "$WATCHDOG_INTERVAL" "$WATCHDOG_FLAG" <<'PY'
import pathlib
import subprocess
import sys
import time

container_name = sys.argv[1]
floor = float(sys.argv[2])
interval = max(1.0, float(sys.argv[3]))
flag_path = pathlib.Path(sys.argv[4])


def memory_fraction() -> tuple[float, float, float]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(value.strip().split()[0])
    total = float(values.get("MemTotal", 0))
    available = float(values.get("MemAvailable", 0))
    if total <= 0:
        return 1.0, total, available
    return available / total, total, available


while True:
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if inspect.returncode != 0 or inspect.stdout.strip() != "true":
        break
    fraction, total, available = memory_fraction()
    if fraction < floor:
        message = (
            f"[model-forge] stopping {container_name}: host MemAvailable "
            f"{available / 1024 / 1024:.1f} GiB fell below floor {floor:.2%}"
        )
        print(message, file=sys.stderr, flush=True)
        flag_path.write_text(message + "\n", encoding="utf-8")
        subprocess.run(["docker", "stop", "--time", "10", container_name], check=False)
        break
    time.sleep(interval)
PY
}

mkdir -p "$MODELS_DIR" "$HF_CACHE"

echo "[model-forge] image: $IMAGE"
echo "[model-forge] runner: $RUNNER"
echo "[model-forge] docker CPU limit: $CPU_LIMIT"
echo "[model-forge] docker memory limit: ${TOTAL_MEM_GB}g"
echo "[model-forge] docker container: $CONTAINER_NAME"
echo "[model-forge] host RAM floor: $RAM_FLOOR"
df -h "$MODELS_DIR"

docker run --rm --gpus all \
  --name "$CONTAINER_NAME" \
  --network host \
  --ipc host \
  --cpus="$CPU_LIMIT" \
  --memory="${TOTAL_MEM_GB}g" \
  --memory-swap="${TOTAL_MEM_GB}g" \
  --shm-size="${MODEL_FORGE_APOSTATE_SHM_SIZE:-32g}" \
  --pids-limit="${MODEL_FORGE_APOSTATE_PIDS_LIMIT:-4096}" \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="$REPO_DIR/src" \
  -e HF_HOME="$HF_CACHE" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  -e MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION="${MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION:-0.05}" \
  -e MODEL_FORGE_MIN_FREE_DISK_FRACTION="${MODEL_FORGE_MIN_FREE_DISK_FRACTION:-0.15}" \
  -v "$REPO_DIR:$REPO_DIR" \
  -v "$MODELS_DIR:$MODELS_DIR" \
  -v "$HF_CACHE:$HF_CACHE" \
  -w "$REPO_DIR" \
  --entrypoint python3 \
  "$IMAGE" \
  "$RUNNER" &
DOCKER_PID=$!
host_memory_watchdog &
WATCHDOG_PID=$!

set +e
wait "$DOCKER_PID"
STATUS=$?
set -e

if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
fi

if [[ -s "$WATCHDOG_FLAG" ]]; then
  cat "$WATCHDOG_FLAG" >&2
  exit 137
fi

exit "$STATUS"
