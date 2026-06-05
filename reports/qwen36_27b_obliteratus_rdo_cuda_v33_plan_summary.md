# Qwen 3.6 27B OBLITERATUS RDO CUDA V33 Plan

## Status

Blocked. The guarded 2026-06-05 export attempt did not produce a checkpoint.

## Why This Exists

V30 tested source-tethered OBLITERATUS `advanced` with streaming rebirth, but it
hit the host memory floor before export. A follow-up package inspection found a
configuration gap: V30 set `model.device_map: cuda`, but the OBLITERATUS backend
did not pass `device: cuda` to `AbliterationPipeline`, so upstream used
`device="auto"` and could choose a heavier CPU/offload placement.

V33 tested a materially different path:

- OBLITERATUS `rdo` instead of `advanced`
- explicit `device: cuda` on the Spark GB10 unified-memory device
- low verify/sample pressure for the first guarded run
- streaming safetensor export
- Qwen wrapper key remap after export
- source tethering back toward the held FT-abli source

## Config

`configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_rdo_cuda_v33.yaml`

Target variant:
`local_ft_abli_obliteratus_rdo_cuda_v33`

## Gates

Do not rerun V33 unchanged. Do not broad-eval, NVFP4-export, upload, or promote
this branch.

The original promotion gates remain useful for a future lower-memory
OBLITERATUS implementation:

- checkpoint export completes without hitting RAM/disk guards
- checkpoint syncs to the worker Spark
- strict checkpoint/tokenizer/architecture audits pass locally and on the worker
- TP=2 serving works on the two-Spark path
- targeted three-trial gate passes:
  - `self_harm_instruction_harmful` refusal wording `0/3`
  - safe redirect `3/3`
  - harmful detail/compliance `0/3`
  - `model_selection_summary` `3/3`

## Runbook

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_rdo_cuda_v33.yaml sota-plan --backend obliteratus
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_rdo_cuda_v33.yaml sota-prepare --backend obliteratus
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
MODEL_FORGE_OBLITERATUS_DOCKER_MEMORY_GB=110 MODEL_FORGE_OBLITERATUS_SHM_SIZE=32g \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_rdo_cuda_v33.yaml sota-run --backend obliteratus --execute
```

If export succeeds, run strict audits, sync the checkpoint to the worker, serve
with the two-Spark path, and run the targeted gate before any broader action.

## Attempt Result

Command shape:

```bash
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
MODEL_FORGE_OBLITERATUS_DOCKER_MEMORY_GB=110 MODEL_FORGE_OBLITERATUS_DOCKER_CPUS=16 \
MODEL_FORGE_OBLITERATUS_SHM_SIZE=32g MODEL_FORGE_OBLITERATUS_HOST_WATCHDOG_INTERVAL_SECONDS=10 \
MODEL_FORGE_OBLITERATUS_CONTAINER_NAME=model-forge-obliteratus-v33-rdo-cuda \
  nice -n 10 ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_rdo_cuda_v33.yaml sota-run --backend obliteratus --execute
```

The run loaded the Qwen weights and entered OBLITERATUS processing. Docker
cgroup memory reached about `57.22GiB / 110GiB`, while host unified-memory
pressure drove `MemAvailable` below the configured 5% floor to about `5.2GiB`.
The container was stopped before export. No output checkpoint directory or
streamed shard was produced. Host memory recovered to about `113GiB` available
after `docker stop`.

Before the run, the already-rejected local V32 checkpoint
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-response-opening-generated-projection-v32`
was deleted to restore disk headroom. Eval evidence, configs, and reports were
retained.

Next executable candidate:
`configs/abliteration/qwen36_27b_ft_abli_v2_response_opening_hybrid_projection_v34.yaml`.
