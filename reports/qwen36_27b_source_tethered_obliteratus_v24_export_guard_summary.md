# Qwen 3.6 27B Source-Tethered OBLITERATUS V24 Export Guard

Date: 2026-06-05

## Goal

Try the source-tethered OBLITERATUS V24 candidate for the Qwen 3.6 27B
local-FT-abli loop before any broad eval, NVFP4 export, or Hugging Face upload.

## Hypothesis

The earlier plain OBLITERATUS diagnostic changed the refusal geometry but failed
the model-forge targeted gate. V24 follows the public OBLITERATUS Qwen3.6
source-tether pattern where it generalizes: run a two-direction regularized
OBLITERATUS pass, remap text-only Qwen keys back to the model-forge wrapper,
then tether the result toward the local FT v4 source with alpha `0.895` and
restore the top `43` highest-drift tensors.

## Attempted Command

```bash
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
MODEL_FORGE_OBLITERATUS_DOCKER_MEMORY_GB=110 MODEL_FORGE_OBLITERATUS_SHM_SIZE=32g \
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_source_tethered_obliteratus_v24.yaml \
  sota-run --backend obliteratus --execute
```

## Result

Blocked by the memory guard before checkpoint export.

- OBLITERATUS loaded all `851` source shards.
- The process entered the projection/export stage without a progress bar.
- Host RAM fell to about `5.0 GiB` available on a `119 GiB` node, below the
  configured `0.05` floor.
- The OBLITERATUS container was stopped to avoid scheduler/SSH starvation.
- The wrapper exited with status `137`.
- No output checkpoint directory was produced at
  `~/models/Qwen3.6-27B-local-ft-v4-abliterated-source-tethered-obliteratus-v24`.
- Host RAM recovered immediately to about `112 GiB` available.

## Decision

Do not rerun this exact single-node 110 GiB OBLITERATUS export shape unchanged.
The method is not behaviorally rejected because V24 never reached checkpoint
audit or targeted eval, but it is operationally blocked until the export path is
made safer.

## Follow-Up

`scripts/run_obliteratus_container.sh` now names the Docker container and runs a
host-side `MemAvailable` watchdog. Future OBLITERATUS runs stop themselves when
the host crosses `MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION`, instead of requiring a
manual stop.

The next OBLITERATUS attempt should use one of these safer shapes:

- a lower Docker memory cap plus the host watchdog, then stop on the first
  resource breach
- a smaller diagnostic model/fewer OBLITERATUS directions to prove the path
  before the 27B checkpoint
- a memory-streamed or sharded OBLITERATUS export path before retrying full Qwen
  27B source-tethered export

Do not broad-eval, quantize, upload, or promote V24 until a checkpoint exists,
strict checkpoint/tokenizer/architecture audits pass, and the three-trial
model-forge targeted gate passes.
