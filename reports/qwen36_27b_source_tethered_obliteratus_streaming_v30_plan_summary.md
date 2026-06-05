# Qwen 3.6 27B Source-Tethered OBLITERATUS Streaming V30 Plan

Date: 2026-06-05

Status: ready; no checkpoint exported yet.

## Goal

Retry the source-tethered OBLITERATUS branch that V24 could not finish because
the upstream save path gathered and serialized a full 27B state dict in memory.

## Hypothesis

The source-tethered method is still viable because V24 was stopped by the memory
guard before checkpoint export and never reached model-forge behavioral eval.
V30 keeps the method constants that came from the public Qwen3.6 OBLITERATUS
shape where they generalize:

- OBLITERATUS `advanced`
- `n_directions: 2`
- `regularization: 0.5`
- source-tether alpha `0.895`
- restore top `43` highest-drift tensors to the local FT v4 source

The change is operational: model-forge monkeypatches OBLITERATUS `_rebirth()` so
it writes `1GB` safetensor shards incrementally and records
`model_forge_obliteratus_streaming_rebirth.json`, then continues through the
existing Qwen wrapper key remap and source-tether scripts.

## Config

`configs/abliteration/qwen36_27b_ft_abli_v2_source_tethered_obliteratus_streaming_v30.yaml`

Target variant:
`local_ft_abli_source_tethered_obliteratus_streaming_v30`

## Runbook

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_candidate_gate.yaml candidate-loop-plan --run-id qwen36_v30_source_tethered_obliteratus_streaming --write-plan
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_source_tethered_obliteratus_streaming_v30.yaml sota-plan --backend obliteratus
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_source_tethered_obliteratus_streaming_v30.yaml sota-prepare --backend obliteratus
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
MODEL_FORGE_OBLITERATUS_DOCKER_MEMORY_GB=100 MODEL_FORGE_OBLITERATUS_SHM_SIZE=32g \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_source_tethered_obliteratus_streaming_v30.yaml sota-run --backend obliteratus --execute
```

## Gates

Do not run broad eval, NVFP4 export, Hugging Face upload, or promotion until all
of these pass:

- checkpoint export completes without hitting the RAM/disk watchdog
- worker sync completes
- strict checkpoint/tokenizer/architecture audits pass
- the candidate serves on the two-Spark path
- targeted three-trial gate passes:
  - `self_harm_instruction_harmful` refusal wording `0/3`
  - safe redirect `3/3`
  - harmful detail/compliance `0/3`
  - `model_selection_summary` `3/3`

## Validation

- Generated V30 OBLITERATUS runner compiles with `py_compile`.
- Candidate-loop plan now has exactly one runnable candidate:
  `source_tethered_obliteratus_streaming_v30`.
