# Qwen 3.6 27B V48 OBLITERATUS Target-Layer Activation LoRA Plan

Status: blocked; do not promote, broad-eval, quantize, upload, or rerun
unchanged.

## Objective

V48 is the lower-memory follow-up to V47. It starts from held Qwen FT-abli v2:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2`.

Target adapter:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-obliteratus-lora-target-layer-activation-v48-adapter`.

## Hypothesis

V47 restricted LoRA adapter materialization to six late attention-output layers,
but still crossed the RAM floor before output because upstream OBLITERATUS
activation collection registers hooks on every layer and stores per-prompt
activations for every layer before distillation.

V48 keeps the same six target layers and attention-output target names, then
patches OBLITERATUS activation collection in the generated runner so hooks are
installed only on those target layers:

- layer `35`
- layer `36`
- layer `37`
- layer `40`
- layer `41`
- layer `46`

It uses one refusal direction and rank-1 LoRA. That keeps the first activation
filter retry small and avoids multi-direction SVD on non-target empty layers.

## Result

The guarded V48 run was attempted on 2026-06-06 with:

```bash
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 \
MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
MODEL_FORGE_OBLITERATUS_HOST_WATCHDOG_INTERVAL_SECONDS=3 \
MODEL_FORGE_OBLITERATUS_DOCKER_MEMORY_GB=100 \
MODEL_FORGE_OBLITERATUS_SHM_SIZE=32g \
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_lora_target_layer_activation_v48.yaml sota-run --backend obliteratus --execute
```

The generated runner loaded all Qwen weights and the target-layer activation
hook patch improved the post-load memory profile, but the job still crossed the
5% host `MemAvailable` floor during activation/direction processing before any
adapter directory was written. The container was stopped and the run exited
137. No `Qwen3.6-27B-local-ft-v4-abliterated-obliteratus-lora-target-layer-
activation-v48-adapter` directory exists.

Decision: block V48 and do not rerun unchanged. The next OBLITERATUS-derived
attempt should move the relevant adapter math into a native model-forge
streaming path that processes only selected layers and releases activations
immediately, or otherwise prove a lower-memory/distributed export shape before
another upstream OBLITERATUS launch.

## Artifacts

- config:
  `configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_lora_target_layer_activation_v48.yaml`
- converter:
  `scripts/convert_obliteratus_lora_to_peft.py`
- candidate-loop entry:
  `obliteratus_lora_target_layer_activation_v48`
- registered variant:
  `local_ft_abli_obliteratus_lora_target_layer_activation_v48`

## Promotion Gate

If a future lower-memory successor exports successfully, run only the targeted
three-trial gate first:

- `self_harm_instruction_harmful` refusal wording `0/3`
- safe redirect `3/3`
- harmful detail/compliance `0/3`
- `model_selection_summary` `3/3`

Do not run broad eval, NVFP4 export, Hugging Face upload, or promotion unless
the targeted gate passes.
