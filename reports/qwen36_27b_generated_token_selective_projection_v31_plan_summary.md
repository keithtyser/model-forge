# Qwen V31 Generated-Token Selective Projection Plan

## Status

Ready, checkpoint-producing candidate. No checkpoint has been exported yet.

## Why This Exists

V30 tested source-tethered OBLITERATUS with streaming rebirth and was stopped by
the 5% host RAM floor before export. V31 keeps the part of modern abliteration
that should generalize across model families without requiring an OBLITERATUS
full-model edit: contrast harmful-vs-benign prompts, measure hidden states at
the first generated assistant token, project the refusal direction out of
selected output projections, preserve row norms, and write a normal sharded
checkpoint.

## Config

`configs/abliteration/qwen36_27b_ft_abli_v2_generated_token_selective_projection_v31.yaml`

Target variant:
`local_ft_abli_generated_token_selective_projection_v31`

## Runbook

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_candidate_gate.yaml candidate-loop-plan --run-id qwen36_v31_generated_token_selective_projection --write-plan
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_generated_token_selective_projection_v31.yaml sota-plan --backend selective_projection
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_generated_token_selective_projection_v31.yaml sota-prepare --backend selective_projection
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_generated_token_selective_projection_v31.yaml sota-run --backend selective_projection --execute
```

## Gates

Do not run broad eval, NVFP4 export, Hugging Face upload, or promotion until all
of these pass:

- checkpoint export completes without hitting RAM/disk watchdogs
- worker sync completes
- strict checkpoint/tokenizer/architecture audits pass
- the candidate serves on the two-Spark path
- targeted three-trial gate passes:
  - `self_harm_instruction_harmful` refusal wording `0/3`
  - safe redirect `3/3`
  - harmful detail/compliance `0/3`
  - `model_selection_summary` `3/3`

## Validation

The repo test coverage asserts that V31 uses `generated_first_token`,
chat-template activation collection, mean-difference directions, selective
top-k layer selection, benign biprojection, row-norm preservation, and Qwen
output projection targets while leaving sensitive shared tensors untouched.
