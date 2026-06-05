# Qwen 3.6 27B Abliterix Response-Opening V26 Plan

Status: prepared; search not yet run.

## Objective

Continue the Qwen FT-abli loop from the held-v2 near miss
`local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`.

The target remains strict:

- `self_harm_instruction_harmful` refusal wording: 0/3
- `self_harm_instruction_harmful` safe redirect: 3/3
- harmful detail and harmful prompt compliance: 0/3
- `model_selection_summary`: 3/3

## Hypothesis

The first Abliterix SRA run found low-KL zero-refusal proxy trials, but it
optimized against one bad eval prompt and the exported trial still failed the
model-forge targeted gate at 1/3 refusal wording. V26 keeps Abliterix SRA/LoRA
but aligns the search objective more closely with the real gate by adding:

- paraphrased self-harm boundary prompts
- observed refusal-opening traces as bad contrast
- no-refusal safe redirect traces as preservation contrast
- 48 search trials instead of 24

## Config

`configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26.yaml`

Source checkpoint:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2`

Planned selected-trial checkpoint:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-abliterix-response-opening-v26-selected`

## Prepare Evidence

Prepared with:

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26.yaml sota-plan --backend abliterix
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26.yaml sota-prepare --backend abliterix
```

Generated artifacts:

- `artifacts/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26/sota_abliterix_response_opening/abliterix.toml`
- `artifacts/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26/sota_abliterix_response_opening/run_abliterix_search.py`
- `artifacts/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26/sota_abliterix_response_opening/model_forge_prompt_datasets/manifest.json`

Prepared prompt counts:

| Section | Count |
| --- | ---: |
| bad train | 48 |
| bad eval | 20 |
| good train | 12 |
| good eval | 3 |

## Resource State

Cluster hardware probed successfully, but cluster health failed consistency
before commit because the coordinator worktree had the V26 changes and the
worker was still at commit `76f3d8f`. Do not run cluster-dependent serving/eval
until the changes are committed, pushed, and synced.

The Abliterix and Apostate Docker wrappers now have the same host
`MemAvailable` watchdog pattern as OBLITERATUS, so future runs can stop
themselves with exit `137` when host RAM falls below
`MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION`.

## Next Commands

After commit/push/sync and confirming no other large model process is active:

```bash
docker build -f docker/abliterix.Dockerfile -t model-forge-abliterix:latest .
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26.yaml sota-run --backend abliterix --execute
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_response_opening_v26.yaml abliterix-search-analyze --backend abliterix --output reports/generated/qwen36_27b_v26_abliterix_response_opening_search_analysis.json
```

Only export one selected trial if the journal analysis recommends it. The
selected-trial checkpoint still must pass strict audits on both Sparks and the
targeted model-forge three-trial candidate gate before broad eval, NVFP4 export,
HF upload, or promotion.
