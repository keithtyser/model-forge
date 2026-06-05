# Qwen 3.6 27B Abliterix Response-Opening V26 Plan

Status: prepared after config repair; first two launch attempts exposed
Abliterix multi-direction backend constraints before any useful trial completed.
The current V26 retry uses `n_directions: 1`.

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
- explicit benign-preservation prompt variants so the search gets 48 good train
  rows and 48 bad train rows

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
| good train | 48 |
| good eval | 3 |

## Failed Launches

The first two-shard run on 2026-06-05 loaded the model on both Sparks, selected
batch size 16, measured baseline refusals at 12/20, and then stopped before
trial search with:

```text
RuntimeError: The size of tensor a (48) must match the size of tensor b (12)
```

Abliterix v1.8.0 computes paired residual differences in its
`n_directions > 1` path, so equal train counts are required. The repo now fails
fast during `sota-prepare` for future Abliterix multi-direction configs with
unequal train counts, and V26 now uses deliberate good prompt variants rather
than duplicate-only cycling.

The second two-shard run passed residual extraction with 48/48 train prompts and
started trial 1, but both shards then failed in Abliterix v1.8.0 LoRA steering:

```text
IndexError: index 10 is out of bounds for dimension 0 with size 2
```

The backend sampled a layer/vector index while the multi-direction SRA tensor
only had size `n_directions=2`. Model-forge now blocks `n_directions > 1` with
`steering_mode: lora` unless the recipe explicitly opts into the experimental
path. V26 has been changed to `n_directions: 1`; the broader prompt objective
and 48-trial budget remain the actual hypothesis under test.

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
