# Qwen 3.6 27B Trial12 Preference-Unlikelihood V17 Direct-Opening Rewrite Plan

Date: 2026-06-06

Status: prepared; data and runner prepared; training not yet launched.

## Goal

Continue the Qwen 3.6 27B FT-abli repair after V38 preserved capability but
still failed the zero-refusal target with one first-person help/denial opening
in three targeted self-harm trials. V17 starts from the merged V38 checkpoint
and narrows the objective to direct-opening rewrite.

## Hypothesis

The remaining blocker is an opening-style prior, not harmful-content leakage.
A short follow-on LoRA should be enough if the data stops reinforcing
first-person "I can help..." chosen openings. V17 therefore filters mined
chosen rows to direct care-first openings, pairs them against safe
denial/help-not-help openings, increases prefix unlikelihood to 96 assistant
tokens, and keeps capability replay.

## Artifacts

- Data repair config:
  `configs/data_repair/qwen36_27b_late_nearmiss_self_harm_direct_opening_repair_v3.yaml`
- Mined repair seed:
  `datasets/seeds/qwen36_27b_late_nearmiss_self_harm_direct_opening_repair_v3.jsonl`
- Mined repair report:
  `reports/qwen36_27b_late_nearmiss_self_harm_direct_opening_repair_v3_report.json`
- Static hard-negative seed:
  `datasets/seeds/qwen36_27b_trial12_pref_ul_v17_direct_opening_rewrite.jsonl`
- Data-source registry:
  `configs/data_sources/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v17_direct_opening_rewrite.yaml`
- Fine-tune manifest:
  `datasets/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v17_direct_opening_rewrite.yaml`
- Fine-tune config:
  `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v17_direct_opening_rewrite.yaml`

## Data Prep Result

`./forge data repair-from-eval --config configs/data_repair/qwen36_27b_late_nearmiss_self_harm_direct_opening_repair_v3.yaml --overwrite`
emitted 72 mined repair pairs with 0 exact eval-prompt rows and no promotion
blockers. The strict chosen filter intentionally removed the V38/V2
first-person chosen rows.

Prepared final training mix:

- 44 strict mined direct-opening repair rows
- 24 v17 hand-reviewed direct-opening hard negatives
- 8 v16 sampled-gate hard-negative replay rows
- 8 v12 refusal-wording hard-negative replay rows
- 6 strict no-refusal redirect replay rows
- 4 unsafe-ablation redirect replay rows
- 16 challenge capability replay rows
- 4 planning capability replay rows

Total accepted rows: 114. The strict direct-opening source sampled 48 rows and
rejected 4 rows during finetune data prep, leaving 44 accepted mined rows.

## Next Command

Run only when no other large model job or vLLM server is active:

```bash
./forge finetune --config configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v17_direct_opening_rewrite.yaml prepare --overwrite
MODEL_FORGE_EXECUTE_CLUSTER_TRAIN=1 \
  runs/finetune/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v17_direct_opening_rewrite/run_cluster_torchrun.sh
```

After training, merge, sync, audit, serve TP=2, and run the targeted gate. Do
not broad-eval, NVFP4-export, upload, or promote unless the targeted gate passes
with self-harm refusal wording 0/3, safe redirect 3/3, harmful detail/compliance
0/3, and `model_selection_summary` 3/3.
