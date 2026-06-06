# Qwen 3.6 27B Trial12 Preference-Unlikelihood V16 Sampled-Gate Repair Plan

Date: 2026-06-06

Status: prepared; training not yet launched.

## Goal

Continue the Qwen 3.6 27B full post-training workflow after V37 rejected the
source-anchored concept-cone projection. The remaining blocker is narrow:
`self_harm_instruction_harmful` still sometimes opens with first-person
refusal wording, while safe redirect and harmful-detail/compliance gates remain
good. V16 changes intervention class from static projection to sampled-gate
pairwise preference plus prefix unlikelihood.

## Hypothesis

The residual blocker is an on-policy stochastic response-opening habit, not a
missing-safety-knowledge problem. A better repair should train from held v2,
mine late near-miss model-forge eval traces, prefer no-denial safe redirects
over denial-first safe redirects, and preserve local FT v4 capability with
replay rows.

## Artifacts

- Data repair config:
  `configs/data_repair/qwen36_27b_late_nearmiss_self_harm_eval_repair_v2.yaml`
- Mined repair seed:
  `datasets/seeds/qwen36_27b_late_nearmiss_self_harm_eval_repair_v2.jsonl`
- Mined repair report:
  `reports/qwen36_27b_late_nearmiss_self_harm_eval_repair_v2_report.json`
- Static hard-negative seed:
  `datasets/seeds/qwen36_27b_trial12_pref_ul_v16_sampled_gate_hard_negatives.jsonl`
- Data-source registry:
  `configs/data_sources/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v16_sampled_gate_repair.yaml`
- Fine-tune manifest:
  `datasets/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v16_sampled_gate_repair.yaml`
- Fine-tune config:
  `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v16_sampled_gate_repair.yaml`
- Prepared run directory:
  `runs/finetune/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v16_sampled_gate_repair`

## Data Prep Result

`./forge data repair-from-eval --config configs/data_repair/qwen36_27b_late_nearmiss_self_harm_eval_repair_v2.yaml --overwrite`
emitted 252 mined repair pairs with 0 exact eval-prompt rows and no promotion
blockers.

`train_trl_sft.py --prepare-data` accepted 150 final training rows:

- 102 late near-miss sampled-gate repair rows
- 8 v16 hand-reviewed late refusal-opening hard negatives
- 10 v12 hard-negative replay rows
- 6 strict no-refusal redirect replay rows
- 4 unsafe-ablation redirect replay rows
- 16 challenge capability replay rows
- 4 planning capability replay rows

## Next Command

Run only when no other large model job or vLLM server is active:

```bash
MODEL_FORGE_EXECUTE_CLUSTER_TRAIN=1 \
  runs/finetune/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v16_sampled_gate_repair/run_cluster_torchrun.sh
```

After training, merge, sync, audit, serve TP=2, and run the targeted gate. Do
not broad-eval, NVFP4-export, upload, or promote unless the targeted gate passes
with self-harm refusal wording 0/3, safe redirect 3/3, harmful detail/compliance
0/3, and `model_selection_summary` 3/3.
