# Qwen 3.6 27B Local FT v4 NVFP4 Public Release Review

Date: 2026-06-06

Status: public quantized-model Hub plan reviewed and unblocked for the
no-ablation Qwen workflow.

## Scope

This review covers the validated Qwen workflow:

`Qwen/Qwen3.6-27B` -> `local_ft_v4` ->
`local_ft_v4_nvfp4_attention_output_bf16_modelopt`

It does not cover the paused FT-ablation/behavior-edit branch.

## Candidate

- Family: `qwen36_27b`
- Source variant: `local_ft_v4`
- Quantized variant: `local_ft_v4_nvfp4_attention_output_bf16_modelopt`
- Local artifact:
  `~/models/model-forge-quantized/qwen36_27b/local_ft_v4_nvfp4_attention_output_bf16_modelopt`
- Release class: `public_quantized_model`
- Planned Hub repo:
  `keithtyser/model-forge-qwen36-27b-local-ft-v4-nvfp4-attention-output-bf16-modelopt`

## Evidence Reviewed

- Validation summary:
  `reports/qwen36_27b_local_ft_v4_nvfp4_modelopt_serving_validation_summary.md`
- Serving benchmark:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_attention_output_bf16_modelopt_tp2_core_20260606/summary.json`
- Serving eval scores:
  `reports/generated/serving_evals/qwen36_27b_local_ft_v4_nvfp4_attention_output_bf16_modelopt_tp2_serving_eval_20260606/scores.csv`
- Quantization card:
  `reports/generated/quantization/qwen36_local_ft_v4_bf16_vs_nvfp4_attention_output_bf16_modelopt_20260606/quantization_card.json`
- NVFP4 evidence gate:
  `reports/generated/quantization/qwen36_local_ft_v4_bf16_vs_nvfp4_attention_output_bf16_modelopt_20260606/nvfp4_evidence_gate.json`
- Hub plan/model card:
  `reports/generated/hub/qwen36_local_ft_v4_nvfp4_attention_output_bf16_modelopt_public_plan_autoscores_20260606`
- Upstream local license evidence:
  `~/models/Qwen3.6-27B/README.md` declares `license: apache-2.0` and
  `~/models/Qwen3.6-27B/LICENSE` is Apache License 2.0.

## Hub Plan Command

```bash
./forge hf plan-model qwen36_27b local_ft_v4_nvfp4_attention_output_bf16_modelopt \
  --release-class public_quantized_model \
  --artifact-path ~/models/model-forge-quantized/qwen36_27b/local_ft_v4_nvfp4_attention_output_bf16_modelopt \
  --validation-state spark_cluster_validated \
  --eval-results reports/generated/serving_evals/qwen36_27b_local_ft_v4_nvfp4_attention_output_bf16_modelopt_tp2_serving_eval_20260606 \
  --serving-card reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_attention_output_bf16_modelopt_tp2_core_20260606/summary.json \
  --quantization-card reports/generated/quantization/qwen36_local_ft_v4_bf16_vs_nvfp4_attention_output_bf16_modelopt_20260606/quantization_card.json \
  --promotion-report reports/generated/quantization/qwen36_local_ft_v4_bf16_vs_nvfp4_attention_output_bf16_modelopt_20260606/nvfp4_evidence_gate.json \
  --source-license-checked \
  --run-id qwen36_local_ft_v4_nvfp4_attention_output_bf16_modelopt_public_plan_autoscores_20260606 \
  --json
```

## Gate Result

The reviewed plan returned `blocked: false` with all release gates passing:

- `validation_state`
- `model_card_complete`
- `source_license_checked`
- `eval_results_present`
- `quantization_card_present`
- `serving_card_present`
- `promotion_gates_passed_or_research_report_only`
- `unsafe_examples_redacted`
- `no_private_tokens_or_paths`
- `variant_promotion_not_blocked`
- `public_checkpoint_release_allowed`

The plan rewrites the serving-eval directory to its sanitized `scores.csv` and
materializes a sanitized copy of the NVFP4 gate JSON under the Hub-plan output
directory because the original gate JSON references local private artifact
paths.

## Decision

Promote `local_ft_v4_nvfp4_attention_output_bf16_modelopt` as the validated
Qwen FT-source Blackwell NVFP4 candidate for the no-ablation workflow. It is
eligible for HF upload through the Hub CLI when `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN` is present in the environment. Do not embed tokens in
commands, docs, reports, or Git history.

No upload was attempted in this review because no Hugging Face token was present
in the process environment.
