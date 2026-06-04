# Qwen 3.6 27B Trial12 Preference-Unlikelihood v3 Summary

Status: rejected.

This is the tracked, portable summary for the v3 sequential repair. The raw
responses, generated manifests, and run artifacts remain in ignored local
`results/` and `runs/` directories because those files can contain unsafe raw
model outputs and machine-specific paths.

## Recipe

- Config: `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v3.yaml`
- Source variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`
- Candidate variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v3`
- Data manifest:
  `datasets/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v3.yaml`
- Primary seed:
  `datasets/seeds/qwen36_27b_trial12_pref_ul_v3_unsafe_no_refusal.jsonl`

## Training

- Steps: 80/80.
- Dataset rows: 68.
- Train loss: 3.0598.
- Runtime: 1150.5678 seconds.
- Method: `qlora_pairwise_preference_unlikelihood`.
- Preference weight: 1.05.
- Unlikelihood weight: 0.22.
- SFT replay weight: 0.70.

## Eval

Scoring version: `model_forge.internal_eval_scoring.v6`.

Targeted buckets:

- `refusal_paired_boundary`
- `unsafe_overcompliance`
- `capability_preservation_challenge`

Scores:

| Bucket | Metric | Value |
| --- | --- | ---: |
| capability_preservation_challenge | normal_use_regression_pass_rate | 0.8125 |
| refusal_paired_boundary | ablation_refusal_suppression_rate | 0.85 |
| refusal_paired_boundary | benign_answer_quality_rate | 0.90 |
| refusal_paired_boundary | benign_refusal_rate | 0.00 |
| refusal_paired_boundary | harmful_detail_rate | 0.10 |
| refusal_paired_boundary | harmful_prompt_compliance_rate | 0.10 |
| refusal_paired_boundary | refusal_rate_harmful | 0.15 |
| refusal_paired_boundary | safe_redirect_rate | 0.85 |
| refusal_paired_boundary | unsafe_overcompliance_rate | 0.10 |
| unsafe_overcompliance | ablation_refusal_suppression_rate | 0.6667 |
| unsafe_overcompliance | harmful_detail_rate | 0.00 |
| unsafe_overcompliance | harmful_prompt_compliance_rate | 0.00 |
| unsafe_overcompliance | refusal_rate_harmful | 0.3333 |
| unsafe_overcompliance | safe_redirect_rate | 1.00 |
| unsafe_overcompliance | unsafe_overcompliance_rate | 0.00 |
| overall | latency_seconds_median | 16.1617 |

## Decision

Reject v3. It improved unsafe-overcompliance refusal from 0.6667 to 0.3333 but
damaged the paired-boundary and challenge gates that v2 had cleared. Keep v2 as
the best held Qwen FT-abli evidence node.
