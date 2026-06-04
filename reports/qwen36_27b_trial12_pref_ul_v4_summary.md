# Qwen 3.6 27B Trial12 Preference-Unlikelihood v4 Summary

Status: rejected.

This is the tracked, portable summary for the v4 micro phrase-repair run. Raw
responses, generated manifests, and run artifacts remain in ignored local
`results/` and `runs/` directories because those files can contain unsafe raw
model outputs and machine-specific paths.

## Recipe

- Config: `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v4.yaml`
- Source variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`
- Candidate variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v4`
- Data manifest:
  `datasets/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v4.yaml`
- Primary seed:
  `datasets/seeds/qwen36_27b_trial12_pref_ul_v4_phrase_repair.jsonl`

## Training

- Steps: 36/36.
- Dataset rows: 72.
- Train loss: 3.6519.
- Runtime: 459.7920 seconds.
- Method: `qlora_pairwise_preference_unlikelihood`.
- Preference weight: 0.45.
- Unlikelihood weight: 0.10.
- SFT replay weight: 1.25.
- LoRA target family: attention-only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

## Eval

Scoring version: `model_forge.internal_eval_scoring.v6`.

Targeted buckets:

- `refusal_paired_boundary`
- `unsafe_overcompliance`
- `capability_preservation_challenge`

Scores:

| Bucket | Metric | Value |
| --- | --- | ---: |
| capability_preservation_challenge | normal_use_regression_pass_rate | 0.8438 |
| refusal_paired_boundary | ablation_refusal_suppression_rate | 0.90 |
| refusal_paired_boundary | benign_answer_quality_rate | 1.00 |
| refusal_paired_boundary | benign_refusal_rate | 0.00 |
| refusal_paired_boundary | harmful_detail_rate | 0.05 |
| refusal_paired_boundary | harmful_prompt_compliance_rate | 0.05 |
| refusal_paired_boundary | refusal_rate_harmful | 0.10 |
| refusal_paired_boundary | safe_redirect_rate | 1.00 |
| refusal_paired_boundary | unsafe_overcompliance_rate | 0.05 |
| unsafe_overcompliance | ablation_refusal_suppression_rate | 0.3333 |
| unsafe_overcompliance | harmful_detail_rate | 0.00 |
| unsafe_overcompliance | harmful_prompt_compliance_rate | 0.00 |
| unsafe_overcompliance | refusal_rate_harmful | 0.6667 |
| unsafe_overcompliance | safe_redirect_rate | 1.00 |
| unsafe_overcompliance | unsafe_overcompliance_rate | 0.00 |
| overall | latency_seconds_median | 27.8799 |

Failed-row summary:

- Unsafe-overcompliance still had explicit refusal wording on 2/3 prompts.
- Paired boundary had 3 failing harmful rows: 2 explicit refusal-wording
  failures and 1 overcompliance/detail failure.
- Challenge capability missed 5/32 rows, mostly required-concept misses.

## Decision

Reject v4. It did not improve the held v2 unsafe-overcompliance refusal rate and
it damaged the v2 paired-boundary and challenge gates. Keep
`local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2` as the best
held Qwen FT-abli evidence node. Do not upload, quantize, or promote v4.

Operational notes from this run:

- The final adapter root on the worker was incomplete/root-owned after cluster
  training and had to be replaced from the coordinator adapter copy before
  merge.
- The merge completed on the worker, then required tokenizer metadata restored
  from v2 for strict tokenizer audit to pass.
- Cluster TP=2 vLLM serving failed twice during NCCL communicator
  initialization, so the v4 quick gate used a behavior-only solo TP=1 fallback
  at 8k context. That fallback is not throughput-comparable to earlier TP=2
  runs.
- `cluster torchrun-smoke` can deadlock if the worker-side container starts and
  waits for a master that does not start.
