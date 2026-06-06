# Eval Provenance Card: qwen36_27b_v0_base_eval

## Run

- Family: `qwen36_27b`
- Variant: `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v18_attention_output_sampled_opening_repair`
- Model: `Qwen/Qwen3.6-27B`
- Objective profile: `general_assistant`
- Trials: `3`
- Total cases: `6`
- Dry run: `false`

## Prompt Suite

- `capability_preservation_challenge`: 1
- `refusal_paired_boundary`: 1

## Judge And Sampling

- Judge type: `deterministic_rule_scoring`
- Scoring version: `model_forge.internal_eval_scoring.v13`
- Sampling: `{"max_tokens": 1200, "temperature": 1.0, "timeout_seconds": 240}`

## Outputs

- `artifact_report`: `artifact_report.html` sha256=`missing` public_safe=`true`
- `artifact_validations`: `artifact_validations.json` sha256=`missing` public_safe=`true`
- `examples`: `examples.md` sha256=`347330bc2655b2fa0b0c00e7ed5880a5bbf558e34c2689e4d8d74534557dff80` public_safe=`false`
- `manifest`: `manifest.json` sha256=`91354d3aa087e40ddbf898c170e0958b744e940b82209c82c7df750252ea4571` public_safe=`true`
- `responses`: `responses.jsonl` sha256=`9ebe2501cf5b2d706182c3b20f98788aa0954578a41cc9b22d9e3d89d23ad098` public_safe=`false`
- `scores`: `scores.csv` sha256=`3e6582067335e604407e6d802cbd5061f2308692ecffdeb58b84a4385edda303` public_safe=`true`

## Publication

Raw `responses.jsonl` and `examples.md` are not public-safe by default. Publish
aggregate scores, manifests, and redacted examples unless a release class allows
private raw-output retention.
