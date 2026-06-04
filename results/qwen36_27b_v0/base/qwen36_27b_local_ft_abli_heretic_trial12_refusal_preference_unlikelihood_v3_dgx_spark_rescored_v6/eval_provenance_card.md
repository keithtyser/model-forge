# Eval Provenance Card: qwen36_27b_v0_base_eval

## Run

- Family: `qwen36_27b`
- Variant: `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v3`
- Model: `Qwen/Qwen3.6-27B`
- Objective profile: `general_assistant`
- Trials: `1`
- Total cases: `75`
- Dry run: `false`

## Prompt Suite

- `capability_preservation_challenge`: 32
- `refusal_paired_boundary`: 40
- `unsafe_overcompliance`: 3

## Judge And Sampling

- Judge type: `deterministic_rule_scoring`
- Scoring version: `model_forge.internal_eval_scoring.v6`
- Sampling: `{"max_tokens": 1200, "temperature": 0.2, "timeout_seconds": 180}`

## Outputs

- `artifact_report`: `artifact_report.html` sha256=`missing` public_safe=`true`
- `artifact_validations`: `artifact_validations.json` sha256=`missing` public_safe=`true`
- `examples`: `examples.md` sha256=`6a058c8b4cd29a4448ded5d8940e0e47f4b945c3e6ef396e173416aa3c65ad54` public_safe=`false`
- `manifest`: `manifest.json` sha256=`f774d1cf8312790d138cf83cd3f52c510d5214a6e79aa4b38a1d4cd6187ca627` public_safe=`true`
- `responses`: `responses.jsonl` sha256=`4f29a2c89c79954dd9986db8677e74e7ac1168f22aece7416abb4e71afdbc9bd` public_safe=`false`
- `scores`: `scores.csv` sha256=`96f042f0bc328c2c11c390bc852b11a01a6bc4963ff7473bf0cef55e0f33cb70` public_safe=`true`

## Publication

Raw `responses.jsonl` and `examples.md` are not public-safe by default. Publish
aggregate scores, manifests, and redacted examples unless a release class allows
private raw-output retention.
