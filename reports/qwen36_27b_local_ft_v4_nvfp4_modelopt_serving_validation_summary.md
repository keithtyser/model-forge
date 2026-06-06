# Qwen 3.6 27B Local FT v4 NVFP4 Serving Validation

Date: 2026-06-06

Status: partial quantization validation complete. Do not treat this as final
promotion evidence until the exact unquantized `local_ft_v4` BF16 source
variant has a matching serving benchmark and sampled serving eval.

## Hypothesis

ModelOpt NVFP4 should preserve the Qwen 3.6 27B local FT v4 serving behavior
closely enough for continued validation while improving tok/s on the two-DGX
Spark Blackwell path.

## Artifact

- Family: `qwen36_27b`
- Source variant: `local_ft_v4`
- Candidate variant: `local_ft_v4_nvfp4_modelopt`
- Config: `configs/quantization/qwen36_27b_local_ft_v4_nvfp4_modelopt.yaml`
- Local artifact:
  `~/models/model-forge-quantized/qwen36_27b/local_ft_v4_nvfp4_modelopt`

## Implementation Notes

The Qwen NVFP4 path exposed and fixed three serving/export issues:

- stock ModelOpt needed a text-only checkpoint view for Qwen wrapper checkpoints
- vLLM needed the exported artifact wrapperized back to
  `Qwen3_5ForConditionalGeneration`
- Qwen wrapper vision-tower metadata needed explicit FP4 exclusion for text-only
  serving

Relevant code:

- `scripts/quantization/qwen_text_modelopt.py`
- `scripts/model_forge_dgx.py`
- `configs/model_families/qwen36_27b.yaml`

## Validation

Strict local and worker checkpoint/tokenizer/architecture audits passed after
the wrapper repair and worker sync. The candidate served with TP=2 across both
DGX Spark nodes through the configured cluster path.

Serving benchmark evidence:

- Smoke run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_smoke_20260606`
- Core run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_core_20260606`
- Serving eval:
  `reports/generated/serving_evals/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_serving_eval_20260606`
- Evidence gate:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_core_20260606/serving_evidence_gate.md`

Core benchmark:

- requests: 9/9 complete
- success rate: 1.0
- mean output tok/s: 13.0560
- mean decode tok/s: 13.9418
- mean total tok/s: 33.2008
- mean TTFT: 0.3264 seconds

Smoke benchmark:

- requests: 3/3 complete
- success rate: 1.0
- mean output tok/s: 13.4769
- mean decode tok/s: 14.4903
- mean total tok/s: 30.3667
- mean TTFT: 0.3276 seconds

Existing Qwen base BF16 smoke baseline:

- run:
  `reports/generated/serve_bench/qwen36_27b_base_bf16_spark_x2_baseline`
- mean output tok/s: 7.1616
- mean decode tok/s: 7.2680
- mean total tok/s: 13.7852

This gives roughly 1.88x output tok/s and 1.99x decode tok/s versus the
existing base BF16 smoke run. That is useful hardware evidence, but it is not a
strict source-vs-quantized comparison because the BF16 run is the base variant,
not `local_ft_v4`.

Sampled serving eval result:

- normal-use regression: passed sampled cases
- capability challenge: passed sampled cases
- benign paired quality: passed sampled cases
- harmful-detail and harmful-compliance sampled rates: 0.0
- structured JSON/tool-use case: failed sampled schema/workflow checks

The structured JSON miss is a quality follow-up. It does not block the serving
evidence gate, but it should be investigated before making broad release-quality
claims for the quantized variant.

## Next Step

Run the exact unquantized `local_ft_v4` BF16 source through the same TP=2
serving benchmark and sampled serving eval, then compare:

- source BF16 `local_ft_v4`
- candidate NVFP4 `local_ft_v4_nvfp4_modelopt`

Only after that comparison should this artifact move toward a quantization card,
HF upload, or promotion.
