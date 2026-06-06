# Qwen 3.6 27B Local FT v4 NVFP4 Serving Validation

Date: 2026-06-06

Status: partial quantization validation complete. The exact unquantized
`local_ft_v4` BF16 source baseline is now available, and the NVFP4 candidate
shows clear source-relative speedup. Do not promote or upload this candidate as
release quality yet because the sampled behavior-preservation report failed on
structured JSON/tool-use checks.

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

The Qwen NVFP4 path exposed and fixed these serving/export issues:

- serving eval needed a per-serving-config timeout override for long Qwen
  generations
- stock ModelOpt needed a text-only checkpoint view for Qwen wrapper checkpoints
- vLLM needed the exported artifact wrapperized back to
  `Qwen3_5ForConditionalGeneration`
- Qwen wrapper vision-tower metadata needed explicit FP4 exclusion for text-only
  serving
- the NVFP4 evidence gate needed configurable source-relative throughput targets
  instead of applying a Gemma-MoE absolute tok/s target to all model families

Relevant code:

- `scripts/quantization/qwen_text_modelopt.py`
- `scripts/model_forge_dgx.py`
- `configs/model_families/qwen36_27b.yaml`

## Validation

Strict local and worker checkpoint/tokenizer/architecture audits passed after
the wrapper repair and worker sync. The candidate served with TP=2 across both
DGX Spark nodes through the configured cluster path.

Candidate NVFP4 serving evidence:

- Smoke run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_smoke_20260606`
- Core run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_core_20260606`
- Serving eval:
  `reports/generated/serving_evals/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_serving_eval_20260606`
- Evidence gate:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_nvfp4_modelopt_tp2_core_20260606/serving_evidence_gate.md`

Exact BF16 source evidence:

- Smoke run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_bf16_lora_tp2_smoke_20260606`
- Core run:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_bf16_lora_tp2_core_20260606`
- Serving eval:
  `reports/generated/serving_evals/qwen36_27b_local_ft_v4_bf16_lora_tp2_serving_eval_20260606_r2`
- Serving evidence gate:
  `reports/generated/serve_bench/qwen36_27b_local_ft_v4_bf16_lora_tp2_core_20260606/serving_evidence_gate.md`

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

Exact source-vs-candidate comparison:

- BF16 source core mean output tok/s: 5.4250
- NVFP4 candidate core mean output tok/s: 13.0560
- core mean output tok/s speedup: about 2.41x
- BF16 source core mean decode tok/s: 5.5714
- NVFP4 candidate core mean decode tok/s: 13.9418
- core mean decode tok/s speedup: about 2.50x
- quantization card output p50 speedup: 2.47x
- quantization card decode-heavy output p50 speedup: 2.61x

The updated NVFP4 evidence gate uses the Qwen config's source-relative
throughput targets:

- output tok/s speedup target: 1.5x; observed 2.47x, pass
- decode-heavy output tok/s speedup target: 1.5x; observed 2.61x, pass
- tokenizer preservation: pass
- behavior preservation: fail

Sampled serving eval result:

- normal-use regression: passed sampled cases
- capability challenge: passed sampled cases
- benign paired quality: passed sampled cases
- harmful-detail and harmful-compliance sampled rates: 0.0
- structured JSON/tool-use case: failed sampled schema/workflow checks

The structured JSON miss is the current blocker. It does not block the raw
serving evidence gate, but it does block NVFP4 promotion because source behavior
was preserved on the BF16 baseline and regressed in the quantized candidate.

## Next Step

Investigate the structured JSON/tool-use regression before any HF upload or
promotion. Likely next checks are a targeted serving eval rerun for only the
JSON/tool case, then a quantization sensitivity pass that preserves the modules
most responsible for format adherence.
