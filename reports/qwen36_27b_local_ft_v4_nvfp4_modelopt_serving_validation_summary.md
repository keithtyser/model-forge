# Qwen 3.6 27B Local FT v4 NVFP4 Serving Validation

Date: 2026-06-06

Status: partial quantization validation complete. The exact unquantized
`local_ft_v4` BF16 source baseline is now available, and the NVFP4 candidate
shows clear source-relative speedup. Do not promote or upload this candidate as
release quality yet because the sampled behavior-preservation report failed on
structured JSON/tool-use checks. The AWQ and W4A16 follow-up candidates were
also rejected: AWQ is not compatible with current vLLM ModelOpt metadata
support, and W4A16 serves fast but generates degenerate repeated punctuation.

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

The serving-eval comparison report is:

`reports/generated/serving_eval_comparisons/qwen36_local_ft_v4_bf16_vs_nvfp4_json_tool_regression_20260606`

It compares 11 matched sampled cases and finds two source-pass/candidate-fail
metrics on `agentic_tool_use_json/model_serve_timeout`: `schema_adherence` and
`workflow_success`. The BF16 source returned strict JSON. The NVFP4 candidate
wrapped the answer in a markdown JSON fence and emitted malformed object keys
(`reason":` without the opening quote) in two steps, so strict JSON parsing
correctly failed.

## Next Step

Investigate the structured JSON/tool-use regression before any HF upload or
promotion. Likely next checks are a targeted serving eval rerun for only the
JSON/tool case, then a quantization sensitivity pass that preserves the modules
most responsible for format adherence.

The next planned quantization candidates are now in the Qwen matrix:

- `local_ft_v4_nvfp4_awq_modelopt`: ModelOpt `nvfp4_awq`; test this first
  because AWQ calibration may preserve format-following better while retaining
  Blackwell FP4 acceleration.
- `local_ft_v4_nvfp4_w4a16_modelopt`: ModelOpt `nvfp4_w4a16`; use this if AWQ
  still fails strict JSON, because weight-only FP4 should keep activations in a
  safer precision regime at possible speed cost.

Plan either candidate with:

```bash
./forge quantize matrix-plan \
  --config configs/quantization/qwen36_27b_local_ft_v4_nvfp4_modelopt.yaml \
  --variants local_ft_v4_nvfp4_awq_modelopt \
  --write-plan
```

An initial AWQ export attempt with the parent calibration defaults
(`calib_size=256,256`, `calib_seq=2048`, `batch_size=4`) reached AWQ activation
statistics and was stopped by the resource watchdog when available memory fell
to 4.57%, below the 5% floor. Partial Docker-root-owned staging files were
removed after the stop. The AWQ and W4A16 matrix entries now override to a
low-memory probe (`calib_size=64`, `calib_seq=1024`, `batch_size=1`) before any
larger promotion-class calibration retry.

The low-memory AWQ retry completed activation-stat collection and AWQ parameter
search, then failed during ModelOpt HF export with `Cannot copy out of meta
tensor; no data!`. This showed that `device_map=auto` can leave offloaded/meta
tensors that AWQ export cannot serialize. The Qwen text ModelOpt script now
supports a full-device map such as `cuda:0` and a `--reject-meta-tensors`
guard; the AWQ/W4A16 matrix entries use both so the next retry either exports
from materialized weights or fails immediately after load with actionable
diagnostics.

The `device_map=cuda:0` AWQ retry exported successfully with low-memory probe
settings (`calib_size=64`, `calib_seq=1024`, `batch_size=1`). It produced
`~/models/model-forge-quantized/qwen36_27b/local_ft_v4_nvfp4_awq_modelopt`
in 2616.392 seconds, wrote one 18.8 GB `model.safetensors` shard, removed the
temporary text-input staging directory, and passed strict checkpoint,
tokenizer, architecture, and source-vs-candidate tokenizer-preservation audits.
The ModelOpt/vLLM compatibility preflight now rejects this artifact before
serving because both `hf_quant_config.json` and `config.json` declare
`quant_algo=NVFP4_AWQ`, which current vLLM ModelOpt does not support.

The W4A16 retry exported successfully in 773.045 seconds and produced
`~/models/model-forge-quantized/qwen36_27b/local_ft_v4_nvfp4_w4a16_modelopt`.
It synced to the worker, passed strict local/worker structural audits, passed
the new ModelOpt/vLLM compatibility report with `quant_algo=NVFP4`, served on
the two-Spark TP=2 path with FlashInfer NVFP4 kernels, and reached roughly
2.46x output tok/s plus 2.50x decode-heavy output tok/s speedup against BF16
`local_ft_v4`. It is still rejected: the sampled serving eval and manual
`temperature=0` probes produced repeated `!` tokens instead of useful answers,
so the behavior-preservation and NVFP4 evidence gates correctly fail.

The next quantization attempt should be a component-sensitivity policy that
keeps format-critical modules in BF16 rather than rerunning default NVFP4, AWQ,
or W4A16 unchanged.
