# Quantization

Model Forge treats quantization as a behavior-preserving model transform, not
just a memory trick. A quantized candidate is useful only after it loads on the
target hardware, serves real requests, improves or preserves serving economics,
and keeps the source model's useful behavior.

## First-Class Path: Blackwell NVFP4

NVFP4 is the priority path for DGX Spark / GB10 and other Blackwell targets.
Use it when a checkpoint is exported with native NVFP4/ModelOpt-compatible
weights, or when a trusted upstream model ships an already-quantized NVFP4
checkpoint.

The default runtime-import profile is:

```bash
./forge quantize plan \
  --config configs/quantization/nvfp4_blackwell_runtime.yaml \
  --write-plan
```

That command does not start a server. It writes a reproducibility plan with an
environment-backed Spark launch command. Set cluster values outside git:

```bash
export MODEL_FORGE_SPARK_VLLM_DOCKER=/path/to/spark-vllm-docker
export MODEL_FORGE_SPARK_CLUSTER_NODES=spark0,spark1
```

Then start exactly one NVFP4 server, run serving benchmarks and sampled behavior
checks, and stop the server before switching variants.

The default profile uses NVIDIA's small official
`nvidia/Llama-3.1-8B-Instruct-NVFP4` checkpoint so agents can validate the
Blackwell NVFP4 path without first downloading a very large MoE. Gemma-specific
NVFP4 import settings live in
`configs/quantization/gemma4_26b_a4b_nvfp4_blackwell_runtime.yaml`.

## Self-Quantizing A Source Model

Runtime-import profiles are not enough for Model Forge promotion. The primary
Gemma path must quantize our own source checkpoints, then compare each
quantized checkpoint against the same unquantized variant:

```bash
docker build \
  -f docker/modelopt-nvfp4.Dockerfile \
  -t model-forge-modelopt-nvfp4:0.43.0 .

./forge quantize matrix-plan \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml
```

Before running a heavy export, write the calibration manifest for the exact
source variant and config:

```bash
./forge quantize calibration-manifest gemma4_26b_a4b base \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --write-manifest
```

The manifest records the source variant, target variant, dataset list, sample
counts, sequence length, batch size, gated/public access classification, and
promotion requirements. If you override calibration with
`MODEL_FORGE_QUANT_CALIB_DATASET`, `MODEL_FORGE_QUANT_CALIB_SIZE`, or
`MODEL_FORGE_QUANT_CALIB_SEQ`, regenerate the manifest before the export so the
evidence matches the actual calibration contract.

The Gemma matrix covers:

- `base -> base_nvfp4_modelopt`
- `local_ft -> local_ft_nvfp4_modelopt`
- `local_abli_sota -> local_abli_sota_nvfp4_modelopt`
- `ft_local_abli_sota_internal_r7_selected_t34_transfer -> ft_local_abli_sota_internal_r7_selected_t34_transfer_nvfp4_modelopt`

Gemma 4 A4B uses the repo-native `gemma4_moe_modelopt` strategy in
`scripts/quantization/gemma4_moe_nvfp4.py`. Stock ModelOpt export can leave
Gemma's fused 3D MoE expert tensors out of the NVFP4 path; that produces a
large checkpoint and only a modest decode-speed gain. The full-MoE strategy
registers a ModelOpt `Gemma4TextExperts` plugin, exposes fused experts as
per-expert linear layers during calibration, exports ModelOpt NVFP4 weights,
and rewrites expert key names into the vLLM `moe.experts.<id>.*` layout.

Serve full-MoE Gemma NVFP4 checkpoints with Marlin on Spark:

```text
VLLM_NVFP4_GEMM_BACKEND=marlin
--quantization modelopt
--kv-cache-dtype fp8
--moe-backend marlin
--language-model-only
```

The previous MLP-only recipe remains useful as loader evidence, but it should
not be promoted as the optimized Gemma path. A real Gemma 4 NVFP4 candidate
should approach the published DGX Spark expectation of roughly 45-60 output
tokens/sec on decode-heavy chat workloads before it is called production-ready.
The published full-MoE reference checkpoint was also served locally on
2026-05-30 with this Marlin stack and reached about 50 output tok/s on the
repo's core serving benchmark. Use that as the near-term apples-to-apples target
for self-quantized Gemma variants.
ModelOpt `--low_memory_mode` was tested on the earlier stock export path and
failed with a meta-tensor dispatch error, so the checked-in recipe uses normal
loading with a full-RAM Spark resource profile: CPU remains capped, disk is
checked, Docker has a hard memory limit, and the watchdog stops the export if
host available RAM drops below 5%.

For a two-Spark cluster, assign independent variant exports across nodes with
an environment variable instead of committing hosts:

```bash
export MODEL_FORGE_QUANT_WORKERS=local,spark-worker
./forge quantize matrix-plan \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --variants base,local_ft
```

Run at most one export per Spark node. ModelOpt export currently runs as one
heavy process per assigned worker; serving and eval should use the two-node vLLM
path when the quantized checkpoint loads. The matrix command is a planner: it
assigns variants to workers and prints the local or SSH command for each worker
without starting a heavy process.

Use `--variants` when only a subset of source checkpoints is present on the
assigned workers. The target variant for single exports can be set explicitly:

```bash
./forge quantize export gemma4_26b_a4b local_ft \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --target-variant local_ft_nvfp4_modelopt \
  --run-id gemma4_local_ft_nvfp4_modelopt \
  --write-plan --execute
```

Production calibration can use NVIDIA Nemotron post-training data when the HF
account has access:

```bash
export HF_HOME=~/cache/model-forge-hf-user
export MODEL_FORGE_QUANT_CALIB_DATASET=cnn_dailymail,nemotron-post-training-dataset-v2
export MODEL_FORGE_QUANT_CALIB_SIZE=4096
export MODEL_FORGE_QUANT_CALIB_SEQ=1024
./forge quantize export gemma4_26b_a4b base \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --run-id gemma4_base_nvfp4_modelopt \
  --write-plan --execute
```

The export runner now has hard host guardrails:

- nonblocking lock under `reports/generated/.locks/` so the same checkout cannot
  start two exports at once
- `systemd-run --user --scope` wrapper with recipe-controlled CPU, memory, and
  IO limits by default; set `export.systemd_scope.user: false` only on hosts
  where noninteractive system scopes are available
- `nice` plus Docker `--cpus`, `--memory`, `--memory-swap`, and `--shm-size`
  limits
- preflight memory and disk checks before Docker starts
- runtime memory watchdog that stops the Docker container if available host RAM
  drops below the configured stop floor

Do not bypass this runner with an ad hoc `docker run` for large checkpoints.
If the guard trips, treat it as a failed run to diagnose before retrying.

## Evidence Contract

Every quantization candidate needs these artifacts:

- source serving benchmark summary
- candidate serving benchmark summary
- source sampled serving eval scores
- candidate sampled serving eval scores
- quantization card comparing latency, throughput, memory, quality, and behavior

For NVFP4 promotion, request/sec is not enough. The candidate should show a
clear `output_tokens_per_second` improvement, especially on the `decode_heavy`
workload, against the matching BF16/FP16 source served with the same model
length, batching, scheduler, prompt set, max tokens, and hardware allocation.
If token throughput does not improve, keep the run as loader evidence instead
of claiming an optimized NVFP4 path.

Write the card:

```bash
./forge quantize card \
  --config configs/quantization/nvfp4_blackwell_runtime.yaml \
  --source-serving-summary reports/generated/source/serve_bench/summary.json \
  --candidate-serving-summary reports/generated/candidate/serve_bench/summary.json \
  --source-serving-eval reports/generated/source/serve_eval \
  --candidate-serving-eval reports/generated/candidate/serve_eval \
  --run-id source_vs_candidate_nvfp4 \
  --write-card
```

For FP8 KV cache runs, also write the focused behavior report:

```bash
./forge quantize fp8-kv-report \
  --config configs/quantization/gemma4_26b_a4b_fp8_runtime.yaml \
  --source-serving-summary reports/generated/source/serve_bench/summary.json \
  --candidate-serving-summary reports/generated/candidate/serve_bench/summary.json \
  --source-serving-eval reports/generated/source/serve_eval \
  --candidate-serving-eval reports/generated/candidate/serve_eval \
  --run-id source_vs_fp8_kv \
  --write-report
```

This report checks that the candidate endpoint actually used FP8 KV settings,
completed all serving requests, and retained normal-use, schema, and workflow
scores within the objective tolerance. It is not a replacement for the broader
quantization card.

Write the behavior-preservation report for every quantized candidate:

```bash
./forge quantize behavior-report \
  --config configs/quantization/fp8_w8a8_modelopt.yaml \
  --source-serving-summary reports/generated/source/serve_bench/summary.json \
  --candidate-serving-summary reports/generated/candidate/serve_bench/summary.json \
  --source-serving-eval reports/generated/source/serve_eval \
  --candidate-serving-eval reports/generated/candidate/serve_eval \
  --run-id source_vs_quantized_behavior \
  --write-report
```

The report applies the `quantized_quality_retention` tolerances: normal use,
challenge capability, schema adherence, workflow success, and benign answer
quality must stay within allowed deltas. Risk metrics such as unsafe
overcompliance are reported separately because ablated models may intentionally
change refusal behavior.

Check tokenizer and chat-template preservation for every quantized or GGUF
export directory:

```bash
./forge quantize tokenizer-report \
  --source-tokenizer-dir /path/to/source-model \
  --candidate-tokenizer-dir /path/to/quantized-or-gguf-export \
  --source-variant base \
  --candidate-variant base_fp8_w8a8_modelopt \
  --run-id source_vs_quantized_tokenizer \
  --write-report
```

Use `--load-tokenizer --strict` for promotion when the tokenizer can be loaded
locally. Configured family variants can still use
`./forge variants tokenizer-audit`; `quantize tokenizer-report` exists for
new export directories before they are added to a family config.

## FP8 W8A8 Checkpoint Pipeline

FP8 W8A8 is a checkpoint-creation path, unlike runtime FP8 KV cache. Use the
generic ModelOpt recipe with an explicit family and source variant:

```bash
./forge quantize plan llama31_8b base \
  --config configs/quantization/fp8_w8a8_modelopt.yaml \
  --write-plan

./forge quantize calibration-manifest llama31_8b base \
  --config configs/quantization/fp8_w8a8_modelopt.yaml \
  --write-manifest

./forge quantize export llama31_8b base \
  --config configs/quantization/fp8_w8a8_modelopt.yaml \
  --target-variant base_fp8_w8a8_modelopt \
  --write-plan --execute
```

The checked-in recipe uses ModelOpt `hf_ptq.py` with `--qformat fp8`,
guarded Docker/systemd execution, FP8 KV casting during export, and a templated
target variant such as `{source_variant}_fp8_w8a8_modelopt`. It is generic by
design: add family-specific overrides in config files or model-family metadata,
not in common quantization code.

The card intentionally reports both safety/refusal behavior and utility metrics.
For ablated models, lower harmful-prompt refusal can be the objective, but only
if normal-use quality, instruction following, structured output, and artifact
behavior remain comparable to the source.

## Runtime Import vs Checkpoint Creation

`nvfp4_runtime` means the checkpoint is already quantized. It validates a real
compressed serving path without pretending Model Forge created the weights.

`nvfp4` checkpoint creation is a separate backend path. When added for a model
family, it must record:

- backend and version, such as ModelOpt, TensorRT-LLM, or a vLLM-compatible
  exporter
- calibration dataset manifest, row count, sequence length, and license tier
- modules kept in BF16 or otherwise excluded from low precision
- exported checkpoint path and loader format
- source-vs-quantized eval and serving deltas

Do not mark a checkpoint-creation recipe complete from a dry run.

## Spark Defaults

DGX Spark defaults live in `src/model_forge/hardware.py` and
`configs/hardware/dgx_spark.yaml`. NVFP4 serving starts conservative:

```text
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_KV_CACHE_DTYPE=fp8_e4m3
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_ENABLE_CHUNKED_PREFILL=1
GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_NUM_SEQS=4
```

Large NVFP4 MoE serving on Spark should generally start lower than dense-model
defaults (`gpu_memory_utilization=0.40-0.60`, `max_num_seqs=1-2`) and only
increase one variable at a time after smoke benchmarks pass.

## Generalization Rule

The reusable recipe is the structure:

```text
choose source and quantized candidate
pin model revision and backend
serve under bounded Spark resources
benchmark the running endpoint
sample behavior through the same endpoint
write a quantization card
promote only with evidence
```

The constants are not universal. Recalibrate model length, sequence count,
calibration samples, and excluded modules for each new family.
