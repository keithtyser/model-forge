# DGX Spark Optimization Notes

model-forge treats DGX Spark as a first-class hardware profile, but the pipeline
must remain model-family agnostic. Spark defaults live in `model_forge.hardware`
and can be overridden from the environment for a specific model or experiment.

These defaults are based on practical settings from AEON-7's public DGX Spark
repos, especially the NVFP4 Gemma 4 and SuperGemma deployment guides:

- https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4
- https://github.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4
- https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4
- https://github.com/AEON-7/Gemma-4-E4B-it-Uncensored-NVFP4
- https://github.com/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4

## Serving Defaults

For `MODEL_FORGE_HARDWARE_PROFILE=dgx_spark`, model-forge recommends:

```bash
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=32768
MAX_NUM_BATCHED_TOKENS=32768
VLLM_MAX_NUM_SEQS=4
VLLM_KV_CACHE_DTYPE=fp8_e4m3
VLLM_DTYPE=auto
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_MATMUL_PRECISION=high
NVIDIA_FORWARD_COMPAT=1
NVIDIA_DISABLE_REQUIRE=1
```

The memory cap is intentionally conservative. AEON-7's examples run between
`0.80` and `0.90` depending on quantization, model architecture, sequence count,
and context length. Start lower, smoke test, then raise one variable at a time.

Use a Spark/GB10-native vLLM build. On Spark, stock vLLM wheels may not be
compiled for SM 12.1, and that can silently fall back to poor paths or fail at
runtime. The repo's Spark docs still prefer `spark-vllm-docker`, and AEON-7's
NVFP4 repos show an alternate source-built container path for Gemma 4 NVFP4.

## NVFP4 / ModelOpt Serving

NVFP4 models need model-specific validation. For ModelOpt checkpoints, serve with:

```bash
VLLM_QUANTIZATION=modelopt
VLLM_DTYPE=auto
VLLM_KV_CACHE_DTYPE=fp8_e4m3
```

Gemma 4 style models may also need:

```bash
VLLM_TRUST_REMOTE_CODE=true
VLLM_ENABLE_AUTO_TOOL_CHOICE=true
VLLM_TOOL_CALL_PARSER=gemma4
VLLM_REASONING_PARSER=gemma4
```

Do not bake those parser names into generic configs. They are family-specific.

For compressed-tensors or patched ModelOpt checkpoints, use the model card and
upstream repo as the source of truth for loader patches. model-forge should record
the selected flags and mounted patches in the run manifest, then validate quality
through the eval suite.

## Speculative Decoding

AEON-7's E4B drafter repos expose vLLM `--speculative-config` recipes. model-forge
does not assume a drafter exists, but the serving wrappers pass one through:

```bash
VLLM_SPECULATIVE_CONFIG='{"method":"eagle3","model":"/models/drafter","num_speculative_tokens":3}' \
./forge serve <family> <variant>
```

Treat speculative decoding as a throughput optimization only. It must not change
promotion criteria: compare the same model with and without the drafter if output
quality or stop behavior looks different.

## Quantization Guidance

AEON-7's SuperGemma notes call out a key MoE lesson: quantizing routers or
multimodal bridge layers can break routing or vision behavior. The Spark profile
therefore recommends keeping these patterns in BF16 unless a family-specific
recipe says otherwise:

```text
router, gate, vision, visual, embed_vision, multi_modal_projector
```

For MoE NVFP4 calibration, prefer an adaptive/batched calibration loop when
available. The profile records:

```bash
MODEL_FORGE_QUANT_CALIBRATION_SAMPLES=512
MODEL_FORGE_QUANT_CALIBRATION_SEQ_LEN=4096
MODEL_FORGE_QUANT_BATCH_SIZE=auto
MODEL_FORGE_MOE_FAST_CALIBRATION=1
```

Those are recommendations for future quantization pipeline work, not a guarantee
that every model should use exactly 512 samples. The recipe structure should
transfer; constants should be recalibrated per model family.

## Training / Ablation Parallelism

DGX Spark can be bandwidth-limited and input-pipeline bound. The profile keeps a
safe default:

```bash
MODEL_FORGE_PARALLELISM=32
```

For stages that are known to benefit from high parallelism, opt in explicitly:

```bash
MODEL_FORGE_ENABLE_HIGH_PARALLELISM=1
```

That promotes `MODEL_FORGE_PARALLELISM` to `192`. You can also set
`MODEL_FORGE_PARALLELISM=192` directly. Do not use high parallelism blindly while
a vLLM server is also loaded.

## Generalization Rule

The Spark profile is a hardware optimization layer. It should not make the repo
Gemma-specific. Family-specific requirements belong in model-family YAML, the
model card, or a named quantization recipe. The core pipeline should keep working
for Qwen, Llama, Mistral, Gemma, and future open models with minimal config.
