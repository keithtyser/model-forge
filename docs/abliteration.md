# Abliteration Workflow

model-forge treats abliteration as a reproducible experiment:

1. collect a refusal direction from harmful-vs-benign contrast prompts
2. review the direction artifacts and candidate layers
3. export a local ablated model only after review
4. run the same internal, artifact, and external eval suite against base, downloaded abli, and local abli

The default command is a dry run and does not load a model:

```bash
./forge ablate gemma4_26b_a4b plan
```

The first executable step is direction collection. It is guarded by `--execute`
and a free-memory check:

```bash
uv pip install -e ".[abliteration]"
./forge ablate gemma4_26b_a4b collect --execute
```

Outputs are written under:

```text
artifacts/abliteration/gemma4_26b_a4b_local_abli/
```

Do not run collection while a vLLM server is active. Keep one large model
process at a time.

## Hardware Profiles

`./forge serve` now detects a hardware profile and fills conservative vLLM
defaults unless you override them. The profile is chosen without allocating GPU
memory.

Useful overrides:

```bash
MODEL_FORGE_HARDWARE_PROFILE=dgx_spark ./forge serve gemma4_26b_a4b base
GPU_MEMORY_UTILIZATION=0.80 MAX_MODEL_LEN=16384 ./forge serve gemma4_26b_a4b base
VLLM_CPU_OFFLOAD_GB=24 ./forge serve gemma4_26b_a4b base
```

The DGX Spark profile follows the practical shape of AEON-style deployments:
one model at a time, `GPU_MEMORY_UTILIZATION=0.85`, prefix caching, chunked
prefill, and `MAX_NUM_BATCHED_TOKENS=32768`. Dedicated Blackwell cards get a
higher starting cap, but still keep the same override path.

## Comparing The Local Ablation

After a reviewed export exists at `~/models/gemma-4-26B-A4B-it-local-abliterated`,
serve and evaluate it like any other variant:

```bash
./forge serve gemma4_26b_a4b local_abli
./forge eval gemma4_26b_a4b local_abli --internal
./forge eval gemma4_26b_a4b local_abli --artifact
./forge eval gemma4_26b_a4b local_abli --external
./forge compare gemma4_26b_a4b
```

For the repo objective, local abli is good only if harmful refusal falls while
normal-use, structured-output, artifact, and external benchmark performance stay
near base. Unsafe compliance is reported as a risk metric, not as an automatic
failure for the ablation-research objective.
