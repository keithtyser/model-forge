# Abliteration Workflow

model-forge treats abliteration as a reproducible experiment:

1. collect a refusal direction from suffix-pooled harmful-vs-benign contrast prompts
2. review the direction artifacts and candidate layers
3. compare the edit geometry against the downloaded abli model for diagnostics
4. export a local ablated model by projecting that direction out of base weights
5. run the same internal, artifact, and external eval suite against base, downloaded abli, and local abli

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

On DGX Spark, run collection in a CUDA-enabled environment. If the host Python
environment has CPU-only PyTorch, use the Spark/vLLM CUDA container and mount
the repo plus model directory. Add `--user "$(id -u):$(id -g)"` to container
runs so generated artifacts remain writable from the host checkout.

Outputs are written under:

```text
artifacts/abliteration/gemma4_26b_a4b_local_abli/
```

The current Gemma recipe is a v3 candidate. It pools activations over short
assistant suffixes instead of only the final prompt token, includes layer 29,
uses norm-preserving biprojection, and refuses to export if any configured
target layer lacks a collected direction. The target modules intentionally
match the downloaded abli checkpoint pattern: `self_attn.o_proj.weight` and
`mlp.down_proj.weight` for language layers 5 through 29. Mixture-of-experts
weights, embeddings, and the LM head stay untouched.

The v3 backend follows the practical shape of OBLITERATUS/HERETIC-style
abliteration without making fine-tuning part of the critical path:

- store harmful and benign activation means in `direction_artifact.pt`
- extract directions with mean-difference, paired SVD, or whitened paired SVD
- optionally orthogonalize the refusal direction against benign activations
- preserve output-row norms after projection
- allow per-module and per-layer strength multipliers
- run reference-alignment sweeps before writing another 49 GB checkpoint

## SOTA Backends

For best available abliteration, model-forge should orchestrate external SOTA
tooling and then evaluate the resulting checkpoint with the same local suite.
The local implementation remains useful for transparent experiments, but the
default SOTA recipe is now:

1. prepare backend config from the model-family config
2. run OBLITERATUS `advanced` as the primary noninteractive backend
3. use Heretic as the KL-optimized baseline/oracle
4. serve the produced `local_abli_sota` checkpoint
5. run internal eval first, then artifact and external evals only if internal
   refusal suppression moves

Both OBLITERATUS and Heretic are AGPL-licensed in their open-source form. Review
their license terms before redistributing modified code, model artifacts, or
running modified tooling as a service.

Prepare backend-specific files:

```bash
./forge ablate gemma4_26b_a4b sota-prepare
```

Run the preferred SOTA backend when the environment is ready:

```bash
./forge ablate gemma4_26b_a4b sota-run --execute
```

Select Heretic explicitly:

```bash
./forge ablate gemma4_26b_a4b sota-run --backend heretic --execute
```

The SOTA output path for Gemma is:

```text
~/models/gemma-4-26B-A4B-it-local-abliterated-sota
```

Do not run collection while a vLLM server is active. Keep one large model
process at a time.

Before exporting, run the reference diagnostic. It compares base-to-downloaded
abli deltas with the configured target pattern and, when local directions exist,
reports cosine similarity between the reference delta and the candidate
projection delta. This is diagnostic only; export still uses base weights and
locally collected directions.

```bash
./forge ablate gemma4_26b_a4b analyze-reference \
  --output artifacts/abliteration/gemma4_26b_a4b_local_abli/reference_diagnostics.json
```

To rank training-free edit settings against the downloaded abli checkpoint
before exporting:

```bash
./forge ablate gemma4_26b_a4b sweep-reference \
  --include-norm-preserve \
  --output artifacts/abliteration/gemma4_26b_a4b_local_abli/reference_sweep.json
```

If earlier root-run containers already own `artifacts/`, either repair the
directory ownership outside the container or write diagnostics to `/tmp` for
inspection.

For exploratory strength sweeps, write each candidate to a distinct output
directory:

```bash
./forge ablate gemma4_26b_a4b export --execute --overwrite \
  --strength 3.0 \
  --output-dir ~/models/gemma-4-26B-A4B-it-local-abliterated-v3
```

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

## Training Parallelism

DGX Spark can be bandwidth-bound during post-training data and activation
pipelines. The hardware profile records a high-throughput setting of `c=192`,
but model-forge does not enable that by default because excessive workers can
raise host memory pressure and make model loading less predictable.

Safe default:

```bash
MODEL_FORGE_HARDWARE_PROFILE=dgx_spark ./forge ablate gemma4_26b_a4b plan
```

Opt in to the high-parallelism recommendation:

```bash
MODEL_FORGE_HARDWARE_PROFILE=dgx_spark \
MODEL_FORGE_ENABLE_HIGH_PARALLELISM=1 \
./forge ablate gemma4_26b_a4b plan
```

Explicit override:

```bash
MODEL_FORGE_PARALLELISM=192 ./forge ablate gemma4_26b_a4b plan
```

Keep activation collection at `batch_size: 1` for Gemma 26B-A4B unless a smoke
run proves there is enough headroom. The `c` setting is for preprocessing and
input-pipeline parallelism, not for multiplying the number of large model
forward passes in memory.

## Comparing The Local Ablation

After a reviewed export exists at `~/models/gemma-4-26B-A4B-it-local-abliterated-v3`,
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
