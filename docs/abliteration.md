# Abliteration Workflow

model-forge treats refusal ablation as a reproducible, model-family-agnostic
experiment:

1. collect a refusal direction from suffix-pooled harmful-vs-benign contrast prompts
2. review the direction artifacts and candidate layers
3. optionally compare edit geometry against a downloaded abli model for diagnostics
4. export or merge a local ablated model from the selected edit
5. run the same internal, artifact, and external eval suite against source,
   reference, and local candidates

## Generalizable Recipe

The portable model-forge recipe is the experiment structure, not a fixed set of
Gemma layer numbers or strengths. The goal is that when a new open model appears
Qwen, Llama, Gemma, Mixtral, Mistral, Phi, or another family, the user adds a
family config and follows the same post-training loop.

For each new model family:

1. Add or update a model-family YAML so base, fine-tuned, downloaded abli, and
   local abli variants have explicit local paths, served names, context length,
   quantization, and report locations.
2. Build a refusal/benign prompt set from the same internal buckets used for
   evaluation, but keep train-style direction prompts and held-out eval prompts
   separate.
3. Compute fresh refusal directions on the exact source checkpoint you intend to
   ablate. A fine-tuned model should get its own directions; do not blindly reuse
   base activations.
4. Pick architecture-specific target modules. Attention output and MLP
   down-projection layers are common starting points, but exact names and safe
   target sets differ by family. Qwen, Llama, Gemma, Mixtral, and other MoE or
   hybrid models can expose different module names, expert layouts, layer
   counts, and hidden sizes, so inspect the module tree before editing weights.
5. Start with conservative memory settings: one model process, batch size 1 for
   activation/residual work, explicit output directories per candidate, and no
   concurrent vLLM server.
6. Run a bounded search or direct transfer. Direct transfer is acceptable only
   for nearby checkpoints after fresh directions are recomputed. For a new
   architecture, run a small Heretic/OBLITERATUS sweep and promote the best
   candidate based on model-forge evals.
7. Evaluate against the source model, not just against the base model. A combined
   fine-tuned/ablated checkpoint succeeds if refusal suppression improves while
   preserving the fine-tuned model's capability and benign-answer quality.
8. Save the recipe, model card, raw responses, scores, and exact served model
   name. A model is not promoted from anecdotes or a single harmful prompt.

Promotion criteria should be explicit:

- unsafe/refusal buckets: ablation refusal suppression should move up
- benign paired boundary: benign refusal should stay low and answer quality
  should stay near the source model
- challenge capability: preserve source-model performance within expected eval
  variance
- normal-use and artifact evals: no material regressions
- external benchmarks: rerun when the internal suite says the candidate is worth
  the cost

For ablation research, unsafe compliance is not automatically a failure. It is
the intended direction for refusal-removal experiments, but it must be reported
separately from capability preservation and should be handled carefully when
publishing models.

## What Transfers

These pieces are expected to generalize:

- the model-family registry and serve/eval/report loop
- harmful/benign direction collection with held-out eval prompts
- exact-module LoRA targeting for Heretic-style edits
- row/norm preservation as the default for capability preservation
- one-model-at-a-time DGX Spark safety discipline
- promotion based on source-model-relative evals

These pieces should be recalibrated per model family:

- layer ranges
- module names and expert handling
- strength, layer weighting, and direction scope
- Heretic/OBLITERATUS search bounds
- prompt mix if the target domain or language differs
- serving settings when context length, attention backend, or quantization
  changes

Example: the Gemma t34 direct-transfer recipe is a warm start for the
Gemma/Gemopus family because the base and FT checkpoints share architecture and
similar refusal geometry. It should not be assumed to work unchanged for Qwen
3.5. For Qwen, add a Qwen family config, inspect Qwen target modules, compute
fresh Qwen directions, run a bounded search, then promote only after the Qwen
base/FT/local-abli comparison shows refusal removal with preserved source-model
performance.

The same principle applies to fine-tuning plus ablation. The ablation baseline
is the checkpoint being ablated. If the source is a fine-tuned model, success
means removing refusals while preserving that fine-tuned model's capability, not
necessarily matching a different downloaded abli model on every metric.

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

Outputs are written under the configured family artifact directory, for example:

```text
artifacts/abliteration/gemma4_26b_a4b_local_abli/
```

The current Gemma recipe is the first validated worked example. It pools activations over short
assistant suffixes instead of only the final prompt token, includes layer 29,
uses norm-preserving biprojection, and refuses to export if any configured
target layer lacks a collected direction. The target modules intentionally
match the downloaded abli checkpoint pattern: `self_attn.o_proj.weight` and
`mlp.down_proj.weight` for language layers 5 through 29. Mixture-of-experts
weights, embeddings, and the LM head stay untouched.

The transparent local backend follows the practical shape of OBLITERATUS/HERETIC-style
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
The local implementation remains useful for transparent experiments. A typical
SOTA path is:

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

The SOTA output path is configured per family. For the validated Gemma base
recipe it is:

```text
~/models/gemma-4-26B-A4B-it-local-abliterated-sota-internal-t34
```

The base Gemma recipe in `configs/abliteration/gemma4_26b_a4b_local_abli.yaml`
uses Heretic with model-forge internal prompts and saves the selected Pareto
trial `[Trial  34] Refusals:  1/27, KL divergence: 0.0183`.

The FT Gemopus recipe in
`configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml` reuses those selected
t34 Heretic parameters, but computes fresh refusal directions on
`Jackrong/Gemopus-4-26B-A4B-it`. This produced the r7 model:

```text
~/models/Gemopus-4-26B-A4B-it-local-abliterated-sota-internal-r7-selected-t34-transfer
```

That direct-transfer path intentionally skips Heretic's full-response
optimization evaluator during export on DGX Spark. The checkpoint should be
judged with model-forge evals after serving. In the recorded r7 run, refusal
suppression matched or beat the downloaded abli while preserving the FT model's
normal-use and challenge capability scores.

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
