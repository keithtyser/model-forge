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
   base activations. Executable stages preflight the local source checkpoint
   before model load; PEFT fine-tunes must be merged into a full checkpoint
   first.
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

OBLITERATUS, Heretic, and Abliterix are external open-source backends with their
own licenses. Review license terms before redistributing modified code, model
artifacts, or running modified tooling as a service.

As of the June 2026 Qwen work, `sota-plan` and `sota-prepare` also know about
method-shift backends:

- `abliterix`: guarded search-only practical toolkit backend with SRA/SOM/OT
  vector methods
- `obliteratus`: guarded advanced baked checkpoint backend
- `apostate`: guarded preservation-direction baked checkpoint backend
- `sra`: surgical refusal ablation / concept-preserving direction cleanup
- `optimal_transport`: native guarded diagnostic for distributional activation
  transport-style checkpoint edits
- `norm_preserving_projection`: native guarded projected/biprojected checkpoint
  edit with optional row-norm preservation for MPOA/NPBA-style method shifts
- `som_projection`: native guarded SOM-style multi-centroid refusal-residual
  projection for cases where one global direction is too blunt
- `qwen_scope_sae`: native guarded SAE dictionary-constrained checkpoint
  projection for Qwen-family residual refusal blockers

For Qwen-family checkpoints, add a contrast-design audit before repeating a
failed direction recipe. The June 2026 Qwen contrast warning says topic-matched
harmful/benign prompt pairs can make refusal directions ineffective. If a
candidate keeps the same stochastic refusal opening after SRA, OBLITERATUS,
native OT, or SOM projection, do not only raise strength or prompt weights.
Instead, test at least one non-topic-matched response-style contrast, or run a
guarded `qwen_scope_sae` dictionary-constrained diagnostic when a compatible
SAE dictionary exists. The candidate still must pass the same model-forge
harmful-detail, safe-redirect, benign-quality, and source-capability gates.

Standalone `sra` remains plan-only until a guarded model-forge runner exists.
OBLITERATUS, Apostate, Abliterix, native optimal transport, and native
norm-preserving/SOM projection now have guarded execution paths. Abliterix first
runs in non-interactive search-only mode; it writes an Optuna journal and exits
without exporting a checkpoint. Use
`abliterix-search-analyze` before `abliterix-export` for a selected trial.
OBLITERATUS, Apostate, native optimal transport, native norm-preserving/SOM
projection, and `qwen_scope_sae` write baked checkpoints directly, so their
backend reports must be followed by source-vs-candidate model-forge targeted
evals before any broader eval, quantization, promotion, or upload. The contract
is the same for every backend: one large model job at a time, source checkpoint
audit, CPU/RAM/disk caps, targeted internal eval before broader eval, and
source-relative promotion gates.

When several diagnostics have completed, rank them with the same case-level gate
used for promotion instead of comparing backend proxy scores or reading reports
by hand:

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_candidate_gate.yaml candidate-gate --write-report
```

`candidate-gate` consumes completed model-forge eval directories with
`responses.jsonl`. It does not start servers or export checkpoints. Add a
`candidate_selection.gate.requirements` block to make the same workflow apply to
another model family, refusal category, or capability-retention case.

To plan the next bounded candidate loop before starting any heavy job:

```bash
./forge ablate --config <config.yaml> candidate-loop-plan --write-plan
```

`candidate-loop-plan` writes a runbook under
`reports/generated/abliteration_candidate_loop/`. It lists preflight checks,
per-candidate export/sync/audit/serve/eval commands, expected eval directories,
the final `candidate-gate` command, and cleanup policy. Candidates with
`status: runner_missing`, `plan_only`, or `blocked` are documented but do not
emit executable heavy-job commands. This is the intended handoff format when the
next method family needs a new guarded runner before model work can continue.

The Qwen-Scope SAE backend uses the same SOTA lifecycle:

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_qwen_scope_sae_v21.yaml sota-plan --backend qwen_scope_sae
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_qwen_scope_sae_v21.yaml sota-prepare --backend qwen_scope_sae
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_qwen_scope_sae_v21.yaml sota-run --backend qwen_scope_sae --execute
```

This backend requires a compatible SAE decoder dictionary. The V21 Qwen config
uses `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` with per-layer
`layer{layer}.sae.pt` files and validates hidden size before export.

Prepare backend-specific files:

```bash
./forge ablate gemma4_26b_a4b sota-prepare
```

Run the preferred SOTA backend when the environment is ready:

```bash
./forge ablate gemma4_26b_a4b sota-run --execute
```

For Heretic recipes with `container_image` set, `sota-run --execute` launches the
generated search/direct runner through `scripts/run_heretic_direct_container.sh`.
That wrapper applies Docker CPU, memory, swap, PID, HF-cache, RAM-floor, and
disk-floor guardrails instead of running large model work through raw host
Python.

For OBLITERATUS recipes with `container_image` set, build the image first:

```bash
docker build -f docker/obliteratus.Dockerfile -t model-forge-obliteratus:latest .
```

Then run through the guarded wrapper:

```bash
./forge ablate --config <config.yaml> sota-run --backend obliteratus --execute
```

The wrapper applies the same Docker CPU, memory, swap, PID, HF-cache, RAM-floor,
and disk-floor guardrails as the other large-model backends. Recipes can
materialize model-forge prompt buckets into OBLITERATUS `harmful_prompts` and
`harmless_prompts`, but the backend report is still not promotion evidence.
Treat the first export as diagnostic until the local targeted eval proves the
source-relative objective.

Some external backends save a different checkpoint namespace than the source
model. In the Qwen 3.6 27B diagnostic, OBLITERATUS flattened the source wrapper
from `model.language_model.*` to `model.*` and wrote a text-only config that
the Spark vLLM build could not serve. Before eval, normalize those exports with
`scripts/remap_safetensors_checkpoint.py`, verify the remapped tensor keys
against the source checkpoint, restore source tokenizer/config metadata, and
rerun strict checkpoint/tokenizer/architecture audits.

For Abliterix recipes with `container_image` set, `sota-run --backend abliterix
--execute` launches the generated non-interactive search runner through
`scripts/run_abliterix_search_container.sh`. The runner does not export a model.
After it finishes, run:

```bash
./forge ablate --config <config.yaml> abliterix-search-analyze --backend abliterix
```

Only build/export a selected trial after the journal gate recommends
`prepare_guarded_export_runner`, then run the model-forge targeted internal eval
before broader evals, quantization, or upload.

```bash
./forge ablate --config <config.yaml> abliterix-export \
  --backend abliterix \
  --trial-index <selected-index> \
  --overwrite
```

The default export command is a dry run and only writes the guarded export
runner. Add `--execute` after checking disk/RAM headroom and confirming no other
large model process is active. Some Abliterix journals record candidate
refusals and KL but not the baseline refusal count; in that case
`abliterix-search-analyze` can still recommend preparing an export runner, but
the required next gate must compare the source checkpoint against the exported
checkpoint with model-forge targeted internal evals.

For native optimal-transport recipes, `sota-prepare --backend
optimal_transport` writes three artifacts under the backend work directory:

- `model_forge_native_prompt_pairs/`: source-relative harmful/benign prompt
  text files and manifest
- `native_optimal_transport_config.yaml`: the derived model-forge
  `collect`/`export` config
- `run_native_optimal_transport.py`: the guarded runner

Run it through:

```bash
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 \
MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 \
./forge ablate --config <config.yaml> sota-run --backend optimal_transport --execute
```

This path approximates activation transport with multi-component
whitened-paired-SVD directions and a norm-preserving baked projection. `sota-run`
launches the generated runner through `scripts/run_native_checkpoint_scope.sh`,
which uses `scripts/run_native_checkpoint_container.sh` when the recipe sets
`container_image` and otherwise uses `scripts/run_native_checkpoint_scope.sh`.
Use the container path for new architectures when the host venv is CPU-only or
does not recognize the model type. It is a diagnostic method-shift backend, not
promotion evidence. After export, register the checkpoint if needed, run strict
checkpoint/tokenizer/architecture audits, then run the source-vs-candidate
targeted gate before broader evals, quantization, upload, or family promotion.
Native activation collection prints model-load, prompt, and layer progress;
override `MODEL_FORGE_NATIVE_PROGRESS_EVERY` or
`MODEL_FORGE_NATIVE_LAYER_PROGRESS_EVERY` only when a very large prompt set needs
less frequent logging.

For native SOM projection recipes, use `sota-prepare --backend som_projection`.
The generated runner is the same guarded checkpoint-export path, but activation
collection uses `direction_extraction: som_centroids`. The extractor learns a
bounded set of refusal residual centroids from harmful prompts relative to the
benign mean, combines them with the global mean direction, orthonormalizes the
resulting basis, then applies the configured projection. Keep `som_neurons`,
`som_steps`, and `direction_components` small until the targeted gate improves;
multi-direction methods increase search surface and can over-edit capability if
the edit scope is broad.

For Apostate recipes with `container_image` set, first build the backend image:

```bash
docker build -f docker/apostate.Dockerfile -t model-forge-apostate:latest .
```

Then launch through the guarded wrapper:

```bash
./forge ablate --config <config.yaml> sota-run --backend apostate --execute
```

The wrapper applies Docker CPU, memory, swap, PID, HF-cache, RAM-floor, and
disk-floor guardrails. Apostate accepts local text prompt files, so model-forge
materializes `model_forge_prompt_datasets` into `harmful_path`,
`harmless_path`, `harmful_test`, `harmless_test`, and `preserve_path` files in
the backend work directory. Keep the backend prompt fit narrow and evaluate the
original held-out model-forge cases after export; do not promote from
Apostate's own refusal/KL report alone.

Recipes can also condition Heretic prompt sections on prior eval traces by
setting keys such as `bad_train_response_source`,
`bad_train_response_case_ids`, and `bad_train_response_score_filters` under
`model_forge_prompt_datasets`. The source is JSONL with `bucket`, `case_id`,
`prompt`, `response_text`, `checks`, and `scores`; matching traces are appended
to the generated section using the configured response template. Use this when
generic refusal/compliance suffixes are too weak and the next search should
target the model's actual residual failure responses.

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
