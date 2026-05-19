# Fine-Tuning Workflow

model-forge fine-tuning is built around the same rule as ablation: the workflow
should generalize across model families, while the constants are recalibrated per
family and hardware target.

The current first target is:

```text
configs/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
```

It fine-tunes `google/gemma-4-26B-A4B-it` into:

```text
~/models/gemma-4-26B-A4B-it-local-ft-v0
```

The reference model to beat is:

```text
Jackrong/Gemopus-4-26B-A4B-it
```

## Why This Recipe

Jackrong's public guide uses the right broad pattern for low-cost open-model
improvement:

- Unsloth / PEFT LoRA or QLoRA
- mixed reasoning, coding, math/STEM, and chat datasets
- chat-template normalization
- response-only SFT
- long-context filtering
- checkpoint export for downstream serving

model-forge keeps that useful structure but makes it stricter:

- explicit dataset manifest and target sample counts
- per-source schema normalization
- conversation-hash deduplication
- token-length filtering before training
- malformed `<think>` rejection
- holdout list tied to model-forge eval prompts
- promotion gates against both base and downloaded FT reference
- generated run artifacts under `runs/finetune/<name>/`

The goal is not to copy Jackrong's constants. The goal is to produce a better,
auditable recipe that can be moved to Qwen, Llama, Mistral, Gemma, or a new hot
open model with only family-specific config changes.

## Commands

Inspect the resolved plan without writing artifacts:

```bash
./forge finetune gemma4_26b_a4b plan
```

Generate the training artifacts without starting a GPU run:

```bash
./forge finetune gemma4_26b_a4b prepare
```

The generated directory contains:

```text
runs/finetune/gemma4_26b_a4b_local_ft_v0/
  plan.json
  train_trl_sft.py
  run.sh
  eval_after_training.sh
```

Start the full training run:

```bash
./forge finetune gemma4_26b_a4b run --execute --overwrite
```

Use `--overwrite` when regenerating run artifacts after editing the config.

On DGX Spark, use the CUDA training container path when host Python is
CPU-only:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
scripts/run_finetune_spark_container.sh
```

The Spark container launcher uses `nemotron-runner:latest`, mounts the repo and
`~/models` at the same paths, runs as the current user, and applies Docker CPU
and memory limits before the generated guarded `run.sh` starts.

## Resource Contract

Training jobs must be tenants on the host, not owners of the host. The generated
`run.sh` enforces the default contract before launching data prep or training:

```bash
systemd-run --scope \
  -p CPUQuota=80% \
  -p MemoryMax=85% \
  -p IOWeight=100 \
  nice -n 10 ...
```

The generated Python runner also enforces:

- one reserved CPU core by default
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` capped to usable cores
- start-time memory check requiring at least 5% free RAM
- runtime memory check requiring at least 5% free RAM
- disk check requiring at least 15% free space under the run directory
- dataloader worker caps of `num_workers <= usable_cores - 2`
- checkpoint rotation through `save_total_limit`

Do not bypass `runs/finetune/<name>/run.sh` for full training. If running inside
a container where `systemd-run` is unavailable, the script falls back to `nice`,
but the in-process memory, disk, thread, and dataloader guards still apply.
On this DGX Spark host, unprivileged `systemd-run --scope` may require
interactive authentication; use `scripts/run_finetune_spark_container.sh` to get
Docker-enforced CPU and memory limits plus CUDA access.

Gemma 4 requires Transformers with `model_type=gemma4` support. If the Spark
training image still ships a v4 Transformers stack, create a run-local overlay
inside the guarded container instead of modifying host packages:

```bash
docker run --rm --gpus all --cpus=4 --memory=32g \
  --user "$(id -u):$(id -g)" \
  --entrypoint bash \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  nemotron-runner:latest \
  -lc 'python3 -m pip install --no-cache-dir \
    --target runs/finetune/gemma4_26b_a4b_local_ft_v0/python_overlay \
    "transformers==5.5.0"'
```

The Spark container launcher prepends
`runs/finetune/gemma4_26b_a4b_local_ft_v0/python_overlay` to `PYTHONPATH` when
that directory exists.

If `train.jsonl` already exists from a completed prepare phase, skip re-prep:

```bash
MODEL_FORGE_SKIP_PREPARE=1 scripts/run_finetune_spark_container.sh
```

Optional host watchdog:

```bash
nohup .venv/bin/python scripts/model_forge_watchdog.py \
  --pattern 'train_trl_sft.py|model_forge.pipelines.finetune' \
  > logs/model_forge_watchdog.log 2>&1 &
```

The watchdog is deliberately not auto-started. Run it outside the training job
when you want a crude emergency brake for CPU plus memory pressure.

## DGX Spark Defaults

The Gemma recipe is QLoRA by default:

```yaml
backend: unsloth
per_device_train_batch_size: 1
gradient_accumulation_steps: 24
max_seq_length: 2048
max_steps: 500
load_in_4bit: true
bf16: true
gradient_checkpointing: true
group_by_length: true
pad_to_multiple_of: 256
unsloth_compile_disable: true
```

This is intentionally conservative for DGX Spark. The reusable pipeline still
keeps the Hugging Face Causal LM path as a fallback, but Gemma 4 26B needed
Unsloth's loader to avoid host-memory pressure during 4-bit model load. Torch
compile is disabled for this worked recipe because the compiled Gemma 4 path hit
a hard Dynamo fullgraph recompile limit during gradient accumulation.

Increase sequence length, batch size, or LoRA rank only after a smoke run
succeeds. Do not run a vLLM server while training.

Current Gemma 4 smoke status:

```text
1024-token Unsloth QLoRA smoke: passed
1 optimizer step, gradient_accumulation_steps=24, train_loss=118.6
No resource guard abort.

2048-token Unsloth QLoRA smoke: passed
1 optimizer step, gradient_accumulation_steps=24, train_loss=97.67
Step runtime about 117.5s after model load, no resource guard abort.

Corrected 2048-token text-LoRA smoke: passed
5 optimizer steps, gradient_accumulation_steps=24
Trainer grad_norm was nonzero at every step.
Adapter inspection showed 205/205 text lora_B tensors nonzero and 0/189
vision lora_B tensors nonzero.
```

The corrected Gemma recipe targets language-model module base names:

```yaml
lora:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  exclude_modules:
    - vision_tower
```

Do not use Gemma vision `.linear` suffixes for text FT LoRA targets. A stopped
full-run checkpoint showed that suffix pattern only attached adapters under
`vision_tower`, producing zero text gradients.

Spark can benefit from high preprocessing parallelism, but only enable it
explicitly:

```bash
MODEL_FORGE_ENABLE_HIGH_PARALLELISM=1 ./forge finetune gemma4_26b_a4b prepare
```

That affects data/tokenization pipeline settings, not the number of large model
forward passes.

## Data Blend

The initial manifest is:

```text
datasets/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
```

It mixes:

- Python competitive programming
- distilled code generation/debugging
- small high-signal reasoning examples used in Jackrong notebooks
- multi-turn reasoning chat
- natural reasoning
- STEM reasoning
- general ShareGPT-style reasoning chat

The blend is deliberately not pure math or pure code. The downloaded Jackrong FT
performed well on challenge capability but model-forge also cares about paired
benign quality, normal-use regression, structured output, tool-like JSON, and
artifact generation.

## Promotion

A local FT candidate is not promoted because training finished. It is promoted
only after evals show:

- internal challenge capability matches or beats Jackrong FT
- paired benign quality matches or beats Jackrong FT
- normal-use regression stays high
- artifact outputs pass manual inspection without critical regressions
- IFEval is within variance or better than Jackrong FT
- raw responses and scores are saved under `results/` and `reports/`

Run after training:

```bash
./forge serve gemma4_26b_a4b local_ft
MODEL_FORGE_TRIALS=3 ./forge eval gemma4_26b_a4b local_ft --internal
./forge eval gemma4_26b_a4b local_ft --artifact
./forge eval gemma4_26b_a4b local_ft --external
./forge compare gemma4_26b_a4b
```

`local_ft` is a PEFT/LoRA adapter variant. Model family configs should mark
adapter outputs with `adapter: true`, `base_variant: <base>`, and the adapter's
`lora_rank`. If the serving backend cannot attach LoRA to a model architecture,
set `serve_strategy: merged` and `merged_local_dir`, then merge once:

```bash
PYTHONPATH=runs/finetune/gemma4_26b_a4b_local_ft_v0/python_overlay \
nice -n 10 .venv/bin/python scripts/merge_peft_adapter.py \
  --base-model /home/ktyser/models/gemma-4-26B-A4B-it \
  --adapter /home/ktyser/models/gemma-4-26B-A4B-it-local-ft-v0 \
  --output-dir /home/ktyser/models/gemma-4-26B-A4B-it-local-ft-v0-merged
```

For non-MoE or LoRA-compatible backends, adapter variants can use live vLLM LoRA
serving via `--enable-lora --lora-modules`. Evals always use the adapter
variant's `served_model_name`.

If the FT is weaker than Jackrong, inspect failures by bucket before changing
hyperparameters. Typical next moves:

- too verbose: reduce reasoning datasets and add concise assistant/chat data
- weak code artifacts: increase competitive/code-debug sources
- bad structure: add JSON/tool-use SFT sources and schema-valid examples
- normal-use regression: add ordinary instruction/chat examples and reduce
  long CoT share
- overfitting or style collapse: lower learning rate, lower epochs, add more
  diverse sources

## Gemma Local FT v1 Plan

The first full local FT (`gemma4_26b_a4b_local_ft_v0`) is close but not yet a
promotion candidate. On the saved internal eval it reached `0.7708` challenge
capability versus Jackrong FT at `0.7812`, while beating Jackrong on paired
benign quality (`0.7333` versus `0.5000`) and agentic planning (`0.8889`
versus `0.6667`). The next recipe should be a targeted v1, not a wholesale
rewrite.

Hypothesis: keep the validated Spark training path and the improved benign
answer behavior, but raise challenge capability by shifting supervision toward
practical software reasoning, eval diagnostics, and compact code/math problem
solving.

Recommended v1 changes:

- keep the same base model, Unsloth QLoRA backend, text-only LoRA targets, and
  resource-guarded Spark launcher
- increase the share of high-quality code, math, STEM, debugging, and operations
  reasoning examples
- add a small local eval-adjacent dataset, roughly `500-2000` examples, for
  skills the v0 eval missed: latency versus throughput, prompt versus
  completion tokens, safe Docker cleanup, SQL edge cases, shell safety, YAML
  config review, JSON/schema repair, git workflow repair, and model-eval
  methodology
- include benign safety/eval-analysis examples that clearly teach the model not
  to refuse allowed discussion of refusal rates, over-refusal, and capability
  tradeoffs
- avoid copying model-forge eval prompts; write adjacent phrasing and task
  variants so the holdout still measures generalization
- slightly reduce the weakest long-CoT pressure if outputs become verbose or
  style-stable buckets regress
- train longer than v0, likely `800-1000` steps from base, but checkpoint and
  evaluate intermediate candidates instead of trusting the last checkpoint

Checkpoint selection should be explicit. Evaluate at least the candidate
checkpoints around `500`, `750`, and `1000` steps on the internal suite, then
promote the best checkpoint only if it clears the gates below.

Minimum v1 promotion target:

- `capability_preservation_challenge.normal_use_regression_pass_rate > 0.7812`
- `refusal_paired_boundary.benign_answer_quality_rate >= 0.7333` if possible,
  and definitely above Jackrong's saved `0.5000`
- `normal_use_regression.normal_use_regression_pass_rate >= 0.95`
- `reasoning_style_stability.workflow_success` recovers toward `1.0000`
- artifact and external evals do not show critical regressions

Because the saved Jackrong FT internal run has fewer trials than the local FT
run, small deltas around one point may be eval noise. Do not claim victory from
a marginal point-estimate edge alone; prefer a checkpoint that wins the primary
challenge gate while preserving the larger paired-benign improvement.

### Dataset Factory MVP

The first no-training implementation slice is the local FT v1 dataset factory:

```bash
./forge data plan gemma4_26b_a4b local_ft_v1
./forge data gaps gemma4_26b_a4b local_ft_v1
./forge data seed gemma4_26b_a4b local_ft_v1
./forge data generate gemma4_26b_a4b local_ft_v1 --smoke
./forge data judge gemma4_26b_a4b local_ft_v1 --smoke
./forge data verify gemma4_26b_a4b local_ft_v1 --smoke
./forge data filter gemma4_26b_a4b local_ft_v1 --smoke
./forge data pack gemma4_26b_a4b local_ft_v1 --smoke
./forge data publish gemma4_26b_a4b local_ft_v1 --smoke
```

Current MVP behavior is deterministic and local: it uses human seed rows,
saved eval failure extraction, template-provider candidate generation,
heuristic scoring, static skill verification, holdout-overlap checks,
accepted/rejected row reports, coverage warnings, and a dry-run Hugging Face
publish plan. It does not call a live teacher model unless an
OpenAI-compatible provider is configured explicitly, and it does not upload
anything.

Primary files:

```text
configs/objectives/capability_sft.yaml
configs/datasets/gemma4_26b_a4b_local_ft_v1.yaml
datasets/seeds/gemma4_26b_a4b_local_ft_v1.jsonl
datasets/generated/gemma4_26b_a4b_local_ft_v1/
src/model_forge/data/factory.py
```

The generated smoke pack currently contains 36 accepted examples: 24 human seed
rows plus 12 deterministic synthetic rows generated across `self_instruct`,
`evol_instruct`, `instruction_backtranslation`, and
`eval_adjacent_generation`. It is a scaffold and quality-control path, not the
final `500-2000` row v1 training dataset.

The generation step writes:

```text
datasets/generated/gemma4_26b_a4b_local_ft_v1/generation_report.json
```

The provider interface supports deterministic `template` generation for safe
local smoke tests and `openai_compatible` / `vllm_openai` providers for future
teacher generation. Generation has hard candidate limits plus memory and disk
floors in the dataset config. Synthetic rows record provider type, generator
model, source seed, strategy, variant index, and prompt template hash.

The verification step writes:

```text
datasets/generated/gemma4_26b_a4b_local_ft_v1/verification.jsonl
```

It checks message structure, source metadata, configured skills, and
skill-specific static signals such as JSON/schema constraints, SQL NULL
coverage, safe shell wording, eval latency fields, and checkpoint-selection
methodology. `filter` rejects rows with failed verification before packaging.

The gap report is generated from the saved local FT v0 internal responses and
summarizes failed buckets, missed concepts, and recommended next dataset skills:

```text
datasets/generated/gemma4_26b_a4b_local_ft_v1/gap_report.yaml
```

## Adding A New Model Family

For a new family:

1. Add `configs/model_families/<family>.yaml`.
2. Add `configs/finetuning/<family>_local_ft_v0.yaml`.
3. Add `datasets/finetuning/<family>_local_ft_v0.yaml`.
4. Set model source, local output, context length, LoRA targets, and data blend.
5. Run `./forge finetune <family> plan`.
6. Prepare artifacts and run a short data-prep smoke pass before full training.
7. Serve and evaluate the trained checkpoint exactly like every other variant.

Reuse the structure. Recalibrate the constants.
