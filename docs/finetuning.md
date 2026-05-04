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
- start-time memory check requiring at least 15% free RAM
- runtime memory check requiring at least 10% free RAM
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
per_device_train_batch_size: 1
gradient_accumulation_steps: 24
max_seq_length: 8192
load_in_4bit: true
bf16: true
gradient_checkpointing: true
```

This is intentionally conservative for DGX Spark. Increase sequence length,
batch size, or LoRA rank only after a smoke run succeeds. Do not run a vLLM
server while training.

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

If the FT is weaker than Jackrong, inspect failures by bucket before changing
hyperparameters. Typical next moves:

- too verbose: reduce reasoning datasets and add concise assistant/chat data
- weak code artifacts: increase competitive/code-debug sources
- bad structure: add JSON/tool-use SFT sources and schema-valid examples
- normal-use regression: add ordinary instruction/chat examples and reduce
  long CoT share
- overfitting or style collapse: lower learning rate, lower epochs, add more
  diverse sources

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
