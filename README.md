# model-forge

model-forge is a reproducible post-training workbench for open models. It is
designed to answer one practical question: did a fine-tune, refusal ablation, or
combined post-training workflow improve the model without breaking capability,
format following, or operational behavior?

The repo is model-family driven. Gemma 4 is the first fully exercised family on
DGX Spark, but the intended shape is general: add a model family, tune or edit
it with calibrated configs, evaluate every candidate, then publish the recipe
and artifacts needed to reproduce the result.

## What It Does

- registers model families and variants in `configs/model_families/`
- builds fine-tuning plans from YAML configs and dataset manifests
- creates eval-adjacent SFT datasets through a gated dataset factory
- runs refusal ablation recipes against source checkpoints
- serves exactly one candidate at a time through hardware-aware vLLM settings
- evaluates internal behavior, artifact quality, and external benchmark results
- compares candidates against the source model and relevant references
- records hypotheses, recipes, validation, and publish state for handoff

## Current State

Gemma 4 A4B has validated paths for base evaluation, downloaded FT/reference
evaluation, downloaded abli evaluation, local base ablation, local FT ablation,
and a guarded local FT recipe. The local FT v0 did not beat Jackrong on the
primary challenge-capability gate, but it was close and improved paired benign
quality. The next FT iteration is the local FT v1 dataset path, which currently
has a smoke-sized gated pack and is ready for a small live-teacher generation
smoke, not a long training run.

Use [docs/status.md](docs/status.md) for the current handoff state and
[docs/experiment-ledger.md](docs/experiment-ledger.md) for detailed experiment
history.

## Quick Start

Install and set up:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/keithtyser/model-forge.git
cd model-forge
./forge setup all
```

List configured families:

```bash
./forge families
```

Download a configured family:

```bash
./forge download gemma4_26b_a4b all
```

Serve one model in one terminal:

```bash
./forge serve gemma4_26b_a4b base
```

Run evals from another terminal:

```bash
./forge eval gemma4_26b_a4b base --internal
./forge eval gemma4_26b_a4b base --artifact
./forge eval gemma4_26b_a4b base --external
./forge compare gemma4_26b_a4b
```

Only run one large model server or training job at a time.

## Core Workflows

Evaluation:

```bash
./forge eval gemma4_26b_a4b base --smoke
./forge eval gemma4_26b_a4b base --internal
./forge eval gemma4_26b_a4b base --artifact
./forge eval gemma4_26b_a4b base --external
./forge compare gemma4_26b_a4b
```

Fine-tuning:

```bash
./forge finetune gemma4_26b_a4b plan
./forge finetune gemma4_26b_a4b prepare
```

On DGX Spark, use the guarded CUDA container launcher for full FT runs:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
scripts/run_finetune_spark_container.sh
```

Dataset factory:

```bash
./forge data plan gemma4_26b_a4b local_ft_v1
./forge data gaps gemma4_26b_a4b local_ft_v1
./forge data generate gemma4_26b_a4b local_ft_v1 --smoke
./forge data verify gemma4_26b_a4b local_ft_v1 --smoke
./forge data review gemma4_26b_a4b local_ft_v1 --smoke --sample 50
./forge data pack gemma4_26b_a4b local_ft_v1 --smoke
./forge data publish gemma4_26b_a4b local_ft_v1 --smoke
```

Abliteration planning:

```bash
./forge ablate gemma4_26b_a4b plan
./forge ablate gemma4_26b_a4b sota-prepare --backend heretic
./forge ablate --config configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml sota-prepare --backend heretic
```

For new families, reuse the workflow, not Gemma constants. Recalibrate module
targets, layer ranges, direction prompts, strength/search bounds, tokenizer
behavior, and serving settings per architecture.

## DGX Spark Rules

DGX Spark settings are hardware policy, not model constants. Default training
and serving paths preserve system headroom with CPU, memory, disk, dataloader,
checkpoint, and one-model-at-a-time guardrails. Spark can benefit from high
parallelism in preprocessing, but that does not mean multiplying large model
forward passes.

Read [docs/dgx-spark.md](docs/dgx-spark.md),
[docs/spark-optimizations.md](docs/spark-optimizations.md), and
[docs/finetuning.md](docs/finetuning.md) before starting long jobs.

## Repository Layout

```text
configs/        Model families, fine-tuning configs, ablation configs, objectives
datasets/       Dataset manifests, seed rows, generated small packs and reports
docs/           Workflow docs, status, roadmap, and experiment ledger
evals/          Internal prompt buckets and rubrics
models/         Directory conventions for local model artifacts
pipelines/      Pipeline design notes
recipes/        Tracked reusable run templates and known-good generated recipes
reports/        Report conventions; generated reports are ignored
results/        Raw eval output conventions; generated result dirs are ignored
scripts/        Operational helpers and compatibility wrappers
src/            `model_forge` Python package
forge           User-facing CLI wrapper
```

See [configs/README.md](configs/README.md) and
[scripts/README.md](scripts/README.md) for directory-specific guidance.

Generated run directories, model weights, tokenized datasets, large reports, and
logs are not committed. See
[docs/artifact-retention.md](docs/artifact-retention.md) before deciding what to
commit, delete, or upload.

## Design Principles

- Optimize for reproducible post-training decisions, not one-off demos.
- Compare every edited model against the source checkpoint it came from.
- Treat ablation success as lower refusal while preserving source capability.
- Report unsafe overcompliance separately from capability.
- Treat fine-tune success as better capability and format quality without
  regressions.
- Keep model-family support in config and reusable pipeline code.
- Push code, configs, docs, recipes, and lightweight manifests to GitHub.
- Upload completed models, datasets, and durable eval artifacts to Hugging Face.
- Never commit tokens or raw model weights.
