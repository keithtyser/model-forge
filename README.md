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
- tracks dataset source registries with licenses, roles, quality tiers, and sampling caps
- tracks a dated research registry so methods, evals, and limitations stay explicit
- runs refusal ablation recipes against source checkpoints
- plans and reports quantization paths, with Blackwell NVFP4 as the first-class Spark target
- serves exactly one candidate at a time through hardware-aware vLLM settings
- plans and validates generic cluster inventories without hard-coded private hosts
- benchmarks already-running OpenAI-compatible serving endpoints
- evaluates internal behavior, artifact quality, and external benchmark results
- compares candidates against the source model and relevant references, with
  manifest provenance, comparability warnings, and research-basis links
- writes promotion reports from saved comparisons
- records hypotheses, recipes, validation, and publish state for handoff

## Current State

Gemma 4 A4B has validated paths for base evaluation, downloaded FT/reference
evaluation, downloaded abli evaluation, local base ablation, local FT ablation,
and a guarded local FT recipe. The local FT v0 did not beat Jackrong on the
primary challenge-capability gate, but it was close and improved paired benign
quality. The next FT iteration is the local FT v1 dataset path, which currently
has deterministic and live-teacher smoke packs with gated review artifacts. The
next step is a medium live-teacher generation pass, not a long training run.

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

Serve a small teacher for dataset generation:

```bash
./forge serve-teacher qwen35_9b
```

Run evals from another terminal:

```bash
./forge eval gemma4_26b_a4b base --internal
./forge eval gemma4_26b_a4b base --artifact
./forge eval gemma4_26b_a4b base --external
./forge compare gemma4_26b_a4b
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
./forge research audit
./forge manifest write --run-type eval --family gemma4_26b_a4b --variant base --command './forge eval gemma4_26b_a4b base --internal'
./forge bench serve --family gemma4_26b_a4b --variant base --dry-run
./forge bench sweep plan --family gemma4_26b_a4b --variant base
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --write-plan
./forge doctor
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
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
```

Fine-tuning:

```bash
./forge finetune gemma4_26b_a4b plan
./forge finetune gemma4_26b_a4b prepare
./forge finetune --config configs/finetuning/gemma4_26b_a4b_local_ft_v1_dryrun.yaml plan
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

Use `generate --overwrite` only when replacing candidates intentionally.
Downstream `--overwrite` refreshes derived artifacts from existing candidates.
`publish` writes a dry-run HF plan by default; `publish --execute` refuses
seed-only and smoke-only datasets.

Promotion reports:

```bash
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
./forge promote gemma4_26b_a4b local_abli_sota_vs_downloaded_abli
```

Research registry:

```bash
./forge research list
./forge research show arditi_2024_refusal_direction
./forge research audit
```

Run manifests:

```bash
./forge manifest write \
  --run-type eval \
  --status planned \
  --family gemma4_26b_a4b \
  --variant base \
  --config configs/experiments/gemma4_26b_a4b_v0.yaml \
  --run-output-dir results/gemma4_26b_a4b_v0/base \
  --command './forge eval gemma4_26b_a4b base --internal'
```

See [docs/run-manifests.md](docs/run-manifests.md) for the canonical schema and
handoff rules.

Cluster planning:

```bash
./forge cluster doctor --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster plan \
  --config configs/clusters/dgx_spark_x2.example.yaml \
  --workload train \
  --launcher torchrun
```

See [docs/cluster.md](docs/cluster.md). Public configs use environment-backed
placeholders; private hostnames, IPs, usernames, tokens, and absolute paths do
not belong in Git.

Serving benchmark:

```bash
./forge bench serve --family gemma4_26b_a4b --variant base --dry-run
MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1 ./forge bench serve --model served/model-name
./forge bench sweep plan --family gemma4_26b_a4b --variant base
```

See [docs/serving-benchmarks.md](docs/serving-benchmarks.md). The benchmark
expects a running OpenAI-compatible endpoint and writes `requests.jsonl`,
`summary.json`, `serving_card.md`, and `manifest.json` under
`reports/generated/serve_bench/`. Reusable workload definitions live under
`configs/serving/workloads/`. Sweep plans expand startup-time server env cases
and the matching `bench serve` commands, but they do not launch vLLM.

Quantization:

```bash
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --write-plan
./forge quantize matrix-plan --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml
./forge quantize card \
  --config configs/quantization/nvfp4_blackwell_runtime.yaml \
  --source-serving-summary <source>/summary.json \
  --candidate-serving-summary <candidate>/summary.json \
  --source-serving-eval <source_eval_dir> \
  --candidate-serving-eval <candidate_eval_dir> \
  --run-id source_vs_nvfp4 \
  --write-card
```

See [docs/quantization.md](docs/quantization.md). NVFP4 is the priority
Blackwell path. Self-quantization must run through the guarded export runner,
not raw Docker, and quantized candidates still need real serving and
behavior-preservation evidence before promotion.

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
[docs/finetuning.md](docs/finetuning.md) before starting long jobs. Read
[docs/serving-benchmarks.md](docs/serving-benchmarks.md) before publishing
serving claims.

## Repository Layout

```text
configs/        Model families, hardware/cluster profiles, training/editing configs
datasets/       Dataset manifests, seed rows, generated small packs and reports
docs/           Workflow docs, research snapshots, status, roadmap, and ledger
docs/roadmaps/  Long-form planning documents and archived roadmap material
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
