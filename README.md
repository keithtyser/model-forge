# model-forge

model-forge is a reproducible post-training workbench for open models. It is
designed to answer one practical question: did a fine-tune, refusal ablation, or
combined post-training workflow improve the model without breaking capability,
format following, or operational behavior?

The repo is model-family driven. Gemma 4 is the first fully exercised family on
DGX Spark, while Qwen and Llama configs now serve as non-Gemma generalization
targets. The intended shape is general: add a model family, tune or edit it with
calibrated configs, evaluate every candidate, then publish the recipe and
artifacts needed to reproduce the result.

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
history. Use [docs/roadmap-status-audit.md](docs/roadmap-status-audit.md) for
the current MF backlog implementation and validation state.

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
./forge variants graph qwen35_9b
./forge variants architecture-audit qwen35_9b
./forge variants graph llama31_8b
```

Download a configured family:

```bash
./forge download gemma4_26b_a4b all
./forge download qwen35_9b base
./forge download llama31_8b base
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
./forge eval qwen35_9b base --smoke
./forge eval llama31_8b base --smoke
./forge eval gemma4_26b_a4b base --artifact
./forge eval gemma4_26b_a4b base --external
./forge compare gemma4_26b_a4b
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
./forge objectives audit
./forge roadmap audit --write-doc
./forge roadmap cli-drift
./forge generalization audit
./forge agent audit
./forge agent optimize-quantization --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --variants base
./forge agent optimize-behavior-edit --family gemma4_26b_a4b
./forge research audit
./forge manifest write --run-type eval --family gemma4_26b_a4b --variant base --command './forge eval gemma4_26b_a4b base --internal'
./forge hf status --offline
./forge hf plan-model gemma4_26b_a4b base --release-class report_only
./forge variants graph gemma4_26b_a4b
./forge artifacts validate reports/generated/.../artifacts/
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

Artifact validation:

```bash
./forge artifacts validate reports/generated/<run>/artifacts/ --strict
./forge artifacts validate reports/generated/<run>/artifacts/ --require-browser
```

See [docs/artifact-validation.md](docs/artifact-validation.md) before using
artifact quality as a promotion claim.

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
./forge data propose gemma4_26b_a4b local_ft_v1
./forge data generate gemma4_26b_a4b local_ft_v1 --smoke
./forge data verify gemma4_26b_a4b local_ft_v1 --smoke
./forge data review gemma4_26b_a4b local_ft_v1 --smoke --sample 50
./forge data pack gemma4_26b_a4b local_ft_v1 --smoke
./forge data publish gemma4_26b_a4b local_ft_v1 --smoke
```

Use `generate --overwrite` only when replacing candidates intentionally.
Downstream `--overwrite` refreshes derived artifacts from existing candidates.
`propose` turns saved eval failures into the next dataset skill targets and a
candidate config patch.
`publish` writes a dry-run HF plan by default; `publish --execute` refuses
seed-only and smoke-only datasets.

Promotion reports:

```bash
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
./forge promote gemma4_26b_a4b local_abli_sota_vs_downloaded_abli
```

Research registry:

```bash
./forge objectives list
./forge objectives audit
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

Variant graph:

```bash
./forge variants graph gemma4_26b_a4b
./forge variants node gemma4_26b_a4b local_ft --write
```

See [docs/variant-graph.md](docs/variant-graph.md). Variant nodes connect source
models, transforms, evidence, artifact checksums, validation state, promotion
decisions, and retention decisions.

Agent experiment plans:

```bash
./forge agent schema
./forge agent audit
./forge agent card recipes/agents/agent_experiment_template.yaml --write-card
./forge agent optimize-serving --family gemma4_26b_a4b --variant base
./forge agent optimize-quantization --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --variants base,local_ft
./forge agent optimize-behavior-edit --family gemma4_26b_a4b --config configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml --source-variant local_ft --target-variant ft_local_abli_sota_internal_r7_selected_t34_transfer --backend heretic
./forge agent init --experiment-id next_step --title "Next step" --family gemma4_26b_a4b --variant base --objective-profile capability_sft --output recipes/agents/next_step.yaml
```

See [docs/agent-experiments.md](docs/agent-experiments.md). Agent plans define
hypothesis, resource policy, evidence, rollback, and handoff before a run starts.

Cluster planning:

```bash
./forge cluster doctor --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster sync --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster health --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster runtime --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest
./forge cluster torchrun-smoke --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest --nccl-socket-ifname <distributed-network-interface>
./forge cluster plan \
  --config configs/clusters/dgx_spark_x2.example.yaml \
  --workload train \
  --launcher torchrun
```

See [docs/cluster.md](docs/cluster.md). Public configs use environment-backed
placeholders; private hostnames, IPs, usernames, tokens, and absolute paths do
not belong in Git. Use `cluster torchrun-smoke` as the gate before claiming a
training, quantization, or benchmark workload used both Spark nodes.

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
./forge quantize matrix-plan --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --variants base,local_ft
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

Variant and tokenizer checks:

```bash
./forge variants graph gemma4_26b_a4b --variant local_abli
./forge variants architecture-audit gemma4_26b_a4b --variant base
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli --load-tokenizer --strict
```

Use `architecture-audit` before reusing LoRA or ablation targets on a new
family. It checks target-discovery metadata and inspects local `config.json`
when present so MoE/router/expert behavior is not missed. Use
`tokenizer-audit` before promoting fine-tuned, ablated, merged-adapter, or
quantized variants. Metadata mode compares tokenizer files, special tokens, and
chat-template hashes; `--load-tokenizer` additionally runs a local
`AutoTokenizer` chat-template round trip when the checkpoint is present.

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
serving claims, and [docs/artifact-validation.md](docs/artifact-validation.md)
before publishing artifact-generation claims.

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
See [docs/adding-model-family.md](docs/adding-model-family.md) before adding a
new architecture or checkpoint family.
See [docs/huggingface-publishing.md](docs/huggingface-publishing.md) for Hub
release planning, model-card generation, and public/private release gates.

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
- Upload completed models, datasets, and durable eval artifacts to Hugging Face
  through dry-run release plans and explicit release-class gates.
- Never commit tokens or raw model weights.
