# model-forge

model-forge is a post-training workbench for open models. The goal is to take a
base checkpoint, improve it with fine-tuning, optionally remove unwanted
refusals with behavior editing, quantize it for fast serving, then evaluate and
publish enough evidence that the result is reproducible.

The repo is model-family driven. Gemma 4 26B-A4B is the first fully worked
family. Qwen and Llama configs are present as generalization targets so new
architectures can follow the same workflow without becoming one-off scripts.

## Start Here

If you found this repo and have a model you want to post-train, use this path:

1. Add the model family in `configs/model_families/<family>.yaml`.
2. Run architecture and tokenizer audits.
3. Serve the base model and run smoke/internal evals.
4. Create a fine-tune plan and dataset plan.
5. Run a bounded fine-tune.
6. Evaluate the fine-tuned model against the base model.
7. Run ablation/abliteration only after the source model has a saved baseline.
8. Quantize only after source and edited variants have comparison evidence.
9. Publish models/datasets through Hugging Face dry-run plans before upload.

Useful docs:

- [Documentation Index](docs/README.md)
- [Add A Model Family](docs/adding-model-family.md)
- [Fine-Tuning](docs/finetuning.md)
- [Abliteration](docs/abliteration.md)
- [Quantization](docs/quantization.md)
- [Evaluation Strategy](docs/evaluation-strategy.md)
- [DGX Spark Rules](docs/dgx-spark.md)
- [Agent Instructions](AGENTS.md)

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/keithtyser/model-forge.git
cd model-forge
./forge setup all
./forge doctor
```

List configured families:

```bash
./forge families
./forge variants graph gemma4_26b_a4b
```

## Use Your Own Model

Copy an existing family config and change the model identity:

```bash
cp configs/model_families/llama31_8b.yaml configs/model_families/my_model.yaml
```

Edit:

- `name`
- `display_name`
- `architecture.family`
- `architecture.context_length`
- `architecture.target_discovery`
- `variants.base.repo_id`
- `variants.base.local_dir`
- `variants.base.served_model_name`
- `serve.script`
- `eval.config`
- `comparison.output_dir`

Then audit before training or editing:

```bash
./forge variants graph my_model
./forge variants architecture-audit my_model --variant base
./forge variants tokenizer-audit my_model --variant base
./forge generalization audit
./forge doctor
```

For a new architecture, do not reuse Gemma layer ranges, LoRA target modules, or
ablation strengths blindly. Inspect the model config and tokenizer first, then
record those choices in the family config and recipe.

## Core Workflow

Serve one model at a time:

```bash
./forge download my_model base
./forge serve my_model base
```

Run evals from another terminal:

```bash
./forge eval my_model base --smoke
./forge eval my_model base --internal
./forge eval my_model base --artifact
./forge compare my_model
```

Plan fine-tuning:

```bash
./forge finetune my_model plan
./forge data plan my_model local_ft_v1
./forge data gaps my_model local_ft_v1
./forge data generate my_model local_ft_v1 --smoke
./forge data verify my_model local_ft_v1 --smoke
./forge data pack my_model local_ft_v1 --smoke
```

Prepare a guarded training run:

```bash
./forge finetune my_model prepare --overwrite
```

On DGX Spark, run heavy jobs through the guarded launchers described in
[docs/dgx-spark.md](docs/dgx-spark.md). Do not run raw training scripts without
CPU, RAM, disk, checkpoint, and watchdog limits.

Plan ablation/abliteration:

```bash
./forge ablate my_model plan
./forge ablate my_model sota-prepare --backend heretic
```

Plan quantization:

```bash
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --write-plan
./forge quantize matrix-plan --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml
```

The NVFP4 path is a priority for Blackwell/DGX Spark, but promotion still
requires source-vs-quantized serving metrics, behavior preservation, tokenizer
preservation, and a quantization card.

## DGX Spark Safety

Heavy jobs must be tenants on the machine, not owners of it.

- Keep one model server or training job active at a time unless a cluster plan
  explicitly says otherwise.
- Reserve CPU and RAM headroom for SSH, the kernel, and control processes.
- Check disk before writing checkpoints.
- Prefer slower runs over a dead machine.
- Use `./forge cluster torchrun-smoke` before claiming a workload used both
  Spark nodes.

See [docs/cluster.md](docs/cluster.md) for two-node setup and
[docs/spark-optimizations.md](docs/spark-optimizations.md) for serving/training
optimization notes.

## Repository Layout

```text
configs/        Model families, objectives, training, ablation, quantization
datasets/       Dataset manifests, seeds, generated packs, publish bundles
docs/           Workflow docs, status, roadmap, and experiment ledger
evals/          Internal eval prompt buckets and rubrics
recipes/        Reusable run templates and agent experiment recipes
reports/        Generated report conventions; large generated outputs ignored
results/        Eval output conventions; generated result dirs ignored
scripts/        Operational launchers and compatibility wrappers
src/            model_forge Python package
tests/          Unit and smoke tests
forge           User-facing CLI wrapper
```

Generated model weights, tokenized caches, raw eval outputs, large reports, and
logs are not committed. Upload durable model and dataset artifacts to Hugging
Face through the release gates in [docs/huggingface-publishing.md](docs/huggingface-publishing.md).

## Design Rules

- Compare every edited model against the source checkpoint it came from.
- Treat ablation success as lower refusal plus retained capability.
- Report unsafe overcompliance separately from capability.
- Treat fine-tune success as better capability and format quality without
  regressions.
- Keep model-family support in config, not hard-coded scripts.
- Push code, configs, docs, recipes, and lightweight manifests to GitHub.
- Never commit tokens, raw model weights, or private infrastructure details.
