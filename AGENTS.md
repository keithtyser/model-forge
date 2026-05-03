# AGENTS.md

This repo is intended to be friendly to AI coding agents. The goal is a
general post-training pipeline for open models: download or register a model,
fine-tune it, ablate refusals, evaluate every candidate, compare against source
models, and publish reproducible artifacts.

## Core Goal

Do not treat this as a Gemma-only repo. Gemma 4 is the first validated worked
example. New model families such as Qwen, Llama, Mistral, Mixtral, Phi, or
future open releases should fit the same family-driven workflow.

When adding support for a new model, prefer model-family configuration and
reusable pipeline code over one-off scripts.

## First Things To Inspect

- `README.md`: project overview and model-agnostic workflow
- `docs/abliteration.md`: refusal-ablation methodology and promotion criteria
- `docs/evaluation-strategy.md`: eval design and interpretation
- `configs/model_families/`: model family registry
- `configs/abliteration/`: ablation recipes
- `evals/prompts/`: internal prompt buckets and rubrics
- `src/model_forge/`: Python package source
- `forge`: user-facing CLI wrapper

## Standard Workflow

1. Add or update a model family config in `configs/model_families/`.
2. Add or update an ablation config in `configs/abliteration/`.
3. Run dry-run planning before loading large models.
4. Serve exactly one large model at a time.
5. Run internal evals before expensive artifact or external evals.
6. Compare against the source checkpoint being modified, not only against an
   unrelated downloaded abli model.
7. Promote only when refusal suppression improves and source-model capability is
   preserved within expected eval variance.
8. Save raw responses, scores, model cards, and exact recipe/config paths.

## Useful Commands

Install/setup:

```bash
./forge setup all
```

List families:

```bash
./forge families
```

Serve and eval a configured variant:

```bash
./forge serve gemma4_26b_a4b base
./forge eval gemma4_26b_a4b base --internal
./forge compare gemma4_26b_a4b
```

Plan ablation without loading a model:

```bash
./forge ablate gemma4_26b_a4b plan
```

Prepare Heretic SOTA artifacts:

```bash
./forge ablate gemma4_26b_a4b sota-prepare --backend heretic
./forge ablate --config configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml sota-prepare --backend heretic
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Abliteration Rules

- The reusable recipe is the structure, not fixed constants.
- Always compute fresh refusal directions on the source checkpoint being
  ablated.
- Direct parameter transfer is only a warm start for nearby checkpoints in the
  same architecture family.
- For new architectures, inspect target module names, layer counts, hidden
  sizes, MoE/expert layouts, and tokenizer/chat templates before editing.
- Recalibrate layer ranges, strengths, direction scope, and search bounds per
  family.
- Keep embeddings, LM heads, routers, and expert weights untouched unless the
  recipe explicitly justifies editing them.
- Unsafe overcompliance is reported separately. For refusal-removal research,
  lower refusal on unsafe prompts is expected, but capability preservation must
  be measured independently.

## Hardware Discipline

- Assume large checkpoints can exhaust memory.
- Keep one large model process or vLLM server active at a time.
- On DGX Spark, prefer conservative settings first: `GPU_MEMORY_UTILIZATION=0.85`,
  prefix caching, chunked prefill, and batch size 1 for activation/residual
  collection.
- `MODEL_FORGE_PARALLELISM=192` is for preprocessing/input-pipeline work, not
  for multiplying large model forward passes.
- Stop `vllm_node` when finished:

```bash
docker stop vllm_node
```

## Current Validated Recipes

Base Gemma 4 A4B local abli:

```text
configs/abliteration/gemma4_26b_a4b_local_abli.yaml
```

FT Gemopus local abli using selected t34 transfer:

```text
configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml
```

These are examples of the general workflow. Do not hard-code future model
support around Gemma-specific layer names or constants.

## Publishing

When publishing a model:

- include a model card linking back to this repo
- include source model, recipe config, eval scores, and intended-use caveats
- keep refusal-ablated models private unless the owner explicitly approves
  public release
- avoid committing raw model weights into this Git repo

Repository link for model cards:

```text
https://github.com/keithtyser/model-forge
```
