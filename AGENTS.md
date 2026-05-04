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
- `docs/finetuning.md`: SFT/QLoRA workflow and promotion gates
- `docs/abliteration.md`: refusal-ablation methodology and promotion criteria
- `docs/evaluation-strategy.md`: eval design and interpretation
- `docs/experiment-ledger.md`: handoff ledger for hypotheses, experiments,
  artifacts, validation, and publish status
- `docs/spark-optimizations.md`: DGX Spark hardware profile, AEON-7-derived
  serving/quantization lessons, and safe overrides
- `configs/model_families/`: model family registry
- `configs/abliteration/`: ablation recipes
- `evals/prompts/`: internal prompt buckets and rubrics
- `src/model_forge/`: Python package source
- `forge`: user-facing CLI wrapper

## Standard Workflow

1. Add or update a model family config in `configs/model_families/`.
2. Add or update fine-tune configs in `configs/finetuning/` and data manifests
   in `datasets/finetuning/` when training a new source model.
3. Add or update an ablation config in `configs/abliteration/`.
4. Run dry-run planning before loading large models.
5. Serve exactly one large model at a time.
6. Run internal evals before expensive artifact or external evals.
7. Compare against the source checkpoint being modified, not only against an
   unrelated downloaded abli model.
8. Promote only when refusal suppression improves and source-model capability is
   preserved within expected eval variance.
9. Save raw responses, scores, model cards, and exact recipe/config paths.
10. Update `docs/experiment-ledger.md` before handing off or starting a long
   run.

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

Plan fine-tuning without loading a model:

```bash
./forge finetune gemma4_26b_a4b plan
./forge finetune gemma4_26b_a4b prepare
```

## Fine-Tuning Rules

- Treat Jackrong's public notebooks as a useful baseline pattern, not a final
  recipe to copy.
- Keep the fine-tune recipe model-family agnostic: source model, LoRA targets,
  context length, data blend, and output variant belong in YAML.
- Use data manifests with explicit source roles, sample targets, schema fields,
  licenses, quality gates, and holdouts.
- Do not train on model-forge eval prompts. Train adjacent skills and let the
  held-out eval suite decide promotion.
- Promote a local FT only if it matches or beats the downloaded FT reference on
  internal challenge capability, paired benign quality, normal-use regression,
  artifact quality, and external benchmarks.

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

- Full fine-tuning must run through generated `runs/finetune/<name>/run.sh`, not
  an ad hoc `python train.py` command. The launcher wraps data prep and training
  in `systemd-run --scope` when available.
- Default hard limits are `CPUQuota=80%`, `MemoryMax=85%`, `IOWeight=100`, and
  `nice -n 10`. Do not raise them casually on shared or remote machines.
- Always leave at least one CPU core free. The fine-tuning runner sets thread
  pools to `max(1, os.cpu_count() - reserve_cores)`.
- Start only if at least 15% RAM and 15% run-directory disk are free.
- Stop the job if runtime available RAM falls below 10%. Treat a resource guard
  trip as a real failure to investigate, not as a warning to ignore.
- Cap dataloaders. `num_workers` must stay below `usable_cores - 2`; keep
  `persistent_workers` off unless memory headroom is known to be safe.
- Keep checkpoint rotation enabled with a small `save_total_limit`.
- Prefer slower over an unreachable machine.
- Assume large checkpoints can exhaust memory.
- Keep one large model process or vLLM server active at a time.
- On DGX Spark, prefer conservative settings first: `GPU_MEMORY_UTILIZATION=0.85`,
  FP8 KV cache, prefix caching, chunked prefill, low `VLLM_MAX_NUM_SEQS`, and
  batch size 1 for activation/residual collection.
- Use Spark/GB10-native vLLM builds. Stock vLLM wheels may not be compiled for
  SM 12.1.
- Treat AEON-7 NVFP4 settings as hardware guidance, not Gemma constants. Put
  parser names, quantization format, loader patches, and drafter paths in family
  config or environment overrides.
- For MoE quantization, keep routers and multimodal projection/vision modules
  in BF16 unless a family-specific recipe and eval pass justify otherwise.
- `MODEL_FORGE_PARALLELISM=192` is for preprocessing/input-pipeline work, not
  for multiplying large model forward passes.
- Optional watchdog, started outside the training job:

```bash
nohup .venv/bin/python scripts/model_forge_watchdog.py \
  --pattern 'train_trl_sft.py|model_forge.pipelines.finetune' \
  > logs/model_forge_watchdog.log 2>&1 &
```

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
- upload completed models, prepared datasets, and needed eval artifacts to
  Hugging Face when the owner provides `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`
- keep refusal-ablated models private unless the owner explicitly approves
  public release
- avoid committing raw model weights into this Git repo
- never write Hugging Face tokens into tracked files or shell scripts

Repository link for model cards:

```text
https://github.com/keithtyser/model-forge
```
