# model-forge

Reusable post-training pipelines for open models. Fine-tuning, abliteration, and evaluation without hand-wavy bullshit.

## v0 opinionated scope

The first milestone is not training. It is evaluation.

Why:
- bad eval makes every later result fake
- once eval works on the base model, every later family plugs into the same harness
- it forces discipline before checkpoint collecting becomes a hobby

## Current default v0 stack

- Model family: Qwen 3.5
- Starter checkpoint: `Qwen/Qwen3.5-9B`
- Fine-tuning: Unsloth QLoRA
- Abliteration: OBLITERATUS
- Eval style: Kyle-inspired prompt buckets with model-forge metrics

Why Qwen 3.5 first:
- much more current than Qwen 2.5
- practical size for repeated local iteration
- cleaner v0 choice than waiting on smaller Qwen 3.6 checkpoints

Gemma 4 and Qwen 3.6 are first-class next targets. The repo structure is built so new model families are config changes, not rewrites.

## First milestone

Produce a base-model eval bundle for one checkpoint:
- scores table
- examples file
- run manifest

No training until that works end to end.

## Repo layout

```
configs/         experiment and backend config
datasets/        training and synthetic data manifests
evals/           prompt sets, rubrics, and eval harness
pipelines/       finetune and abliteration pipeline code
scripts/         shell entrypoints
reports/         report templates and generated outputs
results/         machine-readable run outputs
models/          local checkpoint conventions
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m model_forge.evals.run_eval --config configs/experiments/qwen35_9b_v0.yaml --dry-run
```

## Initial experiment matrix

- base
- ft
- abli
- ft_then_abli

The harness is eval-first. Pipelines plug into the same output contract.
