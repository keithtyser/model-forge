# model-forge

Reusable post-training pipelines for open models.

model-forge provides a structured framework for:
- fine-tuning
- abliteration
- evaluation
- experiment tracking

The repository is designed to make post-training workflows reproducible across model families, with a shared configuration format and a consistent evaluation contract.

## Current scope

The initial repository scaffold focuses on evaluation-first development.

Default starter configuration:
- Model family: Qwen 3.5
- Base model: `Qwen/Qwen3.5-9B`
- Fine-tuning path: Unsloth QLoRA
- Abliteration path: OBLITERATUS

Additional starter configs are included for:
- `Qwen/Qwen3.6-27B`
- `google/gemma-4-E4B-it`

## Repository layout

```text
configs/         Experiment and backend configuration
datasets/        Dataset manifests and metadata
evals/           Prompt sets and scoring rubrics
pipelines/       Pipeline notes and entrypoint plans
scripts/         Shell entrypoints
reports/         Report templates and generated outputs
results/         Machine-readable run outputs
models/          Local checkpoint directory conventions
src/             Python package source
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m model_forge.evals.run_eval --config configs/experiments/qwen35_9b_v0.yaml --dry-run
```

## Initial experiment matrix

- `base`
- `ft`
- `abli`
- `ft_then_abli`

## Status

This repository is currently a scaffold for the first evaluation harness and pipeline interfaces. Training and abliteration implementations will be added behind the same experiment configuration structure.
