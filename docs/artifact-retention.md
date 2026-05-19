# Artifact Retention Policy

This repo should be enough for another agent to reproduce the workflow without
turning Git into a model/checkpoint store.

## Commit To Git

Commit lightweight, durable project state:

- source code, tests, and CLI wrappers
- model family configs, fine-tuning configs, ablation configs, objectives, and
  dataset manifests
- seed datasets and small smoke/generated packs when they are useful for tests
  or handoff
- recipes under `recipes/` that capture reusable generated run templates
- docs, status notes, experiment ledger entries, model-card templates, and
  publish plans
- compact result summaries, golden baselines, and manifests that point to full
  external artifacts

## Do Not Commit

Keep large or local-machine-specific outputs out of Git:

- model checkpoints, merged models, quantized models, and LoRA adapter weights
- full `runs/` directories, tokenized dataset caches, Python overlays, and
  checkpoint rotations
- raw high-volume eval outputs under `results/`
- generated HTML reports and large report bundles under `reports/generated/`
- logs, container homes, package caches, pycache, and local `.env` files
- Hugging Face tokens or any other credentials

These locations are intentionally ignored by `.gitignore`.

## Upload Externally

Use Hugging Face or another durable artifact store for completed artifacts that
must survive the local machine:

- promoted model checkpoints or adapters
- completed prepared datasets used for training
- important eval/report bundles that are too large for Git
- model cards and dataset cards that link back to
  `https://github.com/keithtyser/model-forge`

Record the repo id, artifact path, source config, validation, and publish status
in `docs/experiment-ledger.md`. Smoke-test artifacts should not be uploaded as
final models or datasets.

## Local Cleanup

Do not delete local ignored artifacts just because they are ignored. They may be
expensive to regenerate. Before cleanup:

```bash
git status --ignored -sb
du -h -d 2 runs artifacts results reports/generated models 2>/dev/null
```

Delete only after confirming the artifact is disposable or already published.
Prefer removing old checkpoints through configured checkpoint rotation instead
of manual deletion during active experiments.

## Recipes Versus Runs

`runs/` is runtime scratch. It is where `./forge finetune ... prepare` writes
materialized run scripts, plans, training JSONL files, tokenized caches, and
local overlays.

`recipes/` is tracked. It stores reusable templates, representative generated
plans, and known-good run recipes that are useful for review or handoff.

If a generated run becomes important, move or copy only the small reusable files
into `recipes/` and keep the bulky runtime outputs in `runs/` or on Hugging
Face.
