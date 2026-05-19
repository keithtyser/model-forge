# Recipes

`recipes/` stores tracked, reusable run templates and representative generated
artifacts that are useful for review, handoff, or reproducing a known workflow.

Use this directory for small files such as:

- generated `plan.json` snapshots
- generated launcher scripts
- generated trainer scripts
- eval-after-run scripts
- notes that explain when a recipe is valid

Do not store bulky runtime artifacts here. Training JSONL files, tokenized
caches, checkpoints, Python overlays, raw eval output directories, and generated
reports belong in ignored runtime locations such as `runs/`, `results/`,
`reports/generated/`, or external storage.

The normal workflow is:

```bash
./forge finetune <family> prepare --overwrite
```

That writes materialized runtime files under `runs/finetune/<name>/`. If the
generated files become a reusable reference, move only the small recipe files
into `recipes/finetuning/<name>/` and leave local scratch outputs in `runs/`.
