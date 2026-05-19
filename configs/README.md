# Configs

Configuration is the main extension point for model-forge. Keep model-family
and experiment constants here instead of hard-coding them in scripts.

## Directories

- `model_families/`: source checkpoints, variants, served model names, report
  paths, and serving defaults.
- `experiments/`: eval matrix definitions and external benchmark settings.
- `finetuning/`: SFT/QLoRA recipes, resource contracts, LoRA targets, and
  promotion gates.
- `datasets/`: dataset factory configs for seed, generation, verification,
  review, pack, and publish planning.
- `objectives/`: reusable training objectives and quality criteria.
- `abliteration/`: refusal-direction and behavior-edit recipes.

## Rules

- Prefer model-family config over one-off script edits.
- Use `~/models/...`, repo-relative paths, or explicit environment overrides
  instead of user-specific absolute paths.
- Put architecture-specific constants in the family or recipe config: target
  modules, layer ranges, tokenizer behavior, context length, quantization, and
  serving settings.
- Read secrets from environment variables only. Do not write tokens into YAML.
- Keep generated artifacts out of `configs/`; store small reusable snapshots in
  `recipes/` and large outputs in ignored runtime directories or Hugging Face.
