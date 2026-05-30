# Configs

Configuration is the main extension point for model-forge. Keep model-family
and experiment constants here instead of hard-coding them in scripts.

## Directories

- `model_families/`: source checkpoints, variants, served model names, report
  paths, and serving defaults.
- `hardware/`: generic hardware profile defaults such as DGX Spark serving,
  training, quantization, and resource-policy recommendations.
- `clusters/`: open-source-safe cluster inventory examples. Private hostnames,
  usernames, IPs, and absolute paths should stay in untracked local copies or
  environment variables.
- `serving/`: generic serving benchmark configs and reusable workload
  definitions. These should describe endpoint/workload shape, not private
  infrastructure.
- `sweeps/`: benchmark sweep matrices. Public sweep configs should describe
  cases, hypotheses, env deltas, resource policy, and follow-up gates, but not
  private hosts or absolute local paths.
- `data_sources/`: dataset source registries with ids, provenance, licenses,
  roles, quality tiers, and sampling caps.
- `research_registry.yaml`: dated research claims with implementation hooks,
  eval hooks, and limitations. Validate with `./forge research audit`.
- `experiments/`: eval matrix definitions and external benchmark settings.
- `finetuning/`: SFT/QLoRA recipes, resource contracts, LoRA targets, and
  promotion gates.
- `datasets/`: dataset factory configs for seed, generation, verification,
  review, pack, and publish planning.
- `release_classes/`: Hugging Face publication gates for report-only releases,
  adapter releases, private research checkpoints, public quantized checkpoints,
  and datasets.
- `objectives/`: reusable objective profiles with implementation status,
  validation state, required evidence, comparison metric preferences, and
  research basis. Validate with `./forge objectives audit`.
- `promotion/`: saved comparison gates for deciding whether a candidate should
  be promoted, held, or investigated.
- `abliteration/`: refusal-direction and behavior-edit recipes.
- `artifact_validation/`: optional fixture configs for standalone HTML/Python
  artifact execution validation.

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
