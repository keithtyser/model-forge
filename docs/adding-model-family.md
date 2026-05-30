# Adding A Model Family

Model Forge should generalize when a new open model drops. Add family support
through config and reusable commands first; avoid one-off scripts unless the
architecture genuinely needs a new backend.

## Required Files

Create or update:

- `configs/model_families/<family>.yaml`
- `configs/experiments/<family>_v0.yaml`
- `configs/experiments/<family>_artifacts_v0.yaml` if artifact quality matters
- `configs/finetuning/<family>_local_ft_v0.yaml` before training
- `configs/abliteration/<family>_local_abli.yaml` before ablation
- `configs/quantization/<family>_nvfp4_modelopt.yaml` before self-quantization
- `configs/promotion/<family>.yaml` before promotion claims

Use `qwen35_9b.yaml` and `qwen36_27b.yaml` as the current non-Gemma examples.

## Family Config Checklist

Every `configs/model_families/<family>.yaml` should define:

- `name` and `display_name`
- `models_dir_env` and `default_models_dir`
- `variants.base.repo_id`, `local_dir`, and `served_model_name`
- derived variants for expected workflow stages:
  - `local_ft`
  - `local_abli`
  - `local_ft_abli`
  - quantized variants when relevant
- `base_variant` on every derived variant
- `serve.script`, `default_gpu_memory_utilization`, and `default_max_model_len`
- `eval.config`, `eval.artifact_config`, output root, and suffix templates
- `comparison.output_dir`
- `external.output_root` and default tasks when external evals are used

`./forge doctor` validates required family fields, derived-variant source
edges, serve script paths, and eval config paths.

## Architecture Checklist

Record architecture-specific facts in the family config or recipe configs:

- architecture family, parameter scale, dense/MoE shape, and context length
- tokenizer family and chat-template defaults
- reasoning parser or tool-call parser needed by serving
- attention target-module patterns
- MLP target-module patterns
- MoE/router/expert exclusion policy
- embedding and LM-head edit policy
- tokenizer/chat-template preservation policy
- quantization feasibility notes, especially NVFP4/FP8/GGUF constraints
- license and publication constraints

Never assume Gemma module names, chat-template behavior, or MoE rules transfer
to Qwen, Llama, Mistral, Mixtral, Phi, or a new architecture.

## Smoke Commands

After adding the family config, run:

```bash
./forge families
./forge variants graph <family> --json
./forge variants tokenizer-audit <family> --variant base --json
./forge doctor --json
```

If the base checkpoint is present locally and the Python environment can load
`transformers`, run:

```bash
./forge variants tokenizer-audit <family> --variant base --load-tokenizer --strict
```

Before spending GPU time, verify the serving plan:

```bash
./forge serve <family> base
./forge eval <family> base --smoke
./forge bench serve --family <family> --variant base --dry-run
```

Run one large server or training job at a time unless a cluster config and
resource contract explicitly describe the distributed workload.

## Promotion Evidence

Do not mark a new family as validated because its YAML parses. Promotion needs
evidence appropriate to the target:

- internal eval results with response artifacts and provenance cards
- artifact execution cards for artifact-generation claims
- external eval output for benchmark claims
- serving cards for throughput/latency claims
- tokenizer audit for every derived or quantized candidate
- variant graph node or manifest linking source, transform, command, metrics,
  and retention decision
- Hub dry-run plan before public upload

If the first run fails, write the exact blocker and keep reusable config in the
repo. A precise blocker is better than an undocumented partial port.
