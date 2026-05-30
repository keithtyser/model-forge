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
- `docs/status.md`: current short handoff state and recommended next work
- `docs/roadmap-status-audit.md`: current MF backlog implementation status and
  validation state
- `docs/artifact-retention.md`: what belongs in Git, local scratch, or Hugging
  Face
- `docs/finetuning.md`: SFT/QLoRA workflow and promotion gates
- `docs/abliteration.md`: refusal-ablation methodology and promotion criteria
- `docs/evaluation-strategy.md`: eval design and interpretation
- `docs/artifact-validation.md`: standalone HTML/Python artifact execution
  validators and Artifact Execution Card rules
- `docs/experiment-ledger.md`: handoff ledger for hypotheses, experiments,
  artifacts, validation, and publish status
- `docs/run-manifests.md`: canonical run manifest schema and handoff rules
- `docs/variant-graph.md`: variant node schema, graph inspection, evidence,
  checksums, promotion, and retention rules
- `docs/cluster.md`: generic cluster inventory, doctor, and dry-run planning
  rules
- `docs/serving-benchmarks.md`: serving benchmark command, outputs, and
  interpretation rules
- `docs/quantization.md`: Blackwell NVFP4, runtime import, checkpoint creation,
  and quantization-card evidence rules
- `docs/roadmaps/`: long-form roadmap and archived planning documents
- `docs/research/sota-2026-05-18.md`: dated SOTA snapshot behind current
  roadmap decisions
- `docs/spark-optimizations.md`: DGX Spark hardware profile, AEON-7-derived
  serving/quantization lessons, and safe overrides
- `configs/README.md`: config directory map and portability rules
- `configs/research_registry.yaml`: machine-readable research basis for
  methods, evals, serving, and quantization work
- `configs/hardware/` and `configs/clusters/`: hardware defaults and
  open-source-safe cluster inventory examples
- `configs/serving/`: serving benchmark configs and reusable workload
  definitions
- `configs/quantization/`: FP8/NVFP4 runtime and checkpoint-creation configs
- `configs/sweeps/`: bounded serving/quantization/benchmark sweep matrices
- `scripts/README.md`: script directory map and operational rules
- `configs/model_families/`: model family registry
- `configs/abliteration/`: ablation recipes
- `recipes/`: tracked reusable run templates and known-good generated recipes
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
10. Run `./forge roadmap cli-drift` after editing roadmap CLI examples so
    target commands cannot be confused with implemented commands.
11. Update `docs/status.md` and `docs/experiment-ledger.md` before handing off
   or starting a long run.

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

Audit roadmap status and command drift:

```bash
./forge roadmap audit --write-doc
./forge roadmap cli-drift
```

Plan fine-tuning without loading a model:

```bash
./forge finetune gemma4_26b_a4b plan
./forge finetune gemma4_26b_a4b prepare
```

Run fine-tuning on DGX Spark only through the guarded CUDA container launcher
when host Python is CPU-only:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
scripts/run_finetune_spark_container.sh
```

For Gemma 4 26B on Spark, the validated FT path currently uses
`trainer.backend=unsloth` with `unsloth_compile_disable=true`,
`max_seq_length=2048`, and `max_steps=500` for the first full attempt. The HF
Causal LM loader hit host-memory guard failures before the first training step;
Unsloth's loader passed 1024-token and 2048-token one-step QLoRA smokes. Keep
the backend choice in YAML so other model families can use HF, Unsloth, or
another backend without hard-coding Gemma behavior.

## Fine-Tuning Rules

- Treat Jackrong's public notebooks as a useful baseline pattern, not a final
  recipe to copy.
- Keep the fine-tune recipe model-family agnostic: source model, LoRA targets,
  context length, data blend, and output variant belong in YAML.
- Fine-tuned PEFT outputs are adapter variants. Mark them with `adapter: true`,
  `base_variant: <base>`, and `lora_rank` in
  `configs/model_families/<family>.yaml`; use `./forge serve <family> local_ft`
  so vLLM loads base weights plus the LoRA adapter instead of trying to serve
  the adapter directory as a full model. If live LoRA serving is unsupported for
  the architecture, set `serve_strategy: merged`, merge with
  `scripts/merge_peft_adapter.py`, and serve the merged checkpoint.
- Use data manifests with explicit source roles, sample targets, schema fields,
  licenses, quality gates, and holdouts.
- Use source registries in `configs/data_sources/` for reusable dataset ids,
  provenance, licenses, quality tiers, and sampling caps. Training manifests
  should reference registry ids and override per-run sample targets.
- Do not train on model-forge eval prompts. Train adjacent skills and let the
  held-out eval suite decide promotion.
- Treat `runs/finetune/<name>/` as local generated scratch. Tracked reusable
  templates belong under `recipes/finetuning/<name>/`.
- For FT data iteration, use
  `./forge data plan|gaps|propose|generate|verify|review|pack <family> <variant> --smoke`
  before editing training configs. The local FT v1 pack defaults to the
  deterministic `template` generation provider; OpenAI-compatible generation is
  available only when configured explicitly. Treat `review_report.json`
  `ready_to_scale_generation=true` as the gate before scaling generation.
  `propose` reads saved eval failures and writes `feedback_proposal.yaml` with
  ranked skill targets, generation scale, and candidate config-patch guidance.
  Run `generate --overwrite` only when replacing candidates intentionally;
  downstream `--overwrite` refreshes derived artifacts from existing
  candidates. Current local FT v1 configs reject assistant length violations
  before packaging.
  `publish` remains a dry-run plan unless `--execute` is explicitly passed, and
  execution refuses seed-only or smoke-only datasets.
- Use `./forge promote <family> <profile>` after `./forge compare <family>` to
  write a promotion report from saved eval results.
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

Run repo hygiene checks:

```bash
./forge doctor
```

Inspect or validate the research basis:

```bash
./forge objectives list
./forge objectives audit
./forge roadmap audit --write-doc
./forge research list
./forge research show arditi_2024_refusal_direction
./forge research audit
```

When adding a new method, benchmark adapter, objective profile, or public report,
connect it to `configs/research_registry.yaml`, include implementation and
validation state, and keep the limitations explicit.

Write or inspect a run manifest:

```bash
./forge manifest write \
  --run-type eval \
  --status planned \
  --family gemma4_26b_a4b \
  --variant base \
  --config configs/experiments/gemma4_26b_a4b_v0.yaml \
  --command './forge eval gemma4_26b_a4b base --internal'
```

Use manifests for planned, running, completed, and failed work. They preserve
git state, config hashes, command lines, hardware, safe environment variables,
outputs, artifacts, metrics, and notes. Never pass secrets through manifest
metadata or notes.

Validate generated artifacts before making artifact-quality claims:

```bash
./forge artifacts validate reports/generated/<run>/artifacts/ --strict
./forge artifacts validate reports/generated/<run>/artifacts/ --require-browser
```

The command writes `artifact_validations.json`, `artifact_execution_card.json`,
and `artifact_execution_card.md`. Browser-skipped validation is acceptable only
for smoke checks; promotion paths should use `--require-browser`.

Inspect or write variant graph nodes:

```bash
./forge variants graph gemma4_26b_a4b
./forge variants node gemma4_26b_a4b local_ft --write
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli
```

Variant nodes record the source variant, transform, checkpoint reference,
validation state, evidence path, artifact checksums, promotion decision, and
retention decision. Keep generated nodes in `reports/generated/` unless a small
example is intentionally promoted. Run `tokenizer-audit --load-tokenizer
--strict` before release gates so adapter merges, ablation exports,
quantization exports, and future GGUF conversions cannot silently lose
chat-template or special-token behavior.

Validate or plan cluster usage:

```bash
./forge cluster doctor --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster sync --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster health --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster runtime --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest
./forge cluster torchrun-smoke --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest --nccl-socket-ifname <distributed-network-interface>
./forge cluster plan \
  --config configs/clusters/dgx_spark_x2.example.yaml \
  --workload train \
  --launcher torchrun
```

Cluster configs must remain generic. Do not commit private hostnames, IPs,
usernames, tokens, or absolute machine-specific paths. Put those values in
environment variables or untracked local copies. Real distributed execution
requires `./forge cluster doctor --strict` and a workload-specific launcher with
resource guardrails. Before claiming that a training, quantization, or benchmark
job used both Spark nodes, run `./forge cluster torchrun-smoke` and cite the
generated evidence path.

Benchmark serving only after an endpoint is already running:

```bash
./forge bench serve --family gemma4_26b_a4b --variant base --dry-run
./forge bench serve --family gemma4_26b_a4b --variant base
./forge bench sweep doctor --config configs/sweeps/dgx_spark_vllm_baseline.yaml --strict
./forge bench sweep plan --family gemma4_26b_a4b --variant base
```

`bench serve` is for OpenAI-compatible endpoint mechanics only. `bench sweep`
expands startup-time server env cases plus matching benchmark commands. Neither
command starts a vLLM server. A good latency result is not a quality or behavior
pass. Use the generated `manifest.json`, `summary.json`, `requests.jsonl`, and
`serving_card.md` with eval results before making serving claims.

Plan and report quantization without loading a model:

```bash
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --write-plan
./forge quantize card \
  --config configs/quantization/nvfp4_blackwell_runtime.yaml \
  --source-serving-summary <source>/summary.json \
  --candidate-serving-summary <candidate>/summary.json \
  --source-serving-eval <source_eval_dir> \
  --candidate-serving-eval <candidate_eval_dir> \
  --run-id source_vs_nvfp4 \
  --write-card
```

NVFP4 is the priority Blackwell path. `nvfp4_runtime` means Model Forge is
validating an already-quantized checkpoint; do not imply the repo created those
weights. A real quantization claim needs a candidate endpoint, serving summary,
sampled behavior scores, and quantization card.
For NVFP4 promotion, require a clear `output_tokens_per_second` improvement
over the matching BF16/FP16 endpoint, especially on `decode_heavy`; otherwise
treat the run as loader evidence only.

For self-quantization, use the ModelOpt export runner and the matrix config:

```bash
./forge quantize matrix-plan \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --variants base,local_ft
```

Set `MODEL_FORGE_QUANT_WORKERS=local,<ssh-host>` to distribute independent
variant exports across a Spark cluster. Do not commit those worker names or IPs.
Run at most one export per Spark node, and do not launch export commands outside
`./forge quantize export`; the runner has a runtime memory watchdog and Docker
cleanup path. The generated command defaults to `systemd-run --user --scope`,
`nice`, Docker CPU/memory limits, and a checkout-local export lock. If the
configured systemd mode fails or asks for interactive authorization, stop and
fix the host execution path; do not rerun the same heavy command without
equivalent limits. Use `--target-variant` on single exports so metadata matches
the actual matrix candidate.

For Gemma 4 A4B NVFP4, use the full-MoE ModelOpt path in
`scripts/quantization/gemma4_moe_nvfp4.py` through
`configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml`. The earlier
MLP-only recipe loaded after metadata fixes but left the fused experts in BF16
and only reached about 25 output tok/s in the repo benchmark. Full-MoE Gemma
NVFP4 should serve with `VLLM_NVFP4_GEMM_BACKEND=marlin`,
`--moe-backend marlin`, `--quantization modelopt`, FP8 KV cache, and
`--language-model-only`. The published full-MoE reference checkpoint reached
about 50 output tok/s on the repo core serving benchmark on 2026-05-30, so use
that as the target when validating self-quantized Gemma variants.
Do not enable ModelOpt `--low_memory_mode` for this recipe without retesting;
it failed with a meta-tensor dispatch error on the earlier stock path. The
normal-mode export uses a full-RAM Spark profile with CPU limits, Docker memory
limits, disk preflight, and a 5% available-RAM watchdog floor.

## Abliteration Rules

- The reusable recipe is the structure, not fixed constants.
- Always compute fresh refusal directions on the source checkpoint being
  ablated.
- Direct parameter transfer is only a warm start for nearby checkpoints in the
  same architecture family.
- For new architectures, inspect target module names, layer counts, hidden
  sizes, MoE/expert layouts, and tokenizer/chat templates before editing.
- Before promoting a derived variant, run `./forge variants tokenizer-audit` to
  verify tokenizer and chat-template preservation against the configured source
  variant.
- Recalibrate layer ranges, strengths, direction scope, and search bounds per
  family.
- Keep embeddings, LM heads, routers, and expert weights untouched unless the
  recipe explicitly justifies editing them.
- Unsafe overcompliance is reported separately. For refusal-removal research,
  lower refusal on unsafe prompts is expected, but capability preservation must
  be measured independently.

## Hardware Discipline

- Full fine-tuning must run through generated `runs/finetune/<name>/run.sh`, not
  an ad hoc `python train.py` command. Reusable examples live under
  `recipes/finetuning/<name>/`, but the active job runs from `runs/`. The
  launcher wraps data prep and training in `systemd-run --scope` when available.
- If unprivileged `systemd-run --scope` needs interactive auth, use
  `scripts/run_finetune_spark_container.sh`; it runs the generated `run.sh`
  inside `nemotron-runner:latest` with Docker CPU/memory limits and CUDA access.
- Default hard limits are `CPUQuota=80%`, `MemoryMax=85%`, `IOWeight=100`, and
  `nice -n 10`. Do not raise them casually on shared or remote machines.
- Always leave at least one CPU core free. The fine-tuning runner sets thread
  pools to `max(1, os.cpu_count() - reserve_cores)`.
- Start only if the recipe-specific RAM floor and 15% run-directory disk are free.
- Stop the job if runtime available RAM falls below the recipe-specific floor.
  Treat a resource guard trip as a real failure to investigate, not as a
  warning to ignore.
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
- For Blackwell NVFP4 serving, start with conservative Spark settings and record
  the actual backend. Full-MoE Gemma4 uses Marlin; MLP-only or dense-only
  fallback tests may use Cutlass.
- For MoE quantization, keep multimodal projection/vision modules in BF16
  unless a family-specific recipe and eval pass justify otherwise. Expert
  tensors require family-specific handling; do not assume stock exporters
  quantize fused expert layouts correctly.
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

Gemma 4 A4B local FT v0, runtime smoke passed but full training/evals still in
progress:

```text
configs/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
```

These are examples of the general workflow. Do not hard-code future model
support around Gemma-specific layer names or constants.

## Publishing

When publishing a model:

- start with `./forge hf status --offline` and
  `./forge hf plan-model <family> <variant> --release-class <class>`
- use release classes from `configs/release_classes/` to decide whether a
  release is report-only, adapter-only, private research, public dataset, or a
  public quantized checkpoint
- never bypass failed `release_gates` in `hub_publish.json`; public full
  checkpoints require explicit allowance plus Spark validation evidence
- for dataset releases, prefer `./forge data publish ... --source-license-checked`
  and inspect `hf_publish_bundle/`; public dataset plans must not include raw
  accepted/rejected rows or unredacted message text unless the release class
  explicitly allows it
- for eval evidence, inspect `eval_provenance_card.json` before making claims;
  raw `responses.jsonl` and `examples.md` need redaction before public release
- include a model card linking back to this repo
- include source model, recipe config, eval scores, and intended-use caveats
- upload completed models, prepared datasets, and needed eval artifacts to
  Hugging Face when the owner provides `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`
- keep refusal-ablated models private unless the owner explicitly approves
  public release
- avoid committing raw model weights into this Git repo
- never write Hugging Face tokens into tracked files or shell scripts
- follow `docs/artifact-retention.md` when deciding whether an artifact belongs
  in Git, local scratch, or Hugging Face

Repository link for model cards:

```text
https://github.com/keithtyser/model-forge
```
