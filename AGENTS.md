# AGENTS.md

This file is the operating guide for AI agents working in this repo. The repo
goal is a general post-training pipeline for open models: register a model
family, fine-tune it, optionally run behavior editing/refusal-style ablation,
quantize it for target hardware, evaluate every candidate, and publish
reproducible artifacts.

Do not treat any workflow as Gemma- or Qwen-specific. Existing Gemma, Qwen, and
Llama configs are examples of a family-driven pattern that should generalize to
new architectures.

## Read First

- `README.md`: concise user workflow.
- `docs/status.md`: current handoff state and active blockers.
- `docs/experiment-ledger.md`: detailed experiment history.
- `docs/adding-model-family.md`: how to add a new architecture.
- `docs/finetuning.md`: SFT/QLoRA workflow and promotion gates.
- `docs/abliteration.md`: behavior-editing and refusal-ablation workflow.
- `docs/quantization.md`: FP8/NVFP4 quantization workflow.
- `docs/evaluation-strategy.md`: internal/external eval design.
- `docs/cluster.md` and `docs/dgx-spark.md`: cluster/resource rules.
- `docs/artifact-retention.md`: what goes in Git, local disk, or Hugging Face.

If these files disagree, prefer `docs/status.md` for current run state and
prefer the task-specific docs for workflow mechanics. Update stale docs as part
of the work.

## Non-Negotiable Rules

- Never commit secrets. Use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` from the
  runtime environment only.
- Keep generated checkpoints, adapters, and large datasets out of Git. Push
  code, configs, docs, lightweight reports, and manifests.
- Upload completed model and dataset artifacts to Hugging Face only when they
  are complete and documented. Do not upload partial or failed candidates as
  releases.
- Run one large model server, training job, or checkpoint edit at a time unless
  a cluster plan explicitly requires multiple workers.
- Use guarded launchers for heavy jobs. Do not run raw training or checkpoint
  conversion scripts without CPU, RAM, disk, and watchdog limits.
- Check the worktree before editing. Do not revert user changes or unrelated
  generated output.
- Push completed code/docs/config changes to `main`.

Behavior-editing work must follow the configured objective profile and report
source-relative tradeoffs. Do not conflate refusal wording suppression, benign
over-refusal reduction, safe redirects, harmful-detail suppression, and harmful
prompt compliance. Promotion requires the metrics named by the active objective,
not anecdotes from a few prompts.

## Resource Contract

Before any heavy run:

```bash
./forge cluster doctor --config <cluster-config>
df -h /
./forge variants checkpoint-audit <family> --variant <variant> --strict
```

For DGX Spark or similar shared machines:

- keep SSH/control-plane headroom
- reserve at least one CPU core for the system
- require disk headroom before writing checkpoints
- keep a memory floor; prefer slower progress over a wedged machine
- use the repo launchers instead of ad hoc Docker or Python commands
- stop a server before launching another large model

When the user asks to use a cluster, prove the run used the cluster:

```bash
export MODEL_FORGE_CLUSTER_CONFIG=<untracked-cluster-config>
export MODEL_FORGE_SERVE_REQUIRE_CLUSTER=1
./forge cluster health --config "$MODEL_FORGE_CLUSTER_CONFIG"
./forge cluster torchrun-smoke --config "$MODEL_FORGE_CLUSTER_CONFIG"
MODEL_FORGE_DRY_RUN=1 ./forge serve <family> <variant>
```

`MODEL_FORGE_SERVE_REQUIRE_CLUSTER=1` should hard-stop instead of silently
falling back to solo mode.

## Full Workflow

Use this sequence for any new model.

### 1. Register The Family

Create or update `configs/model_families/<family>.yaml`.

Minimum fields:

- model identity and display name
- architecture family, context length, and target-discovery patterns
- base variant repo/local path/served name
- fine-tuned, behavior-edited, and quantized variant slots as needed
- serve/eval defaults
- comparison/report locations

Audit before training or serving:

```bash
./forge families
./forge variants graph <family>
./forge variants architecture-audit <family> --variant base
./forge variants tokenizer-audit <family> --variant base
./forge variants checkpoint-audit <family> --variant base --strict
./forge generalization audit
./forge doctor
```

### 2. Establish The Base Baseline

Download or register the base checkpoint on every node that will use it:

```bash
./forge download <family> base
./forge variants wait-checkpoint <family> --variant base
```

Serve and benchmark exactly the base model:

```bash
./forge serve <family> base
./forge eval <family> base --smoke
./forge eval <family> base --internal
./forge bench serve --config configs/serving/serve_bench_smoke.yaml \
  --family <family> --variant base --base-url http://127.0.0.1:8000/v1
```

Save the run IDs and evidence paths in `docs/experiment-ledger.md`.

### 3. Fine-Tune

Plan and prepare data before training:

```bash
./forge finetune <family> plan
./forge data plan <family> <ft-variant>
./forge data gaps <family> <ft-variant>
./forge data generate <family> <ft-variant> --smoke
./forge data verify <family> <ft-variant> --smoke
./forge data pack <family> <ft-variant> --smoke
./forge finetune <family> prepare --overwrite
```

For a real run, use the generated guarded launcher or cluster-aware training
entrypoint. After training:

```bash
./forge variants checkpoint-audit <family> --variant <ft-variant> --strict
./forge variants tokenizer-audit <family> --variant <ft-variant> --strict
./forge eval <family> <ft-variant> --internal
./forge compare <family>
```

A fine-tuned model is only a promotion candidate if it beats the base on the
target objective while preserving normal-use and benign-answer quality.

### 4. Behavior Edit / Ablate

Start from the source checkpoint you intend to modify. If the source is a
fine-tuned model, compare against that fine-tuned model, not only against base.

```bash
./forge objectives list
./forge objectives show <objective-profile>
./forge ablate <family> plan
./forge ablate --config configs/abliteration/<recipe>.yaml sota-prepare --backend <backend>
```

Use `--execute` only after the dry-run plan, resource preflight, and source
baseline are present. Backend reports are diagnostics; promotion requires
model-forge evals against the source checkpoint.

Minimum gate for a behavior-edited candidate:

- source-relative capability stays within the expected eval variance
- benign refusal stays low and benign quality is preserved
- configured refusal/redirect metrics improve according to the objective
- harmful-detail and harmful-compliance metrics are reported separately
- stochastic blockers are tested with multi-trial runs when they matter

Do not rerun a rejected candidate unchanged. If a candidate fails, change the
method, objective contrast, target layer set, data, or backend before spending
another heavy run.

### 5. Quantize

Quantize only after the source variant has baseline eval/serving evidence.

For Blackwell NVFP4:

```bash
./forge quantize plan --config configs/quantization/<nvfp4-config>.yaml --write-plan
./forge quantize export <family> <source-variant> \
  --config configs/quantization/<nvfp4-config>.yaml --write-plan --execute
./forge variants checkpoint-audit <family> --variant <nvfp4-variant> --strict
./forge variants tokenizer-audit <family> --variant <nvfp4-variant> --strict
./forge variants architecture-audit <family> --variant <nvfp4-variant> --strict
```

Serve and compare the quantized model against the exact unquantized source
variant. Do not use an unrelated base model as the formal source baseline.

```bash
./forge serve <family> <nvfp4-variant>
./forge bench serve --config configs/serving/serve_bench_core.yaml \
  --family <family> --variant <nvfp4-variant> \
  --base-url http://127.0.0.1:8000/v1
./forge bench serve-eval run \
  --config configs/serving/serve_eval_quality_behavior.yaml \
  --serving-summary <serving-summary.json> \
  --family <family> --variant <nvfp4-variant> \
  --base-url http://127.0.0.1:8000/v1
./forge bench serve --evidence-gate \
  --summary <serving-summary.json> \
  --serving-eval <serving-eval-dir> \
  --write-gate
```

Then write release evidence from the exact source-vs-candidate pair:

```bash
./forge bench serve-eval compare \
  --source-eval <source-serving-eval-dir> \
  --candidate-eval <candidate-serving-eval-dir> \
  --run-id <run-id> --write-report
./forge quantize card \
  --config configs/quantization/<nvfp4-config>.yaml \
  --source-serving-summary <source-summary.json> \
  --candidate-serving-summary <candidate-summary.json> \
  --source-serving-eval <source-serving-eval-dir> \
  --candidate-serving-eval <candidate-serving-eval-dir> \
  --run-id <run-id> --write-card
./forge quantize behavior-report \
  --config configs/quantization/<nvfp4-config>.yaml \
  --source-serving-summary <source-summary.json> \
  --candidate-serving-summary <candidate-summary.json> \
  --source-serving-eval <source-serving-eval-dir> \
  --candidate-serving-eval <candidate-serving-eval-dir> \
  --run-id <run-id> --write-report
./forge quantize tokenizer-report \
  --source-tokenizer-dir <source-tokenizer-dir> \
  --candidate-tokenizer-dir <candidate-tokenizer-dir> \
  --run-id <run-id> --write-report --strict
./forge quantize nvfp4-gate \
  --export-plan <quantization-export-plan.json> \
  --serving-summary <candidate-summary.json> \
  --serving-eval <candidate-serving-eval-dir> \
  --quantization-card <quantization-card.json> \
  --behavior-report <behavior-preservation-report.json> \
  --tokenizer-report <tokenizer-preservation-report.json> \
  --run-id <run-id> --write-gate
```

NVFP4 configs may declare `gates.nvfp4.min_output_tokens_per_second` for
absolute targets or `gates.nvfp4.min_output_speedup` plus
`gates.nvfp4.min_decode_heavy_output_speedup` for source-relative targets.

A quantized model is only a promotion candidate if behavior stays close to the
source and tok/s improves on the target hardware.

### 6. Publish

Before publishing:

```bash
./forge compare <family>
git diff --check
git status --short
```

Document:

- source model and exact variant graph edge
- training/edit/quantization configs
- dataset manifest or HF dataset ID
- eval and serving report paths
- known gaps and rejected candidates
- Hugging Face repo IDs for completed artifacts

Use the helper for uploads:

```bash
.venv/bin/python scripts/publish_hf_artifact.py \
  --repo-id <user-or-org>/<artifact-name> \
  --folder <local-folder> \
  --repo-type model \
  --commit-message "Upload model-forge artifact"
```

Prepared datasets use `--repo-type dataset`.

## Handoff Checklist

Before ending work:

- stop any live model server unless the user explicitly wants it left running
- write or update a tracked report under `reports/` for important ignored
  generated evidence
- update `docs/status.md` for the current short state
- update `docs/experiment-ledger.md` with hypothesis, config, commands,
  result, evidence paths, and publish status
- run focused tests for changed code
- run `git diff --check`
- commit and push to `main`

If the next agent needs to resume, they should be able to read this file,
`docs/status.md`, and `docs/experiment-ledger.md` without chat history.
