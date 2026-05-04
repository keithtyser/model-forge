# Experiment Ledger

This file is the handoff ledger for agents. Every material experiment should
record its hypothesis, recipe/config, artifact path, validation, and publish
status so another agent can resume without relying on chat history.

## Publish Rule

- Push code, configs, docs, and lightweight run metadata to GitHub.
- Upload completed model checkpoints and completed prepared datasets to Hugging
  Face.
- Use the provided Hugging Face token from the environment for future uploads.
  Do not write the token to a file, command log, model card, config, or commit.
- Do not upload partial or smoke-test artifacts as final models or datasets.
- Use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`; never commit tokens.

Publishing helper:

```bash
.venv/bin/python scripts/publish_hf_artifact.py \
  --repo-id <user-or-org>/<artifact-name> \
  --folder <local-folder> \
  --repo-type model \
  --commit-message "Upload model-forge artifact"
```

For prepared datasets, pass `--repo-type dataset`.

## Fine-Tune: Gemma 4 26B A4B Local FT v0

Status: planned and smoke-tested; no full FT model has been produced yet.

Hypothesis: a stricter SFT/QLoRA recipe using Jackrong-style mixed reasoning,
code, STEM, and chat sources, plus model-forge quality gates and holdouts, can
match or beat `Jackrong/Gemopus-4-26B-A4B-it` on the model-forge eval suite.

Primary config:

```text
configs/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
datasets/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
```

Generated run artifacts:

```text
runs/finetune/gemma4_26b_a4b_local_ft_v0/
```

Validation completed:

- fine-tune plan resolves against `google/gemma-4-26B-A4B-it`
- data-prep smoke with `--data-limit 2` accepted rows from multiple sources
- one-step QLoRA smoke loaded the 26B base checkpoint and completed one trainer
  step with temporary `max_seq_length=512`
- no full data preparation completed
- no full training completed
- no local FT checkpoint exists yet

Current blocker: resource governance was added after the host became
unresponsive under load. Future full FT runs must use the generated guarded
`run.sh`, not a direct trainer invocation.

Publish status:

- GitHub: recipe, data manifest, pipeline, docs, and guardrails pushed
- Hugging Face dataset: pending because no complete prepared FT dataset exists
- Hugging Face model: pending because no complete FT model exists

Next run:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
scripts/run_finetune_spark_container.sh
```

Reason: host Python is currently CPU-only, while `nemotron-runner:latest`
exposes `NVIDIA GB10` and includes the required TRL/PEFT/bitsandbytes training
stack. The container launcher mounts repo/model/cache paths at their host paths,
runs as the current user, and applies Docker CPU/memory limits before invoking
the generated guarded `run.sh`.

Current data-prep result:

```text
runs/finetune/gemma4_26b_a4b_local_ft_v0/train.jsonl
rows: 40189
size: 801 MB
```

Training blocker found after data prep: the base Spark training image has
Transformers 4.57.6, which does not recognize `model_type=gemma4`. A run-local
Python overlay was created at:

```text
runs/finetune/gemma4_26b_a4b_local_ft_v0/python_overlay
```

It pins `transformers==5.5.0`, which registers Gemma4 while leaving the host and
base Docker image unchanged. The Spark container launcher prepends this overlay
to `PYTHONPATH` when present.

## Resource Guardrails

Status: implemented and pushed.

Hypothesis: training must run as a tenant with hard CPU, memory, IO, disk, and
thread controls so the host remains reachable during long jobs.

Implementation:

```text
src/model_forge/pipelines/finetune.py
scripts/model_forge_watchdog.py
docs/finetuning.md
AGENTS.md
```

Default contract:

```text
CPUQuota=80%
MemoryMax=85%
IOWeight=100
nice=10
reserve_cores=1
min_memory_available_start=15%
min_memory_available_runtime=10%
min_disk_free=15%
```

Validation:

```text
.venv/bin/python -m py_compile src/model_forge/pipelines/finetune.py scripts/model_forge_watchdog.py
.venv/bin/python -m unittest discover -s tests
git diff --check
```

Result: all checks passed.

## Ablation: Gemma 4 26B A4B Base Local Abli

Status: completed before FT work.

Purpose: ablate refusals from the base model while preserving source-model
capability.

Primary config:

```text
configs/abliteration/gemma4_26b_a4b_local_abli.yaml
```

Representative local artifact:

```text
~/models/gemma-4-26B-A4B-it-local-abliterated-sota-internal-t34
```

Observed result: local recipe surpassed the downloaded abli model on the
model-forge internal suite while preserving capability sufficiently for the
repo's refusal-removal objective.

Publish status: already uploaded to Hugging Face before the FT handoff. No
additional upload needed for this completed ablated checkpoint unless the model
card or files need revision.

## Ablation: Gemopus FT Local Abli

Status: completed before FT work.

Purpose: ablate the already fine-tuned `Jackrong/Gemopus-4-26B-A4B-it` while
preserving its FT capability.

Primary config:

```text
configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml
```

Representative selected artifact:

```text
~/models/Gemopus-4-26B-A4B-it-local-abliterated-sota-internal-r7-selected-t34-transfer
```

Observed result: selected run reduced refusals as intended and approximately
preserved the FT model's paired benign quality and challenge capability, making
it a successful no-finetune abliteration of the FT checkpoint.

Publish status: already uploaded to Hugging Face before the FT handoff. No
additional upload needed for this completed ablated checkpoint unless the model
card or files need revision.
