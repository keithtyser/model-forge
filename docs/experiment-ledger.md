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

Status: corrected text-LoRA recipe smoke-tested; no full FT model has been
produced yet.

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
- full data preparation completed: 40,189 rows, 801 MB JSONL, 2048-token
  tokenized cache created
- 1024-token and 2048-token Unsloth QLoRA smoke tests completed under resource
  guardrails
- corrected 2048-token, 5-step text-LoRA smoke produced nonzero text
  `lora_B` tensors and nonzero gradients
- no valid full training completed
- no local FT checkpoint exists yet

Current blocker: the first full run was stopped after checkpoint 100 because the
original Gemma target modules used `.linear` suffixes. Those matched the
`vision_tower` path only, so text loss produced zero-gradient LoRA updates.
The recipe now targets text modules by base names and excludes `vision_tower`.
Future full FT runs must use the generated guarded `run.sh`, not a direct
trainer invocation.

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

Second training blocker found after model load: TRL tokenized the raw text
dataset after loading the 26B model, pushing available host memory below the
10% runtime floor. The trainer now uses a lean HF causal-LM `Trainer` path:
tokenize/cache `train.jsonl` to `tokenized_train` before model load, release raw
text, then load the QLoRA model and train.

Resume training from the completed prepared dataset with:

```bash
MODEL_FORGE_SKIP_PREPARE=1 scripts/run_finetune_spark_container.sh
```

Active full-run attempt:

```text
started: 2026-05-18 03:51 America/New_York
container: model-forge-ft-local-v0
output: /home/ktyser/models/gemma-4-26B-A4B-it-local-ft-v0
command: guarded Docker run equivalent to MODEL_FORGE_SKIP_PREPARE=1 scripts/run_finetune_spark_container.sh
goal: 500 optimizer steps, checkpoint every 100 steps
checkpoint gate: inspect checkpoint-100 adapter_model.safetensors before trusting the run
```

Checkpoint-100 gate result:

```text
status: passed
timestamp: 2026-05-18 07:01 America/New_York
trainer_state: global_step=100, max_steps=500
loss tail: step 100 loss=24.896150207519533
grad_norm tail: step 100 grad_norm=0.508475661277771
text lora_B tensors: 205/205 nonzero, max_abs=0.2190384417772293
vision lora_B tensors: 0/189 nonzero
decision: continue full run to step 500
```

Checkpoint-200 gate result:

```text
status: passed
timestamp: 2026-05-18 10:04 America/New_York
trainer_state: global_step=200, max_steps=500
loss tail: step 200 loss=22.157656860351562
grad_norm tail: step 200 grad_norm=0.5061691403388977
text lora_B tensors: 205/205 nonzero, max_abs=0.2329128384590149
vision lora_B tensors: 0/189 nonzero
decision: continue full run to step 500
```

Checkpoint-300 gate result:

```text
status: passed
timestamp: 2026-05-18 13:08 America/New_York
trainer_state: global_step=300, max_steps=500
loss tail: step 300 loss=22.977902221679688
grad_norm tail: step 300 grad_norm=0.20519530773162842
text lora_B tensors: 205/205 nonzero, max_abs=0.23511211574077606
vision lora_B tensors: 0/189 nonzero
decision: continue full run to step 500
```

The invalid earlier full-run output was moved aside to:

```text
/home/ktyser/models/gemma-4-26B-A4B-it-local-ft-v0.failed-vision-only-20260518-034146
```

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
min_memory_available_runtime=5%
min_disk_free=15%
```

Validation:

```text
.venv/bin/python -m py_compile src/model_forge/pipelines/finetune.py scripts/model_forge_watchdog.py
.venv/bin/python -m unittest discover -s tests
git diff --check
```

Result: all checks passed.

## Fine-Tuning: Gemma 4 26B A4B Local FT Runtime Bring-Up

Status: in progress.

Purpose: train a local FT from the base model and compare it against the
downloaded Jackrong FT reference without rerunning already-saved baseline evals.

Findings so far:

```text
HF Causal LM 4-bit loader:
- 2048-token one-step smoke stopped at resource guard: memory available 0.3% < 5%.
- 1024-token one-step smoke stopped at resource guard: memory available 1.9% < 5%.
- Root issue was model-load host memory pressure, not sequence length alone.

Unsloth 4-bit loader:
- Gemma 4 26B load-only probe succeeded with about 61 GiB host memory available after load.
- 1024-token one-step QLoRA smoke passed with gradient_accumulation_steps=24.
- Smoke train metrics: train_runtime=61.03s, train_samples_per_second=0.393,
  train_steps_per_second=0.016, train_loss=118.6.
- 2048-token one-step QLoRA smoke passed with gradient_accumulation_steps=24,
  but it was later found to be vision-only LoRA due to bad target modules.
- 2048 one-step smoke train metrics: train_runtime=117.5s,
  train_samples_per_second=0.204, train_steps_per_second=0.009,
  train_loss=97.67.
- Full 500-step attempt was stopped after checkpoint 100. Checkpoint inspection
  showed all 189 LoRA tensors were under `vision_tower`, all `lora_B` tensors
  were zero, and trainer logs reported grad_norm=0.0.
- Corrected 2048-token, 5-step text-LoRA smoke passed. Trainer logs showed
  nonzero grad_norm at every step; final loss decreased from 97.67 to 49.93.
- Corrected smoke adapter inspection:
  text `lora_B` tensors: 205/205 nonzero, max_abs=0.008142.
  vision `lora_B` tensors: 0/189 nonzero.
```

Recipe changes:

```text
model.max_seq_length=2048
trainer.max_steps=500 for the first full local FT attempt
trainer.backend=unsloth
trainer.unsloth_compile_disable=true
trainer.group_by_length=true
trainer.pad_to_multiple_of=256
trainer.torch_dynamo_recompile_limit=128
tokenized_train caches are keyed by max_seq_length
lora.target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora.exclude_modules=vision_tower
```

Justification: Unsloth preserves the guarded Spark runtime while avoiding the
HF loader's host-memory spike. Compile is disabled for this Gemma 4 recipe
because the compiled Unsloth Gemma 4 path hit a hard Torch Dynamo fullgraph
recompile limit during gradient accumulation. Revisit compile after a successful
full FT and eval pass. Gemma 4 target modules must use language-model module
base names, not vision `.linear` suffixes; the 5-step smoke is the guard against
silently training the wrong adapter path.

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
