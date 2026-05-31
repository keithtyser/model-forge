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
Current non-Gemma family configs are `qwen35_9b`, `qwen36_27b`, and
`llama31_8b`; use them to check that new work remains generic rather than
Gemma-specific.

## First Things To Inspect

- `README.md`: project overview and model-agnostic workflow
- `docs/README.md`: task-oriented documentation index
- `docs/status.md`: current short handoff state and recommended next work
- `docs/artifact-retention.md`: what belongs in Git, local scratch, or Hugging
  Face
- `docs/adding-model-family.md`: checklist for adding a portable non-Gemma
  family
- `docs/finetuning.md`: SFT/QLoRA workflow and promotion gates
- `docs/abliteration.md`: refusal-ablation methodology and promotion criteria
- `docs/evaluation-strategy.md`: eval design and interpretation
- `docs/artifact-validation.md`: standalone HTML/Python artifact execution
  validators and Artifact Execution Card rules
- `docs/experiment-ledger.md`: handoff ledger for hypotheses, experiments,
  artifacts, validation, and publish status
- `docs/run-manifests.md`: canonical run manifest schema and handoff rules
- `docs/agent-experiments.md`: pre-run experiment schema for agent work
- `docs/variant-graph.md`: variant node schema, graph inspection, evidence,
  checksums, promotion, and retention rules
- `docs/cluster.md`: generic cluster inventory, doctor, and dry-run planning
  rules
- `docs/serving-benchmarks.md`: serving benchmark command, outputs, and
  interpretation rules
- `docs/profiling.md`: Nsight Systems / Nsight Compute profile planning rules
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

For a full new-model run, do not stop at planning:

1. Resolve/download the base checkpoint on every node that will train, serve, or
   quantize it.
2. Run the base model eval/serve benchmarks and save the comparison baseline.
3. Fine-tune from the base model and prove the FT beats the base on the
   objective-specific evals.
4. Ablate the fine-tuned model, then compare against the FT source. A good
   ablation removes refusals while retaining FT capability and benign quality.
5. Quantize the final candidate, then compare against the unquantized
   FT-ablated source. A good NVFP4 result keeps quality close while improving
   output tok/s on Blackwell.
6. Record every pain point as either a config fix, code fix, or explicit
   follow-up in the ledger before moving on.

For Hugging Face downloads, use `./forge hf login` once or set `HF_TOKEN` in the
runtime environment. Do not put tokens in commands, configs, docs, reports, or
git. In unattended runs set `MODEL_FORGE_HF_ALLOW_PROMPT=0`; the download command
will use cached auth when present and fail fast if no token is available. If the
default Xet path stalls on a checkpoint, retry with `HF_HUB_DISABLE_XET=1` and
keep the job bounded with CPU, memory, and disk guardrails.
After starting or resuming a large download, use
`./forge variants wait-checkpoint <family> --variant <variant>` before model
sync, serving, training, ablation, or quantization. It polls the strict
checkpoint audit and prevents agents from racing partially downloaded HF shards.

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
./forge finetune --config configs/finetuning/qwen36_27b_local_ft_v1.yaml plan
./forge finetune --config configs/finetuning/qwen36_27b_local_ft_v1.yaml prepare --overwrite
```

`prepare` writes `training_method_card.md` beside the generated plan, trainer,
runner, and eval scripts. Treat it as method and guardrail documentation; it is
not proof that training ran or that distributed correctness was validated.

Run fine-tuning on DGX Spark only through the guarded CUDA container launcher
when host Python is CPU-only:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
scripts/run_finetune_spark_container.sh
```

For two-node Spark fine-tunes, prefer generated cluster artifacts when the
recipe has a `cluster:` block:

```bash
./forge cluster doctor --config <private-cluster.yaml> --strict
./forge cluster health --config <private-cluster.yaml>
./forge cluster runtime --config <private-cluster.yaml> --image nemotron-runner:latest
./forge cluster torchrun-smoke --config <private-cluster.yaml> --image nemotron-runner:latest --nccl-socket-ifname <direct-link-iface>
MODEL_FORGE_CLUSTER_CONFIG=<private-cluster.yaml> \
MODEL_FORGE_EXECUTE_CLUSTER_TRAIN=1 \
runs/finetune/<run>/run_cluster_torchrun.sh
```

The cluster script prepares data once, syncs the generated run directory to
worker nodes, and launches Docker-backed `torch.distributed.run` on every node.
If it falls back to host Python or a single node, fix the repo or config before
starting the long run.
For large local checkpoints, set `cluster.sync_model_to_workers: true` in the
fine-tune YAML so the generated launcher runs checkpoint-gated
`./forge cluster model-sync` before distributed training. This avoids a common
two-node failure mode where the coordinator has the model but workers do not.

For newer architectures that require Transformers 5 model classes, build and
copy the generic post-training image before launching the cluster run:

```bash
docker build \
  --cpuset-cpus 0-3 \
  --memory 24g \
  -f docker/posttrain-transformers5.Dockerfile \
  -t model-forge-posttrain-tf5:latest .
```

The image starts from a Spark/vLLM Transformers-5 base and adds PEFT, TRL,
bitsandbytes, datasets, accelerate, and ModelOpt. Use it only when the target
family needs the newer model registry; keep the selected image in the
fine-tune YAML instead of hard-coding it in pipeline code.

Before ablating or quantizing a PEFT fine-tune, merge the adapter into a full
checkpoint and point the next recipe at that merged directory:

```bash
nice -n 10 .venv/bin/python scripts/merge_peft_adapter.py \
  --base-model ~/models/<base-model-dir> \
  --adapter ~/models/<adapter-dir> \
  --output-dir ~/models/<adapter-dir>-merged \
  --dtype bf16 \
  --trust-remote-code
./forge variants checkpoint-audit <family> --variant local_ft --strict
```

The adapter directory remains useful for live LoRA serving. The merged directory
is the source for behavior edits, full-checkpoint quantization, and model
uploads that should no longer depend on separate adapter loading.

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
- Do not mark a dataset recipe validated from static pack quality alone. Run
  `./forge data training-gate <family> <variant> --finetune-plan <run>/plan.json --data-summary <run>/data_summary.json --promotion-report <promotion>.json --write-gate`
  after a bounded Spark fine-tune. The gate checks dataset usage, `max_steps`,
  row bounds, Spark evidence, resource guardrails, materialized train rows, and
  source-relative promotion results. It rejects seed-only and smoke-only packs.

## Behavior Editing Scorecards

Use the behavior scorecard before claiming an ablation is successful:

```bash
./forge behavior doctor --config configs/behavior_edit/gemma4_26b_a4b_scorecard.yaml --strict
./forge behavior score --config configs/behavior_edit/gemma4_26b_a4b_scorecard.yaml local_abli_sota_vs_base --write-card
./forge behavior frontier --config configs/behavior_edit/gemma4_26b_a4b_scorecard.yaml local_abli_sota_vs_base --write-report
./forge behavior risk-report --config configs/behavior_edit/gemma4_26b_a4b_scorecard.yaml local_abli_sota_vs_base --write-report
```

For refusal-removal objectives, lower harmful-prompt refusal can be success.
The scorecard still requires source-relative capability and benign-quality
retention, and it reports unsafe overcompliance or harmful detail as explicit
risks rather than silently ignoring them. Use `frontier` when multiple
candidates exist; it selects from actual saved comparison rows instead of
claiming a single winner by hand. Use public `risk-report` for aggregate-only
release evidence; raw harmful prompts/outputs stay in private ignored artifacts.
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
./forge schema audit
./forge roadmap audit --write-doc
./forge research list
./forge research show arditi_2024_refusal_direction
./forge research audit
./forge research watch
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

Write or validate an agent experiment plan before starting material work:

```bash
./forge agent schema
./forge agent audit
./forge agent init \
  --experiment-id next_agent_step \
  --title "Next agent step" \
  --family gemma4_26b_a4b \
  --variant base \
  --objective-profile capability_sft \
  --output recipes/agents/next_agent_step.yaml
./forge agent optimize-serving --family gemma4_26b_a4b --variant base
./forge agent optimize-quantization --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --variants base,local_ft
./forge agent optimize-behavior-edit --family gemma4_26b_a4b
./forge agent card recipes/agents/next_agent_step.yaml --write-card --update-ledger
```

Agent plans are pre-run contracts. They should state the hypothesis, resource
policy, validation commands, evidence, rollback plan, and handoff requirements.
Agent Run Cards summarize the selected plan, heavy commands, required evidence,
schema findings, and Git state for handoff. Use `--update-ledger` so the
handoff block in `docs/experiment-ledger.md` is inserted or refreshed
automatically instead of relying on chat history.

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
./forge variants architecture-audit gemma4_26b_a4b --variant base
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli
./forge variants checkpoint-audit gemma4_26b_a4b --variant base --strict
./forge variants wait-checkpoint gemma4_26b_a4b --variant base --timeout-seconds 0
```

Variant nodes record the source variant, transform, checkpoint reference,
validation state, evidence path, artifact checksums, promotion decision, and
retention decision. Keep generated nodes in `reports/generated/` unless a small
example is intentionally promoted. Run `tokenizer-audit --load-tokenizer
--strict` before release gates so adapter merges, ablation exports,
quantization exports, and future GGUF conversions cannot silently lose
chat-template or special-token behavior.
Run `checkpoint-audit --strict` before serving or training a downloaded model;
it catches missing safetensor shards, missing index files, missing config or
tokenizer markers, and active Hugging Face `.incomplete` downloads.
Use `wait-checkpoint` in unattended scripts after `./forge download` or
`huggingface_hub.snapshot_download` so the next step starts only after the
strict audit passes.

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

For Qwen-family serving on Spark, `configs/model_families/qwen35_9b.yaml` and
`configs/model_families/qwen36_27b.yaml` use the generic
`scripts/dgx_spark_serve_qwen.sh` launcher. It is solo by default. To use both
Spark nodes, set `MODEL_FORGE_SPARK_CLUSTER=1`,
`MODEL_FORGE_SPARK_CLUSTER_NODES=<coordinator-ip>,<worker-ip>`,
`MODEL_FORGE_SPARK_ETH_IF=<direct-link-interface>`, and
`MODEL_FORGE_TENSOR_PARALLEL_SIZE=2` outside Git, then run
`./forge serve <family> <variant>`. The same model directory must exist on both
nodes under `MODEL_FORGE_MODELS_DIR`; if only the coordinator has HF egress,
download once there and run `./forge cluster model-sync --source <model-dir>
--execute` to copy the completed checkpoint to workers. Use `model-sync` instead
of hand-written `rsync` where possible so generated evidence captures what was
copied.
Do not run `model-sync` on an active or incomplete HF download; first run
`./forge variants wait-checkpoint qwen36_27b --variant base`.
When the source directory corresponds to a configured family variant, pass the
same identity into `model-sync` so the command enforces the checkpoint gate:

```bash
./forge cluster model-sync \
  --config <private-cluster.yaml> \
  --source <models-dir>/Qwen3.6-27B \
  --family qwen36_27b \
  --variant base \
  --models-dir <models-dir> \
  --execute
```

Before launching a large Qwen server, dry-run the exact command and inspect the
vLLM image, chat-template JSON, tensor parallel size, GPU memory utilization,
batched tokens, and max sequence count:

```bash
MODEL_FORGE_DRY_RUN=1 \
MODEL_FORGE_SPARK_CLUSTER=1 \
MODEL_FORGE_SPARK_CLUSTER_NODES=<coordinator-ip>,<worker-ip> \
MODEL_FORGE_SPARK_ETH_IF=<direct-link-interface> \
MODEL_FORGE_TENSOR_PARALLEL_SIZE=2 \
./forge serve qwen36_27b base
```

Family `serve:` defaults are intentional safety bounds and should win over
generic hardware recommendations. For Qwen 3.6 27B the repo defaults to the
Transformers-5 Spark vLLM image, `GPU_MEMORY_UTILIZATION=0.78`,
`MAX_NUM_BATCHED_TOKENS=16384`, and `VLLM_MAX_NUM_SEQS=4`; raise them only after
baseline serving works and a benchmark proves the change helps.

Benchmark serving only after an endpoint is already running:

```bash
./forge serving doctor --config configs/serving/backends/sglang_openai.yaml
./forge serving plan --config configs/serving/backends/sglang_openai.yaml --family gemma4_26b_a4b --variant base --write-plan
./forge serving doctor --config configs/serving/backends/tensorrt_llm_openai.yaml
./forge serving plan --config configs/serving/backends/tensorrt_llm_openai.yaml --family gemma4_26b_a4b --variant base --write-plan
./forge serving architecture-doctor --config configs/serving/architectures/distributed_kv_placeholder.yaml --strict
./forge bench serve --family gemma4_26b_a4b --variant base --dry-run
./forge bench serve --family gemma4_26b_a4b --variant base
./forge bench serve --evidence-gate --summary reports/generated/serve_bench/<run>/summary.json --serving-eval reports/generated/serve_eval/<run> --write-gate
./forge bench sweep doctor --config configs/sweeps/dgx_spark_vllm_baseline.yaml --strict
./forge bench sweep plan --family gemma4_26b_a4b --variant base
./forge bench sweep doctor --config configs/sweeps/dgx_spark_vllm_disagg_prefill_decode.yaml --strict
./forge bench sweep plan --config configs/sweeps/dgx_spark_vllm_disagg_prefill_decode.yaml --family gemma4_26b_a4b --variant base --cluster-config configs/clusters/dgx_spark_x2.example.yaml --write-plan
./forge bench kernel rmsnorm --dry-run --json
./forge bench kernel rope --dry-run --json
./forge bench kernel dequant --dry-run --json
./forge bench kernel kv-layout --dry-run --json
./forge bench kernel card --summary reports/generated/kernel_benchmarks/<run>/summary.json --write-card
./forge profile nsight plan --config configs/profiling/nsight_serving_smoke.yaml --write-plan
./forge profile nsight summarize --plan reports/generated/profiles/nsight/<run>/nsight_profile_plan.json --write-summary
```

`bench serve` is for OpenAI-compatible endpoint mechanics only. `bench sweep`
expands startup-time server env cases plus matching benchmark commands. Neither
command starts a vLLM server. `bench kernel rmsnorm`, `bench kernel rope`,
`bench kernel dequant`, and `bench kernel kv-layout` are microbenchmark
harnesses; use them to produce `summary.json`, `kernel_card.json`, and
`kernel_card.md`, then connect the result to profile evidence before making
optimization claims. `bench kernel card` can regenerate a Kernel Card from an
existing summary and optionally attach a profile summary. `profile nsight`
writes profiler command plans around existing benchmark commands; it does not
start servers or profilers by default. `profile nsight summarize` inventories
expected and present profiler artifacts; it is not a kernel interpretation by
itself. A good latency result is not a quality or behavior pass. Use the
generated `manifest.json`, `summary.json`, `requests.jsonl`,
`serving_card.md`, kernel card, profile plan, and profile summary with eval
results before making serving claims.

Use `./forge bench serve --evidence-gate` before marking serving work complete.
Without sampled quality/behavior evidence under the same served model and base
URL, the gate should fail; do not use `--allow-missing-serving-eval` for
promotion or completion claims.

`./forge serving plan` currently supports SGLang and TensorRT-LLM planning. It
writes launch and benchmark commands but does not start a backend. Start at most
one serving backend at a time and benchmark it through the same `bench serve`
configs used for vLLM before claiming engine comparisons.

The disaggregated prefill/decode sweep profile is a plan-only advanced serving
profile. Compare it against the single-endpoint control under the same model,
precision, benchmark config, and quality/behavior sample before claiming the
split improved Spark throughput or latency.

The distributed-KV architecture file is a placeholder contract. Do not treat it
as implementation evidence; use it to check that a future LMCache/NIXL/Dynamo
or vLLM-disaggregated run records topology, transport, control, and promotion
blockers before claiming success.

Plan and report quantization without loading a model:

```bash
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --write-plan
./forge quantize plan llama31_8b base --config configs/quantization/fp8_w8a8_modelopt.yaml --write-plan
./forge quantize export llama31_8b base --config configs/quantization/gguf_llama_cpp_q4_k_m.yaml --write-plan
./forge quantize calibration-manifest gemma4_26b_a4b base --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --write-manifest
./forge quantize card \
  --config configs/quantization/nvfp4_blackwell_runtime.yaml \
  --source-serving-summary <source>/summary.json \
  --candidate-serving-summary <candidate>/summary.json \
  --source-serving-eval <source_eval_dir> \
  --candidate-serving-eval <candidate_eval_dir> \
  --run-id source_vs_nvfp4 \
  --write-card
./forge quantize fp8-kv-report \
  --config configs/quantization/gemma4_26b_a4b_fp8_runtime.yaml \
  --source-serving-summary <source>/summary.json \
  --candidate-serving-summary <candidate>/summary.json \
  --source-serving-eval <source_eval_dir> \
  --candidate-serving-eval <candidate_eval_dir> \
  --run-id source_vs_fp8_kv \
  --write-report
./forge quantize behavior-report \
  --config configs/quantization/fp8_w8a8_modelopt.yaml \
  --source-serving-summary <source>/summary.json \
  --candidate-serving-summary <candidate>/summary.json \
  --source-serving-eval <source_eval_dir> \
  --candidate-serving-eval <candidate_eval_dir> \
  --run-id source_vs_quantized_behavior \
  --write-report
./forge quantize tokenizer-report \
  --source-tokenizer-dir <source_model_dir> \
  --candidate-tokenizer-dir <quantized_or_gguf_dir> \
  --run-id source_vs_quantized_tokenizer \
  --write-report
./forge quantize sensitivity-report \
  --config configs/quantization/sensitivity_scan.yaml \
  --baseline-serving-summary <source>/summary.json \
  --baseline-serving-eval <source_eval_dir> \
  --candidate name=mlp_only,component=mlp,summary=<candidate>/summary.json,eval=<candidate_eval_dir> \
  --run-id quant_sensitivity \
  --write-report
./forge quantize nvfp4-gate \
  --export-plan <export_plan.json> \
  --serving-summary <serve>/summary.json \
  --serving-eval <serve_eval_dir> \
  --quantization-card <quantization_card.json> \
  --behavior-report <behavior_preservation_report.json> \
  --tokenizer-report <tokenizer_preservation_report.json> \
  --run-id nvfp4_gate \
  --write-gate
```

NVFP4 is the priority Blackwell path. `nvfp4_runtime` means Model Forge is
validating an already-quantized checkpoint; do not imply the repo created those
weights. A real quantization claim needs a candidate endpoint, serving summary,
sampled behavior scores, and quantization card.
For NVFP4 promotion, require a clear `output_tokens_per_second` improvement
over the matching BF16/FP16 endpoint, especially on `decode_heavy`; otherwise
treat the run as loader evidence only.
Before starting a self-quantization export, write a calibration manifest for the
same family/variant/config. If you override datasets through
`MODEL_FORGE_QUANT_CALIB_*`, regenerate the manifest so the export and evidence
card point at the same calibration contract.
For FP8 KV cache experiments, write `./forge quantize fp8-kv-report` from
completed source and candidate endpoint evidence. Treat it as a behavior report,
not as a checkpoint quantization claim.
For FP8 W8A8 checkpoint creation, use
`configs/quantization/fp8_w8a8_modelopt.yaml` with an explicit family and
variant. The config is intentionally generic; do not add Gemma-only defaults to
common quantization code.
For every quantized candidate, write a behavior-preservation report from the
same source/candidate eval evidence used by the quantization card. Throughput
does not compensate for failing the required behavior-retention checks.
For quantized or GGUF export directories that are not yet configured variants,
use `./forge quantize tokenizer-report` to compare tokenizer files, special
tokens, and chat-template metadata directly against the source model directory.
Use `./forge quantize sensitivity-report` after candidate runs exist to rank
component policies such as all-linear, MLP-only, attention-only, or experts-only.
Do not infer component sensitivity from a single candidate.
For GGUF exports, set `MODEL_FORGE_LLAMA_CPP_DIR` outside git, run the guarded
`./forge quantize export ... --config configs/quantization/gguf_llama_cpp_q4_k_m.yaml`
path, and attach tokenizer, llama-cli load, llama-bench, behavior, and
quantization-card evidence before promotion.
For Blackwell NVFP4, run `./forge quantize nvfp4-gate` before promotion. The
gate must see ModelOpt NVFP4 export evidence, completed serving/eval artifacts,
quantization card, behavior report, tokenizer report, and a clear output tok/s
win against the configured threshold.

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

Plan upstream PRs only when there is evidence:

```bash
./forge upstream audit --config configs/upstream/pr_candidates.yaml
./forge upstream plan --config configs/upstream/pr_candidates.yaml --candidate dgx_spark_vllm_serving_recipe --write-plan
./forge upstream verify-pr --config configs/upstream/pr_candidates.yaml --candidate dgx_spark_vllm_serving_recipe --offline --write-report
```

Do not mark `MF-0808` complete from a local plan alone. It requires a real
external pull request URL plus benchmark, profiler, Kernel Card, or serving
evidence suitable for the target upstream project.
Run `./forge upstream audit --config configs/upstream/pr_candidates.yaml --strict`
after replacing placeholder targets and before claiming completion.
Opened or merged upstream records must use a real GitHub pull request URL and
existing local evidence paths; unresolved `<run>` placeholders are planning
markers, not completion evidence.
`verify-pr --offline` is useful while drafting, but MF-0808 completion requires
a non-offline verification report so the GitHub API has confirmed the external
pull request metadata.

## Abliteration Rules

- The reusable recipe is the structure, not fixed constants.
- Always compute fresh refusal directions on the source checkpoint being
  ablated.
- Direct parameter transfer is only a warm start for nearby checkpoints in the
  same architecture family.
- For new architectures, inspect target module names, layer counts, hidden
  sizes, MoE/expert layouts, and tokenizer/chat templates before editing.
- Run `./forge variants architecture-audit <family> --variant base` before
  reusing LoRA targets, ablation target modules, or quantization exclusions.
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

Qwen 3.6 27B full-workflow starter configs:

```text
configs/finetuning/qwen36_27b_local_ft_v1.yaml
configs/abliteration/qwen36_27b_ft_local_abli.yaml
configs/quantization/qwen36_27b_nvfp4_modelopt.yaml
```

These are examples of the general workflow. Do not hard-code future model
support around Gemma-specific layer names or constants.

## Publishing

When publishing a model:

- start with `./forge hf status --offline` and
  `./forge hf plan-model <family> <variant> --release-class <class>`
- run `./forge hf release-classes --audit` after editing release-class YAMLs
- use release classes from `configs/release_classes/` to decide whether a
  release is report-only, adapter-only, private research, public dataset, or a
  public quantized checkpoint
- public behavior-edited releases require a risk report or behavior-edit
  scorecard path through `--risk-report`
- never bypass failed `release_gates` in `hub_publish.json`; public full
  checkpoints require explicit allowance plus Spark validation evidence
- for dataset releases, prefer `./forge data publish ... --source-license-checked`
  and inspect `hf_publish_bundle/`; public dataset plans must not include raw
  accepted/rejected rows or unredacted message text unless the release class
  explicitly allows it
- before uploading any dataset, run
  `./forge hf publish-dataset <dataset_path> --repo-id <namespace>/<repo> --dry-run`
  and fix every failed gate instead of overriding the generated
  `hub_dataset_plan.json`
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
