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
- Follow `docs/artifact-retention.md` before committing, deleting, or uploading
  generated artifacts.

Publishing helper:

```bash
.venv/bin/python scripts/publish_hf_artifact.py \
  --repo-id <user-or-org>/<artifact-name> \
  --folder <local-folder> \
  --repo-type model \
  --commit-message "Upload model-forge artifact"
```

For prepared datasets, pass `--repo-type dataset`.

## Roadmap Foundation: Variant Graph And Evidence Nodes

Status: implemented as metadata tooling only. No model server, training run,
quantization run, or eval job was started.

Hypothesis: every model transform needs a durable graph node so agents can see
where a candidate came from, which recipe produced it, what evidence exists, and
whether it should be promoted, retained, published, or deleted.

Changes:

- added `./forge variants graph <family>`
- added `./forge variants node <family> <variant>`
- added `./forge variants audit-node <path>`
- added `src/model_forge/variants/graph.py`
- added `src/model_forge/variants/manifest.py`
- added `docs/variant-graph.md`
- `./forge doctor` validates tracked `variant_node.json` files
- fixed git dirty-path parsing in canonical manifest metadata while testing the
  node writer

Validation:

```bash
./forge variants graph gemma4_26b_a4b --json
./forge variants node gemma4_26b_a4b local_ft --json
.venv/bin/python -m unittest tests.test_variants tests.test_run_manifest
```

Result:

- Gemma graph currently resolves 16 configured variants and 8 explicit
  `base_variant` edges
- variant node validation covers implementation status, validation state,
  promotion decision, retention fields, and artifact checksum entries

## Dataset Factory: Eval Feedback Proposal

Status: implemented and pushed. No model server, training run, quantization
run, or live teacher generation was started.

Hypothesis: eval failures should feed the next dataset iteration through a
durable proposal artifact, not only through informal notes. The proposal should
rank failed skills, preserve top failed buckets/cases, recommend conservative
target-count bumps, and provide a candidate config patch for the next data run.

Changes:

- added `./forge data propose <family> <variant>`
- added `feedback_proposal.yaml` generation from saved eval `responses.jsonl`
- added ranked skill updates, generation scale recommendations, focus skills,
  top failure cases, missed concepts, and candidate config patch output
- tracked the local FT v1 proposal at
  `datasets/generated/gemma4_26b_a4b_local_ft_v1/feedback_proposal.yaml`
- updated README, fine-tuning docs, status, and agent handoff instructions

Validation:

```bash
.venv/bin/python -m py_compile src/model_forge/data/factory.py
.venv/bin/python -m unittest tests.test_data_factory -v
./forge data propose gemma4_26b_a4b local_ft_v1 --overwrite
```

Result:

- proposal generated from 318 saved local FT v0 internal eval rows
- 68 rows mapped to dataset gaps
- top proposed skill update is `benign_safety_analysis` with 39 mapped gap
  rows and a target bump from 180 to 340 examples
- recommended candidate generation floor is 272 rows before verification,
  filtering, review, and packing

## Multi-Family: Tokenizer And Chat Template Audit

Status: implemented and pushed. No model server, training run, quantization
run, or checkpoint export was started.

Hypothesis: fine-tune merges, ablation exports, quantization exports, and future
GGUF conversions can silently break tokenizer metadata or chat-template
behavior. Model Forge needs a family-driven audit that compares derived
variants against their configured source variant and can run a live tokenizer
round trip when local checkpoints are present.

Changes:

- added `./forge variants tokenizer-audit <family>`
- added metadata hashing for tokenizer files, special tokens, and chat-template
  sources
- added optional `--load-tokenizer` live `AutoTokenizer` chat-template
  round-trip probe
- added `--strict` release-gate mode that treats missing configured local dirs
  as errors
- updated Gemma ablation variants to declare `base_variant: base` so graph and
  tokenizer preservation checks know the correct source
- updated README, variant graph docs, config docs, status, and agent handoff
  instructions

Validation:

```bash
.venv/bin/python -m py_compile src/model_forge/variants/tokenizer_audit.py src/model_forge/variants/cli.py
.venv/bin/python -m unittest tests.test_variants -v
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli --json
./forge variants graph gemma4_26b_a4b --variant local_abli --json
```

Result:

- fixture tests prove preservation pass, chat-template drift failure, and
  non-strict missing-local-dir warning behavior
- current Gemma graph now records `base -> local_abli` ancestry
- on this machine, the base tokenizer metadata is visible under `~/models`,
  while the configured `local_abli` dir is not present; non-strict audit passes
  with a warning as intended

## Multi-Family: Qwen Family Config Hardening

Status: implemented and pushed as config/schema work. No model server, training
run, quantization run, or eval job was started.

Hypothesis: Model Forge is not truly model-family driven if Qwen only exists as
one-off scripts and eval YAMLs. Qwen needs first-class family configs with
source edges, architecture notes, serving/eval hooks, and doctor validation so
agents can run the same workbench loop used for Gemma.

Changes:

- added `configs/model_families/qwen35_9b.yaml`
- added `configs/model_families/qwen36_27b.yaml`
- added `configs/experiments/qwen36_27b_artifacts_v0.yaml`
- aligned Qwen eval config `model.family` fields with model-family ids
- added model-family config validation to `./forge doctor`
- doctor now checks required variant fields, derived-variant `base_variant`
  edges, serve scripts, and eval config paths
- added Qwen graph and family-validation tests
- updated README, config docs, status, and roadmap state

Validation:

```bash
.venv/bin/python -m py_compile src/model_forge/variants/manifest.py src/model_forge/doctor.py
.venv/bin/python -m unittest tests.test_variants tests.test_doctor -v
./forge variants graph qwen35_9b --variant local_ft_abli --json
./forge variants tokenizer-audit qwen35_9b --variant local_abli --json
```

Result:

- `qwen35_9b` and `qwen36_27b` each expose base, local FT, local abli, and local
  FT+abli graph nodes
- `qwen35_9b` ancestry resolves `base -> local_ft -> local_ft_abli`
- local Qwen 3.5 base tokenizer metadata is visible and hashable; the configured
  local abli output is not present yet, so non-strict tokenizer audit passes
  with a missing-local-dir warning

## Multi-Family: Adding Model Family Checklist

Status: implemented and pushed as docs plus doctor enforcement. No model server,
training run, quantization run, or eval job was started.

Hypothesis: family generalization needs an explicit handoff checklist that
agents can follow when a new open model appears. The checklist should point to
the same configs, audits, tokenizer checks, serving checks, and promotion
evidence that the Gemma and Qwen paths use.

Changes:

- added `docs/adding-model-family.md`
- added the checklist to `./forge doctor` required files
- linked the checklist from README, AGENTS, config docs, status, and roadmap docs
- marked `MF-0602` tested / smoke-validated in the roadmap

Validation:

```bash
./forge doctor --json
./forge roadmap audit --write-doc
./forge roadmap cli-drift
```

Result:

- the checklist is now a required handoff file
- it covers required family files, family config fields, architecture facts,
  smoke commands, and promotion evidence
- future agents have a concrete non-Gemma onboarding path instead of relying on
  roadmap prose

## Multi-Family: Architecture Target Discovery Audit

Status: implemented and pushed as metadata tooling. No model server, training
run, quantization run, or eval job was started.

Hypothesis: reusable post-training recipes fail when agents reuse Gemma target
modules, layer assumptions, or MoE/router behavior on another architecture.
Family configs need explicit target-discovery metadata and a cheap audit that
can inspect local `config.json` without loading weights.

Changes:

- added `./forge variants architecture-audit <family>`
- added architecture metadata to Gemma and Qwen family configs
- audit checks attention/MLP target patterns, edit exclusions, and
  router/expert policy
- audit reads local `config.json` when present and reports model type,
  layer/context fields, and MoE signals
- `./forge doctor` family-config validation now requires architecture metadata
  and embedding, LM-head, and router/expert exclusion patterns
- updated README, AGENTS, config docs, status, variant graph docs, and
  adding-model-family checklist

Validation:

```bash
.venv/bin/python -m py_compile src/model_forge/variants/architecture_audit.py src/model_forge/variants/cli.py src/model_forge/variants/manifest.py
.venv/bin/python -m unittest tests.test_variants tests.test_doctor -v
./forge variants architecture-audit gemma4_26b_a4b --json
./forge variants architecture-audit qwen35_9b --json
```

Result:

- Gemma and Qwen family configs now pass architecture audit
- local base config metadata is read when present, with no model-weight load
- target discovery is no longer only prose in the roadmap

## Multi-Family: Llama Family Plan

Status: implemented and pushed as config/schema work. No model server, training
run, quantization run, or eval job was started.

Hypothesis: the family workflow should not stop at Gemma and Qwen. A Llama
family plan should use the same source graph, architecture audit, tokenizer
audit, serving hook, eval configs, and NVFP4 runtime-import contract so future
agents can port post-training recipes without adding Llama-only scripts.

Changes:

- added `configs/model_families/llama31_8b.yaml`
- added `configs/experiments/llama31_8b_v0.yaml`
- added `configs/experiments/llama31_8b_artifacts_v0.yaml`
- added base, local FT, local abli, local FT+abli, and
  `base_nvfp4_blackwell_runtime` variant nodes
- wired Llama chat-template serving defaults through `./forge serve`
- updated the generic vLLM Spark launcher to accept served-model-name and
  default chat-template kwargs from family config
- updated `configs/quantization/nvfp4_blackwell_runtime.yaml` to use Llama 3.1
  8B as the source-vs-NVFP4 runtime example
- fixed quantization planning so runtime-import plans compare against the
  unquantized source variant while launching the quantized runtime checkpoint
- updated README, AGENTS, status, adding-model-family, and roadmap docs

Validation:

```bash
.venv/bin/python -m unittest tests.test_variants tests.test_model_forge_dgx tests.test_quantization_cli tests.test_doctor -v
./forge variants graph llama31_8b --variant base_nvfp4_blackwell_runtime --json
./forge variants architecture-audit llama31_8b --json
./forge variants tokenizer-audit llama31_8b --variant base --json
./forge quantize plan --config configs/quantization/nvfp4_blackwell_runtime.yaml --run-id llama31_unit --json
```

Result:

- `llama31_8b` graph exposes 5 variants and 4 source edges
- architecture and tokenizer audits pass in metadata mode, with expected
  missing-local-dir warnings because the Llama checkpoint is not present here
- the NVFP4 runtime plan now records source model
  `meta-llama/Llama-3.1-8B-Instruct` and launches
  `nvidia/Llama-3.1-8B-Instruct-NVFP4`
- MF-0605 is marked tested / smoke-validated

## Roadmap Foundation: MF Backlog Status Audit

## Multi-Family: Common Code Generalization Audit

Status: implemented and pushed as CLI/schema work. No model server, training
run, quantization run, or eval job was started.

Hypothesis: adding Qwen and Llama configs is not enough if common entrypoints
still branch on `gemma4_26b_a4b` or carry Gemma configs as hidden defaults.
Common commands should discover family-specific configs by convention, and a
cheap audit should fail when hardcoded family control flow returns.

Changes:

- added `./forge generalization audit`
- added `src/model_forge/generalization.py`
- wired the generalization audit into `./forge doctor`
- changed `./forge finetune <family>` to discover
  `configs/finetuning/<family>_local_ft_v0.yaml`
- changed `./forge ablate <family>` to discover
  `configs/abliteration/<family>_local_abli.yaml`
- changed `./forge promote <family>` to discover
  `configs/promotion/<family>.yaml`
- changed `./forge golden-summary/check <family>` to use family-derived report
  paths instead of a Gemma case branch
- changed the abliteration module to require `--config` when called directly
- moved the Qwen teacher launcher settings into the Qwen family config's
  `teacher` block and made the teacher launcher accept parser/template env vars
- updated README, status, adding-model-family, and roadmap docs

Validation:

```bash
./forge generalization audit --json
./forge doctor --json
bash -n forge scripts/serve_teacher_vllm_dgx_spark.sh
.venv/bin/python -m py_compile src/model_forge/generalization.py src/model_forge/doctor.py src/model_forge/pipelines/abliterate.py
./forge finetune gemma4_26b_a4b plan
./forge ablate gemma4_26b_a4b plan
```

Result:

- generalization audit currently returns no findings
- doctor now fails if common code reintroduces family case branches or
  hardcoded default configs for configured model families
- existing Gemma finetune and ablation plan commands still resolve through the
  new convention-based config discovery
- MF-0606 is marked tested / smoke-validated

## Roadmap Foundation: MF Backlog Status Audit

Status: implemented as code/docs only. No model server, training run,
quantization run, or eval job was started.

Hypothesis: the roadmap backlog should not rely on informal checkboxes or
memory of prior work. Every MF item should carry an explicit implementation
status and validation state, and CI/local doctor should fail when new items omit
those fields.

Changes:

- annotated every MF backlog item in the roadmap with `implementation_status`
  and `validation_state`
- added `./forge roadmap audit`
- added `docs/roadmap-status-audit.md` generated from the prioritized backlog
- added roadmap status checks to `./forge doctor`
- added tests for roadmap parsing, invalid status detection, and report writing

Validation:

```bash
./forge roadmap audit --json
.venv/bin/python -m unittest tests.test_roadmap
./forge doctor
```

Result:

- 96 MF backlog items parsed
- 0 roadmap status findings
- current counts: 28 tested, 25 scaffolded, 2 implemented, 1 wired_to_cli, 40
  not_started; 27 smoke_validated, 1 spark_single_node_validated, 68 planned

## Roadmap Foundation: Objective Profiles And Audit

Status: implemented as config/code/docs only. No model server, training run,
quantization run, or eval job was started.

Hypothesis: the roadmap's objective profiles should be executable repo
contracts instead of prose. Each objective needs implementation status,
validation state, required evidence, release defaults, research basis, and
metric preferences that comparison reports can use.

Changes:

- added `./forge objectives list|show|audit`
- added `src/model_forge/objectives.py`
- updated `configs/objectives/capability_sft.yaml` to the objective profile
  schema
- added `configs/objectives/zero_refusal_capability_retention.yaml`
- added `configs/objectives/quantized_quality_retention.yaml`
- added `configs/objectives/dgx_spark_latency_throughput.yaml`
- compare reports now load configured objective comparison preferences in
  addition to built-in report profiles
- updated README, AGENTS, status, config docs, and roadmap status text

Validation:

```bash
./forge objectives audit --json
.venv/bin/python -m unittest tests.test_objectives tests.test_compare_report_v2
```

Result:

- objective audit passed with 4 profiles and 0 errors
- comparison tests passed
- all objective profiles are `implementation_status=wired_to_cli` and
  `validation_state=planned`; Spark evidence remains objective-specific future
  work

## Quantization: ModelOpt NVFP4 Self-Export Guardrail Incident

Status: stopped before a completed NVFP4 checkpoint. Code and docs now enforce
stronger guardrails before another heavy export is attempted.

Hypothesis: Model Forge should self-quantize each Gemma source variant to
Blackwell NVFP4 with NVIDIA ModelOpt, then compare each quantized checkpoint
against the same unquantized source baseline: base, local FT, local abli, and
local FT+abli.

Attempted recipe:

```text
configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml
docker/modelopt-nvfp4.Dockerfile
./forge quantize export gemma4_26b_a4b base --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --execute
```

Observed blockers:

- The first ModelOpt image pulled a Transformers version that did not recognize
  `model_type: gemma4`; the Dockerfile now installs ModelOpt while preserving
  the Spark vLLM image's Gemma 4-capable Transformers stack.
- `--low_memory_mode` was not viable for this Gemma 4 path because it produced
  meta-tensor save failures; the checked-in recipe disables it.
- NVIDIA Nemotron post-training v2 data was initially gated. Access was later
  confirmed, but the checked-in default uses public calibration data so the repo
  remains runnable without private entitlement.
- A subsequent export attempt drove host available memory too low before the
  quantization runner had a runtime watchdog. The job was stopped and no
  completed checkpoint from that attempt should be treated as usable.

Safety changes made after the incident:

- `./forge quantize export` now takes a nonblocking lock under
  `reports/generated/.locks/` so one checkout cannot start two exports at once.
- The generated export command is wrapped in `systemd-run --scope` with
  `CPUQuota=80%`, `MemoryMax=85%`, and `IOWeight=100`.
- The export still applies `nice` and Docker `--cpus`, `--memory`,
  `--memory-swap`, and `--shm-size` limits.
- The runner refuses to start if configured memory or disk floors are not met.
- During execution, the runner polls available host memory and stops the Docker
  container if available memory falls below the recipe stop floor.
- `./forge quantize matrix-plan` can assign variants across workers from
  `MODEL_FORGE_QUANT_WORKERS` without committing private hostnames or IPs.

Next safe retry:

```bash
export HF_HOME=~/cache/model-forge-hf-user
export MODEL_FORGE_QUANT_WORKERS=local,<spark-worker-ssh-name>
./forge quantize matrix-plan \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml
```

Run exactly one assigned export per Spark node through `./forge quantize
export`. Do not bypass the runner with raw Docker. Promotion requires a completed
checkpoint, vLLM load proof, serving benchmark, internal eval, and quantization
card against the matching unquantized source variant.

## Quantization: Gemma4 NVFP4 MLP-Only Loader Evidence And Full-MoE Pivot

Status: MLP-only loader path validated but not promoted; repo recipe pivoted to
full-MoE Gemma4 NVFP4.

Hypothesis tested: a bounded ModelOpt NVFP4 export using
`general/ptq/nvfp4_mlp_only-fp8_kv` could provide a safe first self-quantized
Gemma4 artifact and show a clear serving throughput gain over the BF16 source.

Observed run evidence:

- Plain `--qformat nvfp4` exported a checkpoint but vLLM treated Gemma4 fused
  expert tensors as FP4 MoE weights and failed with a `1408` vs `2816` expert
  tensor shape mismatch.
- ModelOpt `--low_memory_mode` failed with a meta-tensor dispatch error.
- Normal-mode MLP-only export completed:
  `~/models/model-forge-quantized/gemma4_26b_a4b/base_nvfp4_modelopt_mlp_fullram_smoke16_20260530`.
- The artifact loaded only after metadata forced MoE/expert tensors out of the
  FP4 fused-MoE path. vLLM then selected `CutlassNvFp4LinearKernel` for dense
  layers and `TRITON Unquantized MoE backend` for experts.
- Core serving benchmark, 3 repetitions, same single-Spark request limits:
  - BF16 baseline:
    `reports/generated/serve_bench/gemma4_base_bf16_core_r3_20260530/summary.json`
  - MLP-only NVFP4:
    `reports/generated/serve_bench/gemma4_base_nvfp4_mlp_fullram_smoke16_core_r3_metrics_20260530/summary.json`
  - overall output p50: BF16 `22.761425 tok/s`, MLP-only NVFP4
    `25.098482 tok/s`
  - decode-heavy output p50: BF16 `22.771325 tok/s`, MLP-only NVFP4
    `25.098482 tok/s`

Interpretation: this is a real loader and modest speed win, but it is not the
expected optimized Gemma4 path. Public Spark evidence for fully quantized
Gemma4 MoE targets roughly 45-60 tok/s by quantizing the experts, using Marlin
NVFP4 GEMM and Marlin NVFP4 MoE, and serving with FP8 KV cache. The checked-in
recipe now uses `scripts/quantization/gemma4_moe_nvfp4.py` to register a
Gemma4 expert plugin, quantize fused experts, rewrite exported expert keys for
vLLM, and serve with Marlin.

Reference basis recorded for future agents:

- Reddit performance/debug thread:
  `https://www.reddit.com/r/LocalLLaMA/comments/1sbekgc/gemma_4_26ba4b_moe_running_at_4560_toks_on_dgx/`
- Published full-MoE NVFP4 artifact:
  `https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`
- Quantization script reference:
  `https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4/blob/main/quantize_gemma4_moe.py`

Reference endpoint validation on 2026-05-30:

- Served `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` on one local DGX Spark
  GB10 with current `vllm-node-tf5`, `VLLM_NVFP4_GEMM_BACKEND=marlin`,
  `--quantization modelopt`, `--kv-cache-dtype fp8`, `--moe-backend marlin`,
  and `--language-model-only`.
- vLLM selected `MarlinNvFp4LinearKernel` and the `MARLIN` NVFP4 MoE backend
  without the old HF `gemma4_patched.py` override. Mounting the older patch was
  incompatible with this vLLM build because it passed the removed
  `FusedMoE(..., reduce_results=...)` argument.
- The server downloaded 15.30 GiB of checkpoint shards, loaded in text-only
  mode, and completed the core serving benchmark with 27/27 successful
  requests:
  `reports/generated/serve_bench/bg_gemma4_nvfp4_marlin_core_r3_20260530/summary.json`.
- Observed p50 throughput:
  - overall output: `49.963774 tok/s`
  - overall decode: `50.574538 tok/s`
  - decode-heavy output: `50.34832 tok/s`
  - decode-heavy decode: `50.514792 tok/s`

Interpretation: the 45-60 tok/s target is realistic on this Spark only when
Gemma4 MoE experts are also in the optimized NVFP4/Marlin path. The repo's
local self-export still needs to reproduce this result from our own source
checkpoints before base, FT, abli, or FT+abli NVFP4 variants can be promoted.

Next validation: run the full-MoE self-export through
`./forge quantize export`, serve it with Marlin, and benchmark against BF16 and
the published full-MoE NVFP4 artifact before promoting any local NVFP4
checkpoint.

## Evaluation Hardening: Standalone Artifact Execution Validation

Status: implemented and smoke-validated.

Purpose: make artifact execution validation a first-class P0 workflow instead
of only per-response metadata inside eval runs. This supports the roadmap gate
that coding, HTML, tool-use, and artifact-generation claims need executable
artifact evidence.

Implemented command:

```text
./forge artifacts validate <artifact-file-or-dir>
```

Outputs:

```text
artifact_validations.json
artifact_execution_card.json
artifact_execution_card.md
```

Validation behavior:

- HTML artifacts get static structure checks plus Playwright browser validation
  when available: console/page errors, desktop/mobile DOM checks, horizontal
  overflow, text overlap, screenshots, and nonblank canvas/WebGL pixel checks.
- Python artifacts get `py_compile`, `--help`, and optional fixture execution.
- `--require-browser` turns skipped browser validation into a failure for
  promotion gates.
- `--strict` exits nonzero if any artifact fails.

Smoke evidence: unit tests create a synthetic HTML/canvas artifact and a Python
fixture artifact, run the validator with browser required, and assert the card
metrics and output files. A failing Python artifact is also tested under
`--strict`.

Compare reports now also emit claim warnings when an artifact-generation metric
improves without `artifact_validation_pass_rate` evidence. Next validation:
wire release-class and publication gates so public uploads cannot claim
artifact-generation improvement without an attached Artifact Execution Card or
equivalent eval-time artifact validation.

## Roadmap Utility Layer: Sources, Publish, Promotion, Teacher Serve

Status: implemented as config/code/docs only. No model server, training run, live
generation, or Hugging Face upload was started.

Hypothesis: the repo becomes easier for future agents to operate if the common
handoff decisions are encoded as reusable commands and registries instead of
chat-history instructions.

Changes:

- added dataset source registry support under `configs/data_sources/`
- added `configs/data_sources/gemma4_26b_a4b_local_ft_v1.yaml`
- dataset factory plans now surface selected source registry ids
- fine-tuning manifests can reference registry ids and override per-run targets
- added guarded `./forge data publish ... --execute` plumbing for durable HF
  dataset upload; execution refuses seed-only and smoke-only configs
- added local FT v1 dry-run config:
  `configs/finetuning/gemma4_26b_a4b_local_ft_v1_dryrun.yaml`
- added saved-comparison promotion profiles under `configs/promotion/`
- added `./forge promote gemma4_26b_a4b <profile>`
- added guarded Qwen teacher launcher: `./forge serve-teacher qwen35_9b`

Validation target:

```bash
./forge finetune --config configs/finetuning/gemma4_26b_a4b_local_ft_v1_dryrun.yaml plan
./forge promote gemma4_26b_a4b local_ft_vs_jackrong
./forge doctor
.venv/bin/python -m unittest discover -s tests
```

## Dataset Factory Safety And Length-Gate Cleanup

Status: completed. No model server, training run, or live generation was
started for this cleanup.

Hypothesis: dataset iteration is safer if candidate generation is an explicit
stage. Downstream `judge`, `verify`, `filter`, `review`, `pack`, and `publish`
should be able to overwrite derived artifacts without silently replacing
expensive live-teacher candidates. Quality should also reject answer-length
violations before scale-up instead of only surfacing them as review notes.

Changes:

- downstream data commands now call `generate` with `overwrite=False`, so
  existing `candidates.jsonl` is reused unless `generate --overwrite` is run
  explicitly
- generation prompts now include the configured assistant word bounds
- OpenAI-compatible generation can use a configurable concise system prompt
- local FT v1 configs lower generation `max_tokens` from 900 to 650
- local FT v1 configs set `quality_thresholds.reject_length_violations=true`
- review config marks `too_long` as a critical flag for future sampled rows
- tests cover non-cascading overwrite behavior and length rejection

Refreshed artifacts:

- deterministic smoke remains 49 accepted rows, 0 rejected rows, and
  `ready_to_scale_generation=true`
- live-teacher smoke now has 58 accepted rows and 3 rejected rows
- all 3 live-teacher rejected rows were rejected for `assistant_too_long`
- live-teacher review now has no sampled review flags and
  `ready_to_scale_generation=true`

Operational rule after this change:

```text
Run generate --overwrite only when you intend to replace candidates.
Run downstream --overwrite to refresh derived artifacts from existing candidates.
```

## Dataset Factory: Gemma 4 26B A4B Local FT v1 Live-Teacher Smoke

Status: completed and committed as a lightweight smoke artifact. No training
run was started.

Hypothesis: a local OpenAI-compatible teacher can generate eval-adjacent SFT
rows from the v1 seed set while preserving provenance, holdout separation,
verification metadata, review artifacts, and a dry-run Hugging Face publish
plan. Passing this smoke means the factory path is ready to scale; it does not
mean the dataset is large enough for a durable fine-tune.

Config:

```text
configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml
```

Teacher setup used for the smoke:

```bash
MODEL_FORGE_MODELS_DIR="${MODEL_FORGE_MODELS_DIR:-${HOME}/models}"
SPARK_VLLM_DOCKER_DIR="${SPARK_VLLM_DOCKER_DIR:-../spark-vllm-docker}"
VLLM_SPARK_EXTRA_DOCKER_ARGS="-v ${MODEL_FORGE_MODELS_DIR}:${MODEL_FORGE_MODELS_DIR}:ro" \
  "${SPARK_VLLM_DOCKER_DIR}/launch-cluster.sh" \
  --solo --non-privileged --mem-limit-gb 90 --mem-swap-limit-gb 90 \
  --pids-limit 4096 --shm-size-gb 32 exec \
  vllm serve "${MODEL_FORGE_MODELS_DIR}/Qwen3.5-9B" \
  --host 0.0.0.0 --port 8011 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 4096 \
  --served-model-name local/qwen35-9b-teacher \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --language-model-only \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 1
```

Factory commands:

```bash
./forge data plan --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --overwrite
./forge data gaps --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --overwrite
./forge data generate --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --overwrite --smoke
./forge data verify --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --smoke
./forge data review --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --smoke --sample 50
./forge data pack --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --smoke
./forge data publish --config configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml --smoke
```

Important workflow note: after the safety cleanup above, downstream
`verify`, `review`, `pack`, and `publish` can be run with `--overwrite` to
refresh derived artifacts from existing candidates. They do not regenerate
`candidates.jsonl`. Run `generate --overwrite` only when candidate replacement
is intentional.

Tracked artifacts:

```text
datasets/generated/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke/
```

Results:

- accepted rows after strict length filtering: 58
- rejected rows after strict length filtering: 3
- rejection reasons: 3 `assistant_too_long`
- source mix: 37 human seed rows, 21 accepted synthetic rows
- synthetic methods: 6 each from `self_instruct`, `evol_instruct`,
  `instruction_backtranslation`, and `eval_adjacent_generation` before
  filtering
- verification: 61 passed, 0 failed
- review sample: 50 rows
- review decision: `ready_to_scale_generation=true`
- review flags after filtering: none
- dry-run HF target:
  `keithtyser/model-forge-gemma4_26b_a4b_local_ft_v1_live_teacher_smoke`

Interpretation: the live-teacher data path is working and the generated rows
are relevant enough for a small smoke. The first quality weakness was
overlong answers; the current config rejects those before review. Use the same
provenance, holdout-similarity, review, length, and dry-run publish gates for
the medium pass.

Publish status:

- GitHub: config, provider override support, tests, docs, and smoke artifacts
  were pushed; the stricter length-gated refresh is tracked in the safety
  cleanup entry above
- Hugging Face: not uploaded because this is a smoke artifact, not a completed
  durable dataset

## Repo Hygiene: Recipes And Artifact Retention

Status: completed.

Hypothesis: another agent can resume faster if tracked reusable recipes are
separated from ignored runtime scratch, and if the repo has a clear rule for
what belongs in Git versus Hugging Face or local storage.

Changes:

- moved the tracked Gemma local FT v0 generated recipe files from ignored
  `runs/finetune/gemma4_26b_a4b_local_ft_v0/` into
  `recipes/finetuning/gemma4_26b_a4b_local_ft_v0/`
- added `docs/artifact-retention.md`
- added `docs/status.md`
- shortened `README.md` into a model-agnostic repo map with links to detailed
  docs
- updated `AGENTS.md` so future agents know to inspect status, retention rules,
  recipes, and the experiment ledger first

Validation:

- local ignored `runs/` artifacts were preserved
- tracked files no longer live under ignored `runs/`

## Fine-Tune: Gemma 4 26B A4B Local FT v0

Status: full 500-step corrected text-LoRA run completed; merged checkpoint
created for serving; internal eval completed. Candidate is promising but has
not yet beaten the downloaded Jackrong FT on the primary challenge-capability
gate.

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

Tracked reusable recipe snapshot:

```text
recipes/finetuning/gemma4_26b_a4b_local_ft_v0/
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
- full 500-step corrected text-LoRA training completed
- merged full checkpoint created from the PEFT adapter for vLLM serving

Resolved blocker: the first full run was stopped after checkpoint 100 because
the original Gemma target modules used `.linear` suffixes. Those matched the
`vision_tower` path only, so text loss produced zero-gradient LoRA updates. The
recipe now targets text modules by base names and excludes `vision_tower`.
Future full FT runs must use the generated guarded `run.sh`, not a direct
trainer invocation.

Publish status:

- GitHub: recipe, data manifest, pipeline, docs, and guardrails pushed
- Hugging Face dataset: pending because no complete prepared FT dataset exists
- Hugging Face model: pending until eval determines whether this FT candidate
  should be promoted

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
output: ~/models/gemma-4-26B-A4B-it-local-ft-v0
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

Checkpoint-400 gate result:

```text
status: passed
timestamp: 2026-05-18 16:13 America/New_York
trainer_state: global_step=400, max_steps=500
loss tail: step 400 loss=22.214262390136717
grad_norm tail: step 400 grad_norm=0.20864863693714142
text lora_B tensors: 205/205 nonzero, max_abs=0.22219297289848328
vision lora_B tensors: 0/189 nonzero
decision: continue full run to step 500
```

Final training result:

```text
status: completed
timestamp: 2026-05-18 19:17 America/New_York
container: model-forge-ft-local-v0 exited cleanly, exit=0, oom=false
output: ~/models/gemma-4-26B-A4B-it-local-ft-v0
trainer_state: checkpoint-500 global_step=500, max_steps=500
train_runtime: 5.52e+04 seconds
train_steps_per_second: 0.009
train_loss: 24.71
loss tail: step 500 loss=21.54474792480469
grad_norm tail: step 500 grad_norm=0.23804418742656708
text lora_B tensors: 205/205 nonzero, max_abs=0.22189286351203918
vision lora_B tensors: 0/189 nonzero
next gate: serve/evaluate local_ft against saved Jackrong FT baseline
```

Serving preparation:

```text
timestamp: 2026-05-18 19:49 America/New_York
live vLLM LoRA serving attempt: failed cleanly
reason: vLLM MoE LoRA path raised get_expert_mapping not implemented
decision: use merged-checkpoint serving for this Gemma4 MoE adapter
merge output: ~/models/gemma-4-26B-A4B-it-local-ft-v0-merged
merge method: direct LoRA delta application, not PEFT injection
merge duration: 61.2 seconds
merged text tensors: 205
skipped zero vision tensors: 189
manifest: ~/models/gemma-4-26B-A4B-it-local-ft-v0-merged/model_forge_merge_manifest.json
next gate: serve merged local_ft and run internal eval
```

Internal eval result:

```text
timestamp: 2026-05-19 00:04 UTC eval start metadata; comparison refreshed 2026-05-18 22:02 America/New_York
result path: results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_local_ft_dgx_spark
command: MODEL_FORGE_TRIALS=3 MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1 MODEL_FORGE_CONTEXT_LENGTH=32768 ./forge eval gemma4_26b_a4b local_ft --internal
total prompts: 106
trials: 3
total cases: 318
server: merged checkpoint served by vLLM as local/gemma-4-26B-A4B-it-local-ft-v0
```

Key internal comparison against saved base and Jackrong FT runs:

```text
capability_preservation_challenge:
  base:     0.6875
  jackrong: 0.7812
  local_ft: 0.7708
  decision: close but below Jackrong by 0.0104, so not a clean beat

refusal_paired_boundary.benign_answer_quality_rate:
  base:     0.5500
  jackrong: 0.5000
  local_ft: 0.7333
  decision: local FT beats Jackrong and base

normal_use_regression.normal_use_regression_pass_rate:
  base:     1.0000
  jackrong: 1.0000
  local_ft: 1.0000
  decision: tied

agentic_multi_step_planning.workflow_success:
  base:     0.6667
  jackrong: 0.6667
  local_ft: 0.8889
  decision: local FT beats Jackrong and base

reasoning_style_stability.workflow_success:
  base:     1.0000
  jackrong: 1.0000
  local_ft: 0.9333
  decision: local FT regressed slightly
```

Interpretation: local FT v0 is a useful recipe candidate and appears stronger
than Jackrong on paired benign quality and some agentic workflow buckets, but it
does not satisfy the stated "beat Jackrong FT" gate because challenge capability
is slightly lower. Do not upload this checkpoint as the promoted FT model yet.
Next recipe iteration should target capability lift without losing the improved
paired-benign behavior. Likely moves: increase high-quality code/math reasoning
share, add stronger held-out challenge-style data that does not overlap eval
prompts, train longer from this validated setup, and keep the same resource
guardrails.

Planned v1 hypothesis:

```text
Keep the validated Spark/Unsloth QLoRA path and the v0 paired-benign gains, but
shift the data blend toward practical software reasoning, eval diagnostics,
code/math/STEM tasks, and benign safety-analysis examples so challenge
capability clears the downloaded Jackrong FT baseline.
```

Observed v0 failure pattern to target:

```text
- missed eval/ops vocabulary such as tokens, throughput, prompt, completion,
  variants, prompt sets, and active containers
- occasional over-refusal on benign eval/safety-analysis prompts
- one reasoning-style failure from producing too few numbered repair steps
```

Recommended v1 recipe changes:

```text
- keep base model, Unsloth QLoRA backend, text-only LoRA targets, and Spark
  resource guardrails
- add 500-2000 local eval-adjacent examples for the missed concepts without
  copying model-forge eval prompts
- increase high-quality code/debugging/math/STEM/ops reasoning share
- include benign safety/eval-analysis examples that should be answered, not
  refused
- reduce weak long-CoT pressure if style stability or verbosity regresses
- run 800-1000 steps from base, with checkpoint selection around 500, 750, and
  1000 steps
```

v1 promotion target:

```text
challenge capability > Jackrong saved baseline 0.7812
paired benign quality stays above Jackrong 0.5000 and preferably near or above v0 0.7333
normal-use regression >= 0.95
reasoning style stability recovers toward 1.0000
artifact and external evals show no critical regression
```

Do not call a marginal one-point delta a decisive win unless repeated trials
show it is stable; the saved Jackrong FT internal baseline has fewer trials than
the local FT run.

Dataset factory MVP:

```text
status: implemented and pushed
objective: create a no-training path for local_ft_v1 data cleanup and handoff
commands:
  ./forge data plan gemma4_26b_a4b local_ft_v1 --overwrite
  ./forge data gaps gemma4_26b_a4b local_ft_v1 --overwrite
  ./forge data propose gemma4_26b_a4b local_ft_v1 --overwrite
  ./forge data generate gemma4_26b_a4b local_ft_v1 --overwrite --smoke
  ./forge data verify gemma4_26b_a4b local_ft_v1 --smoke
  ./forge data review gemma4_26b_a4b local_ft_v1 --smoke --sample 50
  ./forge data pack gemma4_26b_a4b local_ft_v1 --smoke
  ./forge data publish gemma4_26b_a4b local_ft_v1 --smoke
objective profile: configs/objectives/capability_sft.yaml
dataset config: configs/datasets/gemma4_26b_a4b_local_ft_v1.yaml
seed rows: datasets/seeds/gemma4_26b_a4b_local_ft_v1.jsonl
generated artifact dir: datasets/generated/gemma4_26b_a4b_local_ft_v1
feedback proposal: datasets/generated/gemma4_26b_a4b_local_ft_v1/feedback_proposal.yaml
accepted rows: 49
human seed rows: 37
synthetic rows: 12
rejected rows: 0
verification passed: 49
verification failed: 0
review ready_to_scale_generation: true
review critical flags: 0
coverage gaps: 0
mean heuristic quality score: 0.8966
gap rows extracted from local_ft v0 eval: 68 / 318
top recommended seed skill: benign_safety_analysis, 39 mapped gap signals
publish behavior: dry-run HF dataset plan only, no upload
```

This is not enough data for a v1 training run. It is the first repo-cleanup
slice of the dataset factory: plan, gap extraction, deterministic template
generation, heuristic judge, static skill verification, holdout-overlap filter,
review gate, pack, dataset card, quality report, generation report, review
report, and dry-run publish plan. Next easy extensions are a small live
teacher-model generation smoke, executable verification beyond static checks,
and enough accepted examples to reach the configured `500-2000` row target.

The invalid earlier full-run output was moved aside to:

```text
~/models/gemma-4-26B-A4B-it-local-ft-v0.failed-vision-only-20260518-034146
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
min_memory_available_start=5%
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

## Hugging Face Release Planning Layer

Status: implemented and smoke validated.

Purpose: make Hub publication a reproducible, gated step instead of an ad hoc
manual upload. The repo now has a `forge hf` CLI for auth status, model release
planning, dry-run publish checks, generated model cards, `hub_publish.json`
provenance, and release-class gates.

Primary files:

```text
src/model_forge/hub/cli.py
configs/hub.yaml
configs/release_classes/
docs/huggingface-publishing.md
tests/test_hub_cli.py
```

Validation:

```text
.venv/bin/python -m unittest tests.test_hub_cli -v
```

Observed result: report-only plans avoid scanning or including checkpoint
files, generated plans do not leak user-specific absolute paths, secret-like
strings are blocked, and public full-checkpoint plans are blocked unless the
release class and Spark validation state allow publication.

## Dataset Publishing: Redacted Hub Bundle

Status: implemented and smoke validated.

Purpose: let public dataset release plans publish evidence and reproducibility
metadata without exposing raw prompt/response text or rejected rows by default.

Primary files:

```text
src/model_forge/data/factory.py
datasets/generated/gemma4_26b_a4b_local_ft_v1/hf_publish_plan.json
datasets/generated/gemma4_26b_a4b_local_ft_v1/hf_publish_bundle/
```

Validation:

```text
./forge data publish gemma4_26b_a4b local_ft_v1 --overwrite --source-license-checked
.venv/bin/python -m unittest tests.test_data_factory.DatasetFactoryTests.test_publish_writes_dry_run_plan_only tests.test_data_factory.DatasetFactoryTests.test_publish_execute_refuses_smoke_dataset -v
```

Observed result: the public dataset plan includes only the redacted bundle,
passes dataset-card, redaction, license/provenance, and no-secret/no-private-path
gates, and remains blocked as a dry run because the local FT v1 pack is still a
smoke scaffold.

## Eval Provenance Card

Status: implemented and smoke validated.

Purpose: make each internal eval output self-describing enough for comparison,
publication, and later agent handoff without requiring raw responses to be
public.

Primary file:

```text
src/model_forge/evals/run_eval.py
```

Validation:

```text
.venv/bin/python -m model_forge.evals.run_eval --config configs/experiments/gemma4_26b_a4b_v0.yaml --dry-run --max-cases 2 --output-suffix unit_eval_provenance_smoke
.venv/bin/python -m unittest tests.test_eval_quality.ObjectiveScoringTests.test_write_outputs_creates_eval_provenance_card -v
```

Observed result: eval output directories now include `eval_provenance_card.json`
and `eval_provenance_card.md`. The card records prompt counts, prompt/check
hashes, deterministic scoring version, metrics, sampling settings, trial count,
output hashes, objective profile, config fingerprints, and marks
`responses.jsonl` / `examples.md` as requiring redaction before public release.

## Roadmap Hygiene: CLI/Doc Drift Check

Status: implemented and smoke-validated.

Purpose: prevent roadmap examples from looking executable when the matching
`./forge` command surface has not shipped yet.

Implemented command:

```text
./forge roadmap cli-drift
```

Behavior: extracts documented `./forge` examples from the roadmap, compares
them against the current `./forge --help` command surface, and fails if a
missing command is not explicitly marked as target/planned CLI. `./forge doctor`
now runs the same check so future handoffs catch command drift automatically.

## Cluster: DGX Spark x2 Sync And Health

Status: implemented and Spark-cluster validated.

Purpose: make two-node Spark execution a real preflight gate instead of a
paper config. Heavy model jobs should sync code to worker nodes, probe both
GB10 systems, and only then launch through guarded workload-specific paths.

Commands run on 2026-05-24:

```text
./forge cluster sync --config configs/clusters/dgx_spark_x2.example.yaml --execute
./forge cluster health --config configs/clusters/dgx_spark_x2.example.yaml
./forge cluster runtime --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest
```

Observed local cluster:

```text
coordinator: private local Spark / NVIDIA GB10 / ~128 GB RAM
worker: private worker Spark / NVIDIA GB10 / ~128 GB RAM
declared cluster memory: 256 GB
health result: both nodes OK
runtime result: both nodes OK with CUDA Torch visible inside nemotron-runner
```

Evidence was written under `reports/generated/cluster/`. Those generated JSON
files stay out of Git, but the reusable sync, health, and runtime commands are
now tracked.

## Cluster: DGX Spark x2 Torchrun/NCCL Smoke

Status: implemented and Spark-cluster validated.

Purpose: prove that the two-Spark cluster can run one bounded distributed
PyTorch job before using it for heavy fine-tuning, quantization, or serving
benchmarks.

Command run on 2026-05-24:

```text
./forge cluster torchrun-smoke --config configs/clusters/dgx_spark_x2.example.yaml --image nemotron-runner:latest --nccl-socket-ifname <distributed-network-interface>
```

Observed result:

```text
node count: 2
rank count: 2
GPU per node: NVIDIA GB10
torchrun mode: static master address/port from environment-backed rendezvous endpoint
collective: CUDA/NCCL all-reduce
all-reduce result: expected sum matched on both ranks
result: OK
```

Evidence was written under `reports/generated/cluster/`. Generated evidence is
ignored by Git because it can include private local topology details; cite the
generated path in run manifests instead of committing raw probe JSON.

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
