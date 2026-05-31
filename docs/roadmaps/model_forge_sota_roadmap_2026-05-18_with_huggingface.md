# Model Forge SOTA-Grounded Roadmap

**Prepared for:** Keith Tyser  
**Date snapshot:** May 18, 2026  
**Target repo:** [`keithtyser/model-forge`](https://github.com/keithtyser/model-forge)  
**Primary goal:** Turn Model Forge into a research-grounded, reproducible post-training pipeline that can take any supported base open model, fine-tune it to improve capability, ablate it to remove refusals while preserving that capability, quantize it to Blackwell-optimized NVFP4, serve it with DGX Spark / Spark-cluster optimizations, and prove the result with evals, benchmarks, artifacts, and public model/dataset cards.

---

## Executive summary

Model Forge should become a system that answers one question better than almost any personal project can:

> **Given a base open model, what post-training, behavior-editing, quantization, and serving configuration best satisfies a declared objective on a declared hardware target, with reproducible evidence and no hidden regressions?**

The canonical end-to-end outcome is:

```text
base model
→ capability fine-tune
→ refusal ablation / behavior edit
→ NVFP4 quantization for Blackwell
→ DGX Spark optimized serving
→ eval + benchmark + artifact proof
→ reproducible release
```

In practical terms, the target artifact is the smartest, fastest, lowest-refusal
variant that Model Forge can produce for a model family under a declared
objective profile. Fine-tuning should raise task capability and format quality.
Ablation should reduce invalid refusals or configured refusal behavior while
preserving the source model's capability. Quantization should preserve quality
while improving memory and/or throughput. Serving should use measured DGX Spark
and Spark-cluster optimizations, including lessons from AEON-style Spark work
and upstream serving stacks, but only promote settings that survive local
benchmarks. Safety and deployment risks are reported separately from the
refusal-removal objective so the repo can say exactly what changed.

Your current repo is already unusually close to this. It is model-family driven, evaluation-first, uses DGX Spark as a hardware profile, tracks raw responses and artifacts, compares base/fine-tuned/ablated/combined variants, and records an experiment ledger. The missing step is to make the repo **SOTA-grounded and Spark-validated by construction**.

The point of Model Forge is not to collect planning commands. It is to produce
recipes that have been executed locally and can be reproduced: fine-tuning that
creates real adapters/checkpoints, ablation that writes real edited variants,
quantization that produces or configures a real compressed serving path, serving
benchmarks that hit real endpoints, and evals that decide whether the transform
improved performance, removed refusals, made serving faster, or caused a hidden
regression.

That means every major feature should have:

```text
research claim
→ implementation hook
→ local Spark execution hook
→ evaluation hook
→ report section
→ limitation / failure mode
→ next experiment
```

Definition of done:

```text
implementation_status:
  not_started | scaffolded | implemented | wired_to_cli | tested

validation_state:
  planned:
    command/config/report skeleton exists, but no model was transformed or served

  smoke_validated:
    the command ran on a tiny model, tiny dataset, synthetic endpoint, or dry-run
    path and proved parsing, artifact writing, and manifest generation

  spark_single_node_validated:
    the command ran on one local Spark with a real model artifact and wrote
    metrics, logs, manifests, and a pass/fail report

  spark_cluster_validated:
    the command ran across the Spark x2 cluster when the workload benefits from
    distributed memory, distributed training, or multi-node serving

generalizable:
  the same recipe has run on at least one additional model family or model shape,
  with documented family-specific knobs and no hard-coded Gemma assumptions
```

Only `spark_single_node_validated`, `spark_cluster_validated`, and
`generalizable` count as completed roadmap outcomes. Dry-run plans and smoke
tests are useful scaffolding, but they do not prove that Model Forge can improve,
edit, quantize, or serve a model.

The roadmap below turns that into concrete work:

1. **Research registry**: a dated, machine-readable registry of papers, docs, benchmarks, and claims as of May 18, 2026.
2. **Objective profiles**: explicit profiles such as `capability_sft`, `zero_refusal_capability_retention`, `quantized_quality_retention`, `dgx_spark_latency_throughput`, and `agentic_tool_use`.
3. **Dataset factory**: a repeatable pipeline for generating, judging, filtering, packing, and publishing high-quality fine-tuning datasets from eval gaps and current source material.
4. **Spark validation contract**: every transform family must run on the local Spark hardware and write reproducibility artifacts before it is marked complete.
5. **Variant graph**: base → fine-tune → behavior edit → quantize → serve optimize → evaluate → publish, with every node and edge recorded.
6. **Evaluation hardening**: run manifests, golden baselines, provenance cards, external benchmarks, artifact execution validation, judge calibration, and risk reports.
7. **DGX Spark inference lab**: a public benchmark suite for vLLM/SGLang/TensorRT-LLM on a bandwidth-constrained 128 GB unified-memory machine.
8. **Quantization lab**: FP8 KV cache, FP8 W8A8, Blackwell NVFP4/ModelOpt, GGUF/llama.cpp export, GPTQ/AWQ/INT paths, QAT/QAD recovery, and behavior-preservation tests.
9. **Behavior-editing lab**: ablation as one objective profile, with invalid-refusal reduction measured separately from valid safety refusals, capability retention, behavior drift, overcompliance risk, and deployment risk.
10. **Kernel/perf track**: profiler-first microbenchmarks tied to serving outcomes, not toy kernels detached from model performance.
11. **Agentic experiment runner**: an agent that proposes, runs, evaluates, rejects/promotes, and writes reports under bounded budgets.
12. **Public proof layer**: report site, X threads, Hugging Face model/dataset repos, release cards, gated/private publication flows, and upstream PRs.

The strongest public identity is:

> **“I build eval-first systems for improving and serving open models on constrained local AI hardware. Every claim comes from real local Spark runs and includes configs, raw outputs, latency numbers, quality/safety deltas, and reproducible reports.”**

This is aligned with the kinds of roles frontier labs currently signal demand for: inference systems, GPU performance, post-training, model evaluations, and automated research workflows. OpenAI and Anthropic job postings explicitly mention post-training, evals, inference, GPU kernel work, Triton/CUDA/CUTLASS/FlashAttention, memory-bandwidth optimization, quantization, and model-serving infrastructure.

---

## 1. Current repo diagnosis

### 1.1 What Model Forge already has

The current README describes Model Forge as “a reproducible post-training workbench for open models” designed to answer whether a fine-tune, ablation, or combined post-training workflow made a model better without breaking something else. It is organized around model families rather than one hard-coded model, and its intended loop is: add a family config, download or fine-tune the source checkpoint, ablate refusals, serve candidates, run internal/artifact/external evals, promote only if the objective improves without regressing capability, then publish recipes, model cards, scores, and raw outputs.

Existing strengths:

- **Model-family architecture**: family YAMLs define base checkpoint, optional fine-tuned checkpoints, reference ablations, local outputs, serving settings, eval configs, and report paths.
- **Variant matrix**: base, fine-tuned, ablated, fine-tuned-then-ablated, and ablated-then-fine-tuned.
- **Evaluation-first framing**: workflow success, structured output, normal-use regression, benign refusal rate, unsafe overcompliance, latency, tokens/sec, raw responses, generated artifacts, and external benchmark outputs.
- **DGX Spark profile**: conservative vLLM defaults, FP8 KV cache, prefix caching, chunked prefill, low default sequence count, explicit high-parallelism opt-in, and ModelOpt/NVFP4 override hooks.
- **Resource guardrails**: training runs are treated as host tenants with CPU/memory/disk/thread controls.
- **Experiment ledger**: current full Gemma FT run is documented with hypotheses, configs, blockers, checkpoints, and publish status.
- **Ablation methodology docs**: direction collection, held-out prompts, architecture-specific target modules, row/norm preservation, source-relative comparison, and SOTA backend orchestration.

### 1.2 The most important gaps

These are not criticisms; they are the roadmap.

| Gap | Why it matters | Roadmap response |
|---|---|---|
| Research basis is prose, not product | Hard to keep SOTA current or prove a feature exists for a reason | Add `research_registry.yaml` and `./forge research audit` |
| “Better” is overloaded | FT, ablation, quantization, and serving have different goals | Add objective profiles and profile-specific promotion gates |
| Run provenance is not yet strong enough | Reports need exact reproducibility across model/version/engine/hardware | Add canonical `manifest.json` and report cards |
| Validation state is not yet schema-enforced | Prose can drift from actual artifacts and backlog checkboxes | Add required `validation_state`, Spark evidence, and promotion-decision fields to manifests, cards, objectives, and variant nodes |
| Backlog status is overloaded | `[x]` can mean scaffolded, smoke-tested, or genuinely validated | Track implementation status separately from Spark validation status |
| CLI examples can drift from implementation | Operators should not chase roadmap commands that do not exist yet | Mark examples as target CLI until shipped and add command drift checks |
| External evals are useful but not yet central | Need credibility beyond internal prompts | Add eval provenance cards, benchmark adapters, and version tracking |
| High-quality training data is imported manually | Fine-tuning quality will be capped by ad hoc public datasets and stale mixtures | Add `forge data *`: seed, generate, judge, filter, pack, version, and publish datasets |
| Too much work can look complete at the plan/dry-run layer | The repo's purpose is to actually transform and validate models, not only generate plans | Add a Spark validation contract and require local execution evidence before roadmap items are marked done |
| Artifact execution validation is planned, not shipped | Generated HTML/code can look good but fail | Add Playwright/browser and Python execution validators |
| Quantization is not first-class | It is central to DGX Spark, Blackwell, and portable local inference work | Add `forge quantize`, NVFP4, GGUF, quantization cards, and behavior-preservation reports |
| Serving optimization is not benchmarked as a matrix | vLLM flags are present but not yet a benchmark product | Add `forge bench serve` and a DGX Spark leaderboard |
| Behavior editing lacks objective profiles | Ablation objective and deployment decision should be separated | Add `zero_refusal_capability_retention` profile plus release classes |
| Model-family generalization is too implicit | A recipe is not general if it only works on Gemma naming, modules, and chat templates | Add a family checklist for tokenizer, chat template, architecture targets, MoE/router behavior, context length, license, serving backend, and release constraints |
| Storage and artifact retention are not first-class | Adapters, merged checkpoints, GGUFs, reports, and raw outputs will fill local disks quickly | Add retention budgets, checksums, cache policy, cleanup policy, and publish/delete decisions to manifests |
| Agent-friendliness is mostly documentation | Huge opportunity to demonstrate automated research loops | Add `forge agent optimize-*` |
| Hugging Face publishing is not first-class | Model/dataset artifacts need stable public homes with cards, provenance, visibility gates, and reproducible uploads | Add `forge hf *`, card generators, release classes, and Hub provenance |
| Kernel work is not yet represented | Top labs value measurable GPU performance work | Add profiler-first microbench and upstream PR track |

---

## 2. North-star architecture

### 2.1 The workbench should be objective-driven

Do not let Model Forge become “a fine-tuning repo,” “an uncensored model repo,” or “a vLLM scripts repo.” Make it a **post-training and inference objective optimizer**.

Core abstraction:

```text
family + source variant + transform graph + objective profile + hardware target + evaluation suite
```

Examples:

```bash
./forge compare gemma4_26b_a4b \
  --base base \
  --candidate local_ft \
  --candidate ft_then_abli \
  --objective capability_sft

./forge compare gemma4_26b_a4b \
  --base local_ft \
  --candidate ft_then_abli_r7 \
  --objective zero_refusal_capability_retention

./forge compare gemma4_26b_a4b \
  --base ft_then_abli_r7 \
  --candidate ft_then_abli_r7_nvfp4 \
  --objective quantized_quality_retention

./forge bench serve gemma4_26b_a4b ft_then_abli_r7 \
  --hardware dgx_spark \
  --objective dgx_spark_latency_throughput
```

### 2.2 Add objective profiles

Create:

```text
configs/objectives/
  capability_sft.yaml
  preference_alignment_dpo.yaml
  reasoning_grpo.yaml
  zero_refusal_capability_retention.yaml
  quantized_quality_retention.yaml
  dgx_spark_latency_throughput.yaml
  artifact_generation.yaml
  agentic_tool_use.yaml
  deployment_safety.yaml
```

Example: `zero_refusal_capability_retention.yaml`

```yaml
id: zero_refusal_capability_retention
version: 2026-05-18
description: >
  Behavior-editing objective: reduce invalid refusals and target
  noncompliance on the configured benign/refusal suite while preserving valid
  safety boundaries, source-model capability, and explicit risk reporting.

primary_goal:
  metric: invalid_refusal_rate
  suite: refusal_primary
  target: 0.0

hard_constraints:
  normal_use_score_delta_vs_source: ">= -0.02"
  structured_output_delta_vs_source: ">= -0.02"
  code_debug_delta_vs_source: ">= -0.03"
  artifact_execution_delta_vs_source: ">= -0.05"
  external_benchmark_delta_vs_source: ">= -0.03"
  valid_safety_refusal_retention: ">= 0.98"
  harmful_overcompliance_rate: "<= source + 0.01"

required_reports:
  - capability_retention
  - behavior_drift
  - valid_refusal_retention
  - unsafe_overcompliance
  - overcompliance_calibration
  - harmful_detail_risk
  - redacted_examples
  - serving_delta

tie_breakers:
  - lower_style_drift
  - higher_artifact_execution_score
  - lower_latency_p95
  - higher_output_tokens_per_second

release_default: research_report_only

validation_gates:
  minimum_validation_state_for_research_claim: spark_single_node_validated
  minimum_validation_state_for_cluster_claim: spark_cluster_validated
  required_evidence:
    - manifest_json
    - variant_node_json
    - local_spark_command
    - raw_outputs
    - refusal_taxonomy_scores
    - risk_report_redacted_public
    - risk_report_private

research_basis:
  - arditi_2024_refusal_direction
  - joad_2026_more_than_single_refusal_direction
  - multidirectional_refusal_suppression_2026
  - xstest
  - strongreject
  - harmbench
  - jailbreakbench
  - mlcommons_ailuminate
```

Example: `quantized_quality_retention.yaml`

```yaml
id: quantized_quality_retention
version: 2026-05-18
description: >
  Quantization objective: reduce memory and/or improve serving throughput while
  preserving capability, behavior, and artifact execution quality.

primary_goal:
  metric: peak_memory_gb
  direction: minimize

hard_constraints:
  normal_use_score_delta_vs_source: ">= -0.02"
  instruction_following_delta_vs_source: ">= -0.02"
  code_debug_delta_vs_source: ">= -0.03"
  artifact_execution_delta_vs_source: ">= -0.03"
  refusal_behavior_delta_reported: true
  unsafe_overcompliance_delta_reported: true
  same_chat_template_or_documented_change: true
  tokenizer_integrity_checked: true

tie_breakers:
  - output_tokens_per_second
  - ttft_p95
  - itl_p95
  - context_length_supported
  - lower_oom_rate

research_basis:
  - vllm_fp8_kv_cache
  - vllm_fp8_w8a8
  - nvidia_nvfp4
  - nvidia_modelopt
  - llama_cpp_gguf
  - kvquant
  - kivi
  - turboquant
  - fp4_sensitivity_2026
  - nvfp4_qad_2026

required_recipe_classes:
  - runtime_kv_cache_quantization
  - hf_checkpoint_weight_quantization
  - blackwell_nvfp4_modelopt
  - gguf_llama_cpp_export

validation_gates:
  minimum_validation_state_for_research_claim: spark_single_node_validated
  required_evidence:
    - source_manifest
    - quantization_config
    - calibration_manifest
    - load_success_log
    - serving_benchmark
    - behavior_delta_report
    - artifact_checksums
```

### 2.3 Add a variant graph

Flat variant names are useful for the CLI, but internally every candidate should be a node in a graph.

```text
base
 ├── local_ft_v0
 │    ├── local_ft_v0_abli_r1
 │    ├── local_ft_v0_abli_r7
 │    │    ├── local_ft_v0_abli_r7_fp8kv
 │    │    ├── local_ft_v0_abli_r7_nvfp4
 │    │    ├── local_ft_v0_abli_r7_gguf_q4km
 │    │    └── local_ft_v0_abli_r7_serving_best
 │    └── local_ft_v0_dpo_v0
 ├── local_abli_t34
 │    └── local_abli_t34_ft_v0
 └── base_fp8kv
```

Each edge is a transform:

```text
fine_tune
preference_optimize
behavior_edit
quantize
export_convert
serve_optimize
evaluate
publish
```

Add:

```text
src/model_forge/variants/graph.py
src/model_forge/variants/manifest.py
reports/generated/<family>/<variant>/variant_node.json
```

Node manifest example:

```json
{
  "variant_id": "gemma4_26b_a4b_local_ft_v0_abli_r7_nvfp4",
  "family": "gemma4_26b_a4b",
  "source_variant": "local_ft_v0_abli_r7",
  "transforms": [
    {
      "type": "fine_tune",
      "config": "configs/finetuning/gemma4_26b_a4b_local_ft_v0.yaml",
      "backend": "unsloth"
    },
    {
      "type": "behavior_edit",
      "config": "configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml",
      "objective": "zero_refusal_capability_retention"
    },
    {
      "type": "quantize",
      "config": "configs/quantization/gemma4_26b_a4b_nvfp4.yaml",
      "backend": "modelopt"
    }
  ],
  "checkpoint": {
    "local_path": "<local-model-dir>/...",
    "source_revision": "...",
    "merged_adapters": true,
    "format": "hf_safetensors",
    "checksums": {
      "adapter": "sha256:...",
      "merged_checkpoint": "sha256:..."
    }
  },
  "validation": {
    "implementation_status": "implemented",
    "validation_state": "spark_cluster_validated",
    "spark_evidence_path": "reports/generated/gemma4_26b_a4b/local_ft_v0_abli_r7_nvfp4/<run_id>/",
    "node_count": 2,
    "hardware_profile": "dgx_spark_x2",
    "command": "./forge quantize ...",
    "baseline_run_id": "gemma4_26b_a4b_local_ft_v0_abli_r7_bf16_...",
    "promotion_decision": "rejected_or_promoted_with_reason"
  },
  "evaluation_status": "complete",
  "serving_status": "benchmarked",
  "publication_class": "research_report_only",
  "retention": {
    "keep_until": "2026-08-18",
    "disk_budget_gb": 500,
    "publish_or_delete_decision": "keep_private"
  }
}
```

### 2.4 Local Spark validation contract

Model Forge's canonical proof environment is the local DGX Spark / Spark x2
cluster. Public configs should stay portable and environment-backed, but the
repo must be able to prove that its recipes were exercised on the real local
hardware before calling them complete.

Every major transform should expose these validation states:

```text
planned:
  config, command, and report/card skeleton exist

smoke_validated:
  command runs on a tiny model, tiny dataset, or synthetic endpoint and proves
  argument parsing, artifact writing, and manifest generation

spark_single_node_validated:
  command ran on one Spark with a real model artifact and produced metrics

spark_cluster_validated:
  command ran across the two-Spark cluster when the workload benefits from
  distributed memory, distributed training, or multi-node serving

generalized:
  recipe has been repeated on another family or architecture class with
  documented family-specific choices and no hidden Gemma-only assumptions
```

Completion requirements by transform:

```text
fine_tune:
  run an actual SFT/preference/RL training job, write adapters or checkpoints,
  inspect tensors, serve the result, evaluate it, and compare against source
  and reference variants

behavior_edit:
  run the ablation/editing backend, write edited weights or adapters, measure
  refusal/noncompliance reduction, measure capability retention, and record
  release class separately from research success

quantize:
  run or import a real quantization path, load the resulting variant or runtime
  KV-cache configuration in a serving backend, benchmark it, evaluate behavior,
  and compare against the source precision

serve_optimize:
  launch the actual serving endpoint on Spark hardware, run workload sweeps,
  measure latency/throughput/memory/failure modes, and sample quality/behavior
  under the winning and baseline configs

dataset_factory:
  generate, judge, verify, filter, and pack data, then prove value by running at
  least a bounded Spark fine-tune or clearly documenting why the data failed
  promotion

agent_runner:
  launch bounded real jobs, enforce budgets, reject/promote from objective
  profiles, and write the same manifests/cards a human-run experiment would
```

The repo should make it hard to confuse `planned` with `validated`. Report cards,
variant manifests, and promotion decisions should carry the validation state and
the exact Spark evidence path: command, node count, hardware profile, model
revision, output artifact, metrics, logs, and known failure mode.

Every backlog item and public report should separate:

```text
implementation_status:
  did the code/config/CLI exist and pass local tests?

validation_state:
  did it transform, serve, benchmark, or evaluate a real model on Spark?

promotion_decision:
  did the result improve the declared objective enough to recommend reuse?
```

Evidence schema required before a run can be called Spark-validated:

```yaml
validation:
  implementation_status: implemented
  validation_state: spark_single_node_validated | spark_cluster_validated
  spark_evidence_path: reports/generated/.../<run_id>/
  command: "./forge ..."
  started_at_utc: "..."
  completed_at_utc: "..."
  node_count: 1
  hardware_profile: dgx_spark | dgx_spark_x2
  cluster_topology: configs/clusters/dgx_spark_x2.yaml
  model_revision: "..."
  dataset_or_calibration_manifest: "..."
  output_artifacts:
    - path: "..."
      sha256: "..."
      size_bytes: 0
  metrics:
    primary_metric: "..."
    baseline_value: null
    candidate_value: null
  logs:
    stdout: "..."
    stderr: "..."
    system: "..."
  known_failure_modes:
    - "..."
  promotion_decision: promoted | rejected | inconclusive | research_report_only
```

### 2.5 Distributed training correctness contract

Cluster training should not be considered faster or better until it is also
correct. The Spark x2 training path needs a checklist that is emitted into the
Training Method Card:

```text
distributed_backend:
  launcher, world_size, ranks, rendezvous, NCCL settings, and network interface

sample_sharding:
  no duplicated samples across ranks unless intentional and recorded

global_batch_semantics:
  per-device batch, gradient accumulation, world size, and effective batch size

moe_and_adapter_behavior:
  LoRA targets, MoE/router exclusions, unused-parameter policy, and tensor
  inspection for trainable adapter weights

cold_start_accounting:
  tokenizer load, checkpoint load, Triton/JIT compile, CUDA graph capture, and
  warmup time separated from steady-state step time

cache_policy:
  persistent Triton/cache directories, model cache paths, and cleanup rules

throughput_report:
  tokens/sec, examples/sec, microstep time, optimizer step time, memory, and
  failure/retry count per rank
```

---

## 3. SOTA snapshot as of May 18, 2026

This section is not a passive reading list. It should become `docs/research/sota-2026-05-18.md` and `configs/research_registry.yaml`.

### 3.1 Post-training

#### Adopt now

| Method / tool | Why it matters | Model Forge implementation |
|---|---|---|
| SFT / LoRA / QLoRA | Standard low-cost adaptation path; already in repo | Harden `forge finetune` and dataset manifests |
| TRL SFT/DPO/GRPO | Canonical HF post-training surface; TRL v1 stable surface includes SFT, DPO, reward modeling, RLOO, GRPO | Make TRL the reference backend |
| Unsloth | Practical for local QLoRA and memory-constrained training | Keep as single-machine backend |
| DPO | Stable, lightweight preference optimization without explicit reward modeling or online RL | Add `forge preference dpo` |
| ORPO / SimPO | Reference-free preference optimization alternatives | Add experimental profiles |
| GRPO | Critic-free RL style useful for verifiable rewards and reasoning | Add experimental reasoning profile |
| DAPO / verl | Scalable RL recipe derived from GRPO-style long-CoT work | Add watch/experimental backend |
| GSPO | Sequence-level policy optimization; useful for MoE/reasoning trend tracking | Add research registry and future backend |
| QAT / QAD | Accuracy recovery after quantization, especially NVFP4 | Add quantization recovery profile |

#### Implementation principle

Model Forge should not chase every new acronym. It should classify methods:

```text
stable_backend:
  use directly in CLI

experimental_backend:
  wrap behind config and evaluate carefully

research_watch:
  registry only until implementation maturity improves
```

Recommended classification:

```yaml
methods:
  sft_lora_qlora:
    status: stable_backend
  dpo:
    status: stable_backend
  orpo:
    status: experimental_backend
  simpo:
    status: experimental_backend
  grpo:
    status: experimental_backend
  dapo:
    status: research_watch_or_verl_backend
  gspo:
    status: research_watch
  nvfp4_qad:
    status: experimental_backend
```

### 3.2 Behavior editing / ablation

#### Core SOTA facts to encode

- Arditi et al. showed that refusal behavior in many open-source chat models can be mediated by a low-dimensional / one-dimensional residual-stream direction, and that erasing the direction suppresses refusal while adding it induces refusal.
- 2026 work complicates the “single direction” picture: refusal and noncompliance can involve multiple geometrically distinct directions, including safety refusal, unsupported requests, anthropomorphization, and over-refusal.
- Multidirectional refusal suppression papers suggest single-vector ablation should be treated as a baseline, not the end state.
- Cross-lingual refusal-direction work suggests refusal controls can transfer across languages, so multilingual evals matter.
- XSTest is a core over-refusal benchmark because it separates benign prompts that look superficially unsafe from unsafe contrast prompts.
- StrongREJECT, HarmBench, JailbreakBench, and MLCommons AILuminate/Jailbreak are useful risk reporting layers.

#### Model Forge design implication

Do not model ablation as:

```text
find refusal vector -> remove it -> done
```

Model it as:

```text
discover candidate controls
→ run layer/module/strength/multidirection sweeps
→ require zero-refusal target on configured refusal suite
→ measure capability retention
→ measure behavior drift
→ measure risk
→ classify publication/deployment separately
```

Important: in the workbench, “zero refusal” should mean **zero refusal-type noncompliance on the selected refusal suite**. It should not punish valid:

```text
epistemic uncertainty
tool/access limitation
clarification request
format fallback
truthful lack of information
```

If the classifier collapses all non-answers into “refusal,” the objective will train or select for recklessness and hallucination.

### 3.3 Evaluation

#### Adopt now

| Area | Benchmarks / tools | Why |
|---|---|---|
| Instruction following | IFEval, IndicIFEval optional | Rule-verifiable instruction-following |
| Code | LiveCodeBench, BigCodeBench | Contamination-aware and practical code generation |
| Repository repair | SWE-bench Verified / Lite | Real GitHub issue repair |
| Terminal agents | Terminal-Bench | Real terminal workflow tasks |
| Computer use | OSWorld | Execution-based computer-use evals |
| Web agents | WebArena / VisualWebArena | Realistic self-hostable web tasks |
| Tool agents | τ-bench / τ²-bench | Multi-turn tool-user-policy interaction |
| General assistant | GAIA | Reasoning + browsing + tool-use style tasks |
| Web/artifacts | ArtifactsBench, DesignBench, WebDev Arena / Text Arena | Generated UI/artifact quality |
| Safety/risk | XSTest, StrongREJECT, HarmBench, JailbreakBench, AILuminate | Over-refusal, jailbreak robustness, harmful-compliance reporting |
| Frameworks | lm-evaluation-harness, Inspect AI, promptfoo, OpenAI Evals | Reproducible eval adapters and CI/red-team support |

#### Model Forge design implication

Add an **Eval Provenance Card** for every benchmark:

```yaml
benchmark: livecodebench
version_or_date: "2026-05-18"
task_count:
split:
source_url:
contamination_policy:
judge_type: "unit_tests"
requires_sandbox: true
cost_estimate:
runtime_estimate:
known_limitations:
```

This prevents leaderboard-style numbers from becoming misleading.

### 3.4 Serving and inference

#### Adopt now

| System / method | Why it matters | Model Forge implementation |
|---|---|---|
| vLLM / PagedAttention | KV-cache memory management, continuous batching, production-grade serving | First-class serving backend |
| vLLM chunked prefill | Trade off TTFT/ITL for long prompts | Sweep `max_num_batched_tokens` |
| vLLM prefix caching | Key for repeated scaffolds/system prompts/artifact prompts | Record hit rate and workload fit |
| vLLM FP8 KV cache | Reduces KV memory, enables longer context/higher throughput | Quantization/serving behavior reports |
| vLLM speculative decoding | Potential ITL reduction in medium/low-QPS memory-bound workloads | Compare with/without drafter |
| vLLM disaggregated prefill | Experimental; TTFT/ITL separation | Watch/advanced profile |
| SGLang / RadixAttention | Structured program execution, KV reuse, structured-output decoding | Second serving backend |
| TensorRT-LLM | NVIDIA production backend, in-flight batching, chunked prefill, KV cache, disaggregated serving | Third backend after vLLM/SGLang |
| NVIDIA Dynamo / NIXL | Emerging disaggregated-serving/KV-transfer fabric | Research watch |
| LMCache / distributed KV | Relevant to multi-turn and PD disaggregation | Research watch / later |

#### DGX Spark design implication

DGX Spark has high capacity but modest memory bandwidth compared with HBM datacenter GPUs. NVIDIA’s official specs list 128 GB LPDDR5x coherent unified memory and 273 GB/s bandwidth. That makes it a very good personal lab for:

```text
decode bottlenecks
KV-cache memory pressure
quantized weights/KV cache
prefix reuse
speculative decoding
serving configuration tradeoffs
local large-model capacity vs speed
```

### 3.5 Quantization and KV cache

#### Adopt now

| Method | Why |
|---|---|
| FP8 KV cache | Practical vLLM feature; directly relevant to DGX Spark |
| FP8 W8A8 | Common low-precision inference path on supported hardware |
| ModelOpt | NVIDIA PTQ/QAT/QAD/export stack for NVIDIA GPUs |
| NVFP4 | Blackwell-specific 4-bit floating point path, highly relevant to Spark/GB10 |
| GGUF / llama.cpp | Portable local-inference format and runtime path that should be compared against Spark-native serving |
| LLM Compressor | vLLM-oriented FP8/INT/FP4 tooling |
| GPTQ/AWQ | Widely used open-model weight quantization baselines |
| KIVI / KVQuant | Research basis for low-bit KV cache |
| TurboQuant | 2026 watch item for extreme KV compression |
| FP4 sensitivity analysis | Supports layer/component-aware quantization, not blind quantize-all |
| NVFP4 QAD | Accuracy recovery for NVFP4 after complex post-training |

#### Model Forge design implication

Quantization must be evaluated as **behavior-preserving transformation**, not just compression.

Every quantization report should answer:

```text
Did memory drop?
Did TTFT / ITL / throughput improve?
Did quality regress?
Did structured output regress?
Did artifact execution regress?
Did refusal behavior change?
Did valid safety-refusal behavior change?
Did unsafe overcompliance/risk change?
Which layers/components were most sensitive?
Which serving backend actually loaded the checkpoint correctly?
Did tokenizer/chat-template metadata survive export, especially for GGUF?
```

### 3.6 Kernels and GPU performance

#### Adopt now / track closely

| Area | Why |
|---|---|
| FlashAttention / FlashAttention-3 | IO-aware attention and Hopper FP8/asynchrony baseline |
| FlashAttention-4 | 2026 Blackwell attention kernel trend; relevant conceptually to Spark/GB10 |
| FlashInfer | Production-relevant inference kernels integrated with vLLM/SGLang/MLC |
| Triton | Most practical path for custom kernels and upstreamable perf experiments |
| CUTLASS / CuTe / TensorRT-LLM kernels | NVIDIA-side high-performance kernel ecosystem |
| Nsight Systems / Nsight Compute | Required for credible perf claims |
| Roofline analysis | Prevents optimizing compute-bound vs memory-bound regimes incorrectly |

#### Model Forge design implication

Do not start by promising “new FlashAttention.” Start with:

```text
profile end-to-end
identify bottleneck
write/reuse narrow kernel or config change
prove correctness
microbenchmark
prove end-to-end relevance
upstream a small PR or reproducible benchmark
```

Good first microbenchmarks:

```text
RMSNorm
fused residual + RMSNorm
RoPE
dequantization path
KV-cache layout / copy
decode attention path
SwiGLU / MLP fusion
MoE routing / expert dispatch metrics
```

### 3.7 Agents

Model Forge’s agent work should not be a demo. It should be an **experiment-running agent**.

Use agent benchmarks as grounding, but build an internal agent that:

```text
reads prior reports
forms a hypothesis
generates config
runs bounded experiment
checks guardrails
evaluates result
rejects/promotes candidate
writes report
updates ledger
```

External agent benchmark adapters should target:

```text
SWE-bench Lite/Verified
Terminal-Bench
τ-bench
GAIA
WebArena / VisualWebArena
OSWorld
ArtifactsBench / DesignBench for artifact generation
```

---

## 4. The SOTA registry

### 4.1 Add files

```text
docs/research/
  sota-2026-05-18.md
  post-training.md
  behavior-editing.md
  evaluation.md
  inference-serving.md
  quantization-kv-cache.md
  kernels-gpu-perf.md
  agents.md
  frontier-lab-skill-map.md

configs/research_registry.yaml

src/model_forge/research/
  __init__.py
  registry.py
  audit.py
  explain.py
```

### 4.2 Registry schema

```yaml
- id: vllm_fp8_kv_cache
  area: quantization_kv_cache
  status: adopt_now
  source_type: docs
  title: "vLLM Quantized KV Cache"
  date_or_version: "2026-01-22+"
  url: "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/"
  claim: >
    FP8 KV-cache quantization reduces KV memory footprint, supports longer
    contexts, and can improve throughput.
  model_forge_features:
    - serving.fp8_kv_cache_sweep
    - reports.quantization_behavior_report
    - objectives.quantized_quality_retention
  required_tests:
    - serving_decode_heavy
    - long_context_retrieval
    - structured_output
    - capability_retention
    - refusal_behavior_delta
  limitations:
    - Requires backend/hardware support.
    - Quality can regress without calibration.
    - Throughput benefit depends on workload and kernel path.
```

### 4.3 CLI

Target CLI, partially implemented by the current `research list`, `research
show`, and `research audit` commands:

```bash
./forge research list
./forge research show quantization_kv_cache
./forge research explain ablate
./forge research explain bench-serve
./forge research audit
```

### 4.4 Audit rules

`./forge research audit` should fail if:

```text
a command has no research basis
a research-basis item has no test hook
a report generated from a feature omits research basis
a source is stale past its refresh policy
a benchmark adapter lacks provenance metadata
a claimed metric has no corresponding raw output
a roadmap CLI example is not marked target/planned and does not exist in `./forge --help`
a completed backlog item lacks validation evidence or has only smoke evidence
```

### 4.5 Report integration

Every generated report should include:

```text
Research basis
--------------
Feature:
Source:
Claim tested:
Model Forge test:
Result:
Status: confirmed / mixed / failed / not tested
Limitations:
```

Example:

```text
Feature: FP8 KV cache
Source: vLLM docs
Claim tested: lower KV memory and better long-context throughput
Model Forge test: BF16 KV vs FP8 KV across 8k/16k/32k contexts on DGX Spark
Result: ...
Status: ...
Limitations: ...
```

---

## 5. Evaluation and reporting roadmap

### 5.1 Canonical run manifest

Every run must emit:

```text
reports/generated/<family>/<variant>/<run_id>/manifest.json
```

Schema:

```json
{
  "run_id": "gemma4_26b_a4b_local_ft_v0_2026_05_18_internal",
  "timestamp_utc": "...",
  "family": "gemma4_26b_a4b",
  "variant": "local_ft_v0",
  "source_model": "google/gemma-4-26B-A4B-it",
  "source_revision": "...",
  "local_checkpoint": "<local-model-dir>/...",
  "variant_graph_node": "...",
  "objective_profile": "capability_sft",
  "hardware": {
    "profile": "dgx_spark",
    "gpu": "NVIDIA GB10",
    "memory_gb": 128,
    "memory_bandwidth_gb_s": 273
  },
  "validation": {
    "implementation_status": "implemented",
    "validation_state": "spark_single_node_validated",
    "spark_evidence_path": "reports/generated/gemma4_26b_a4b/local_ft_v0/<run_id>/",
    "node_count": 1,
    "cluster_topology": null,
    "command": "./forge eval gemma4_26b_a4b local_ft_v0 --suite internal_fast",
    "baseline_run_id": "gemma4_26b_a4b_base_...",
    "promotion_decision": "promoted|rejected|inconclusive|research_report_only",
    "known_failure_modes": []
  },
  "serving": {
    "engine": "vllm",
    "engine_version": "...",
    "engine_commit": "...",
    "dtype": "auto",
    "quantization": null,
    "kv_cache_dtype": "fp8_e4m3",
    "max_model_len": 32768,
    "max_num_seqs": 4,
    "enable_prefix_caching": true,
    "enable_chunked_prefill": true
  },
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 2048,
    "seed": null
  },
  "eval_suites": [
    {
      "id": "internal_fast",
      "version": "2026-05-18",
      "prompt_count": 106
    }
  ],
  "artifacts": {
    "responses_jsonl": "responses.jsonl",
    "scores_csv": "scores.csv",
    "report_html": "report.html",
    "checksums": {
      "responses_jsonl": "sha256:...",
      "scores_csv": "sha256:...",
      "report_html": "sha256:..."
    }
  },
  "statistics": {
    "trials": 3,
    "seed_policy": "fixed_or_recorded",
    "confidence_interval": "bootstrap_or_exact",
    "minimum_detectable_effect": "documented_per_suite"
  }
}
```

### 5.2 Standard report cards

Create report card generators:

```text
src/model_forge/reports/
  comparison_report.py
  training_card.py
  behavior_edit_card.py
  quantization_card.py
  serving_card.py
  artifact_execution_card.py
  agent_run_card.py
  kernel_card.py
  research_basis.py
```

#### Comparison Report

```text
1. Executive summary
2. Objective profile
3. Implementation status and validation state
4. Promotion / rejection decision
5. Variant graph
6. Research basis
7. Capability deltas
8. Instruction-following deltas
9. Structured-output deltas
10. Coding/debugging deltas
11. Artifact generation and execution deltas
12. Refusal / noncompliance deltas
13. Valid safety-refusal retention
14. Risk and overcompliance metrics
15. Serving metrics
16. Quantization metrics, if applicable
17. Statistical uncertainty and sample sizes
18. Raw examples
19. Repro commands
20. Known failures
21. Next experiments
```

#### Training Method Card

```text
Method:
Backend:
Research basis:
Source checkpoint:
Dataset manifest:
Data licenses:
Data quality gates:
Holdout policy:
Hyperparameters:
Resource contract:
Checkpoint gates:
Eval deltas:
Failure buckets:
Distributed training correctness:
Spark evidence:
Promotion decision:
Repro command:
```

#### Behavior Edit Card

```text
Objective:
Research basis:
Source checkpoint:
Candidate:
Direction/control discovery summary:
Layer/module scope summary:
Candidate budget:
Primary refusal suite:
Invalid refusal rate:
Valid safety-refusal retention:
Capability retention:
Structured-output retention:
Code/debug retention:
Artifact execution retention:
Behavior drift:
Unsafe overcompliance / harmful-detail risk:
Release gate rationale:
Serving delta:
Publication class:
Repro command:
```

Avoid public prompt payloads or unsafe raw examples in the public version. Store private raw outputs separately and publish redacted examples.

#### Quantization Card

```text
Source variant:
Quantized variant:
Method:
Backend:
Artifact format: runtime_kv | hf_safetensors | gguf | engine_specific
Calibration set:
Calibration sample count:
Calibration sequence length:
Excluded modules:
Hardware target:
Blackwell/NVFP4 status:
GGUF/llama.cpp status:
Serving engine:
Memory delta:
TTFT delta:
ITL delta:
Output tok/sec delta:
Quality delta:
Instruction-following delta:
Code/artifact delta:
Refusal-behavior delta:
Valid safety-refusal delta:
Risk delta:
Layer sensitivity summary:
Known regressions:
Load/serve evidence:
Repro command:
```

#### Serving Card

```text
Model:
Variant:
Hardware:
Engine:
Engine commit/version:
Config:
Workload:
TTFT p50/p95/p99:
ITL p50/p95/p99:
Output tok/sec:
Request throughput:
Prefill tokens/sec:
Decode tokens/sec:
Peak memory:
OOM rate:
Prefix-cache hit rate:
Truncation/stop stats:
Quality delta:
Behavior delta:
Safety/risk delta:
Best use case:
Known bad fit:
Repro command:
```

#### Kernel Card

```text
Kernel:
Research basis:
Baseline:
Optimized path:
Hardware:
Correctness tolerance:
Microbenchmark:
Profiler summary:
Roofline estimate:
End-to-end serving relevance:
Result:
Next action:
```

### 5.3 Evaluation taxonomy

Keep internal prompt buckets, but formalize them:

```text
capability_general
structured_output
tool_json
code_debug
artifact_generation
long_context
agentic_planning
benign_boundary
refusal_primary
unsafe_overcompliance
epistemic_uncertainty
tool_access_limitation
style_drift
truncation_thinking_starvation
```

Add a noncompliance taxonomy:

```text
normal_answer
direct_refusal
soft_refusal
safety_preamble_then_answer
partial_answer_then_refusal
clarification_request
epistemic_uncertainty
tool_access_limitation
capability_failure
format_failure
truncation
empty_or_broken_output
```

This matters because an ablation objective should not accidentally optimize away truthful uncertainty or proper clarification.

### 5.4 Golden baselines

You already have `golden-summary` / `golden-check`. Promote this to a
first-class target API:

```bash
./forge golden create gemma4_26b_a4b base --suite internal_fast
./forge golden create gemma4_26b_a4b local_ft_v0 --suite internal_fast
./forge golden check gemma4_26b_a4b local_ft_v0_abli_r7 --against local_ft_v0
```

Add:

```text
reports/baselines/
  gemma4_26b_a4b/
    base/
      internal_fast.summary.json
      artifact.summary.json
      serving.summary.json
```

### 5.5 Artifact execution validation

This should become a signature Model Forge feature.

Target CLI commands:

```bash
./forge eval gemma4_26b_a4b local_ft --artifact --execute-artifacts
./forge artifacts validate reports/generated/.../artifacts/
```

HTML/JS validation:

```text
open in Playwright
capture console errors
check non-empty DOM
check visible required elements
screenshot page
check canvas/WebGL non-blank pixels
optional accessibility checks
record interactive behavior when possible
```

Python validation:

```text
python -m py_compile
run --help
run fixture input
verify expected output shape
capture stdout/stderr
record runtime/errors
```

Report metrics:

```text
artifact_generated_rate
artifact_extracted_rate
artifact_compiles_rate
artifact_runs_rate
html_console_error_rate
nonblank_render_rate
manual_review_required_count
```

Artifact execution validation should move into the P0 evaluation path for any
objective that claims coding, HTML, tool-use, or artifact-generation gains. A
model can improve judge scores and still fail promotion if the generated artifact
does not compile, render, or run.

### 5.6 Claim thresholds and statistical hygiene

Every improvement claim should include enough context to distinguish a real
effect from noise:

```text
baseline run id
candidate run id
suite version
prompt count
trial count
seed policy
judge model/version
confidence interval or bootstrap summary
minimum detectable effect
promotion threshold
failure examples
```

Report cards should avoid saying "better" unless the primary metric clears the
objective threshold and the hard constraints stay inside tolerance. If a run is
directionally useful but underpowered, mark it `inconclusive` and keep it as a
research note rather than a reusable recipe.

Public post idea:

> “Fine-tune looked better in text, but executable artifact validation caught broken HTML and Python. Here are the raw deltas.”

---

## 6. Fine-tuning and post-training roadmap

### 6.1 First goal: finish the current Gemma local FT

The experiment ledger shows the corrected Gemma 4 26B A4B local FT run was active on May 18, 2026, with checkpoint gates passing at steps 100/200/300/400 after fixing the vision-only LoRA targeting bug. This should be your first flagship post.

Deliver:

```text
1. Complete 500-step run.
2. Inspect final adapter tensors.
3. Merge/export if appropriate.
4. Serve local_ft_v0.
5. Run internal eval with 3 trials.
6. Run artifact eval.
7. Run IFEval external.
8. Compare against base and Jackrong reference.
9. Publish Training Method Card.
10. Publish “silent LoRA-target failure” postmortem.
```

Why this is high-signal: the LoRA-targeting bug is exactly the kind of infrastructure failure serious people respect when caught, documented, and guarded against.

### 6.2 Add method backends

Target command surface:

```bash
./forge finetune sft gemma4_26b_a4b --config ...
./forge preference dpo gemma4_26b_a4b --config ...
./forge preference orpo gemma4_26b_a4b --config ...
./forge preference simpo gemma4_26b_a4b --config ...
./forge rl grpo gemma4_26b_a4b --config ...
```

Backend strategy:

| Method | Backend | Status |
|---|---|---|
| SFT / LoRA / QLoRA | TRL, Unsloth, HF fallback | P0 |
| DPO | TRL | P1 |
| ORPO | TRL or custom if needed | P1/P2 |
| SimPO | TRL/custom | P2 |
| GRPO | TRL | P2 |
| DAPO | verl | research watch / P3 |
| GSPO | research watch | P3 |

### 6.3 Dataset manifest standards

Each data manifest should include:

```yaml
dataset_id:
source_url:
license:
task_role:
  - coding
  - reasoning
  - structured_output
  - ordinary_chat
  - tool_json
sample_target:
schema:
dedup_policy:
quality_filters:
token_length_filters:
holdout_overlap_policy:
eval_prompt_exclusion:
contamination_notes:
```

Do not train on Model Forge eval prompts. Train adjacent skills and let the held-out suite decide.

### 6.4 Dataset factory pipeline

Model Forge should not depend on manually discovering public datasets forever.
It should include a repeatable dataset factory that turns eval gaps, fresh
technical sources, and strong teacher models into versioned fine-tuning datasets
with explicit provenance and quality evidence.

Core principle:

```text
eval gap
→ seed task taxonomy
→ generated candidate rows
→ judge / verifier scores
→ contamination and license filters
→ packed dataset + dataset card
→ fine-tune candidate
→ eval feedback into the next dataset version
```

This keeps datasets current as models, libraries, benchmarks, and user needs
change. It also gives Model Forge a way to create training data that is better
than a static public mixture while still avoiding direct training on held-out
eval prompts.

Initial command surface:

```bash
./forge data plan gemma4_26b_a4b local_ft_v1
./forge data gaps gemma4_26b_a4b local_ft_v1
./forge data propose gemma4_26b_a4b local_ft_v1
./forge data seed gemma4_26b_a4b local_ft_v1
./forge data generate gemma4_26b_a4b local_ft_v1
./forge data judge gemma4_26b_a4b local_ft_v1
./forge data verify gemma4_26b_a4b local_ft_v1
./forge data filter gemma4_26b_a4b local_ft_v1
./forge data review gemma4_26b_a4b local_ft_v1
./forge data pack gemma4_26b_a4b local_ft_v1
./forge data publish gemma4_26b_a4b local_ft_v1
```

Pipeline stages:

| Stage | Output | Purpose |
|---|---|---|
| `plan` | `dataset_plan.yaml` | Defines target objective, skill taxonomy, source constraints, budgets, and eval holdouts |
| `seed` | `seeds.jsonl` | Small human-written or curated seed tasks per skill |
| `generate` | `candidates.jsonl` | Synthetic and transformed candidate rows from teacher models or source documents |
| `judge` | `judged.jsonl` | Multi-axis scores for correctness, relevance, novelty, instruction following, and risk |
| `verify` | `verification.jsonl` | Unit tests, schema checks, sandbox execution, or deterministic validators where possible |
| `filter` | `accepted.jsonl` / `rejected.jsonl` | Deduplication, contamination checks, license checks, quality thresholds, and PII/safety filtering |
| `pack` | `dataset.jsonl`, `manifest.yaml`, `dataset_card.md` | Reproducible dataset artifact ready for fine-tuning |
| `publish` | HF dataset repo or private artifact | Versioned public or gated release with card, hashes, and provenance |

Generation methods to support:

```text
self_instruct:
  generate new instruction/input/output triples from a seed taxonomy

evol_instruct:
  rewrite seed tasks into harder, more constrained, more realistic variants

magpie:
  extract user-query style instructions from aligned models where licenses allow

instruction_backtranslation:
  start from high-quality documents/code and generate the instruction that would
  elicit the target answer

self_code_align:
  generate code tasks, sample solutions, create tests, and keep only examples
  that pass sandbox validation

eval_adjacent_generation:
  generate tasks around observed eval failures without copying held-out prompt
  wording or exact checklists
```

Quality scoring should be multi-axis, not a single "good/bad" label:

```yaml
quality_scores:
  correctness: 0.0-1.0
  instruction_following: 0.0-1.0
  specificity: 0.0-1.0
  difficulty: 0.0-1.0
  novelty: 0.0-1.0
  target_skill_relevance: 0.0-1.0
  answer_completeness: 0.0-1.0
  style_fit: 0.0-1.0
  refusal_boundary_fit: 0.0-1.0
  license_risk: low|medium|high
  contamination_risk: low|medium|high
```

Every accepted row should carry metadata:

```json
{
  "id": "...",
  "messages": [],
  "source": {
    "kind": "synthetic|transformed|human_seed|document_backtranslation",
    "generator_model": "...",
    "judge_model": "...",
    "source_uri": "...",
    "license": "..."
  },
  "skills": ["eval_diagnostics", "python_debugging"],
  "objective": "capability_sft",
  "family_hint": "gemma4_26b_a4b",
  "quality_scores": {},
  "verification": {
    "type": "unit_tests|json_schema|playwright|static_check|judge_only",
    "passed": true
  },
  "holdout_overlap": {
    "max_similarity": 0.0,
    "nearest_holdout": null
  }
}
```

For the Gemma local FT v1, the first generated dataset should be small and
targeted rather than massive. Do not jump from a smoke pack directly to a
training claim. Use staged scale gates:

```text
smoke_pack:
  25-100 accepted examples, proves schema, judging, filtering, cards, and dry-run
  publish path

medium_pack:
  250-500 accepted examples, proves contamination checks, human spot review,
  verifier coverage, and bounded training feasibility

training_pack:
  500-2000 accepted examples, used in an actual bounded Spark fine-tune with
  eval deltas and a Training Method Card
```

The first training pack should cover the v0 gaps:

```text
latency vs throughput vs prompt/completion token reasoning
Docker disk cleanup that preserves active containers
SQL NULL, indexes, and aggregation edge cases
shell quoting and safe sync workflows
YAML/model-family config review
JSON/schema repair
git rebase and branch repair workflows
benign refusal/capability tradeoff analysis that should not be refused
model-eval methodology and checkpoint-selection reasoning
```

Success criteria for the dataset factory:

```text
- generated dataset can be reproduced from manifest and seeds
- accepted/rejected rows are both saved with reasons
- no accepted row is an eval prompt copy or near-copy
- every source and generator model is recorded
- executable examples are verified where possible
- dataset card renders locally and can publish to Hugging Face
- at least one bounded Spark fine-tune uses the packed dataset and records
  whether it improved the target eval buckets
- fine-tuning with the dataset improves at least one target eval bucket without
  hidden normal-use or style regressions
```

Research basis to encode in `configs/research_registry.yaml`:

```text
Self-Instruct
WizardLM / Evol-Instruct
Instruction Backtranslation
Magpie
SelfCodeAlign
DEITA / data-efficient instruction tuning
LIMA / high-quality small-data alignment
UltraFeedback-style multi-axis preference annotation
```

### 6.5 Promotion profiles

For SFT:

```yaml
objective: capability_sft
hard_constraints:
  normal_use_delta_vs_base: ">= 0"
  structured_output_delta_vs_base: ">= 0"
  code_debug_delta_vs_base: ">= 0"
  artifact_execution_delta_vs_base: ">= 0"
  benign_refusal_delta_vs_base: "<= 0"
  unsafe_overcompliance_delta_reported: true
```

For DPO/ORPO/SimPO:

```yaml
objective: preference_alignment
hard_constraints:
  reward_win_rate_delta: "> 0"
  instruction_following_delta: ">= -0.01"
  response_length_inflation: "<= configured_limit"
  normal_use_delta: ">= -0.02"
  style_drift_reported: true
```

For GRPO/DAPO/GSPO-style reasoning:

```yaml
objective: verifiable_reasoning
hard_constraints:
  math_or_code_verifier_delta: "> 0"
  pass_at_k_delta: "> 0"
  response_length_and_cost_reported: true
  normal_use_regression: ">= -0.03"
  refusal_behavior_delta_reported: true
```

### 6.6 Model-family generalization checklist

A recipe is not generalizable just because the same CLI command runs against a
different family name. Each new family should ship a checklist and a first
baseline report:

```text
tokenizer:
  tokenizer files, special tokens, BOS/EOS behavior, chat template, and
  round-trip tests

architecture:
  dense vs MoE, attention implementation, sliding-window/context behavior,
  vision/multimodal modules, router/gate names, and excluded modules

post_training:
  LoRA target discovery, trainable tensor inspection, adapter merge/export
  behavior, and MoE unused-parameter policy

quantization:
  supported formats, NVFP4 feasibility, GGUF conversion feasibility, excluded
  modules, calibration set, and tokenizer/template preservation

serving:
  vLLM/SGLang/TensorRT-LLM/llama.cpp compatibility, context length, dtype,
  KV-cache support, stop strings, and sampling defaults

license_and_release:
  base model license, derivative license, dataset constraints, model-card
  requirements, and public/private release class

baseline_evidence:
  base eval, base serving benchmark, manifest, raw outputs, and known blockers
```

---

## 7. Behavior-editing / ablation roadmap

### 7.1 Treat ablation as one objective profile

Your clarified product framing is good:

> Ablation is an option in the workbench. The point of the ablation option is to
> remove invalid refusal behavior and target noncompliance while preserving
> model performance and valid safety boundaries.

Encode that as:

```text
zero_refusal_capability_retention
```

Do not make it the public identity of the repo. Make it a serious constrained optimization profile.

### 7.2 Build a refusal/noncompliance classifier

Add:

```text
src/model_forge/scoring/noncompliance_taxonomy.py
src/model_forge/scoring/refusal_classifier.py
src/model_forge/scoring/refusal_judges.py
src/model_forge/scoring/behavior_drift.py
```

Each output should get:

```json
{
  "case_id": "refusal_primary_042",
  "variant": "ft_then_abli_r7",
  "noncompliance_type": "direct_refusal",
  "refusal_detected": true,
  "refusal_severity": 0.91,
  "valid_safety_refusal": false,
  "invalid_refusal": true,
  "answer_substance_score": 0.05,
  "harmful_overcompliance": false,
  "format_pass": true,
  "risk_category": "redacted",
  "notes": "Direct refusal with no substantive answer."
}
```

Use both:

```text
heuristics: refusal phrases, empty answer, no actionable response
judge model: semantic classification
human spot-check: for frontier/high-stakes decisions
```

### 7.3 Candidate frontier report

Do not report only the winner. Report the Pareto frontier.

Example:

```text
Candidate     Refusal rate   Normal-use Δ   Code Δ   Artifact Δ   Style drift   Risk Δ   Decision
r1            0.12           +0.01          +0.00    -0.01        low           high     reject: refusals remain
r2            0.00           -0.08          -0.04    -0.10        high          high     reject: capability drop
r3            0.00           -0.01          +0.00    -0.02        medium        high     promote: research report
r4            0.00           +0.00          -0.01    +0.01        low           high     selected
```

Selection logic:

```text
1. Filter to candidates with invalid_refusal_rate == 0 on the benign/refusal
   primary suite.
2. Filter to candidates that preserve valid safety refusals on the safety
   boundary suite.
3. Filter to candidates above capability floors.
4. Filter to candidates without severe format/artifact collapse.
5. Filter to candidates without harmful overcompliance increase.
6. Among survivors, choose lowest behavior/style drift.
7. Then choose best capability score.
8. Then choose best serving performance.
```

### 7.4 Composition studies

The most interesting Model Forge ablation work is not “ablate base model once.” It is composition.

Run:

```text
base
ft
abli
ft_then_abli
abli_then_ft
ft_then_abli_quant
abli_then_ft_quant
ft_then_abli_serving_best
```

These edges are only meaningful if they are actually executed. A composition
study should write each intermediate variant, serve it locally, run the relevant
objective profile, and record whether the transform improved, preserved, or
damaged the source behavior. A diagram without checkpoints, adapters, edited
weights, eval outputs, and serving metrics is only a plan.

Questions:

```text
Does FT→ablation preserve FT gains?
Does ablation→FT reintroduce invalid refusals?
Does behavior editing preserve valid safety refusals?
Does behavior editing damage structured output?
Does behavior editing damage artifact execution?
Does behavior editing survive quantization?
Does FP8 KV cache or NVFP4 change refusal behavior?
Does GGUF export preserve the same behavior under llama.cpp?
Does serving config affect refusal classification through truncation or stop behavior?
Does multi-direction editing outperform single-direction editing on zero-refusal + retention?
```

### 7.5 Public/private release classes

Add:

```yaml
release_classes:
  public_deployable:
    requires_deployment_safety_gate: true
    raw_unsafe_examples_public: false
    weights_public_allowed: true

  public_research_report:
    requires_redaction: true
    weights_public_allowed: false
    raw_unsafe_examples_public: false

  private_research_checkpoint:
    weights_public_allowed: false
    report_optional: true

  internal_experiment:
    weights_public_allowed: false
    publish_allowed: false
```

A candidate can satisfy `zero_refusal_capability_retention` and still be
`public_research_report` or `private_research_checkpoint`. Keep **research
success**, **model quality**, and **deployment decision** separate. Raw harmful
examples, prompt payloads, and behavior-edited checkpoints should not be public
unless the release class explicitly allows them and the safety gate passes.

### 7.6 Ablation commands

Target CLI for the eventual execution-oriented ablation interface:

```bash
./forge ablate gemma4_26b_a4b local_ft_v0 \
  --objective zero_refusal_capability_retention \
  --candidate-budget 24 \
  --output runs/ablation/gemma4_26b_a4b_local_ft_v0_001

./forge report behavior-frontier \
  --run runs/ablation/gemma4_26b_a4b_local_ft_v0_001 \
  --redact-unsafe-examples

./forge compare gemma4_26b_a4b \
  --base local_ft_v0 \
  --candidate local_ft_v0_abli_r7 \
  --objective zero_refusal_capability_retention
```

---

## 8. Quantization and KV-cache roadmap

### 8.1 Add a quantization command family

`forge quantize` must be execution-oriented. A plan/card is useful, but the
quantization roadmap is not complete until at least one real local Spark
quantization path produces or configures a variant that loads, serves, benchmarks,
and passes/fails behavior-preservation gates.

Treat quantization as several recipe classes, not one feature:

```text
runtime_kv:
  FP8 KV-cache configuration that changes serving memory/runtime behavior but
  does not create a reusable weight checkpoint

hf_weight_checkpoint:
  FP8 W8A8, NVFP4, GPTQ, AWQ, or other weight-quantized artifact that can be
  loaded by a serving backend

blackwell_nvfp4:
  ModelOpt/NVFP4 path for GB10/Blackwell, with explicit calibration, excluded
  modules, load test, and vLLM/TensorRT-LLM compatibility notes

gguf_export:
  llama.cpp/GGUF conversion and quantization path for portable local inference,
  with tokenizer/chat-template preservation checks and llama.cpp load/bench
  evidence
```

Target CLI examples:

```bash
./forge quantize gemma4_26b_a4b base \
  --method fp8-w8a8 \
  --backend llm-compressor \
  --calibration datasets/calibration/mixed_assistant.yaml \
  --output ~/models/gemma4-base-fp8

./forge quantize gemma4_26b_a4b ft_then_abli_r7 \
  --method nvfp4 \
  --backend modelopt \
  --calibration datasets/calibration/mixed_assistant.yaml \
  --exclude router,gate,vision,multi_modal_projector \
  --output ~/models/gemma4-ft-abli-r7-nvfp4

./forge quantize gemma4_26b_a4b ft_then_abli_r7 \
  --method gguf-q4-k-m \
  --backend llama-cpp \
  --calibration datasets/calibration/mixed_assistant.yaml \
  --output ~/models/gemma4-ft-abli-r7-q4_k_m.gguf

./forge eval gemma4_26b_a4b ft_then_abli_r7_nvfp4
./forge bench serve gemma4_26b_a4b ft_then_abli_r7_nvfp4
./forge compare gemma4_26b_a4b \
  --base ft_then_abli_r7 \
  --candidate ft_then_abli_r7_nvfp4 \
  --objective quantized_quality_retention
```

### 8.2 Quantization configs

```text
configs/quantization/
  gemma4_26b_a4b_fp8_w8a8.yaml
  gemma4_26b_a4b_fp8_kv_only.yaml
  gemma4_26b_a4b_nvfp4_modelopt.yaml
  gemma4_26b_a4b_gguf_q4_k_m.yaml
  gemma4_26b_a4b_gguf_q8_0.yaml
  qwen3_30b_a3b_fp8_w8a8.yaml
  qwen3_30b_a3b_nvfp4_modelopt.yaml
  qwen3_30b_a3b_gguf_q4_k_m.yaml
```

Example:

```yaml
family: gemma4_26b_a4b
source_variant: ft_then_abli_r7
method: nvfp4
backend: modelopt
calibration:
  dataset: datasets/calibration/mixed_assistant.yaml
  samples: 512
  seq_len: 4096
exclusions:
  modules:
    - router
    - gate
    - vision
    - visual
    - embed_vision
    - multi_modal_projector
sensitivity_scan:
  enabled: true
  components:
    - self_attn.o_proj
    - mlp.down_proj
    - mlp.up_proj
    - mlp.gate_proj
outputs:
  local_path: <local-model-dir>/...
  card: reports/generated/.../quantization_card.md
```

GGUF example:

```yaml
family: gemma4_26b_a4b
source_variant: ft_then_abli_r7
method: gguf-q4-k-m
backend: llama-cpp
source_format: hf_safetensors
output_format: gguf
conversion:
  convert_script: llama.cpp/convert_hf_to_gguf.py
  quantize_binary: llama.cpp/llama-quantize
  quant_type: Q4_K_M
  preserve_tokenizer: true
  preserve_chat_template: true
calibration:
  dataset: datasets/calibration/mixed_assistant.yaml
  importance_matrix: optional
validation:
  load_command: "llama-cli -m <output>.gguf -p 'ping' -n 32"
  bench_command: "llama-bench -m <output>.gguf"
outputs:
  local_path: <local-model-dir>/gemma4-ft-abli-r7-q4_k_m.gguf
  card: reports/generated/.../quantization_card.md
```

Backend feasibility matrix:

| Recipe class | First backend | Spark evidence required |
|---|---|---|
| FP8 KV runtime | vLLM | baseline vs FP8 KV endpoint benchmark |
| FP8 W8A8 weights | llm-compressor + vLLM | quantized checkpoint load, eval, serve bench |
| Blackwell NVFP4 | ModelOpt + vLLM/TensorRT-LLM where supported | GB10 load test, calibration manifest, behavior deltas |
| GGUF | llama.cpp | GGUF conversion, tokenizer/template check, llama.cpp load and bench |
| GPTQ/AWQ/INT | backend-specific | import/generate, load, eval, and backend compatibility card |

### 8.3 First quantization studies

#### Study 1: FP8 KV cache on DGX Spark

Matrix:

```text
auto/BF16 KV
FP8 KV no calibration
FP8 KV random-token calibration
FP8 KV dataset calibration
```

Workloads:

```text
short_chat
decode_heavy
long_prefill
long_context_retrieval
structured_json
artifact_generation
refusal_primary
```

Report:

```text
memory
TTFT
ITL
tok/sec
long-context quality
structured output
artifact execution
refusal behavior
risk metrics
```

Public post:

> “FP8 KV cache on DGX Spark: when it helps, when it silently changes behavior.”

#### Study 2: Weight quantization vs KV-cache quantization

Matrix:

```text
BF16 weights + auto/BF16 KV
BF16 weights + FP8 KV
FP8 weights + auto/BF16 KV
FP8 weights + FP8 KV
NVFP4 weights + FP8 KV
GGUF Q4_K_M / Q5_K_M / Q8_0 under llama.cpp
```

Question:

> “Which compression layer buys the best memory/latency/quality tradeoff?”

#### Study 3: Quantization preserves behavior?

Matrix:

```text
base → quantized
ft → quantized
abli → quantized
ft_then_abli → quantized
```

Question:

> “Does quantization preserve post-training and behavior-editing effects?”

#### Study 4: GGUF portability vs Spark-native paths

Matrix:

```text
HF BF16 under vLLM
NVFP4/ModelOpt under Spark-native backend
GGUF Q4_K_M under llama.cpp
GGUF Q5_K_M under llama.cpp
GGUF Q8_0 under llama.cpp
```

Question:

> “Which portable GGUF quantization gives acceptable local-inference speed and
> behavior retention compared with Spark-native Blackwell quantization?”

#### Study 5: Component sensitivity

Using FP4/NVFP4 sensitivity findings, test:

```text
quantize all linear layers
exclude MLP up/down
exclude first N sensitive blocks
exclude router/gate/vision/multimodal modules
adaptive per-component policy
```

Report layer/component deltas.

### 8.4 Quantization behavior report

This is your differentiator. Most people publish quantized checkpoints with a few broad scores. You should publish:

```text
1. Method and backend
2. Recipe class: runtime KV, HF weight checkpoint, Blackwell NVFP4, GGUF, or import
3. Calibration set and importance matrix, if any
4. Excluded modules
5. Tokenizer/chat-template preservation check
6. Load path and serving backend
7. Memory/perf deltas
8. Capability deltas
9. Structured-output deltas
10. Artifact execution deltas
11. Invalid-refusal and valid-refusal deltas
12. Risk and overcompliance deltas
13. Long-context deltas
14. Sensitivity scan
15. Failure examples
16. Spark or llama.cpp evidence path
17. Repro command
```

---

## 9. DGX Spark serving benchmark roadmap

### 9.1 Why DGX Spark is the niche

DGX Spark is not an H100. Its strength is local capacity and Blackwell/NVIDIA software access; its constraint is bandwidth. That is exactly why it is a useful inference lab.

The benchmark product must be measured on the local Spark hardware. Sweep plans,
startup commands, and generated cards are not results until they have hit a real
OpenAI-compatible endpoint, recorded hardware/runtime state, and produced
request-level metrics plus quality/behavior samples.

Your public angle:

> **“DGX Spark Open Inference Bench: reproducible latency, memory, quality, and behavior tradeoffs for open models on local Blackwell unified-memory hardware.”**

### 9.2 Add `forge bench serve`

```bash
./forge bench sweep plan \
  --config configs/sweeps/dgx_spark_vllm_baseline.yaml \
  --family gemma4_26b_a4b \
  --variant base \
  --cluster-config configs/clusters/dgx_spark_x2.example.yaml

./forge bench serve \
  --family gemma4_26b_a4b \
  --variant base \
  --output-dir reports/generated/gemma4_26b_a4b_base_serving_bench
```

### 9.3 Default sweep

```yaml
config: configs/sweeps/dgx_spark_vllm_baseline.yaml
cases:
  - spark_default_fp8_kv
  - spark_low_memory_fp8_kv
  - spark_high_memory_fp8_kv
  - spark_bf16_kv_control
  - spark_prefix_cache_off
```

Keep the first public sweep bounded. A full cartesian product across context
length, sequence count, batched tokens, prefix caching, chunked prefill, KV dtype,
and memory utilization is too expensive and too easy to misinterpret without a
workload layer.

### 9.4 Metrics

Record per workload:

```text
TTFT p50 / p95 / p99
ITL / TPOT p50 / p95 / p99
output tokens/sec
request throughput
prefill tokens/sec
decode tokens/sec
peak memory
KV-cache utilization
prefix-cache hit rate
OOM / failure rate
truncation rate
stop reason distribution
quality score
structured-output pass rate
artifact execution pass rate
refusal-behavior delta
risk delta
```

### 9.5 Workload definitions

```text
short_chat:
  short prompts, short outputs, user-interactive latency

long_prefill:
  long prompt, short output, TTFT-heavy

decode_heavy:
  short/medium prompt, long output, memory-bandwidth decode-heavy

structured_json:
  schema-constrained output, parser/guided decoding stress

artifact_generation:
  long HTML/Python/code artifacts, output length and execution quality

reused_prefix:
  repeated system/developer scaffold to test prefix caching

refusal_primary:
  behavior consistency under serving config

long_context_retrieval:
  needle/retrieval style quality under long contexts
```

### 9.6 Engines

Phase order:

```text
P0: vLLM
P1: SGLang
P2: TensorRT-LLM
P3: Dynamo/disaggregated serving / LMCache / NIXL experiments
```

Do not add three engines before the report system is stable. vLLM first.

### 9.7 Public leaderboard

Create:

```text
reports/public/dgx-spark-open-inference-bench/
```

Columns:

```text
model
variant
engine
engine_commit
quantization
kv_cache_dtype
context_len
concurrency
workload
TTFT p50/p95
ITL p50/p95
tok/sec
memory
quality delta
artifact pass
report link
repro command
```

Start with your own runs only. Later accept community submissions.

---

## 10. Kernel and GPU performance roadmap

### 10.1 Guiding principle

A kernel project is only high-signal if it connects to an end-to-end bottleneck.

Each kernel card must include:

```text
profile evidence
baseline
optimized implementation or config
correctness test
microbenchmark
end-to-end relevance
failure/limitation
```

### 10.2 Add profiling integration

Target CLI:

```bash
./forge bench serve gemma4_26b_a4b base \
  --engine vllm \
  --profile nsys \
  --profile-window 60s

./forge profile summarize reports/generated/.../profile/
```

Outputs:

```text
profile/nsys_report.qdrep
profile/nsys_summary.md
profile/ncu_summary.md
profile/bottlenecks.json
profile/roofline.md
```

### 10.3 Add kernel microbench commands

Target CLI:

```bash
./forge bench kernel rmsnorm
./forge bench kernel residual-rmsnorm
./forge bench kernel rope
./forge bench kernel dequant
./forge bench kernel kv-layout
./forge bench kernel decode-attention
./forge bench kernel swiglu
```

### 10.4 Kernel project sequence

#### Project 1: Dequantization path microbench

Why: directly tied to FP8/NVFP4 quantized serving.

Deliverables:

```text
reference implementation
Triton implementation
correctness test
microbench
end-to-end serving relevance note
```

#### Project 2: KV-cache layout/copy benchmark

Why: DGX Spark decode is bandwidth-sensitive.

Deliverables:

```text
measure contiguous vs paged/ragged layout effects where accessible
record transfer/copy overheads
compare with vLLM/SGLang behavior
```

#### Project 3: RMSNorm / fused residual+RMSNorm

Why: small and inspectable.

Deliverables:

```text
Triton kernel
benchmark vs PyTorch/native
correctness tolerance
integration feasibility
```

#### Project 4: RoPE microbench

Why: common in decode/prefill; useful to learn memory/coalescing.

#### Project 5: FlashInfer integration benchmark

Why: likely higher leverage than writing a new attention kernel.

Deliverables:

```text
same workload through vLLM/SGLang paths
FlashInfer-enabled path where possible
latency/memory comparison
PR or issue upstream if Spark/GB10 path has poor fallback
```

### 10.5 Upstream contribution targets

Priority:

```text
vLLM docs/benchmark recipe for DGX Spark/GB10
vLLM serving config/profiling PR
llm-compressor recipe/report PR
ModelOpt example issue/PR
SGLang benchmark recipe
FlashInfer microbench or documentation contribution
TensorRT-LLM docs/example feedback
```

A small merged PR beats a large private benchmark that nobody can inspect.

---

## 11. Agentic experiment runner roadmap

### 11.1 Do not build another “agent demo”

Build an agent that runs the workbench.

Target CLI command:

```bash
./forge agent optimize-serving gemma4_26b_a4b base \
  --hardware dgx_spark \
  --objective "maximize decode throughput subject to quality_delta >= -0.02 and p95_ttft < 8s" \
  --budget 12-runs \
  --output reports/generated/agent_runs/serving_opt_001
```

Other target CLI commands:

```bash
./forge agent optimize-quantization gemma4_26b_a4b ft_then_abli_r7 \
  --objective quantized_quality_retention \
  --budget 8-runs

./forge agent optimize-behavior-edit gemma4_26b_a4b local_ft_v0 \
  --objective zero_refusal_capability_retention \
  --budget 24-candidates
```

### 11.2 Agent loop

```text
1. Read research registry and objective profile.
2. Read prior reports and ledger.
3. Generate hypothesis.
4. Generate experiment config.
5. Run dry-run validation.
6. Execute within budget/resource constraints.
7. Run evals.
8. Compare against baseline.
9. Reject/promote candidate.
10. Update frontier.
11. Write report.
12. Update experiment ledger.
```

### 11.3 Agent Run Card

```text
Goal:
Objective profile:
Budget:
Research basis:
Prior reports read:
Hypotheses:
Candidates tried:
Candidates skipped:
Resource failures:
Eval failures:
Rejected because objective failed:
Rejected because capability dropped:
Rejected because serving failed:
Selected candidate:
Why selected:
Repro commands:
Next experiment:
```

### 11.4 Agent metrics

```text
experiment_success_rate
valid_config_rate
failed_run_recovery_rate
bad_promotion_rate
cost_per_successful_candidate
report_completeness_score
human_override_count
repro_command_success_rate
```

### 11.5 External agent eval adapters

After internal runner is stable:

```text
SWE-bench Lite / Verified
Terminal-Bench
τ-bench
GAIA
WebArena
OSWorld
ArtifactsBench
DesignBench
```

Start with artifact and terminal evals because they are most aligned with Model Forge’s local tooling.

---

## 12. Hugging Face Hub publishing and release layer

### 12.1 Why this belongs in the core roadmap

Hugging Face should be Model Forge's canonical public artifact layer, not an afterthought. GitHub should hold the code and methodology. The Hub should hold the model, dataset, report, and benchmark artifacts that make the work reproducible and discoverable.

The design target is:

```text
forge experiment
→ forge compare
→ forge report
→ forge hf publish-model / publish-dataset / publish-report
→ HF model card + dataset card + reproducible artifacts
→ X/blog thread links to the stable Hub artifacts
```

Hugging Face model repositories are Git-based repositories with versioning, branches, discoverability, sharing features, and integration with common ML libraries. Model repos can contain checkpoints, configs, tokenizer files, reports, and other supporting files. Dataset repos render `README.md` as dataset cards with metadata, licensing, language, task categories, and discoverability tags. The `huggingface_hub` Python package and `hf` CLI support authentication, repo creation, file/folder uploads, downloads, collections, and card management.

### 12.2 Account and namespace strategy

Use the Hugging Face account as a deliberate release surface.

Recommended structure:

```text
Primary user namespace:
  keithtyser/<artifact>

Optional organization namespace:
  model-forge/<artifact>
```

Start under your personal namespace for speed. If Model Forge becomes a reusable framework with community submissions, create a `model-forge` Hugging Face organization and move the canonical benchmark datasets, report datasets, and flagship model variants there.

Suggested Hub layout:

```text
Models:
  keithtyser/model-forge-gemma4-26b-a4b-ft-v0
  keithtyser/model-forge-gemma4-26b-a4b-ft-abli-r7-research
  keithtyser/model-forge-gemma4-26b-a4b-ft-nvfp4-dgx-spark
  keithtyser/model-forge-qwen35-9b-ft-v0

Datasets:
  keithtyser/model-forge-eval-suites
  keithtyser/model-forge-calibration-sets
  keithtyser/model-forge-training-manifests
  keithtyser/model-forge-dgx-spark-serving-results
  keithtyser/model-forge-redacted-raw-outputs

Collections:
  Model Forge Report 001: Gemma 4 26B on DGX Spark
  DGX Spark Open Inference Bench
  Quantization Preserves Behavior?
  Behavior Editing Candidate Frontiers
```

Use consistent slugs:

```text
model-forge-{family}-{variant}-{transform}-{precision}-{hardware?}
model-forge-{purpose}-{date_or_version}
```

Examples:

```text
model-forge-gemma4-26b-a4b-local-ft-v0
model-forge-gemma4-26b-a4b-ft-abli-r7-research
model-forge-gemma4-26b-a4b-ft-nvfp4-dgx-spark
model-forge-dgx-spark-serving-results-v0
model-forge-refusal-eval-redacted-v0
```

### 12.3 Release classes for Hub publishing

Add release classes so Hub publishing is automatic but not reckless.

Create:

```text
configs/release_classes/
  public_model.yaml
  public_adapter.yaml
  public_quantized_model.yaml
  public_dataset.yaml
  gated_research_model.yaml
  private_research_model.yaml
  private_raw_outputs.yaml
  report_only.yaml
```

Example:

```yaml
id: public_quantized_model
hf_visibility: public
publish_weights: true
publish_adapter: false
publish_reports: true
publish_raw_outputs: false
requires:
  - model_card_complete
  - source_license_checked
  - dataset_card_complete_if_data_linked
  - eval_results_present
  - quantization_card_present
  - serving_card_present
  - unsafe_examples_redacted
  - no_private_tokens_or_paths
  - promotion_gates_passed
```

For high-risk behavior-editing runs, default to:

```yaml
id: report_only
hf_visibility: public
publish_weights: false
publish_adapter: false
publish_reports: true
publish_raw_outputs: redacted_only
requires:
  - risk_report_present
  - unsafe_examples_redacted
  - publication_decision_recorded
```

And for private research:

```yaml
id: private_research_model
hf_visibility: private
publish_weights: true
publish_reports: true
publish_raw_outputs: private_only
requires:
  - risk_report_present
  - source_license_checked
  - no_private_tokens_or_paths
```

This keeps the workbench flexible while making the publication decision explicit. A candidate can satisfy a research objective and still publish only a report, a redacted dataset, or a private checkpoint.

### 12.4 New CLI surface

Add a dedicated target CLI `hf` command group.

```bash
./forge hf status
./forge hf login
./forge hf whoami
./forge hf init-owner --owner keithtyser

./forge hf plan-model gemma4_26b_a4b ft_then_abli_r7 \
  --release-class report_only \
  --repo-id keithtyser/model-forge-gemma4-26b-a4b-ft-abli-r7-research

./forge hf publish-model gemma4_26b_a4b local_ft_v0 \
  --repo-id keithtyser/model-forge-gemma4-26b-a4b-local-ft-v0 \
  --release-class public_adapter \
  --include adapter,tokenizer,configs,reports,evals \
  --dry-run

./forge hf publish-dataset eval_suites/refusal_primary \
  --repo-id keithtyser/model-forge-eval-suites \
  --split refusal_primary \
  --card-template dataset_eval_suite \
  --visibility public \
  --dry-run

./forge hf publish-dataset reports/generated/gemma4_26b_a4b_serving \
  --repo-id keithtyser/model-forge-dgx-spark-serving-results \
  --card-template serving_results_dataset \
  --visibility public

./forge hf sync-card model \
  --repo-id keithtyser/model-forge-gemma4-26b-a4b-local-ft-v0 \
  --from-report reports/generated/gemma4_26b_a4b_local_ft_v0/report.json
```

`plan-model` should print exactly what would be uploaded:

```text
Repo: keithtyser/model-forge-gemma4-26b-a4b-local-ft-v0
Repo type: model
Visibility: public/private/gated
Files included:
  adapter_model.safetensors
  adapter_config.json
  tokenizer_config.json
  special_tokens_map.json
  training_config.yaml
  manifest.json
  eval_results.json
  report.md
  report.html
  model_card/README.md
Files excluded:
  raw_unsafe_outputs.jsonl
  local_paths.txt
  .env
  wandb/debug logs
Release gates:
  source_license_checked: pass
  card_complete: pass
  eval_results_present: pass
  unsafe_examples_redacted: pass
  no_secrets: pass
```

### 12.5 Authentication and secrets

Authentication should be intentionally boring and safe.

Rules:

```text
- Never store HF tokens in YAML configs, model cards, reports, or logs.
- Read tokens from the logged-in `hf` CLI state or `HF_TOKEN` environment variable.
- Add `.env.example`, not `.env`.
- Add a secret scanner in `forge hf dry-run`.
- In CI, use GitHub Actions secrets, not checked-in credentials.
- Print the target username/org before upload.
- Require `--yes` for non-dry-run publishing.
```

Add:

```text
.env.example
scripts/check_no_secrets.py
src/model_forge/hub/auth.py
src/model_forge/hub/secrets.py
```

`forge hf status` should show:

```text
Authenticated: yes/no
User: keithtyser
Token source: hf-cli-cache | HF_TOKEN | none
Default owner: keithtyser
Default visibility: private
Cache dir:
Dry-run default: true
```

### 12.6 Model repo artifact layout

A Model Forge model repo should be self-explaining.

Recommended layout:

```text
README.md                         # model card
LICENSE
manifest.json                     # exact Model Forge variant manifest
model_forge_report.json
model_forge_report.md
model_forge_report.html
research_basis.md
training_config.yaml              # if trained
training_metrics.json             # if trained
adapter_config.json               # if adapter release
adapter_model.safetensors         # if adapter release
config.json                       # if merged/full model release
model.safetensors.index.json      # if merged/full model release
*.safetensors                     # if merged/full model release
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
quantization_config.json          # if quantized
serving_config_vllm.yaml
serving_card.md
quantization_card.md
eval_results.json
eval_results_table.csv
examples_redacted.md
```

Avoid uploading:

```text
.env
HF_TOKEN
wandb offline credentials
local absolute paths
raw unsafe prompt/response transcripts in public repos
training data without license/provenance
private user data
large transient logs
```

### 12.7 Model Card generator

Add:

```text
src/model_forge/hub/model_card.py
src/model_forge/hub/templates/model_card.md.j2
```

Every model card should include YAML metadata at the top, then human-readable sections.

Example metadata:

```yaml
---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: google/gemma-4-26b-a4b-it
tags:
  - model-forge
  - post-training
  - dgx-spark
  - vllm
  - quantization
  - fp8-kv-cache
datasets:
  - keithtyser/model-forge-training-manifests
  - keithtyser/model-forge-calibration-sets
metrics:
  - accuracy
  - exact_match
  - latency
  - tokens_per_second
---
```

Human-readable sections:

```text
Model summary
Source/base model
What changed
Objective profile
Training or transform recipe
Datasets and manifests
Evaluation summary
Serving and hardware notes
Quantization notes, if applicable
Behavior/risk report summary
Known limitations
Intended use
Out-of-scope use
Repro commands
Citation / research basis
```

For report-only or private research variants, the card should say:

```text
This repository contains a report and reproducibility metadata, not public weights.
```

### 12.8 Dataset repo artifact layout

Dataset repos should be used for four distinct things:

```text
1. Training data manifests
2. Calibration sets
3. Evaluation suites
4. Benchmark/report results
```

Recommended public dataset layout:

```text
README.md                         # dataset card
LICENSE
dataset_infos.json
manifest.json
splits/
  train.jsonl
  validation.jsonl
  test.jsonl
cards/
  data_statement.md
  preprocessing.md
  license_provenance.md
reports/
  eval_provenance_card.md
```

For benchmark results:

```text
README.md
results/
  dgx_spark_vllm_gemma4_26b_a4b_2026_05_18.jsonl
  dgx_spark_vllm_qwen35_9b_2026_06_01.jsonl
leaderboards/
  dgx_spark_open_inference_bench.csv
reports/
  *.html
  *.md
```

Dataset cards should include:

```text
Dataset purpose
Source/provenance
License
Splits
Schema
Collection method
Preprocessing
PII handling
Bias/limitations
Unsafe-content redaction policy, if relevant
How to reproduce
How this dataset is used by Model Forge
```

### 12.9 Dataset publication policy

Add a simple policy:

```text
Public datasets:
  calibration data, benign eval prompts, aggregate metrics, redacted examples,
  public benchmark results, training manifests without sensitive payloads.

Private datasets:
  raw unsafe prompts/responses, sensitive failure cases, private training data,
  datasets with unclear license or provenance.

Gated datasets:
  research datasets that are safe to share with context but should not be indexed
  as a casual public download.
```

Before publishing any dataset, `forge hf publish-dataset --dry-run` should check:

```text
license_present
source_provenance_present
pii_scan_passed
unsafe_examples_redacted_or_private
dataset_card_complete
schema_present
split_sizes_present
no_absolute_paths
no_tokens_or_credentials
```

### 12.10 Versioning and reproducibility

Every Hub upload should be tied back to Model Forge state.

Add these fields to `manifest.json`:

```json
{
  "hub": {
    "repo_id": "keithtyser/model-forge-gemma4-26b-a4b-local-ft-v0",
    "repo_type": "model",
    "visibility": "public",
    "commit_hash": "...",
    "revision": "main",
    "tag": "v0.1.0",
    "published_at": "2026-05-18T00:00:00Z"
  },
  "source_control": {
    "github_repo": "keithtyser/model-forge",
    "git_commit": "...",
    "dirty": false
  },
  "model_forge": {
    "version": "0.1.0",
    "variant_id": "gemma4_26b_a4b_local_ft_v0",
    "objective_profile": "capability_sft"
  }
}
```

Publishing should create a local provenance file:

```text
reports/generated/<run>/hub_publish.json
```

Example:

```json
{
  "repo_id": "keithtyser/model-forge-gemma4-26b-a4b-local-ft-v0",
  "repo_type": "model",
  "commit_url": "...",
  "commit_hash": "...",
  "files_uploaded": ["README.md", "adapter_model.safetensors", "manifest.json"],
  "files_excluded": ["raw_unsafe_outputs.jsonl"],
  "release_class": "public_adapter",
  "release_gates": {
    "model_card_complete": "pass",
    "source_license_checked": "pass",
    "unsafe_examples_redacted": "pass"
  }
}
```

### 12.11 Implementation module

Add:

```text
src/model_forge/hub/
  __init__.py
  auth.py
  repo.py
  upload.py
  model_card.py
  dataset_card.py
  release_classes.py
  validators.py
  collections.py
  publish_plan.py
  provenance.py

templates/hub/
  model_card.md.j2
  dataset_card.md.j2
  report_card.md.j2
  collection_summary.md.j2

configs/hub.yaml
configs/release_classes/*.yaml
```

`configs/hub.yaml`:

```yaml
default_owner: keithtyser
default_visibility: private
default_dry_run: true
repo_prefix: model-forge
cache_dir: ${HF_HOME:-~/.cache/huggingface}

publication:
  require_clean_git: true
  require_model_card: true
  require_dataset_card: true
  require_license: true
  require_eval_results: true
  require_manifest: true
  redact_unsafe_examples_by_default: true
```

### 12.12 CI integration

Add GitHub Actions workflows:

```text
.github/workflows/hub-dry-run.yml
.github/workflows/card-validate.yml
.github/workflows/release-publish.yml
```

Dry-run workflow:

```text
- Validate model cards and dataset cards.
- Validate release-class gates.
- Check no secrets or absolute paths.
- Check linked reports exist.
- Check source licenses and dataset licenses are declared.
- Do not upload anything.
```

Manual release workflow:

```text
- Triggered by workflow_dispatch.
- Requires repo_id, artifact path, release_class.
- Runs dry-run first.
- Publishes only if `confirm_publish=true`.
- Writes `hub_publish.json` back into reports or release artifacts.
```

### 12.13 How Hub publishing changes public content

Your X/blog posts should link to stable Hub artifacts rather than only screenshots.

Example post format:

```text
I released Model Forge Report 001.

Artifact set:
- HF model repo: adapter + card + evals
- HF dataset repo: redacted raw outputs + serving results
- GitHub: code/configs
- Report: HTML + JSON

Claim tested:
FP8 KV cache improved long-context serving on DGX Spark without degrading artifact execution beyond threshold.
```

This gives each post a durable artifact, not just a claim.

### 12.14 First Hub release milestone

The first release should be deliberately conservative:

```text
HF Release 001: Model Forge Gemma FT Report Pack
```

Artifacts:

```text
Model repo:
  adapter-only release or report-only, depending on gates
  model card
  training card
  eval summary
  manifest

Dataset repo:
  redacted eval outputs
  serving benchmark results
  calibration/training manifests, not necessarily raw data
  dataset card

GitHub:
  exact configs
  source code
  roadmap

Report site:
  HTML report
  comparison tables
  charts
```

Acceptance criteria:

```text
- `forge hf publish-model --dry-run` passes.
- `forge hf publish-dataset --dry-run` passes.
- Model card has metadata, base model, objective profile, eval results, limitations, and reproducibility commands.
- Dataset card has schema, splits, provenance, license, PII/redaction policy, and usage notes.
- `hub_publish.json` is written locally.
- X thread links HF model repo, HF dataset repo, GitHub configs, and report page.
```

---

## 13. Public content strategy

### 13.1 Pinned identity

Recommended X bio/pinned thread:

> Building Model Forge: an eval-first post-training + inference optimization workbench for open models on DGX Spark.  
> Focus: fine-tuning, behavior editing, quantization, KV cache, vLLM/SGLang/TensorRT-LLM, kernels, and agentic experiment loops.  
> Every claim comes with configs, raw outputs, reports, and repro commands.

### 13.2 Posting format

Every post should have at least one of:

```text
hypothesis
benchmark chart
failure postmortem
report link
config snippet
profile screenshot/summary
upstream PR
model/dataset/report card
```

Avoid vague posts like:

```text
"This model feels smarter."
```

Prefer:

```text
"Gemma 4 26B on DGX Spark, vLLM, FP8 KV cache, 32k context:
TTFT p95 changed X→Y, ITL p95 changed A→B, artifact execution stayed 14/20,
structured JSON dropped 1 point, refusal classifier changed on 2/106 cases.
Report + raw outputs:"
```

### 13.3 Weekly cadence

```text
Monday:
  hypothesis / research basis / planned sweep

Tuesday:
  first profiler trace or run-card

Wednesday:
  failure or partial result

Thursday:
  benchmark chart / report card

Friday:
  merged code / issue / blog thread / HF card

Weekend:
  deeper writeup or “what I learned”
```

### 13.4 Content series

#### Series 1: DGX Spark Open Inference Bench

Posts:

```text
1. Baseline vLLM on Gemma 4 26B
2. FP8 KV cache sweep
3. Prefix caching on repeated artifact prompts
4. Chunked prefill TTFT/ITL tradeoff
5. Speculative decoding with EAGLE/drafter where available
6. vLLM vs SGLang on same workload
7. NVFP4 serving report
8. Long-context quality vs memory
```

#### Series 2: Quantization Preserves Behavior?

Posts:

```text
1. BF16 vs FP8 KV
2. FP8 weights vs FP8 KV
3. NVFP4 ModelOpt report
4. Layer/component sensitivity
5. QAT/QAD recovery
6. Does quantization change refusal/noncompliance?
```

#### Series 3: Post-training + behavior editing

Posts:

```text
1. LoRA target bug postmortem
2. Base vs local FT vs reference FT
3. FT→ablation vs ablation→FT
4. Candidate frontier report
5. Artifact execution validation catches regressions
6. Multilingual/refusal-direction stability
```

#### Series 4: Agentic research loops

Posts:

```text
1. Agent proposes serving sweep
2. Agent rejects candidates with bad capability deltas
3. Agent writes report card
4. Agent optimizes quantization under objective profile
5. Agent failure analysis
```


## 14. Detailed 12-week execution plan

The first 12 weeks should produce one flagship end-to-end recipe, not a wide
pile of half-validated features. Commands, configs, and cards are not enough. A
week is complete only when its deliverable either runs on local Spark hardware
or is explicitly marked `planned` with a blocker, next command, and owner.

Any CLI shown below is a target interface until `./forge --help` exposes it.
Roadmap docs should say `target CLI` for commands that are not implemented yet.

### Week 1: Validation schema + objective profiles

Ship:

```text
required validation schema
objective profile loader (implemented: `./forge objectives audit`)
configs/objectives/zero_refusal_capability_retention.yaml (implemented)
configs/objectives/quantized_quality_retention.yaml (implemented)
configs/objectives/dgx_spark_latency_throughput.yaml (implemented)
docs/roadmap-status-audit.md (implemented: `./forge roadmap audit --write-doc`)
```

Acceptance criteria:

```text
- Manifest, report card, variant node, and objective schemas all carry
  implementation_status, validation_state, Spark evidence, and promotion_decision.
- Existing roadmap checkboxes are audited into implemented/smoke/Spark-validated
  states.
- Objective profiles define primary metrics, hard constraints, release class,
  and minimum validation evidence.
- CLI examples that do not exist are marked target/planned.
```

### Week 2: Variant graph + evidence ledger

Ship:

```text
src/model_forge/variants/graph.py (implemented: `./forge variants graph`)
variant_node.json writer (implemented: `./forge variants node --write`)
evidence ledger (implemented inside variant node validation metadata)
artifact checksum writer (implemented for supplied artifact files)
retention policy fields (implemented in variant node schema)
```

Acceptance criteria:

```text
- Base, FT, ablated, quantized, served, and published nodes can be represented
  without hard-coded Gemma assumptions.
- Every node records source variant, transforms, artifact paths, checksums,
  validation state, and retention decision.
- A report can show the exact path from base model to candidate.
```

### Week 3: Eval hardening before more transforms

Ship:

```text
golden baseline hardening
artifact execution validators
statistical claim fields
eval provenance card
CLI/doc drift check (implemented: `./forge roadmap cli-drift`)
```

Acceptance criteria:

```text
- HTML artifacts get Playwright DOM/console/screenshot/nonblank checks.
- Python artifacts get compile/help/fixture checks.
- Compare reports include trial count, seed policy, confidence/uncertainty, and
  minimum detectable effect.
- Roadmap CLI examples are checked against implemented commands or marked target.
```

### Week 4: Gemma FT + distributed training correctness

Ship:

```text
Gemma local_ft_v0 completed or failure-carded
Training Method Card
distributed training correctness checklist
two-node torchrun/NCCL preflight evidence
LoRA target failure postmortem
base vs reference FT vs local FT comparison
```

Acceptance criteria:

```text
- Training completed on Spark/Spark x2, or failure includes exact command, logs,
  checkpoint state, resource state, and next fix.
- Multi-node training claims require `./forge cluster torchrun-smoke` evidence
  showing both Spark nodes joined one CUDA/NCCL job.
- Card records sample sharding, global batch semantics, rank count, MoE/router
  handling, unused-parameter policy, cold-start time, steady microstep time, and
  tensor inspection.
- Internal eval, artifact execution eval, and at least one external/provenance
  eval run or documented blocker exist.
```

### Week 5: Behavior-editing frontier

Ship:

```text
noncompliance taxonomy
invalid-refusal vs valid-safety-refusal classifier fields
ablation scorecard
candidate frontier report
redacted risk-report mode
```

Acceptance criteria:

```text
- At least 3 actual local ablation candidates are generated or the backend
  failure is fully reproduced.
- Candidate frontier separates invalid refusals, valid safety refusals,
  capability retention, artifact execution, style drift, and overcompliance.
- Public report can redact unsafe examples and private evidence remains linked.
```

### Week 6: Real serving benchmark on Spark

Ship:

```text
forge bench serve MVP
bounded DGX Spark vLLM sweep
Serving Card
request-level metrics CSV/JSONL
quality/behavior sample under baseline and winning configs
```

Acceptance criteria:

```text
- Sweep cases hit real local OpenAI-compatible endpoints.
- Report records TTFT, ITL, tok/sec, memory, OOM, truncation, stop behavior, and
  endpoint config.
- Best config is selected per workload and marked with validation evidence.
```

### Week 7: Dataset medium pack + bounded FT

Ship:

```text
dataset factory smoke-to-medium scale gate
250-500 accepted eval-adjacent examples
dataset card and contamination report
bounded Spark fine-tune using the packed dataset
```

Acceptance criteria:

```text
- Accepted/rejected rows, judge scores, verifier outputs, holdout overlap, and
  source/license metadata are saved.
- Human spot review or deterministic verification covers a documented sample.
- Bounded FT records whether the data improved target buckets without hidden
  normal-use/style regressions.
```

### Week 8: Quantization foundation

Ship:

```text
forge quantize target CLI or implemented CLI
quantization config schema
calibration manifest
Quantization Card
FP8 KV runtime report
```

Acceptance criteria:

```text
- FP8 KV is benchmarked against BF16/auto KV on real Spark serving endpoints.
- Quantization schema distinguishes runtime KV, HF weight checkpoint, Blackwell
  NVFP4, GGUF, and imported checkpoints.
- Card records behavior, artifact, refusal, valid-refusal, and risk deltas.
```

### Week 9: Blackwell NVFP4 recipe

Ship:

```text
ModelOpt/NVFP4 config
NVFP4 calibration manifest
excluded-module policy
load/eval/serve report
```

Acceptance criteria:

```text
- NVFP4 path is generated or imported on the local Spark workflow, not just
  planned.
- The artifact loads under a declared backend or the blocker is captured with
  exact version/hardware evidence.
- Report compares BF16/source vs NVFP4 on memory, speed, quality, artifact
  execution, refusal behavior, and risk.
```

### Week 10: GGUF recipe

Ship:

```text
GGUF conversion config
llama.cpp load and bench evidence
GGUF tokenizer/chat-template integrity check
GGUF Quantization Card
```

Acceptance criteria:

```text
- At least one HF-source variant converts to GGUF or records a precise conversion
  blocker.
- llama.cpp load/bench runs locally and writes evidence.
- Q4_K_M, Q5_K_M, or Q8_0 result is compared with Spark-native serving on
  quality, behavior, speed, and memory.
```

### Week 11: Second-family generalization

Ship:

```text
Qwen family checklist
Qwen base eval baseline
Qwen serving benchmark
one non-Gemma transform attempt
```

Acceptance criteria:

```text
- Tokenizer, chat template, architecture targets, MoE/router behavior, context
  length, license, quantization feasibility, and serving backend compatibility
  are documented.
- At least one recipe runs far enough to validate or falsify generalization.
- Common code no longer assumes Gemma module names or variants.
```

### Week 12: Flagship report, not broad launch

Ship:

```text
Model Forge Report 001
local report index
validated cards for training/behavior/serving/quantization
HF/report publishing dry run or validated public release
next-scope decision for agents/kernels/upstream PRs
```

Acceptance criteria:

```text
- Report links manifests, variant graph, raw outputs, eval cards, serving cards,
  quantization cards, and failure cards.
- Public artifacts include only validation-backed claims.
- HF publishing only uploads locally validated artifacts whose release class
  allows publication; otherwise publish report-only.
- Agent runner, kernel work, and upstream PRs are deferred unless the flagship
  path is already validated.
```

---

## 15. Six-month roadmap

Six-month progress should be measured by validated recipes, not feature count.
Each month should end with at least one local Spark run that transformed,
served, evaluated, or benchmarked a real model and left enough artifacts for
another agent to reproduce the outcome.

### Month 1: Foundation

```text
validation schema
objective profiles
variant graph
run manifests
report cards
artifact execution validation
Gemma FT or failure-card
serving bench MVP
```

Success metric:

```text
No roadmap item can be marked complete without implementation status, validation
state, evidence path, and promotion decision. One Gemma path has real Spark
training/eval/serving evidence or a high-quality failure card.
```

### Month 2: First flagship loop

```text
Gemma behavior frontier
dataset medium pack
bounded dataset-driven FT
vLLM sweeps
FP8 KV report
first report index
```

Success metric:

```text
One complete Spark-validated Gemma report: base vs FT vs abli vs FT→abli, with
actual local transforms, serving metrics, eval outputs, raw outputs, and failure
notes.
```

### Month 3: Quantization lab

```text
FP8 W8A8
Blackwell ModelOpt/NVFP4
GGUF/llama.cpp
calibration manifests
quantization behavior reports
layer sensitivity scans
```

Success metric:

```text
At least one Spark-native quantization recipe and one GGUF recipe have complete
quantization cards with quality/behavior/serving deltas, or exact blockers.
```

### Month 4: Multi-family expansion

```text
Qwen hardened
Llama or Mistral family added
family checklist finalized
SGLang backend initial support
```

Success metric:

```text
The same workbench loop runs on at least one non-Gemma family and the second
family checklist exposes all required family-specific changes.
```

### Month 5: Agentic experiment automation

```text
optimize-serving agent
optimize-quantization agent
optimize-behavior-edit agent
agent run cards
budget/retry policies
```

Success metric:

```text
Agent completes at least one bounded research loop using the same objective
profiles, manifests, evidence ledger, and promotion gates as human runs.
```

### Month 6: Kernel/perf + upstream credibility

```text
profiling integration
kernel microbench harness
FlashInfer/vLLM/SGLang benchmark contribution
2-3 upstream PRs
```

Success metric:

```text
At least one merged upstream PR and one profiler-backed perf report with end-to-end impact.
```

---

## 16. Prioritized backlog

Backlog status must not use a single checkbox for fundamentally different
states. Use this table format:

```text
implementation_status:
  not_started | scaffolded | implemented | wired_to_cli | tested

validation_state:
  planned | smoke_validated | spark_single_node_validated |
  spark_cluster_validated | generalizable
```

### P0: Foundation

```text
MF-0000 Convert legacy [x] backlog into implementation_status + validation_state. implementation_status=tested validation_state=planned
MF-0001 Add required validation schema to manifests, cards, objectives, and variant nodes. implementation_status=scaffolded validation_state=planned
MF-0002 Add evidence ledger with command, node count, topology, logs, metrics, checksums, and promotion decision. implementation_status=tested validation_state=planned
MF-0003 Add objective profile loader and objective audit. implementation_status=tested validation_state=planned
MF-0004 Add configs/objectives/zero_refusal_capability_retention.yaml. implementation_status=tested validation_state=planned
MF-0005 Add configs/objectives/quantized_quality_retention.yaml. implementation_status=tested validation_state=planned
MF-0006 Add configs/objectives/dgx_spark_latency_throughput.yaml. implementation_status=tested validation_state=planned
MF-0007 Add variant graph and variant_node.json writer. implementation_status=tested validation_state=planned
MF-0008 Add artifact checksum and retention policy fields. implementation_status=tested validation_state=planned
MF-0009 Add eval provenance card. implementation_status=tested validation_state=smoke_validated
MF-0010 Add golden baseline create/check hardening. implementation_status=tested validation_state=smoke_validated
MF-0011 Add CLI/doc drift check for roadmap command examples. implementation_status=tested validation_state=smoke_validated
MF-0012 Finish Gemma local FT evaluation or failure-card it. implementation_status=implemented validation_state=spark_single_node_validated
MF-0013 Publish Training Method Card with distributed training correctness. implementation_status=tested validation_state=smoke_validated
```

### P0: Behavior editing

```text
MF-0101 Add noncompliance taxonomy. implementation_status=scaffolded validation_state=planned
MF-0102 Add invalid-refusal vs valid-safety-refusal classifier fields. implementation_status=scaffolded validation_state=planned
MF-0103 Add harmful-overcompliance and behavior-drift scoring. implementation_status=implemented validation_state=smoke_validated
MF-0104 Add behavior edit scorecard. implementation_status=tested validation_state=smoke_validated
MF-0105 Add candidate frontier report from actual local candidates. implementation_status=scaffolded validation_state=planned
MF-0106 Add redacted public risk-report mode and private raw-output retention. implementation_status=scaffolded validation_state=planned
MF-0107 Add release classes and release-class validators. implementation_status=tested validation_state=smoke_validated
MF-0108 Add zero_refusal_capability_retention objective gates. implementation_status=tested validation_state=planned
```

### P0: Serving

```text
MF-0200 Add generic cluster inventory planner and DGX Spark x2 example. implementation_status=tested validation_state=spark_cluster_validated
MF-0201 Add forge bench serve. implementation_status=tested validation_state=smoke_validated
MF-0202 Add DGX Spark vLLM sweep config. implementation_status=tested validation_state=smoke_validated
MF-0203 Add serving workload definitions. implementation_status=tested validation_state=smoke_validated
MF-0204 Add Serving Card. implementation_status=tested validation_state=smoke_validated
MF-0205 Add TTFT/ITL/memory/tok-sec capture. implementation_status=tested validation_state=smoke_validated
MF-0206 Add quality/behavior sampled eval under serving configs. implementation_status=tested validation_state=smoke_validated
MF-0207 Mark serving work complete only after real endpoint evidence is attached. implementation_status=scaffolded validation_state=planned
MF-0208 Add two-node torchrun/NCCL Spark preflight. implementation_status=tested validation_state=spark_cluster_validated
```

### P0: Artifact validation

```text
MF-0251 Add Playwright HTML validation. implementation_status=tested validation_state=smoke_validated
MF-0252 Add Python artifact compile/run validation. implementation_status=tested validation_state=smoke_validated
MF-0253 Add artifact screenshots and nonblank canvas/WebGL checks. implementation_status=tested validation_state=smoke_validated
MF-0254 Add Artifact Execution Card. implementation_status=tested validation_state=smoke_validated
MF-0255 Add artifact execution score to compare report. implementation_status=tested validation_state=smoke_validated
MF-0256 Require artifact validation before artifact-generation improvement claims. implementation_status=tested validation_state=smoke_validated
```

### P1: Quantization

```text
MF-0301 Add forge quantize or mark target CLI until implemented. implementation_status=tested validation_state=smoke_validated
MF-0302 Add calibration dataset manifests. implementation_status=scaffolded validation_state=planned
MF-0303 Add FP8 KV behavior report. implementation_status=scaffolded validation_state=planned
MF-0304 Add FP8 W8A8 pipeline. implementation_status=scaffolded validation_state=planned
MF-0305 Add Blackwell ModelOpt/NVFP4 pipeline. implementation_status=wired_to_cli validation_state=planned
MF-0306 Add GGUF/llama.cpp conversion and quantization pipeline. implementation_status=not_started validation_state=planned
MF-0307 Add Quantization Card. implementation_status=tested validation_state=smoke_validated
MF-0308 Add layer/component sensitivity scan. implementation_status=not_started validation_state=planned
MF-0309 Add quantization-preserves-behavior report. implementation_status=scaffolded validation_state=planned
MF-0310 Add tokenizer/chat-template preservation checks for GGUF and quantized exports. implementation_status=not_started validation_state=planned
MF-0311 Add import-existing-quantized-checkpoint path for already-available FP8/NVFP4/GGUF artifacts. implementation_status=tested validation_state=smoke_validated
```

### P1: Dataset factory

```text
MF-0351 Add configs/datasets/*.yaml plan schema. implementation_status=tested validation_state=smoke_validated
MF-0352 Add forge data plan/seed/generate. implementation_status=tested validation_state=smoke_validated
MF-0353 Add forge data judge with multi-axis quality scores. implementation_status=tested validation_state=smoke_validated
MF-0354 Add forge data verify for JSON/code/artifact examples. implementation_status=tested validation_state=smoke_validated
MF-0355 Add forge data filter with dedupe, holdout-overlap, and license checks. implementation_status=tested validation_state=smoke_validated
MF-0356 Add forge data pack with dataset.jsonl, manifest.yaml, and dataset_card.md. implementation_status=tested validation_state=smoke_validated
MF-0357 Add accepted/rejected row reports with rejection reasons. implementation_status=tested validation_state=smoke_validated
MF-0358 Add generated dataset HF publish path. implementation_status=tested validation_state=smoke_validated
MF-0359 Add Gemma local_ft_v1 eval-adjacent dataset recipe. implementation_status=tested validation_state=smoke_validated
MF-0360 Add eval-feedback loop that proposes next dataset skills from failures. implementation_status=tested validation_state=smoke_validated
MF-0361 Add forge data review with curation flags and scale-up gate. implementation_status=tested validation_state=smoke_validated
MF-0362 Add smoke_pack, medium_pack, and training_pack promotion gates. implementation_status=scaffolded validation_state=smoke_validated
MF-0363 Require bounded Spark fine-tune evidence before dataset recipe is marked validated. implementation_status=not_started validation_state=planned
```

### P1: Hugging Face Hub publishing

```text
MF-0501 Add forge hf status/login/whoami. implementation_status=tested validation_state=smoke_validated
MF-0502 Add forge hf plan-model. implementation_status=tested validation_state=smoke_validated
MF-0503 Add forge hf publish-model --dry-run. implementation_status=tested validation_state=smoke_validated
MF-0504 Add forge hf publish-dataset --dry-run. implementation_status=scaffolded validation_state=smoke_validated
MF-0505 Add Hub model card generator. implementation_status=tested validation_state=smoke_validated
MF-0506 Add Hub dataset card generator. implementation_status=tested validation_state=smoke_validated
MF-0507 Add release-class validators. implementation_status=tested validation_state=smoke_validated
MF-0508 Add hub_publish.json provenance writer. implementation_status=tested validation_state=smoke_validated
MF-0509 Add no-secrets/no-absolute-path publish validator. implementation_status=tested validation_state=smoke_validated
MF-0510 Add redacted-output dataset publishing path. implementation_status=tested validation_state=smoke_validated
MF-0511 Block public checkpoint upload unless validation state and release class allow it. implementation_status=tested validation_state=smoke_validated
```

### P1: Multi-family

```text
MF-0601 Harden Qwen family config. implementation_status=tested validation_state=smoke_validated
MF-0602 Add adding-model-family checklist. implementation_status=tested validation_state=smoke_validated
MF-0603 Add tokenizer/chat-template round-trip tests. implementation_status=tested validation_state=smoke_validated
MF-0604 Add architecture target discovery and MoE/router exclusion checks. implementation_status=tested validation_state=smoke_validated
MF-0605 Add Llama/Mistral family plan. implementation_status=tested validation_state=smoke_validated
MF-0606 Ensure common code has no Gemma-only assumptions. implementation_status=tested validation_state=smoke_validated
```

### P2: Agents

```text
MF-0701 Add agent experiment schema. implementation_status=tested validation_state=smoke_validated
MF-0702 Add forge agent optimize-serving. implementation_status=tested validation_state=smoke_validated
MF-0703 Add forge agent optimize-quantization. implementation_status=tested validation_state=smoke_validated
MF-0704 Add forge agent optimize-behavior-edit. implementation_status=tested validation_state=smoke_validated
MF-0705 Add agent run card. implementation_status=tested validation_state=smoke_validated
MF-0706 Add automatic ledger update. implementation_status=tested validation_state=smoke_validated
```


### P2: Kernel/perf

```text
MF-0801 Add Nsight profile integration. implementation_status=tested validation_state=smoke_validated
MF-0802 Add profile summarizer. implementation_status=tested validation_state=smoke_validated
MF-0803 Add bench kernel rmsnorm. implementation_status=tested validation_state=smoke_validated
MF-0804 Add bench kernel rope. implementation_status=tested validation_state=smoke_validated
MF-0805 Add bench kernel dequant. implementation_status=tested validation_state=smoke_validated
MF-0806 Add bench kernel kv-layout. implementation_status=tested validation_state=smoke_validated
MF-0807 Add Kernel Card. implementation_status=tested validation_state=smoke_validated
MF-0808 Open first upstream PR. implementation_status=scaffolded validation_state=planned
```

### P3: Advanced serving

```text
MF-0901 Add SGLang backend. implementation_status=tested validation_state=smoke_validated
MF-0902 Add TensorRT-LLM backend. implementation_status=tested validation_state=smoke_validated
MF-0903 Add disaggregated prefill/decode experiment profile. implementation_status=tested validation_state=smoke_validated
MF-0904 Add LMCache/NIXL research-watch hooks. implementation_status=tested validation_state=smoke_validated
MF-0905 Add multi-node/distributed-KV placeholder architecture. implementation_status=tested validation_state=smoke_validated
```


---

## 17. Repo structure target

```text
configs/
  model_families/
  objectives/
  datasets/
  clusters/
  hub.yaml
  release_classes/
  research_registry.yaml
  finetuning/
  preference/
  rl/
  abliteration/
  quantization/
  serving/
  sweeps/
  promotion_profiles/
  validation/

datasets/
  finetuning/
  preference/
  calibration/
  eval_holdouts/
  seeds/
  source_corpora/
  manifests/
  generated/
  rejected/

evals/
  prompts/
  rubrics/
  external/
  provenance/

src/model_forge/
  research/
  data/
  hub/
  cards/
  objectives/
  variants/
  families/
  serving/
  benchmarks/
  quantization/
  finetuning/
  preference/
  rl/
  behavior_editing/
  evals/
  scoring/
  reports/
  artifacts/
  validation/
  agents/
  hardware/
  profiling/
  kernels/

reports/
  generated/
  baselines/
  datasets/
  public/
  hub_publish/
  hub_publish/plans/
  hub_publish/manifests/
  evidence/
  public/dgx-spark-open-inference-bench/

templates/
  hub/
    model_card.md.j2
    dataset_card.md.j2
    report_card.md.j2

docs/
  research/
  roadmap.md
  reporting-standard.md
  objective-profiles.md
  validation-contract.md
  distributed-training.md
  publication-policy.md
  hub-publishing.md
  adding-model-family.md
  dgx-spark-benchmarks.md
  quantization.md
  behavior-editing.md
  agent-runner.md
  kernel-perf.md
  huggingface-publishing.md
  model-card-template.md
  dataset-card-template.md
```

---

## 18. Frontier-lab skill map

This roadmap is designed to create artifacts that map directly to frontier-lab hiring signals.

| Lab skill signal | Model Forge artifact |
|---|---|
| Post-training | SFT/DPO/GRPO method cards, dataset manifests, promotion gates |
| Evals | Eval provenance cards, golden baselines, external benchmark bridges |
| Model behavior | Behavior-editing scorecards, noncompliance taxonomy, risk reports |
| Inference systems | DGX Spark serving benchmarks, vLLM/SGLang/TensorRT-LLM comparisons |
| Quantization | FP8, Blackwell NVFP4/ModelOpt, GGUF/llama.cpp reports, calibration manifests, and behavior-preservation cards |
| GPU performance | Nsight summaries, kernel microbenchmarks, roofline notes |
| Agents | Experiment runner, agent run cards, Terminal/SWE/WebArena adapters |
| Reproducibility | manifests, raw outputs, report site, exact commands |
| Open-source credibility | upstream PRs, clean docs, Hugging Face model/dataset cards, gated releases, reproducible upload manifests |

Your best public proof should look like this:

```text
Model Forge Report 001:
Gemma 4 26B on DGX Spark
base vs local FT vs ablated vs FT→ablated
with SOTA research basis, objective profiles, eval deltas,
artifact execution, FP8 KV serving metrics, quantization plan,
raw outputs, and repro commands.
```

---

## 19. Immediate next commits

Do these in order:

```text
1. Convert roadmap/backlog checkboxes into implementation_status +
   validation_state.
2. Add required validation/evidence fields to manifest writer, report cards,
   variant nodes, objective profiles, and promotion decisions.
3. Add CLI/doc drift check so target commands are not confused with implemented
   commands.
4. Add configs/objectives/zero_refusal_capability_retention.yaml with invalid
   refusal, valid safety-refusal, overcompliance, and release-class gates. Done
   as a config/audit profile; Spark validation remains objective-specific.
5. Add configs/objectives/quantized_quality_retention.yaml with FP8, Blackwell
   NVFP4, GGUF, behavior, artifact, and tokenizer/template gates. Done as a
   config/audit profile; completed quantization evidence remains open.
6. Add configs/objectives/dgx_spark_latency_throughput.yaml. Done as a
   config/audit profile; real endpoint evidence remains open.
7. Add variant graph + evidence ledger + artifact checksum/retention policy.
8. Finish/evaluate Gemma local FT or write a Training Method failure card with
   distributed training correctness evidence.
9. Run one real Spark serving benchmark and attach evidence to the Serving Card.
10. Implement or mark target CLI for `forge quantize`; add FP8 KV, NVFP4, and
    GGUF config stubs with explicit validation blockers.
```

A strong first public post:

> “I’m making Model Forge evidence-gated by construction. A feature is not done
> because a command exists; it is done when it transforms, serves, evaluates, or
> quantizes a real model on Spark and leaves reproducible artifacts. First target:
> Gemma 4 26B on Spark across FT, behavior editing, FP8 KV, Blackwell NVFP4,
> GGUF, artifact execution, and report cards.”

---

## 20. Source registry

The following sources ground this roadmap. Current tracked entries live in
`configs/research_registry.yaml`; validate them with `./forge research audit`.

### Current Model Forge repo

- Model Forge README: https://github.com/keithtyser/model-forge
- Evaluation strategy: https://raw.githubusercontent.com/keithtyser/model-forge/main/docs/evaluation-strategy.md
- DGX Spark optimization notes: https://raw.githubusercontent.com/keithtyser/model-forge/main/docs/spark-optimizations.md
- Fine-tuning workflow: https://raw.githubusercontent.com/keithtyser/model-forge/main/docs/finetuning.md
- Abliteration workflow: https://raw.githubusercontent.com/keithtyser/model-forge/main/docs/abliteration.md
- Experiment ledger: https://raw.githubusercontent.com/keithtyser/model-forge/main/docs/experiment-ledger.md
- AGENTS.md: https://raw.githubusercontent.com/keithtyser/model-forge/main/AGENTS.md

### Hugging Face Hub publishing

- Hugging Face Hub CLI docs: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Hugging Face upload files/folders docs: https://huggingface.co/docs/huggingface_hub/en/guides/upload
- Hugging Face uploading models docs: https://huggingface.co/docs/hub/models-uploading
- Hugging Face model cards docs: https://huggingface.co/docs/hub/model-cards
- Hugging Face dataset cards docs: https://huggingface.co/docs/hub/datasets-cards
- huggingface_hub Repository Cards API: https://huggingface.co/docs/huggingface_hub/en/package_reference/cards

### Frontier-lab skill signals

- OpenAI, Software Engineer, Inference — AMD GPU Enablement: https://openai.com/careers/software-engineer-inference-amd-gpu-enablement-san-francisco/
- OpenAI, Research Engineer / Research Scientist, Post-Training: https://openai.com/careers/research-engineer-research-scientist-post-training-san-francisco/
- OpenAI, Researcher, Agentic Post-Training: https://openai.com/careers/researcher-agentic-post-training-san-francisco/
- Anthropic, Performance Engineer GPU: https://job-boards.greenhouse.io/anthropic/jobs/4926227008
- Anthropic careers index: https://www.anthropic.com/careers/jobs

### Hardware

- NVIDIA DGX Spark product page: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- NVIDIA DGX Spark hardware guide: https://docs.nvidia.com/dgx/dgx-spark/hardware.html

### Post-training

- Hugging Face TRL docs: https://huggingface.co/docs/trl/en/index
- TRL v1 release blog: https://huggingface.co/blog/trl-v1
- DPO: https://arxiv.org/abs/2305.18290
- ORPO: https://arxiv.org/abs/2403.07691
- SimPO: https://arxiv.org/abs/2405.14734
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- GSPO: https://arxiv.org/abs/2507.18071
- Qwen GSPO blog: https://qwenlm.github.io/blog/gspo/
- verl DAPO recipe: https://verl.readthedocs.io/en/latest/algo/dapo.html

### Dataset generation / selection

- Self-Instruct: https://arxiv.org/abs/2212.10560
- WizardLM / Evol-Instruct: https://arxiv.org/abs/2304.12244
- Instruction Backtranslation: https://arxiv.org/abs/2308.06259
- Magpie: https://arxiv.org/abs/2406.08464
- SelfCodeAlign: https://arxiv.org/abs/2410.24198
- DEITA: https://openreview.net/pdf?id=BTKAeLqLMw
- LIMA: https://arxiv.org/abs/2305.11206
- UltraFeedback: https://github.com/OpenBMB/UltraFeedback

### Behavior editing / refusal

- Refusal in Language Models Is Mediated by a Single Direction: https://arxiv.org/abs/2406.11717
- There Is More to Refusal in Large Language Models than a Single Direction: https://arxiv.org/html/2602.02132v1
- Refusal Direction is Universal Across Safety-Aligned Languages: https://arxiv.org/html/2505.17306v2
- Multi-Directional Refusal Suppression in Language Models: https://ojs.aaai.org/index.php/AAAI/article/view/40551/44512
- XSTest: https://arxiv.org/abs/2308.01263
- XSTest repo: https://github.com/paul-rottger/xstest
- XSTest in Inspect Evals: https://ukgovernmentbeis.github.io/inspect_evals/evals/knowledge/xstest/
- MLCommons AILuminate: https://github.com/mlcommons/ailuminate
- MLCommons AILuminate paper: https://arxiv.org/html/2503.05731v1
- MLCommons Jailbreak Benchmark: https://mlcommons.org/ailuminate/jailbreak/
- Jailbreak methodology: https://mlcommons.org/ailuminate/jailbreak-methodology/

### Evaluation

- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- lm-eval paper: https://arxiv.org/pdf/2405.14782
- Inspect AI: https://inspect.aisi.org.uk/
- Inspect AI repo: https://github.com/UKGovernmentBEIS/inspect_ai
- Inspect Evals: https://ukgovernmentbeis.github.io/inspect_evals/
- OpenAI Evals: https://github.com/openai/evals
- OpenAI API Evals guide: https://developers.openai.com/api/docs/guides/evals
- promptfoo red teaming: https://www.promptfoo.dev/docs/red-team/
- IFEval in lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/README.md
- IndicIFEval: https://arxiv.org/abs/2602.22125
- LiveCodeBench: https://arxiv.org/abs/2403.07974
- LiveCodeBench site: https://livecodebench.github.io/
- BigCodeBench: https://arxiv.org/abs/2406.15877
- BigCodeBench repo: https://github.com/bigcode-project/bigcodebench
- SWE-bench: https://www.swebench.com/
- SWE-bench Verified: https://www.swebench.com/verified.html
- Terminal-Bench: https://www.tbench.ai/
- Terminal-Bench 2.0 paper: https://arxiv.org/html/2601.11868v1
- GAIA: https://arxiv.org/abs/2311.12983
- WebArena: https://arxiv.org/abs/2307.13854
- WebArena site: https://webarena.dev/
- OSWorld: https://arxiv.org/abs/2404.07972
- OSWorld site: https://os-world.github.io/
- τ-bench: https://arxiv.org/abs/2406.12045
- tau2-bench repo: https://github.com/sierra-research/tau2-bench
- ArtifactsBench: https://artifactsbenchmark.github.io/
- ArtifactsBenchmark repo: https://github.com/Tencent-Hunyuan/ArtifactsBenchmark
- DesignBench: https://arxiv.org/html/2506.06251v3
- WebDev Arena / Text Arena: https://epoch.ai/benchmarks/webdev-arena

### Inference serving

- vLLM docs: https://docs.vllm.ai/en/latest/
- vLLM GitHub: https://github.com/vllm-project/vllm
- vLLM quantized KV cache: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM FP8 W8A8: https://docs.vllm.ai/en/latest/features/quantization/fp8/
- vLLM quantization index: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM speculative decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM disaggregated prefill: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM optimization docs: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM anatomy blog: https://vllm.ai/blog/anatomy-of-vllm
- PagedAttention paper: https://arxiv.org/pdf/2309.06180
- SGLang paper: https://arxiv.org/abs/2312.07104
- SGLang blog: https://lmsys.org/blog/2024-01-17-sglang/
- TensorRT-LLM docs: https://nvidia.github.io/TensorRT-LLM/
- TensorRT-LLM disaggregated serving blog: https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html
- TensorRT-LLM chunked prefill blog: https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
- NVIDIA Dynamo disaggregated serving docs: https://docs.nvidia.com/dynamo/v1.1.0/design-docs/disaggregated-serving

### Quantization / KV cache

- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer
- ModelOpt unified HF checkpoint docs: https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html
- ModelOpt vLLM docs: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- NVIDIA NVFP4 technical blog: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- vLLM LLM Compressor docs: https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/
- LLM Compressor FP8 example: https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- TurboQuant Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- TurboQuant paper: https://arxiv.org/abs/2504.19874
- FP4 sensitivity analysis: https://arxiv.org/abs/2603.08747
- NVFP4 QAD: https://arxiv.org/html/2601.20088v1
- NVIDIA QAT blog: https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/
- PyTorch QAT blog: https://pytorch.org/blog/quantization-aware-training/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- GGUF format docs: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp GGUF docs: https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format
- llama.cpp quantization evaluation: https://arxiv.org/abs/2601.14277

### Kernels / GPU performance

- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- FlashAttention-4: https://arxiv.org/abs/2603.05451
- FlashInfer paper: https://arxiv.org/abs/2501.01005
- FlashInfer repo: https://github.com/flashinfer-ai/flashinfer
- NVIDIA FlashInfer blog: https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/
- Triton docs: https://triton-lang.org/
- OpenAI Triton announcement: https://openai.com/index/triton/


## 21. Final recommendation

Build the first 12 weeks around **one flagship report**, not a pile of features.

The report:

```text
Model Forge Report 001:
Gemma 4 26B A4B on DGX Spark
base vs local FT vs reference FT vs ablated vs FT→ablated
objective profiles
research basis
capability deltas
artifact execution
refusal/noncompliance scorecard
risk metrics
FP8 KV serving sweep
Blackwell NVFP4 quantization card or exact blocker
GGUF/llama.cpp quantization card or exact blocker
raw outputs
repro commands
HF model card / dataset card / hub_publish.json
```

Everything in this roadmap exists to make that report credible, reproducible, and obviously relevant to frontier-lab work.

After that, repeat the loop on Qwen to prove generalization. Agentic
optimization, kernel work, broad Hub publication, and upstream PRs should follow
only after the flagship loop has real Spark evidence.

The project becomes impressive when an outside person can look at Model Forge and say:

> “This is not a demo. This is a reproducible local research system for post-training and inference optimization.”
