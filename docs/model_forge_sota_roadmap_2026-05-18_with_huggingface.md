# Model Forge SOTA-Grounded Roadmap

**Prepared for:** Keith Tyser  
**Date snapshot:** May 18, 2026  
**Target repo:** [`keithtyser/model-forge`](https://github.com/keithtyser/model-forge)  
**Primary goal:** Turn Model Forge into a research-grounded, reproducible workbench for post-training, behavior editing, quantization, serving optimization, kernel/performance experiments, and agentic experiment automation.

---

## Executive summary

Model Forge should become a system that answers one question better than almost any personal project can:

> **Given a base open model, what post-training, behavior-editing, quantization, and serving configuration best satisfies a declared objective on a declared hardware target, with reproducible evidence and no hidden regressions?**

Your current repo is already unusually close to this. It is model-family driven, evaluation-first, uses DGX Spark as a hardware profile, tracks raw responses and artifacts, compares base/fine-tuned/ablated/combined variants, and records an experiment ledger. The missing step is to make the repo **SOTA-grounded by construction**.

That means every major feature should have:

```text
research claim
→ implementation hook
→ evaluation hook
→ report section
→ limitation / failure mode
→ next experiment
```

The roadmap below turns that into concrete work:

1. **Research registry**: a dated, machine-readable registry of papers, docs, benchmarks, and claims as of May 18, 2026.
2. **Objective profiles**: explicit profiles such as `capability_sft`, `zero_refusal_capability_retention`, `quantized_quality_retention`, `dgx_spark_latency_throughput`, and `agentic_tool_use`.
3. **Dataset factory**: a repeatable pipeline for generating, judging, filtering, packing, and publishing high-quality fine-tuning datasets from eval gaps and current source material.
4. **Variant graph**: base → fine-tune → behavior edit → quantize → serve optimize → evaluate → publish, with every node and edge recorded.
5. **Evaluation hardening**: run manifests, golden baselines, provenance cards, external benchmarks, artifact execution validation, judge calibration, and risk reports.
6. **DGX Spark inference lab**: a public benchmark suite for vLLM/SGLang/TensorRT-LLM on a bandwidth-constrained 128 GB unified-memory machine.
7. **Quantization lab**: FP8 KV cache, FP8 W8A8, NVFP4/ModelOpt, GPTQ/AWQ/INT paths, QAT/QAD recovery, and behavior-preservation tests.
8. **Behavior-editing lab**: ablation as one objective profile, with zero-refusal targets measured separately from capability retention, behavior drift, and deployment risk.
9. **Kernel/perf track**: profiler-first microbenchmarks tied to serving outcomes, not toy kernels detached from model performance.
10. **Agentic experiment runner**: an agent that proposes, runs, evaluates, rejects/promotes, and writes reports under bounded budgets.
11. **Public proof layer**: report site, X threads, Hugging Face model/dataset repos, release cards, gated/private publication flows, and upstream PRs.

The strongest public identity is:

> **“I build eval-first systems for improving and serving open models on constrained local AI hardware. Every claim comes with configs, raw outputs, latency numbers, quality/safety deltas, and reproducible reports.”**

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
| External evals are useful but not yet central | Need credibility beyond internal prompts | Add eval provenance cards, benchmark adapters, and version tracking |
| High-quality training data is imported manually | Fine-tuning quality will be capped by ad hoc public datasets and stale mixtures | Add `forge data *`: seed, generate, judge, filter, pack, version, and publish datasets |
| Artifact execution validation is planned, not shipped | Generated HTML/code can look good but fail | Add Playwright/browser and Python execution validators |
| Quantization is not first-class | It is central to DGX Spark and frontier inference work | Add `forge quantize`, quantization cards, and behavior-preservation reports |
| Serving optimization is not benchmarked as a matrix | vLLM flags are present but not yet a benchmark product | Add `forge bench serve` and a DGX Spark leaderboard |
| Behavior editing lacks objective profiles | Ablation objective and deployment decision should be separated | Add `zero_refusal_capability_retention` profile plus release classes |
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
  Behavior-editing objective: drive refusal behavior to zero on the configured
  refusal suite while preserving source-model capability and reporting risk.

primary_goal:
  metric: refusal_rate
  suite: refusal_primary
  target: 0.0

hard_constraints:
  normal_use_score_delta_vs_source: ">= -0.02"
  structured_output_delta_vs_source: ">= -0.02"
  code_debug_delta_vs_source: ">= -0.03"
  artifact_execution_delta_vs_source: ">= -0.05"
  external_benchmark_delta_vs_source: ">= -0.03"

required_reports:
  - capability_retention
  - behavior_drift
  - unsafe_overcompliance
  - harmful_detail_risk
  - redacted_examples
  - serving_delta

tie_breakers:
  - lower_style_drift
  - higher_artifact_execution_score
  - lower_latency_p95
  - higher_output_tokens_per_second

release_default: research_report_only

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
  - kvquant
  - kivi
  - turboquant
  - fp4_sensitivity_2026
  - nvfp4_qad_2026
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
    "local_path": "/home/ktyser/models/...",
    "source_revision": "...",
    "merged_adapters": true
  },
  "evaluation_status": "complete",
  "serving_status": "benchmarked",
  "publication_class": "research_report_only"
}
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
Did unsafe overcompliance/risk change?
Which layers/components were most sensitive?
Which serving backend actually loaded the checkpoint correctly?
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
  "local_checkpoint": "/home/ktyser/models/...",
  "variant_graph_node": "...",
  "objective_profile": "capability_sft",
  "hardware": {
    "profile": "dgx_spark",
    "gpu": "NVIDIA GB10",
    "memory_gb": 128,
    "memory_bandwidth_gb_s": 273
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
    "report_html": "report.html"
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
3. Promotion / rejection decision
4. Variant graph
5. Research basis
6. Capability deltas
7. Instruction-following deltas
8. Structured-output deltas
9. Coding/debugging deltas
10. Artifact generation and execution deltas
11. Refusal / noncompliance deltas
12. Risk metrics
13. Serving metrics
14. Quantization metrics, if applicable
15. Raw examples
16. Repro commands
17. Known failures
18. Next experiments
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
Refusal rate:
Capability retention:
Structured-output retention:
Code/debug retention:
Artifact execution retention:
Behavior drift:
Unsafe overcompliance / harmful-detail risk:
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
Calibration set:
Calibration sample count:
Calibration sequence length:
Excluded modules:
Hardware target:
Serving engine:
Memory delta:
TTFT delta:
ITL delta:
Output tok/sec delta:
Quality delta:
Instruction-following delta:
Code/artifact delta:
Refusal-behavior delta:
Risk delta:
Layer sensitivity summary:
Known regressions:
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

You already have `golden-summary` / `golden-check`. Promote this to a first-class API:

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

Commands:

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
./forge data seed gemma4_26b_a4b local_ft_v1
./forge data generate gemma4_26b_a4b local_ft_v1
./forge data judge gemma4_26b_a4b local_ft_v1
./forge data filter gemma4_26b_a4b local_ft_v1
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
targeted rather than massive. Start with roughly `500-2000` accepted examples
covering the v0 gaps:

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

---

## 7. Behavior-editing / ablation roadmap

### 7.1 Treat ablation as one objective profile

Your clarified product framing is good:

> Ablation is an option in the workbench. The whole point of the ablation option is to remove refusal behavior while preserving model performance.

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
  "answer_substance_score": 0.05,
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
1. Filter to candidates with refusal_rate == 0 on primary refusal suite.
2. Filter to candidates above capability floors.
3. Filter to candidates without severe format/artifact collapse.
4. Among survivors, choose lowest behavior/style drift.
5. Then choose best capability score.
6. Then choose best serving performance.
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

Questions:

```text
Does FT→ablation preserve FT gains?
Does ablation→FT reintroduce refusals?
Does behavior editing damage structured output?
Does behavior editing damage artifact execution?
Does behavior editing survive quantization?
Does FP8 KV cache or NVFP4 change refusal behavior?
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

A candidate can satisfy `zero_refusal_capability_retention` and still be `public_research_report` or `private_research_checkpoint`. Keep **research success**, **model quality**, and **deployment decision** separate.

### 7.6 Ablation commands

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
  qwen3_30b_a3b_fp8_w8a8.yaml
  qwen3_30b_a3b_nvfp4_modelopt.yaml
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
  local_path: /home/ktyser/models/...
  card: reports/generated/.../quantization_card.md
```

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

#### Study 4: Component sensitivity

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
2. Calibration set
3. Excluded modules
4. Load path and serving backend
5. Memory/perf deltas
6. Capability deltas
7. Structured-output deltas
8. Artifact execution deltas
9. Refusal/noncompliance deltas
10. Risk deltas
11. Long-context deltas
12. Sensitivity scan
13. Failure examples
14. Repro command
```

---

## 9. DGX Spark serving benchmark roadmap

### 9.1 Why DGX Spark is the niche

DGX Spark is not an H100. Its strength is local capacity and Blackwell/NVIDIA software access; its constraint is bandwidth. That is exactly why it is a useful inference lab.

Your public angle:

> **“DGX Spark Open Inference Bench: reproducible latency, memory, quality, and behavior tradeoffs for open models on local Blackwell unified-memory hardware.”**

### 9.2 Add `forge bench serve`

```bash
./forge bench serve gemma4_26b_a4b base \
  --engine vllm \
  --hardware dgx_spark \
  --workload short_chat,long_prefill,decode_heavy,structured_json,artifact_generation \
  --sweep configs/sweeps/dgx_spark_vllm_baseline.yaml \
  --output reports/generated/gemma4_26b_a4b_base_serving_bench
```

### 9.3 Default sweep

```yaml
sweep:
  engine:
    - vllm
  kv_cache_dtype:
    - auto
    - fp8_e4m3
  max_model_len:
    - 8192
    - 16384
    - 32768
  max_num_seqs:
    - 1
    - 2
    - 4
  max_num_batched_tokens:
    - 2048
    - 4096
    - 8192
    - 16384
    - 32768
  enable_prefix_caching:
    - true
    - false
  enable_chunked_prefill:
    - true
    - false
  gpu_memory_utilization:
    - 0.80
    - 0.85
    - 0.90
```

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

Command:

```bash
./forge agent optimize-serving gemma4_26b_a4b base \
  --hardware dgx_spark \
  --objective "maximize decode throughput subject to quality_delta >= -0.02 and p95_ttft < 8s" \
  --budget 12-runs \
  --output reports/generated/agent_runs/serving_opt_001
```

Other commands:

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

Add a dedicated `hf` command group.

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

### Week 1: Research registry + objective profiles

Ship:

```text
docs/research/sota-2026-05-18.md
configs/research_registry.yaml
configs/objectives/*.yaml
src/model_forge/research/registry.py
src/model_forge/research/audit.py
```

Commands:

```bash
./forge research list
./forge research show behavior_editing
./forge research audit
./forge objective list
./forge objective show zero_refusal_capability_retention
```

Acceptance criteria:

```text
- Every existing major command maps to at least one research-basis item.
- `forge research audit` prints missing implementation/test hooks.
- Objective profiles can be loaded and rendered in reports.
```

Public artifact:

```text
“Model Forge now has a SOTA registry: every feature must cite a claim, test it, and report whether it held.”
```

### Week 2: Run manifest + report cards

Ship:

```text
canonical manifest.json
comparison report v2
research-basis report section
training card
behavior edit card
Hugging Face card templates for model/dataset/report artifacts
```

Acceptance criteria:

```text
- Every eval run emits manifest.json.
- Compare report includes objective profile and research basis.
- Report contains exact model/engine/hardware/sampling metadata.
- Model/dataset/report card templates can render from a run manifest.
```

### Week 3: Refusal/noncompliance taxonomy + scorecards

Ship:

```text
noncompliance taxonomy
refusal classifier
ablation scorecard
candidate frontier report
redacted risk-report mode
```

Acceptance criteria:

```text
- Each refusal-suite output is classified into taxonomy.
- Candidate frontier ranks at least 3 candidates.
- Public report can redact unsafe examples.
```

### Week 4: Finish Gemma local FT and publish postmortem

Ship:

```text
local_ft_v0 completed or clearly documented if failed
Training Method Card
LoRA target failure postmortem
base vs reference FT vs local FT comparison
local_ft_v1 dataset factory plan from observed eval gaps
```

Acceptance criteria:

```text
- Final adapter tensor inspection passes.
- Internal eval with repeated trials.
- Artifact eval.
- External IFEval run or documented blocker.
- v1 dataset plan lists target skills, seed examples, quality filters, and
  holdout-overlap policy.
```

### Week 5: Dataset factory MVP + `forge bench serve` MVP

Ship:

```text
forge data plan/seed/generate/judge/filter/pack skeleton
Gemma local_ft_v1 seed taxonomy and 500-2000 accepted eval-adjacent examples
serving benchmark harness
short_chat / long_prefill / decode_heavy workloads
Serving Card
CSV + HTML report
```

Acceptance criteria:

```text
- Dataset pack includes manifest, accepted/rejected rows, judge scores, and
  dataset card.
- Holdout overlap check rejects near-copies of model-forge eval prompts.
- Runs vLLM benchmark on at least one family/variant.
- Reports TTFT, ITL, tok/sec, memory, OOM, truncation.
- Repro command works from report.
```

### Week 6: DGX Spark vLLM sweep

Ship:

```text
dgx_spark_vllm_baseline.yaml
FP8 KV vs auto KV report
prefix caching on/off report
chunked prefill on/off report
```

Acceptance criteria:

```text
- At least 12 serving configs tested.
- Report identifies best config per workload.
- Quality/behavior suite sampled under best and baseline configs.
```

### Week 7: Quantization MVP

Ship:

```text
forge quantize
quantization configs
Quantization Card
calibration manifest
eval-after-quant
bench-after-quant
```

Acceptance criteria:

```text
- At least one FP8 or NVFP4 quantized variant generated or imported.
- Quantized variant loads under serving backend.
- Quantization report includes quality and behavior deltas.
```

### Week 8: Quantization behavior study

Ship:

```text
BF16 vs FP8 KV vs weight quantization matrix
behavior-preservation report
Pareto chart
```

Acceptance criteria:

```text
- Memory/perf/quality/behavior deltas reported together.
- At least one negative or mixed result documented honestly.
```

### Week 9: Artifact execution validation

Ship:

```text
Playwright HTML validator
Python compile/run validator
artifact execution card
screenshots
```

Acceptance criteria:

```text
- HTML artifacts get DOM/console/screenshot validation.
- Python artifacts get compile/help/fixture validation.
- Artifact pass rate appears in compare reports.
```

### Week 10: Qwen family hardening

Ship:

```text
Qwen family YAML
Qwen serving config
Qwen eval baseline
Qwen FT/ablation planning configs
```

Acceptance criteria:

```text
- Qwen base can be served/evaluated.
- No Gemma-specific assumptions in common code.
- Model family checklist is followed.
```

### Week 11: Agent runner MVP

Ship:

```text
forge agent optimize-serving
experiment proposal JSON
budget enforcement
automatic report
ledger update
```

Acceptance criteria:

```text
- Agent runs bounded serving sweep.
- Agent rejects/promotes based on objective profile.
- Agent writes reproducible report and updates ledger.
```

### Week 12: Public report site + Hugging Face publishing + upstream PR

Ship:

```text
reports/index.html
DGX Spark leaderboard page
model-family report pages
forge hf doctor / plan-upload / upload-report MVP
first Hugging Face report or dataset repo
first upstream PR
```

Acceptance criteria:

```text
- Public index links training/behavior/quant/serving cards.
- At least one Hugging Face repo is created from a dry-run upload plan.
- Hugging Face repo includes README.md card, manifest, report summary, and hashes.
- At least one vLLM/SGLang/ModelOpt/llm-compressor/docs PR opened.
- Pinned X thread links report site and Hugging Face artifact.
```

---

## 15. Six-month roadmap

### Month 1: Foundation

```text
research registry
objective profiles
run manifests
report cards
Hub card templates
finish Gemma FT
dataset factory MVP
behavior scorecards
serving bench MVP
```

Success metric:

```text
One complete model-family report: base vs FT vs abli vs FT→abli, with serving metrics and raw outputs.
```

### Month 2: DGX Spark benchmark product

```text
vLLM sweeps
FP8 KV reports
prefix/chunked prefill reports
first public leaderboard
artifact execution validation
HF dataset repo for benchmark results
first HF dataset repo for a generated fine-tuning dataset
```

Success metric:

```text
At least 5 high-quality public serving reports, one community-reproducible
benchmark command, and one generated dataset with manifest, card, quality
report, and fine-tuning outcome.
```

### Month 3: Quantization lab

```text
FP8 W8A8
ModelOpt/NVFP4
calibration manifests
quantization behavior reports
layer sensitivity scans
```

Success metric:

```text
At least one quantized variant with a complete quantization card and quality/behavior/serving deltas.
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
The same workbench loop runs on at least two non-Gemma families.
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
Agent completes 3 bounded research loops and produces useful reports without manual reconstruction.
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

### P0: Foundation

```text
[ ] MF-0001 Add docs/research/sota-2026-05-18.md
[ ] MF-0002 Add configs/research_registry.yaml
[ ] MF-0003 Add ./forge research list/show/audit
[ ] MF-0004 Add objective profile loader
[ ] MF-0005 Add configs/objectives/*.yaml
[ ] MF-0006 Add canonical run manifest
[ ] MF-0007 Add comparison report v2
[ ] MF-0008 Add research-basis report section
[ ] MF-0009 Add eval provenance card
[ ] MF-0010 Add golden baseline create/check hardening
[ ] MF-0011 Finish Gemma local FT evaluation
[ ] MF-0012 Publish Training Method Card
[ ] MF-0013 Add Hub release-class loader
[ ] MF-0014 Add model card and dataset card templates
```

### P0: Behavior editing

```text
[ ] MF-0101 Add noncompliance taxonomy
[ ] MF-0102 Add refusal classifier
[ ] MF-0103 Add behavior edit scorecard
[ ] MF-0104 Add candidate frontier report
[ ] MF-0105 Add redacted risk-report mode
[ ] MF-0106 Add release classes
[ ] MF-0107 Add zero_refusal_capability_retention objective gates
```

### P0: Serving

```text
[ ] MF-0201 Add forge bench serve
[ ] MF-0202 Add DGX Spark vLLM sweep config
[ ] MF-0203 Add serving workload definitions
[ ] MF-0204 Add Serving Card
[ ] MF-0205 Add TTFT/ITL/memory/tok-sec capture
[ ] MF-0206 Add quality/behavior sampled eval under serving configs
```

### P1: Quantization

```text
[ ] MF-0301 Add forge quantize
[ ] MF-0302 Add calibration dataset manifests
[ ] MF-0303 Add FP8 KV behavior report
[ ] MF-0304 Add FP8 W8A8 pipeline
[ ] MF-0305 Add ModelOpt/NVFP4 pipeline
[ ] MF-0306 Add Quantization Card
[ ] MF-0307 Add layer/component sensitivity scan
[ ] MF-0308 Add quantization-preserves-behavior report
```

### P1: Dataset factory

```text
[ ] MF-0351 Add configs/datasets/*.yaml plan schema
[ ] MF-0352 Add forge data plan/seed/generate
[ ] MF-0353 Add forge data judge with multi-axis quality scores
[x] MF-0354 Add forge data verify for JSON/code/artifact examples
[ ] MF-0355 Add forge data filter with dedupe, holdout-overlap, and license checks
[ ] MF-0356 Add forge data pack with dataset.jsonl, manifest.yaml, and dataset_card.md
[ ] MF-0357 Add accepted/rejected row reports with rejection reasons
[ ] MF-0358 Add generated dataset HF publish path
[ ] MF-0359 Add Gemma local_ft_v1 eval-adjacent dataset recipe
[ ] MF-0360 Add eval-feedback loop that proposes next dataset skills from failures
```

### P1: Artifact validation

```text
[ ] MF-0401 Add Playwright HTML validation
[ ] MF-0402 Add Python artifact compile/run validation
[ ] MF-0403 Add artifact screenshots
[ ] MF-0404 Add Artifact Execution Card
[ ] MF-0405 Add artifact execution score to compare report
```

### P1: Hugging Face Hub publishing

```text
[ ] MF-0501 Add forge hf status/login/whoami
[ ] MF-0502 Add forge hf plan-model
[ ] MF-0503 Add forge hf publish-model --dry-run
[ ] MF-0504 Add forge hf publish-dataset --dry-run
[ ] MF-0505 Add Hub model card generator
[ ] MF-0506 Add Hub dataset card generator
[ ] MF-0507 Add release-class validators
[ ] MF-0508 Add hub_publish.json provenance writer
[ ] MF-0509 Add no-secrets/no-absolute-path publish validator
[ ] MF-0510 Add redacted-output dataset publishing path
```

### P1: Multi-family

```text
[ ] MF-0601 Harden Qwen family config
[ ] MF-0602 Add adding-model-family checklist
[ ] MF-0603 Add Llama/Mistral family plan
[ ] MF-0604 Ensure common code has no Gemma-only assumptions
```

### P2: Agents

```text
[ ] MF-0701 Add agent experiment schema
[ ] MF-0702 Add forge agent optimize-serving
[ ] MF-0703 Add forge agent optimize-quantization
[ ] MF-0704 Add forge agent optimize-behavior-edit
[ ] MF-0705 Add agent run card
[ ] MF-0706 Add automatic ledger update
```


### P2: Kernel/perf

```text
[ ] MF-0801 Add Nsight profile integration
[ ] MF-0802 Add profile summarizer
[ ] MF-0803 Add bench kernel rmsnorm
[ ] MF-0804 Add bench kernel rope
[ ] MF-0805 Add bench kernel dequant
[ ] MF-0806 Add bench kernel kv-layout
[ ] MF-0807 Add Kernel Card
[ ] MF-0808 Open first upstream PR
```

### P3: Advanced serving

```text
[ ] MF-0901 Add SGLang backend
[ ] MF-0902 Add TensorRT-LLM backend
[ ] MF-0903 Add disaggregated prefill/decode experiment profile
[ ] MF-0904 Add LMCache/NIXL research-watch hooks
[ ] MF-0905 Add multi-node/distributed-KV placeholder architecture
```


---

## 17. Repo structure target

```text
configs/
  model_families/
  objectives/
  datasets/
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
| Quantization | FP8/NVFP4/ModelOpt reports, calibration and behavior-preservation cards |
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
1. Add docs/research/sota-2026-05-18.md.
2. Add configs/research_registry.yaml.
3. Add configs/objectives/zero_refusal_capability_retention.yaml.
4. Add configs/objectives/quantized_quality_retention.yaml.
5. Add configs/objectives/dgx_spark_latency_throughput.yaml.
6. Add canonical manifest writer.
7. Make compare report include Objective Profile and Research Basis.
8. Add noncompliance taxonomy.
9. Finish/evaluate Gemma local FT.
10. Add forge bench serve MVP.
11. Add configs/publishing/huggingface.example.yaml.
12. Add `./forge hf doctor` and `./forge hf plan-upload`.
13. Generate Hugging Face Model Card / Dataset Card / Report Card from manifests.
```

A strong first public post:

> “I’m making Model Forge SOTA-grounded by construction. Every feature now has a research claim, implementation hook, eval hook, and report section. First target: Gemma 4 26B on DGX Spark across FT, behavior editing, FP8 KV serving, artifact execution, and Hugging Face report/model/dataset cards.”

---

## 20. Source registry

The following sources ground this roadmap. The project should copy these into `configs/research_registry.yaml` with feature/test hooks.

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
quantization plan
raw outputs
repro commands
HF model card / dataset card / hub_publish.json
```

Everything in this roadmap exists to make that report credible, reproducible, and obviously relevant to frontier-lab work.

After that, repeat the loop on Qwen. Then add quantization. Then add agentic optimization. Then upstream the tooling.

The project becomes impressive when an outside person can look at Model Forge and say:

> “This is not a demo. This is a reproducible local research system for post-training and inference optimization.”
