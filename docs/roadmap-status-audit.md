# Roadmap Status Audit

This file is generated from the prioritized backlog in
`docs/roadmaps/model_forge_sota_roadmap_2026-05-18_with_huggingface.md`.

## Summary

- Items: 97
- Findings: 0

## Implementation Status

| Status | Count |
|---|---:|
| implemented | 2 |
| not_started | 1 |
| scaffolded | 8 |
| tested | 86 |

## Validation State

| State | Count |
|---|---:|
| planned | 15 |
| smoke_validated | 79 |
| spark_cluster_validated | 2 |
| spark_single_node_validated | 1 |

## Findings

No status audit findings.

## Items

| Item | Section | Implementation | Validation | Title |
|---|---|---|---|---|
| MF-0000 | P0: Foundation | tested | planned | Convert legacy [x] backlog into implementation_status + validation_state |
| MF-0001 | P0: Foundation | scaffolded | planned | Add required validation schema to manifests, cards, objectives, and variant nodes |
| MF-0002 | P0: Foundation | tested | planned | Add evidence ledger with command, node count, topology, logs, metrics, checksums, and promotion decision |
| MF-0003 | P0: Foundation | tested | planned | Add objective profile loader and objective audit |
| MF-0004 | P0: Foundation | tested | planned | Add configs/objectives/zero_refusal_capability_retention.yaml |
| MF-0005 | P0: Foundation | tested | planned | Add configs/objectives/quantized_quality_retention.yaml |
| MF-0006 | P0: Foundation | tested | planned | Add configs/objectives/dgx_spark_latency_throughput.yaml |
| MF-0007 | P0: Foundation | tested | planned | Add variant graph and variant_node.json writer |
| MF-0008 | P0: Foundation | tested | planned | Add artifact checksum and retention policy fields |
| MF-0009 | P0: Foundation | tested | smoke_validated | Add eval provenance card |
| MF-0010 | P0: Foundation | tested | smoke_validated | Add golden baseline create/check hardening |
| MF-0011 | P0: Foundation | tested | smoke_validated | Add CLI/doc drift check for roadmap command examples |
| MF-0012 | P0: Foundation | implemented | spark_single_node_validated | Finish Gemma local FT evaluation or failure-card it |
| MF-0013 | P0: Foundation | tested | smoke_validated | Publish Training Method Card with distributed training correctness |
| MF-0101 | P0: Behavior editing | scaffolded | planned | Add noncompliance taxonomy |
| MF-0102 | P0: Behavior editing | scaffolded | planned | Add invalid-refusal vs valid-safety-refusal classifier fields |
| MF-0103 | P0: Behavior editing | implemented | smoke_validated | Add harmful-overcompliance and behavior-drift scoring |
| MF-0104 | P0: Behavior editing | tested | smoke_validated | Add behavior edit scorecard |
| MF-0105 | P0: Behavior editing | scaffolded | planned | Add candidate frontier report from actual local candidates |
| MF-0106 | P0: Behavior editing | scaffolded | planned | Add redacted public risk-report mode and private raw-output retention |
| MF-0107 | P0: Behavior editing | tested | smoke_validated | Add release classes and release-class validators |
| MF-0108 | P0: Behavior editing | tested | smoke_validated | Add zero_refusal_capability_retention objective gates |
| MF-0200 | P0: Serving | tested | spark_cluster_validated | Add generic cluster inventory planner and DGX Spark x2 example |
| MF-0201 | P0: Serving | tested | smoke_validated | Add forge bench serve |
| MF-0202 | P0: Serving | tested | smoke_validated | Add DGX Spark vLLM sweep config |
| MF-0203 | P0: Serving | tested | smoke_validated | Add serving workload definitions |
| MF-0204 | P0: Serving | tested | smoke_validated | Add Serving Card |
| MF-0205 | P0: Serving | tested | smoke_validated | Add TTFT/ITL/memory/tok-sec capture |
| MF-0206 | P0: Serving | tested | smoke_validated | Add quality/behavior sampled eval under serving configs |
| MF-0207 | P0: Serving | tested | smoke_validated | Mark serving work complete only after real endpoint evidence is attached |
| MF-0208 | P0: Serving | tested | spark_cluster_validated | Add two-node torchrun/NCCL Spark preflight |
| MF-0251 | P0: Artifact validation | tested | smoke_validated | Add Playwright HTML validation |
| MF-0252 | P0: Artifact validation | tested | smoke_validated | Add Python artifact compile/run validation |
| MF-0253 | P0: Artifact validation | tested | smoke_validated | Add artifact screenshots and nonblank canvas/WebGL checks |
| MF-0254 | P0: Artifact validation | tested | smoke_validated | Add Artifact Execution Card |
| MF-0255 | P0: Artifact validation | tested | smoke_validated | Add artifact execution score to compare report |
| MF-0256 | P0: Artifact validation | tested | smoke_validated | Require artifact validation before artifact-generation improvement claims |
| MF-0301 | P1: Quantization | tested | smoke_validated | Add forge quantize or mark target CLI until implemented |
| MF-0302 | P1: Quantization | tested | smoke_validated | Add calibration dataset manifests |
| MF-0303 | P1: Quantization | tested | smoke_validated | Add FP8 KV behavior report |
| MF-0304 | P1: Quantization | tested | smoke_validated | Add FP8 W8A8 pipeline |
| MF-0305 | P1: Quantization | tested | smoke_validated | Add Blackwell ModelOpt/NVFP4 pipeline |
| MF-0306 | P1: Quantization | tested | smoke_validated | Add GGUF/llama.cpp conversion and quantization pipeline |
| MF-0307 | P1: Quantization | tested | smoke_validated | Add Quantization Card |
| MF-0308 | P1: Quantization | tested | smoke_validated | Add layer/component sensitivity scan |
| MF-0309 | P1: Quantization | tested | smoke_validated | Add quantization-preserves-behavior report |
| MF-0310 | P1: Quantization | tested | smoke_validated | Add tokenizer/chat-template preservation checks for GGUF and quantized exports |
| MF-0311 | P1: Quantization | tested | smoke_validated | Add import-existing-quantized-checkpoint path for already-available FP8/NVFP4/GGUF artifacts |
| MF-0351 | P1: Dataset factory | tested | smoke_validated | Add configs/datasets/*.yaml plan schema |
| MF-0352 | P1: Dataset factory | tested | smoke_validated | Add forge data plan/seed/generate |
| MF-0353 | P1: Dataset factory | tested | smoke_validated | Add forge data judge with multi-axis quality scores |
| MF-0354 | P1: Dataset factory | tested | smoke_validated | Add forge data verify for JSON/code/artifact examples |
| MF-0355 | P1: Dataset factory | tested | smoke_validated | Add forge data filter with dedupe, holdout-overlap, and license checks |
| MF-0356 | P1: Dataset factory | tested | smoke_validated | Add forge data pack with dataset.jsonl, manifest.yaml, and dataset_card.md |
| MF-0357 | P1: Dataset factory | tested | smoke_validated | Add accepted/rejected row reports with rejection reasons |
| MF-0358 | P1: Dataset factory | tested | smoke_validated | Add generated dataset HF publish path |
| MF-0359 | P1: Dataset factory | tested | smoke_validated | Add Gemma local_ft_v1 eval-adjacent dataset recipe |
| MF-0360 | P1: Dataset factory | tested | smoke_validated | Add eval-feedback loop that proposes next dataset skills from failures |
| MF-0361 | P1: Dataset factory | tested | smoke_validated | Add forge data review with curation flags and scale-up gate |
| MF-0362 | P1: Dataset factory | scaffolded | smoke_validated | Add smoke_pack, medium_pack, and training_pack promotion gates |
| MF-0363 | P1: Dataset factory | not_started | planned | Require bounded Spark fine-tune evidence before dataset recipe is marked validated |
| MF-0501 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add forge hf status/login/whoami |
| MF-0502 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add forge hf plan-model |
| MF-0503 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add forge hf publish-model --dry-run |
| MF-0504 | P1: Hugging Face Hub publishing | scaffolded | smoke_validated | Add forge hf publish-dataset --dry-run |
| MF-0505 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add Hub model card generator |
| MF-0506 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add Hub dataset card generator |
| MF-0507 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add release-class validators |
| MF-0508 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add hub_publish.json provenance writer |
| MF-0509 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add no-secrets/no-absolute-path publish validator |
| MF-0510 | P1: Hugging Face Hub publishing | tested | smoke_validated | Add redacted-output dataset publishing path |
| MF-0511 | P1: Hugging Face Hub publishing | tested | smoke_validated | Block public checkpoint upload unless validation state and release class allow it |
| MF-0601 | P1: Multi-family | tested | smoke_validated | Harden Qwen family config |
| MF-0602 | P1: Multi-family | tested | smoke_validated | Add adding-model-family checklist |
| MF-0603 | P1: Multi-family | tested | smoke_validated | Add tokenizer/chat-template round-trip tests |
| MF-0604 | P1: Multi-family | tested | smoke_validated | Add architecture target discovery and MoE/router exclusion checks |
| MF-0605 | P1: Multi-family | tested | smoke_validated | Add Llama/Mistral family plan |
| MF-0606 | P1: Multi-family | tested | smoke_validated | Ensure common code has no Gemma-only assumptions |
| MF-0701 | P2: Agents | tested | smoke_validated | Add agent experiment schema |
| MF-0702 | P2: Agents | tested | smoke_validated | Add forge agent optimize-serving |
| MF-0703 | P2: Agents | tested | smoke_validated | Add forge agent optimize-quantization |
| MF-0704 | P2: Agents | tested | smoke_validated | Add forge agent optimize-behavior-edit |
| MF-0705 | P2: Agents | tested | smoke_validated | Add agent run card |
| MF-0706 | P2: Agents | tested | smoke_validated | Add automatic ledger update |
| MF-0801 | P2: Kernel/perf | tested | smoke_validated | Add Nsight profile integration |
| MF-0802 | P2: Kernel/perf | tested | smoke_validated | Add profile summarizer |
| MF-0803 | P2: Kernel/perf | tested | smoke_validated | Add bench kernel rmsnorm |
| MF-0804 | P2: Kernel/perf | tested | smoke_validated | Add bench kernel rope |
| MF-0805 | P2: Kernel/perf | tested | smoke_validated | Add bench kernel dequant |
| MF-0806 | P2: Kernel/perf | tested | smoke_validated | Add bench kernel kv-layout |
| MF-0807 | P2: Kernel/perf | tested | smoke_validated | Add Kernel Card |
| MF-0808 | P2: Kernel/perf | scaffolded | planned | Open first upstream PR |
| MF-0901 | P3: Advanced serving | tested | smoke_validated | Add SGLang backend |
| MF-0902 | P3: Advanced serving | tested | smoke_validated | Add TensorRT-LLM backend |
| MF-0903 | P3: Advanced serving | tested | smoke_validated | Add disaggregated prefill/decode experiment profile |
| MF-0904 | P3: Advanced serving | tested | smoke_validated | Add LMCache/NIXL research-watch hooks |
| MF-0905 | P3: Advanced serving | tested | smoke_validated | Add multi-node/distributed-KV placeholder architecture |
