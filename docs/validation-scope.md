# Validation Scope

This repo separates implementation status from validation strength.

`implementation_status=tested` means the command, schema, config, report writer,
or planning path has automated coverage. It does not automatically mean a full
model run completed on DGX Spark or a two-node Spark cluster.

## Validation States

| State | Meaning |
|---|---|
| `planned` | Design or candidate exists, but completion evidence is missing. |
| `smoke_validated` | Unit tests, CLI smoke runs, dry runs, tiny generated artifacts, or offline checks pass. |
| `spark_single_node_validated` | A real Spark node ran the relevant model/training/eval path with saved evidence. |
| `spark_cluster_validated` | Both Spark nodes were used for the relevant distributed preflight or workload evidence. |

Use the lowest applicable state when documenting a feature. Do not upgrade a
feature to Spark validation because adjacent infrastructure passed.

## Current High-Confidence Evidence

Gemma 4 26B-A4B is the worked model family:

- base, downloaded FT, downloaded abli, local base abli, local FT, and local FT
  abli variants have saved comparison history
- local base ablation beat the downloaded abli reference in the saved internal
  comparison
- local FT ablation preserved the source FT closely enough to count as a
  successful ablation of an already fine-tuned model
- local FT v0 completed under the guarded Spark training path, but did not beat
  Jackrong on the primary challenge-capability promotion gate

DGX Spark cluster infrastructure:

- two-node inventory, health checks, runtime checks, and torchrun/NCCL smoke
  passed with the private Spark pair
- cluster configs remain env-backed examples, not hard-coded hostnames

Serving and NVFP4:

- Model Forge has BF16 and NVFP4 serving benchmark artifacts for the Gemma path
- a published full-MoE NVFP4 reference checkpoint served with Marlin and reached
  about 50 output tok/s on the core benchmark
- self-quantization planning and gating exist, but a new self-quantized model
  still needs source-vs-quantized serving, behavior, tokenizer, and quantization
  card evidence before promotion

## Smoke-Validated Areas

These areas have useful scaffolding and automated checks, but should not be
marketed as fully proven on arbitrary models yet:

- non-Gemma family support beyond config/audit/smoke paths
- dataset factory scale-up beyond deterministic and live-teacher smoke packs
- Hugging Face publish execution beyond dry-run/release-gate paths unless a
  concrete upload artifact is recorded
- SGLang and TensorRT-LLM backend planning
- disaggregated prefill/decode planning
- Nsight profile planning and summarization
- kernel microbenchmarks as optimization diagnostics
- full self-quantization export/promotion for each new model family
- agent optimization plans
- upstream PR tracking, until a real external PR URL is recorded

## Promotion Rules

Fine-tune promotion requires:

- source-relative eval comparison
- challenge/capability gates for the selected objective
- benign quality and format-following checks
- dataset manifest and training method card
- bounded training evidence for any Spark validation claim

Ablation promotion requires:

- lower invalid refusals or unsafe suppression for the intended objective
- retained source capability
- retained benign paired quality
- overcompliance and harmful-detail risk reporting
- tokenizer/chat-template preservation checks

Quantization promotion requires:

- source-vs-quantized serving benchmark evidence
- clear tok/s and latency comparison
- behavior preservation report
- tokenizer preservation report
- quantization card
- objective-specific quality-retention gate

Cluster performance claims require:

- `./forge cluster torchrun-smoke` evidence for the target cluster
- workload evidence showing both nodes were used
- serving/training logs or cards that identify topology and command

## Before Making Claims

Run:

```bash
./forge roadmap audit --write-doc
./forge roadmap cli-drift
./forge doctor
```

Then inspect:

- [roadmap-status-audit.md](roadmap-status-audit.md)
- [status.md](status.md)
- [experiment-ledger.md](experiment-ledger.md)
- generated run cards, serving cards, quantization cards, and promotion reports
