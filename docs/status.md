# Current Status

Last updated: 2026-06-01.

This is the short handoff state for humans and agents. Use
`docs/experiment-ledger.md` for detailed run history and raw observations.

## Validated So Far

- The repo is organized around model families, not Gemma-only scripts.
- Qwen 3.5 9B and Qwen 3.6 27B now have model-family configs with base,
  local-FT, local-abli, and local-FT-abli variant nodes, Qwen chat-template
  defaults, serving/eval hooks, and doctor-audited source edges.
- Llama 3.1 8B Instruct now has the same first-class family plan shape,
  including base, local-FT, local-abli, local-FT-abli, and Blackwell NVFP4
  runtime-import variants. Its NVFP4 plan compares against the unquantized base
  source while launching the quantized runtime checkpoint.
- `docs/adding-model-family.md` now defines the portable checklist for adding
  non-Gemma families and is required by `./forge doctor`.
- Architecture target-discovery checks are now wired through
  `./forge variants architecture-audit`; family configs record attention/MLP
  target patterns plus embedding, LM-head, and router/expert exclusions.
- Common-code family assumptions are now checked by `./forge generalization
  audit`, which is also part of `./forge doctor`. The top-level `./forge`
  finetune, ablate, promote, and golden commands discover family configs by
  convention instead of branching on Gemma.
- Agent experiment plans are now schema-backed through `./forge agent
  schema|audit|init`; `./forge doctor` validates tracked agent templates.
- `./forge agent optimize-serving` now writes a validated pre-run agent plan
  for serving sweeps, marking server starts as heavy execute-only steps and
  requiring serving cards, manifests, and sampled quality checks before
  promotion.
- `./forge agent optimize-quantization` now writes a validated pre-run agent
  plan for quantization matrices, export plans, serving checks, quantization
  cards, and release evidence while marking exports/server starts as heavy.
- `./forge agent optimize-behavior-edit` now writes a validated pre-run agent
  plan for refusal-ablation behavior edits, SOTA backend prep/run commands,
  source-vs-edited eval evidence, comparison refreshes, and promotion gates.
- `./forge agent card` now writes Agent Run Cards from experiment plans,
  summarizing plan identity, heavy commands, resource policy, evidence
  requirements, schema findings, and Git state for handoff.
- Agent Run Cards can now update `docs/experiment-ledger.md` with
  `--update-ledger`; updates are idempotent by experiment id and replace the
  prior marked block instead of duplicating entries.
- Gemma 4 A4B is the first worked family for base, downloaded FT, downloaded
  abli, local base abli, local FT, and local FT abli comparisons.
- Internal evals now cover refusal suppression, benign paired quality,
  normal-use regression, challenge capability, reasoning style stability,
  artifact quality, multi-trial variance, and golden comparisons.
- Internal eval keyword checks now support explicit concept alternatives and a
  `scripts/rescore_internal_eval.py` path for rescoring saved responses after
  rubric fixes without rerunning large model servers. Scoring v3 normalizes
  smart punctuation in refusal phrases and separates direct refusal suppression
  from actual harmful compliance or harmful detail. The capability challenge
  and paired-boundary checks now avoid brittle exact-word failures discovered
  during Qwen 3.6 FT-ablation evaluation.
- Local base ablation beat the downloaded abli reference on the saved internal
  comparison while preserving stronger behavior than expected.
- Local FT ablation preserved the source FT's primary internal behavior closely
  enough to count as a successful ablation of an already fine-tuned model.
- Gemma local FT v0 completed under the guarded Spark training path. It was
  close to Jackrong on challenge capability and better on paired benign quality,
  but it did not clear the primary challenge-capability promotion gate.
- Qwen 3.6 27B local FT v4 is the promoted Qwen FT source. It beat the base on
  internal challenge capability while preserving paired benign quality,
  planning, normal-use behavior, and harmful-prompt refusal behavior.
- Qwen 3.6 27B local FT v4 trial2 scale0.75 ablation completed the full
  internal suite on the two-Spark cluster. After scorer/rubric v3 rescoring, it
  preserves or improves measured capability versus base and FT, but still
  refuses 60% of paired harmful prompts, so it is a hold, not the final
  zero-refusal FT-abli target.
- A trial2 scale1.0 follow-up config exists, but the guarded merge helper
  blocked export on the coordinator because projected free disk would fall to
  14.2%, below the 15% floor. Free coordinator disk or relocate old local model
  artifacts before running another two-node checkpoint export.
- The local FT v1 dataset factory MVP is implemented with planning, gap
  extraction, feedback proposals, seed rows, generation adapters, verification,
  filtering, review, packing, dry-run publish planning, non-cascading overwrite
  semantics, and length-violation rejection gates.
- The repo now has reusable dataset source registries, guarded HF dataset
  publish execution, a local FT v1 dry-run config, saved-comparison promotion
  reports, and a safe Qwen 3.5 9B teacher launcher.
- The roadmap foundation now has a dated SOTA snapshot, a machine-readable
  research registry, and `./forge research list/show/audit` for checking that
  objective profiles reference known research entries.
- Canonical run manifest writing is implemented through `./forge manifest
  write/show`; eval manifests now include the shared `canonical` provenance
  block while preserving the existing eval manifest layout.
- Comparison reports now include report-v2 provenance and research basis:
  canonical manifest summaries, config hashes, comparability warnings, and
  selected `configs/research_registry.yaml` entries.
- Generic cluster planning is now present for open-source-safe inventories:
  `configs/hardware/dgx_spark.yaml`, `configs/clusters/*.example.yaml`, and
  `./forge cluster plan/doctor`. DGX Spark x2 is represented as an example
  config with env-backed hosts, not hard-coded private infrastructure.
- Serving benchmark MVP is now present through `./forge bench serve`,
  `configs/serving/serve_bench_smoke.yaml`, and
  `src/model_forge/benchmarks/serve.py`. It benchmarks an already-running
  OpenAI-compatible endpoint and writes request, summary, serving-card, and
  canonical manifest artifacts under `reports/generated/serve_bench/`.
- The baseline DGX Spark vLLM serving sweep is now present through
  `configs/sweeps/dgx_spark_vllm_baseline.yaml` and `./forge bench sweep`.
  It expands bounded startup-time vLLM env cases plus matching `bench serve`
  commands and can attach the two-node env-backed Spark cluster inventory.
- Two-node Spark readiness is now executable through `./forge cluster sync`
  `./forge cluster health`, `./forge cluster runtime`, and
  `./forge cluster torchrun-smoke`. On 2026-05-24, the
  repo was synced to the private worker Spark and both GB10 nodes passed health
  with ~256 GB declared cluster memory, visible GPUs, repo checkout, RAM
  headroom, and disk headroom. Both nodes also passed a bounded
  `nemotron-runner:latest` GPU container probe with CUDA Torch visible. The
  two-node torchrun smoke then joined both GB10s into one `world_size=2`
  CUDA/NCCL all-reduce job through the guarded container launcher.
- Serving workload definitions are now present under
  `configs/serving/workloads/`, with smoke and core benchmark configs loading
  reusable workload files instead of hard-coding all requests inline.
- Nsight profile planning is now present through `./forge profile nsight`,
  `configs/profiling/nsight_serving_smoke.yaml`, and
  `docs/profiling.md`; it writes `nsys`/`ncu` command plans around existing
  benchmark commands without starting profilers by default.
- Profile summarization is now present through `./forge profile nsight
  summarize`; it inventories expected and present profiler artifacts and writes
  `profile_summary.json` plus `profile_summary.md`.
- RMSNorm kernel microbenchmarking is now present through `./forge bench kernel
  rmsnorm`; it supports dry-run planning plus Torch-backed runs that emit
  `summary.json` and `kernel_card.md`.
- RoPE kernel microbenchmarking is now present through `./forge bench kernel
  rope`; it follows the same dry-run, correctness, latency, and kernel-card
  pattern as RMSNorm.
- Dequantization microbenchmarking is now present through `./forge bench kernel
  dequant`; it uses a packed NVFP4 E2M1 proxy with local/global scales and the
  same kernel-card artifact pattern.
- KV-cache layout microbenchmarking is now present through `./forge bench
  kernel kv-layout`; it compares contiguous cache reads with a paged/gathered
  proxy layout.
- Kernel Cards now have a reusable structured generator in
  `src/model_forge/reports/kernel_card.py`; benchmark writes include
  `kernel_card.json` and `kernel_card.md`, and `./forge bench kernel card` can
  regenerate cards from existing summaries with optional profile summaries.
- Upstream PR planning is scaffolded through `./forge upstream`; it audits
  candidate contribution plans and writes `upstream_pr_plan.json`/`.md`, but
  MF-0808 is not complete until a real external PR URL is recorded.
- SGLang backend planning is present through `./forge serving`; it audits
  `configs/serving/backends/sglang_openai.yaml` and writes SGLang launch plus
  matching `bench serve` commands without starting a server.
- TensorRT-LLM backend planning is present through `./forge serving`; it audits
  `configs/serving/backends/tensorrt_llm_openai.yaml` and writes `trtllm-serve`
  launch plus matching `bench serve` commands without starting a server.
- Disaggregated prefill/decode planning is present through `./forge bench
  sweep` with `configs/sweeps/dgx_spark_vllm_disagg_prefill_decode.yaml`; it
  expands a single-endpoint control and two Spark split cases without starting
  servers.
- Serving completion is gated through `./forge bench serve --evidence-gate`;
  completion-ready evidence requires successful endpoint metrics plus
  same-endpoint sampled quality/behavior artifacts.
- LMCache/NIXL/Dynamo are tracked through `./forge research watch` and
  `configs/research_watch/advanced_serving.yaml`; these are watch hooks, not
  validated backends.
- Distributed-KV placeholder architecture is tracked through
  `./forge serving architecture-doctor` and
  `configs/serving/architectures/distributed_kv_placeholder.yaml`; it documents
  roles, gates, and blockers only.
- Fine-tune `prepare` now writes `training_method_card.md` beside generated run
  artifacts. The card records recipe, data, LoRA, eval commands, and Spark
  resource guardrails, but does not claim training completed.
- Behavior-edit scorecards are present through `./forge behavior`; they read
  comparison reports and write objective-specific ablation scorecards that
  separate refusal suppression, capability retention, benign quality, and
  reported overcompliance risks.
- Behavior-edit reports now include a reusable noncompliance taxonomy, aggregate
  invalid-refusal vs valid-safety-refusal classifier fields, candidate frontier
  selection from saved comparison rows, and public redacted risk reports with
  private raw-output retention policy.
- The `zero_refusal_capability_retention` objective is wired into behavior
  scorecard gates, including structured output, artifact reporting, valid
  safety-refusal reporting, and overcompliance risk reporting.
- Release classes are audited through `./forge hf release-classes --audit`.
  Public behavior-edited releases now require a risk report or behavior-edit
  scorecard path before publish plans can pass.
- Serving Card generation now writes a structured `serving_card.md` for each
  `bench serve` run with identity, hardware/config, overall metrics,
  per-workload metrics, artifacts, and promotion gates.
- Quantization planning is now first-class for Blackwell NVFP4 runtime import
  and ModelOpt self-quantization. The self-quantization export path has a
  checkout-local lock, preflight memory/disk gates, `systemd-run --scope`,
  Docker CPU/memory limits, and a runtime memory watchdog. The initial Gemma4
  MLP-only NVFP4 export loaded and showed a modest decode-speed gain, but it did
  not meet the expected fully quantized MoE Spark target. The active Gemma4
  self-quantization path now uses a full-MoE ModelOpt plugin exporter plus
  Marlin serving. The published full-MoE NVFP4 reference checkpoint served with
  Marlin on 2026-05-30 and reached about 50 output tok/s on the core serving
  benchmark, confirming the target path.
- Objective profiles are now config-backed and auditable through
  `./forge objectives audit`: `capability_sft`,
  `zero_refusal_capability_retention`, `quantized_quality_retention`, and
  `dgx_spark_latency_throughput`. Compare reports load objective metric
  preferences from these configs.
- Required validation schemas are auditable through `./forge schema audit`
  across run manifests, objective profiles, variant nodes, and generated card
  schema versions.
- The prioritized roadmap backlog now has explicit status fields on every MF
  item, and roadmap command examples are checked by `./forge roadmap cli-drift`.
- Variant graph metadata is now wired through `./forge variants graph|node`.
  Variant nodes can record source variant, transform, artifact checksums,
  validation state, Spark evidence, promotion decision, and retention decision.
- Tokenizer and chat-template preservation checks are now wired through
  `./forge variants tokenizer-audit`. Metadata mode compares tokenizer files,
  special tokens, and chat-template hashes against each variant's configured
  source; `--load-tokenizer --strict` adds a local AutoTokenizer round trip for
  release gates.
- Standalone artifact execution validation is now wired through
  `./forge artifacts validate`. It validates HTML artifacts with Playwright
  browser checks, screenshots, and nonblank canvas checks when available;
  validates Python artifacts with compile/help/fixture checks; and writes
  `artifact_execution_card.json` / `.md`. Compare reports also emit claim
  warnings when an artifact-generation metric improves without
  `artifact_validation_pass_rate` evidence.
- Hugging Face model release planning is now wired through `./forge hf`.
  `status`, `whoami`, `login`, `plan-model`, and dry-run `publish-model`
  generate model cards, `hub_publish.json` provenance, no-secret/no-private-path
  checks, and release-class gates from `configs/release_classes/`.
- Dataset Hub dry runs now create a public redacted bundle for `public_dataset`
  release plans. The bundle keeps provenance, hashes, verification, quality, and
  review evidence while excluding raw accepted/rejected rows and message text.
- Internal eval runs now write `eval_provenance_card.json` and `.md` next to
  `manifest.json`, `scores.csv`, and `responses.jsonl`. The card records prompt
  counts, case hashes, scoring version, sampling settings, trials, output
  hashes, objective profile, and raw-output publication status.

## Current Dataset State

The deterministic smoke local FT v1 pack contains 49 accepted examples:

- 37 human seed rows
- 12 deterministic synthetic rows
- generation methods: `self_instruct`, `evol_instruct`,
  `instruction_backtranslation`, and `eval_adjacent_generation`
- review gate: `ready_to_scale_generation=true`
- known limitation: this is a scaffold and QC path, not a final training
  dataset

The live-teacher local FT v1 smoke is now complete. It used a local
OpenAI-compatible Qwen 3.5 9B teacher endpoint and did not start a training
run. The tracked config is:

```text
configs/datasets/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml
```

It defaults to a local OpenAI-compatible endpoint at
`http://127.0.0.1:8011/v1` and can be redirected with
`MODEL_FORGE_DATA_PROVIDER_BASE_URL` and `MODEL_FORGE_DATA_GENERATOR_MODEL`.

The live-teacher smoke pack now contains 58 accepted examples after stricter
length filtering:

- 37 human seed rows
- 21 accepted synthetic rows from the live teacher
- 3 rejected synthetic rows for `assistant_too_long`
- generation methods: `self_instruct`, `evol_instruct`,
  `instruction_backtranslation`, and `eval_adjacent_generation`
- review gate: `ready_to_scale_generation=true`
- review flags: none after length rejection
- publish status: dry-run HF plan only; this remains a smoke artifact

## Recommended Next Work

1. Attach variant nodes to completed and planned Gemma base, FT, abli, and
   quantization artifacts as generated evidence, then connect report cards to
   those nodes.
2. Attach variant nodes to generated eval provenance cards and publish redacted
   eval-output bundles for report releases.
3. Add guarded non-dry-run model Hub upload and CI dry-run workflows after
   release plans have human-reviewed evidence.
4. Finish/evaluate Gemma local FT or write a Training Method failure card with
   distributed Spark correctness evidence.
5. Run one real Spark serving benchmark and attach endpoint evidence to the
   Serving Card.
6. Run the guarded full-MoE Gemma4 ModelOpt NVFP4 export through the Spark path,
   serve it with Marlin, and compare tok/s against BF16 and the now-validated
   published full-MoE NVFP4 reference. Quantization remains incomplete until
   base, FT, abli, and FT+abli checkpoints load and match their unquantized
   baselines.
7. Scale the local FT v1 dataset through medium-pack review before treating it
   as a training dataset.
8. Continue Qwen 3.6 FT-ablation search from the promoted local FT v4 source.
   Trial2 scale0.75 passed capability retention on the full internal suite but
   only reduced paired harmful-prompt refusal from 1.0 to 0.6. The next ablation
   experiment should increase refusal suppression while keeping the full-suite
   capability and benign-quality gains; do not quantize or upload this candidate
   as the final FT-abli model yet.

## Operational Guardrails

- Run only one large model server or training job at a time.
- Do not bypass guarded run scripts for full training.
- Do not bypass the PEFT merge disk preflight for full-checkpoint ablation
  exports. If it blocks at the 15% floor, clear reviewed local artifacts first
  rather than lowering the guard.
- Keep `runs/`, `results/`, `reports/generated/`, and local model artifacts out
  of Git unless a small reusable file is intentionally moved into `recipes/`.
- Check `docs/artifact-retention.md` before deleting or uploading artifacts.
- Push code, configs, docs, recipes, and lightweight manifests to GitHub before
  handing off.
- Run `./forge doctor` before handoff to catch tracked ignored files, secret
  literals, nonportable paths, and accidental generated dataset commits.
- Do not bypass quantization or fine-tuning launchers with raw `docker run` or
  `python train.py` for large jobs.
