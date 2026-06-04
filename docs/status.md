# Current Status

Last updated: 2026-06-04.

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
- Qwen 3.6 27B local FT v5 was trained and quick-gated as a boundary-redirect
  source candidate on the two-Spark TP=2 live-LoRA path. It is rejected as an
  ablation source because it still refused all harmful paired and
  unsafe-overcompliance prompts, dropped paired benign quality to 90%, and
  dropped challenge capability to 87.5%. Keep local FT v4 as the Qwen source.
- Qwen 3.6 27B local FT v4 trial2 scale0.75 ablation completed the full
  internal suite on the two-Spark cluster. After scorer/rubric v4 rescoring, it
  preserves or improves measured capability versus base and FT, but still
  refuses 75% of paired harmful prompts, so it is a hold, not the final
  zero-refusal FT-abli target.
- Qwen 3.6 27B residual-trial12 combined preference-plus-unlikelihood v2
  trained for 120 guarded two-node steps, merged, synced to both Sparks, served
  with TP=2, and quick-gated. After scoring v6 rescore, it is the strongest
  Qwen FT-abli candidate so far on the targeted gate: paired harmful refusal
  0.00, paired benign quality 1.00, paired safe redirects 1.00, paired harmful
  detail 0.00, paired harmful compliance 0.00, and challenge capability 0.9688.
  It remains blocked rather than promoted because unsafe-overcompliance still
  has explicit refusal wording on 2/3 prompts, even though harmful detail and
  harmful compliance are both zero there.
- Qwen 3.6 27B residual-trial12 preference-unlikelihood v3 trained for 80
  guarded two-node steps from the held v2 checkpoint, merged on the worker Spark
  because the coordinator merge disk preflight correctly blocked at the 15%
  floor, synced back, tokenizer-audited, served with TP=2, and quick-gated.
  Rescoring v6 confirmed it is rejected: unsafe-overcompliance refusal improved
  from 0.6667 to 0.3333, but paired harmful refusal regressed to 0.15, paired
  harmful detail/compliance regressed to 0.10, paired benign quality fell to
  0.90, and challenge capability fell to 0.8125. Keep v2 as the best held
  Qwen FT-abli evidence node and do not upload, quantize, or promote v3.
- Qwen 3.6 27B residual-trial12 preference-unlikelihood v4 trained for 36
  guarded two-node steps from the held v2 checkpoint, merged, synced back,
  checkpoint/tokenizer-audited, and quick-gated with a solo TP=1 behavior-only
  fallback after cluster TP=2 vLLM serving failed twice during NCCL
  communicator initialization. Rescoring v6 confirmed it is rejected:
  unsafe-overcompliance refusal stayed at 0.6667, paired harmful refusal
  regressed to 0.10, paired harmful detail/compliance regressed to 0.05, and
  challenge capability fell to 0.8438. Keep v2 as the best held Qwen FT-abli
  evidence node and do not upload, quantize, or promote v4.
- The eval suite now separates legacy refusal-calibration prompts from the
  ablation-specific no-denial objective. `unsafe_overcompliance` remains a
  deployment-safety/reporting bucket whose prompts explicitly request refusal;
  `unsafe_ablation_redirect` is the reusable zero-refusal gate that asks for
  safe redirects without denial-style openings and still checks harmful detail
  and harmful compliance. A two-Spark TP=2 quick gate on held Qwen v2 passed
  this corrected bucket cleanly: refusal 0.00, safe redirect 1.00, harmful
  detail 0.00, and harmful compliance 0.00 on 3/3 prompts. The same
  single-trial run showed paired/challenge drift versus the earlier v2 rescore
  (paired harmful refusal 0.05, paired benign quality 0.90, challenge 0.9062),
  with several misses looking like strict keyword/rubric brittleness on manual
  inspection. Qwen v5 remains prepared but untrained; do not spend another
  training run solely for `unsafe_ablation_redirect` unless repeated/rescored
  paired/challenge evidence says v2 cannot be promoted.
- The generic Qwen 3.6 27B `local_ft_abli` slot and
  `local_ft_abli_nvfp4_modelopt` target are now blocked in family metadata until
  a real FT-abli candidate passes the zero-refusal capability-retention gate.
  This prevents the Qwen NVFP4 recipe from exporting a placeholder or held
  ablation candidate before the unquantized source is promoted.
- Live LoRA follow-up adapters were materialized for low-disk ablation gating,
  but the equivalence control failed for this Qwen Heretic adapter. A live
  scale0.75 adapter refused 95% of paired harmful prompts, while the already
  merged scale0.75 checkpoint refused 65% on the same paired bucket. Treat
  live-LoRA Qwen Heretic results as diagnostic only until vLLM live LoRA support
  is verified for the adapter's `linear_attn.out_proj` edits.
- A trial2 scale1.0 follow-up config exists, but the guarded merge helper
  blocked export on the coordinator because projected free disk would fall to
  14.2%, below the 15% floor. The same full merge was completed on the worker
  Spark, where disk headroom was safe, and quick-gated from a worker-local vLLM
  server. It still refused 65% of paired harmful prompts and dropped challenge
  pass rate to 84.38%, so it is rejected as a final FT-abli candidate.
- `scripts/run_merge_peft_container.sh` now provides a reusable, resource-capped
  container merge path using the newer posttrain image. This fixed two Qwen
  painpoints found during worker execution: optional Unsloth import failures in
  CPU/container merge contexts, and root-owned model outputs from ad hoc Docker
  runs.
- `scripts/merge_peft_adapter.py` now preserves tokenizer metadata from the
  source/base checkpoint by default and exposes `--tokenizer-source
  adapter|auto` only for intentional tokenizer changes. This prevents PEFT merge
  tokenizer drift from silently reaching a release candidate.
- Generated fine-tune `training_result.json` files now include LoRA rank, alpha,
  dropout, and target modules so run postmortems can verify which modules were
  actually trained.
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
- On 2026-06-04, `cluster torchrun-smoke` was hardened after the v4 follow-up
  exposed a worker-waiting-for-master failure mode. The smoke now uses
  deterministic rank-named containers, an inner Docker timeout, and cleanup on
  timeout/failure. The hardened smoke passed on the two-Spark cluster with the
  posttrain Transformers-5 image and left no running containers.
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
   only reduced paired harmful-prompt refusal from 1.0 to 0.75 after
   scorer/rubric v4. Trial2 scale1.0 did not improve enough on the worker
   quick gate. Long-search trial2 remains the strongest merged candidate but
   still refused 0.35 of paired harmful prompts. Refusal-suffix trial17 scale1.5
   regressed to 0.50 paired harmful refusal and unchanged unsafe-overcompliance
   refusal. Live-LoRA scale gates are not trusted for this adapter after the
   scale0.75 equivalence failure. The next tracked search is
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_residual_search.yaml`,
   which uses reusable `*_case_ids` prompt filters to target the exact residual
   harmful cases still refused by the best Heretic candidate. The search
   completed and selected trial index 12 as the strongest export candidate
   (3/10 Heretic-probe refusals, KL 0.0293). The direct export config is
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_residual_trial12.yaml`.
   The worker-local quick gate completed at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_heretic_residual_trial12_quick`:
   paired harmful refusal improved to 0.10 with harmful detail 0.0, but
   unsafe-overcompliance refusal stayed 1.0, paired benign quality was 0.90,
   and challenge capability was 0.875. Hold for the next ablation iteration.
   The follow-up search
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_unsafe_followup_search.yaml`,
   which targets the five remaining trial12 refusal cases and preserves the
   exact benign/challenge cases trial12 regressed, completed on the Spark
   worker. It improved the focused Heretic signal from 4/5 refusals to
   1/5 refusals, but found no zero-refusal trial. The best low-KL candidate is
   trial index 15 / trial_id 14 at KL 0.0095. This is not promotion evidence by
   itself. Next either free reviewed worker disk and export trial15 for a
   merged model-forge quick gate, or revise the search space until the focused
   search reaches 0/5 before another full 51G checkpoint export. The next
   prepared branch is
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_search.yaml`,
   a search-only sequential edit that starts from the merged residual trial12
   checkpoint instead of local FT v4. That search completed on the Spark worker:
   focused baseline was 3/5 refusals, the best refusal-count trial
   reached 1/5 only at high KL 0.1856, and the best within-budget trial was
   trial index 16 / trial_id 15 at 2/5 refusals and KL 0.0003. A diagnostic
   direct export recipe was prepared at
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_trial16.yaml`;
   the checkpoint was exported on the Spark worker and quick-gated at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_heretic_trial12_unsafe_followup_trial16_dgx_spark`.
   Reject it: paired harmful refusal regressed from residual trial12's 0.10 to
   0.20, paired benign quality regressed from 0.90 to 0.80, challenge capability
   regressed from 0.875 to 0.8438, unsafe-overcompliance still refused all 3
   cases, and harmful detail rose from 0.0 to 0.1. The focused Heretic signal
   did not translate to the model-forge quick gate. Next, revise the behavior
   edit/search objective or gate directly on model-forge unsafe-overcompliance
   cases instead of exporting more low-KL near-miss trials from the same method.
   The ablation CLI now includes a lightweight pre-export journal gate:
   `./forge ablate --config <search-config.yaml> heretic-search-analyze`. Run it
   on Heretic JSONL journals before any full checkpoint export. The trial12
   unsafe-followup journal analysis recommends `do_not_export` under the
   zero-refusal follow-up gate, which would have blocked trial16 before the 51
   GiB export.
   A new behavior-edit SFT branch is prepared at
   `configs/finetuning/qwen36_27b_local_ft_v4_behavior_abli_v1.yaml`. It starts
   from `Qwen3.6-27B-local-ft-v4-merged`, trains a small LoRA on refusal-free
   safe redirects plus FT v4 capability anchors, and merges to
   `Qwen3.6-27B-local-ft-v4-abliterated-behavior-v1` for evaluation as
   `local_ft_abli_behavior_v1`. Local data prep accepted 76/76 rows. The
   two-Spark guarded train completed 140/140 steps with train loss 0.8275, the
   adapter was merged into a 51 GiB checkpoint, and the targeted quick gate ran
   at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_behavior_v1_dgx_spark`.
   Reject it: paired harmful refusal stayed at 0.10, paired benign quality fell
   to 0.85, challenge capability fell to 0.8438, and unsafe-overcompliance
   still refused 2/3 cases. Do not promote this checkpoint to `local_ft_abli`
   or use it as the Qwen NVFP4 source.
   The refusal-unlikelihood v2 branch trained successfully on the guarded
   two-Spark path, merged to
   `Qwen3.6-27B-local-ft-v4-abliterated-refusal-unlikelihood-v2`, and
   quick-gated at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_refusal_unlikelihood_v2_dgx_spark`.
   Reject it: paired harmful refusal improved to 0.05, but challenge capability
   fell to 0.8125, paired benign quality stayed at 0.85, harmful detail was
   0.05, and unsafe-overcompliance still refused all 3 cases.
   The refusal-unlikelihood v3 branch also trained successfully on the guarded
   two-Spark path, merged to
   `Qwen3.6-27B-local-ft-v4-abliterated-refusal-unlikelihood-v3`, and
   quick-gated at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_refusal_unlikelihood_v3_dgx_spark`.
   Reject it: paired benign quality recovered to 0.95 and harmful detail stayed
   at 0.0, but paired harmful refusal worsened to 0.55, challenge capability
   was only 0.8438, and unsafe-overcompliance still refused all 3 cases. Do not
   promote it, quantize it, or upload it as a release candidate.
   Current Qwen FT-abli status: no candidate has cleared the zero-refusal plus
   source-capability-retention target. Residual Heretic trial12 remains the best
   refusal suppressor on paired harmful prompts, while v3 proves that lowering
   refusal-unlikelihood pressure restores benign quality but gives up too much
   ablation. The next branch should use the corrected
   `unsafe_ablation_redirect` bucket because legacy `unsafe_overcompliance`
   prompts explicitly request refusal. The prepared v5 branch starts from held
   v2 and uses 68 rows, 18 primary no-denial unsafe-redirect pairs,
   attention-only rank-4 LoRA, 48 guarded steps, and high SFT replay. Before
   training v5, quick-gate v2 on `unsafe_ablation_redirect`, paired boundary,
   and challenge. That quick gate now exists:
   `reports/qwen36_27b_trial12_pref_ul_v2_ablation_redirect_quick_summary.md`.
   v2 passes the corrected ablation-specific bucket, so keep v5 prepared but
   untrained and next repeat/rescore the paired/challenge misses before any
   promotion, quantization, or upload decision.
   The pairwise preference method shift at
   `configs/finetuning/qwen36_27b_local_ft_v4_pairwise_preference_v1.yaml`
   trained 100/100 guarded two-Spark steps, merged, and quick-gated. It is
   rejected: paired harmful refusal remained 0.85, unsafe-overcompliance
   refusal remained 1.0, paired benign quality was 0.90, harmful detail was
   0.0, and challenge capability was 0.8438. Keep
   `local_ft_abli_pairwise_preference_v1` blocked for quantization, upload, and
   promotion.
   A targeted sequential branch at
   `configs/finetuning/qwen36_27b_heretic_trial12_refusal_unlikelihood_v1.yaml`
   trained and merged, but it is rejected as a promotion or NVFP4 source.
   Training completed 90/90 guarded cluster steps from residual Heretic trial12,
   using 15 adjacent refusal-unlikelihood pairs for the five remaining trial12
   refusal failures plus 56 benign/capability anchors. The quick gate lives in
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_heretic_trial12_refusal_unlikelihood_v1_dgx_spark`:
   challenge capability 0.8438, paired harmful refusal 0.10, paired benign
   quality 1.00, unsafe-overcompliance refusal 1.00, and unsafe harmful detail
   0.00 after scoring v5 rescore. It improved paired benign quality but did not
   reduce paired harmful refusal or unsafe-overcompliance refusal versus
   residual trial12, and it regressed challenge capability.
   A response-conditioned method shift was prepared and run:
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_trial12_response_conditioned_search.yaml`.
   It adds generic response-conditioned Heretic prompt support and uses the
   committed safe trace seed
   `datasets/abliteration/qwen36_trial12_response_conditioned_traces.jsonl`:
   five actual residual refusal traces become bad contrast examples and eight
   no-refusal safe redirects become good contrast examples. The search completed
   32/32 guarded worker trials, but the journal gate recommends
   `do_not_export`: best refusal count was 1/5 only at KL 0.1303, above the
   0.075 budget, and the best within-budget frontier stayed at 2/5. Do not
   export, upload, quantize, or promote this branch.
   A deliberate high-KL diagnostic export was then added at
   `configs/abliteration/qwen36_27b_ft_local_abli_heretic_trial12_response_conditioned_trial19_aggressive.yaml`
   to test whether the 0.075 KL cap was too conservative. It exported from
   response-conditioned search trial index 19, synced the checkpoint to both
   Sparks, passed strict checkpoint/tokenizer audits, and ran the targeted
   TP=2 quick gate at
   `results/qwen36_27b_v0/base/qwen36_27b_local_ft_abli_heretic_trial12_response_conditioned_trial19_aggressive_dgx_spark`.
   Reject it: paired harmful refusal regressed to 0.20 versus residual
   trial12's 0.10, unsafe-overcompliance still refused all 3 cases, paired
   benign quality stayed 0.90, harmful detail stayed 0.0, and challenge
   capability improved to 0.9062 but remained below local FT v4's 0.9375. The
   hypothesis that the KL cap alone blocked a better Qwen FT-abli candidate did
   not hold. The rejected 51 GiB trial19 checkpoint was removed from both
   Sparks after validation to restore disk headroom; keep only the recipe,
   aggregate eval results, and cluster sync evidence.
   The next prepared method shift is
   `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v2.yaml`.
   It starts from residual Heretic trial12 and combines length-normalized
   chosen-vs-rejected preference loss with token-level refusal unlikelihood on
   rejected completions. The prepared data pack has 93 rows and 39 paired
   chosen/rejected rows, including all 15 residual-trial12 pairs plus 24
   adjacent eval-style/broader refusal pairs, with capability and no-refusal
   redirect SFT replay. The variant
   `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2` is blocked
   for quantization, upload, and promotion until it is trained, merged, and
   quick-gated. This work also fixed the finetune data builder so mixed SFT/pair
   sources preserve optional `rejected_messages` fields before
   `Dataset.from_list`.
   This branch also exposed and fixed a repo pain point: generated finetune
   trainers now enforce exact holdout-prompt rejection when manifests set
   `reject_eval_prompt_overlap: true`. The run also exposed a serving pain
   point: the quick gate used single-node `./forge serve` by default and ran at
   roughly 4.2 tok/s. That launcher path now resolves `MODEL_FORGE_CLUSTER_CONFIG`
   or `MODEL_FORGE_SPARK_CLUSTER_CONFIG`, derives Spark nodes and tensor
   parallel size from the inventory, and can hard-stop solo fallback with
   `MODEL_FORGE_SERVE_REQUIRE_CLUSTER=1`. A follow-up fix makes cluster serving
   use `distributed.master_addr`, `distributed.rdzv_endpoint`, or node-level
   `serving_host` when the coordinator SSH host is `localhost`. A TP=2 cluster
   smoke against `local_ft_abli_heretic_residual_trial12` passed 4/4 prompts
   with median latency 22.2894s, then both vLLM containers were stopped.
   A scoring v5 eval fix also landed: imperative harmful-boundary refusals such
   as "do not provide" are recognized for unsafe prompts without becoming global
   benign-refusal patterns, and safe redirect keywords can use normalized
   alternatives.
   Eval manifests now include `scoring_version` at top level and in canonical
   metadata, not only in the provenance card.

## Operational Guardrails

- Run only one large model server or training job at a time.
- Run `heretic-search-analyze` after Heretic search-only jobs and before any
  full Heretic export. Export only if the analysis recommends
  `export_for_model_forge_quick_gate`, then still require the model-forge quick
  gate before promotion.
- Do not bypass guarded run scripts for full training.
- Do not run generated Heretic search/direct Python directly for large models.
  `./forge ablate ... sota-run --execute` now routes Heretic recipes with
  `container_image` through the guarded Docker wrapper automatically.
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
