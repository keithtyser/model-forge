# Current Status

Last updated: 2026-05-23.

This is the short handoff state for humans and agents. Use
`docs/experiment-ledger.md` for detailed run history and raw observations.

## Validated So Far

- The repo is organized around model families, not Gemma-only scripts.
- Gemma 4 A4B is the first worked family for base, downloaded FT, downloaded
  abli, local base abli, local FT, and local FT abli comparisons.
- Internal evals now cover refusal suppression, benign paired quality,
  normal-use regression, challenge capability, reasoning style stability,
  artifact quality, multi-trial variance, and golden comparisons.
- Local base ablation beat the downloaded abli reference on the saved internal
  comparison while preserving stronger behavior than expected.
- Local FT ablation preserved the source FT's primary internal behavior closely
  enough to count as a successful ablation of an already fine-tuned model.
- Gemma local FT v0 completed under the guarded Spark training path. It was
  close to Jackrong on challenge capability and better on paired benign quality,
  but it did not clear the primary challenge-capability promotion gate.
- The local FT v1 dataset factory MVP is implemented with planning, gap
  extraction, seed rows, generation adapters, verification, filtering, review,
  packing, dry-run publish planning, non-cascading overwrite semantics, and
  length-violation rejection gates.
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
- Serving workload definitions are now present under
  `configs/serving/workloads/`, with smoke and core benchmark configs loading
  reusable workload files instead of hard-coding all requests inline.
- Serving Card generation now writes a structured `serving_card.md` for each
  `bench serve` run with identity, hardware/config, overall metrics,
  per-workload metrics, artifacts, and promotion gates.
- Quantization planning is now first-class for Blackwell NVFP4 runtime import
  and ModelOpt self-quantization. The self-quantization export path has a
  checkout-local lock, preflight memory/disk gates, `systemd-run --scope`,
  Docker CPU/memory limits, and a runtime memory watchdog. No completed
  self-quantized Gemma NVFP4 checkpoint has passed the evidence contract yet.

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

1. Convert roadmap/backlog status from single checkboxes to separate
   `implementation_status` and `validation_state` fields.
2. Add required validation/evidence fields to manifests, report cards, variant
   nodes, objective profiles, and promotion decisions.
3. Add objective profiles for `zero_refusal_capability_retention`,
   `quantized_quality_retention`, and `dgx_spark_latency_throughput`.
4. Finish/evaluate Gemma local FT or write a Training Method failure card with
   distributed Spark correctness evidence.
5. Run one real Spark serving benchmark and attach endpoint evidence to the
   Serving Card.
6. Re-run one guarded ModelOpt NVFP4 export only after confirming the two-node
   worker plan and host guardrails. Quantization remains incomplete until base,
   FT, abli, and FT+abli checkpoints load and match their unquantized baselines.
7. Scale the local FT v1 dataset through medium-pack review before treating it
   as a training dataset.

## Operational Guardrails

- Run only one large model server or training job at a time.
- Do not bypass guarded run scripts for full training.
- Keep `runs/`, `results/`, `reports/generated/`, and local model artifacts out
  of Git unless a small reusable file is intentionally moved into `recipes/`.
- Check `docs/artifact-retention.md` before deleting or uploading artifacts.
- Push code, configs, docs, recipes, and lightweight manifests to GitHub before
  handing off.
- Run `./forge doctor` before handoff to catch tracked ignored files, secret
  literals, nonportable paths, and accidental generated dataset commits.
- Do not bypass quantization or fine-tuning launchers with raw `docker run` or
  `python train.py` for large jobs.
