# Current Status

Last updated: 2026-05-19.

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
  packing, and dry-run publish planning.

## Current Dataset State

The smoke local FT v1 pack contains 49 accepted examples:

- 37 human seed rows
- 12 deterministic synthetic rows
- generation methods: `self_instruct`, `evol_instruct`,
  `instruction_backtranslation`, and `eval_adjacent_generation`
- review gate: `ready_to_scale_generation=true`
- known limitation: this is a scaffold and QC path, not a final training
  dataset

The next dataset step should be a small live-teacher generation smoke, roughly
20-50 candidates, followed by verification and review. Do not start a long
training run for that step.

## Recommended Next Work

1. Run a live-teacher dataset generation smoke for `local_ft_v1`.
2. Inspect generated rows manually and tighten weak templates or skill checks.
3. Scale to a 500-2000 accepted-row dataset only after the small live smoke
   passes review.
4. Publish completed durable datasets to Hugging Face and record repo ids in
   `docs/experiment-ledger.md`.
5. Run a short FT candidate only after dataset quality is established and Spark
   guardrails are active.

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
