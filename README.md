# model-forge

model-forge is a reproducible post-training workbench for open models.

It helps answer one question: did a fine-tune, ablation, or combined
post-training workflow make the model better without breaking something else?

The repo is evaluation-first today. Fine-tuning and ablation workflows will use
the same family registry and result structure as they are added.

## Current Focus

The first supported family is Gemma 4 on DGX Spark:

- `google/gemma-4-26B-A4B-it`
- `Jackrong/Gemopus-4-26B-A4B-it`
- `huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated`

Qwen configs are included for lower-level experiments, but the simple workflow
currently targets Gemma 4.

## Quick Start

Install with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/keithtyser/model-forge.git
cd model-forge
./forge setup all
```

Download models:

```bash
./forge download gemma4_26b_a4b all
```

Run each model. Keep `serve` running in one terminal, then run `eval` in another.
`eval` runs the built-in checks, artifact generation, external benchmarks, and
refreshes the comparison report for the served variant. The CLI prints phase
progress, per-case progress, elapsed time, and ETA while it runs.

```bash
./forge serve gemma4_26b_a4b base
./forge eval gemma4_26b_a4b base
```

Plan a local Gemma refusal ablation without loading the model:

```bash
./forge ablate gemma4_26b_a4b plan
```

Prepare a SOTA abliteration recipe for the base or fine-tuned model:

```bash
./forge ablate gemma4_26b_a4b sota-prepare --backend heretic
./forge ablate --config configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml sota-prepare --backend heretic
```

The ablation workflow is not "copy Gemma constants to every model." The reusable
part is the loop:

1. pick the source checkpoint, usually base or an already fine-tuned model
2. collect model-specific refusal directions from matched harmful/benign prompts
3. apply a transparent edit or run a bounded Heretic/OBLITERATUS search
4. serve exactly one candidate at a time
5. promote only if refusals drop while normal-use, challenge, artifact, and
   external benchmark scores stay near the source model

For nearby checkpoints, a known-good recipe can be used as a warm start. The
Gemopus r7 recipe does this: it computes fresh directions on
`Jackrong/Gemopus-4-26B-A4B-it`, then applies the selected base Gemma t34
Heretic parameters. For a new architecture such as Qwen, reuse the workflow and
prompt/eval harness, but recalibrate targets, layer ranges, strengths, and
search parameters before trusting the result.

```bash
./forge serve gemma4_26b_a4b ft
./forge eval gemma4_26b_a4b ft
```

```bash
./forge serve gemma4_26b_a4b abli
./forge eval gemma4_26b_a4b abli
```

`compare` prints a Rich terminal summary with internal scores, external
benchmark scores, deltas against base, and recommendations. It also writes the
HTML reports:

```text
reports/generated/gemma4_26b_a4b_comparison/comparison_report.html
```

Generated artifacts can be reviewed side by side here:

```text
reports/generated/gemma4_26b_a4b_comparison/artifact_compare.html
```

## Focused Commands

The default `eval` command is the recommended path. Focused commands are useful
for quick checks or debugging:

```bash
./forge eval gemma4_26b_a4b base --smoke
./forge eval gemma4_26b_a4b base --internal
./forge eval gemma4_26b_a4b base --artifact
./forge eval gemma4_26b_a4b base --external
./forge compare gemma4_26b_a4b
```

For a stronger internal baseline, run three sampled trials per prompt. This is
intentionally expensive because the Gemma internal suite is 106 prompts per
variant, so three trials is 318 generations per variant:

```bash
MODEL_FORGE_TRIALS=3 ./forge eval gemma4_26b_a4b base --internal
MODEL_FORGE_TRIALS=3 ./forge eval gemma4_26b_a4b ft --internal
MODEL_FORGE_TRIALS=3 ./forge eval gemma4_26b_a4b abli --internal
./forge compare gemma4_26b_a4b
./forge golden-summary gemma4_26b_a4b
```

Later, compare a refreshed report against that compact baseline without
rerunning the models:

```bash
./forge golden-check gemma4_26b_a4b
```

## External Benchmarks

Run IFEval through `lm-evaluation-harness` against the served model:

```bash
./forge eval gemma4_26b_a4b base --external
./forge eval gemma4_26b_a4b ft --external
./forge eval gemma4_26b_a4b abli --external
```

The external command checks the active server first, so the requested variant
must match the model currently served by `./forge serve`.

For a quick check:

```bash
MODEL_FORGE_EXTERNAL_LIMIT=20 ./forge external gemma4_26b_a4b base
```

External outputs are written to:

```text
reports/generated/gemma4_26b_a4b_external/
```

## What Gets Measured

model-forge tracks:

- workflow success
- structured output adherence
- normal-use regression
- benign refusal rate
- unsafe overcompliance
- latency and tokens/sec
- raw responses and generated artifacts
- external benchmark outputs

The built-in evals are a screening harness. Promotion decisions should also use
external benchmarks and human inspection of raw outputs.

## Families

List configured families:

```bash
./forge families
```

Family definitions live in:

```text
configs/model_families/
```

A family file defines variants, local model paths, served aliases, eval configs,
external benchmark defaults, and report locations. Adding a new model family
should usually mean adding one YAML file plus, if needed, a serving profile.

## Useful Overrides

```bash
MODEL_FORGE_MODELS_DIR=/data/models ./forge serve gemma4_26b_a4b base
MODEL_FORGE_HARDWARE_PROFILE=dgx_spark ./forge serve gemma4_26b_a4b base
GPU_MEMORY_UTILIZATION=0.80 ./forge serve gemma4_26b_a4b base
MAX_MODEL_LEN=16384 ./forge serve gemma4_26b_a4b base
VLLM_CPU_OFFLOAD_GB=24 ./forge serve gemma4_26b_a4b base
MODEL_FORGE_ENABLE_HIGH_PARALLELISM=1 ./forge ablate gemma4_26b_a4b plan
MODEL_FORGE_PARALLELISM=192 ./forge ablate gemma4_26b_a4b plan
HF_MAX_WORKERS=16 HF_XET_NUM_CONCURRENT_RANGE_GETS=16 ./forge download gemma4_26b_a4b base
```

## Repository Layout

```text
configs/                 Experiment and family configuration
docs/                    Detailed workflow notes
evals/                   Prompt sets and scoring rubrics
pipelines/               Fine-tuning and ablation pipeline notes
reports/                 Generated comparison and external reports
results/                 Raw evaluation outputs
scripts/                 Internal helpers and compatibility wrappers
src/model_forge/         Python package source
forge                    Main user-facing command
```

## Design Principles

- Keep the public workflow short and reproducible.
- Preserve raw outputs so results can be inspected.
- Use external benchmarks instead of trusting only local checks.
- Treat ablation success as fewer refusals while preserving capability; report
  unsafe compliance separately as risk.
- Treat fine-tune success as better capability and structure without regressions.
