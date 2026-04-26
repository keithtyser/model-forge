# model-forge

model-forge is a reproducible post-training evaluation workbench for open
models. It is designed to compare a base model against fine-tuned, ablated, and
combined variants using the same serving stack, prompts, runtime metadata,
external benchmarks, and reports.

The immediate focus is Gemma 4 on DGX Spark:

- base: `google/gemma-4-26B-A4B-it`
- fine-tune: `Jackrong/Gemopus-4-26B-A4B-it`
- ablation: `huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated`

Qwen experiment configs are also present. The reusable family runner is
config-driven, but the first fully populated family registry is Gemma 4.

## What It Measures

model-forge is meant to answer practical post-training questions:

- Did the fine-tune improve reasoning structure, workflow quality, or long-form stability?
- Did the ablation reduce false refusals?
- Did either variant regress normal-use capability?
- Did unsafe overcompliance increase?
- Do external benchmark signals agree with the local workbench?
- Are raw outputs and artifacts preserved so humans can inspect failures?

The in-repo evals are a fast screening harness. They are not a replacement for
external benchmarks or human review.

## Install

Use `uv` for reproducible setup.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/keithtyser/model-forge.git
cd model-forge
./scripts/setup.sh
```

Optional extras:

```bash
./scripts/setup.sh external
./scripts/setup.sh artifacts
./scripts/setup.sh all
```

Profiles:

- `base`: core model-forge CLI and eval harness
- `external`: `lm-evaluation-harness`, LightEval, Inspect AI dependencies
- `artifacts`: Playwright browser validation for generated HTML/Canvas/WebGL
- `all`: external plus artifact dependencies

## Gemma 4 Quick Start

Download the three Gemma 4 variants:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b download base
./scripts/model_forge_dgx.py gemma4_26b_a4b download ft
./scripts/model_forge_dgx.py gemma4_26b_a4b download abli
```

The downloader prompts for `HF_TOKEN` if it is not already set. It stores models
under `~/models` by default and uses Hugging Face Xet high-performance mode.

Serve and evaluate each variant. Run `serve` in one terminal, then run evals in
another terminal while the server is up.

Base:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b serve base
./scripts/model_forge_dgx.py gemma4_26b_a4b smoke base
./scripts/model_forge_dgx.py gemma4_26b_a4b full base
```

Fine-tune:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b serve ft
./scripts/model_forge_dgx.py gemma4_26b_a4b smoke ft
./scripts/model_forge_dgx.py gemma4_26b_a4b full ft
```

Ablation:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b serve abli
./scripts/model_forge_dgx.py gemma4_26b_a4b smoke abli
./scripts/model_forge_dgx.py gemma4_26b_a4b full abli
```

Compare:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b compare
```

For Gemma only, `./scripts/gemma4_dgx.sh ...` remains available as a short
compatibility alias for the generic family runner.

Open:

```text
reports/generated/gemma4_26b_a4b_comparison/comparison_report.html
```

## External Benchmarks

Install external dependencies once:

```bash
./scripts/model_forge_dgx.py gemma4_26b_a4b external-install
```

Run IFEval through `lm-evaluation-harness` against the currently served model:

```bash
MODEL_FORGE_EXTERNAL_LIMIT=20 ./scripts/model_forge_dgx.py gemma4_26b_a4b external base ifeval
MODEL_FORGE_EXTERNAL_LIMIT=20 ./scripts/model_forge_dgx.py gemma4_26b_a4b external ft ifeval
MODEL_FORGE_EXTERNAL_LIMIT=20 ./scripts/model_forge_dgx.py gemma4_26b_a4b external abli ifeval
```

Remove `MODEL_FORGE_EXTERNAL_LIMIT` for a full run.

External outputs are written under:

```text
reports/generated/gemma4_26b_a4b_external/
```

The wrapper checks `/v1/models` before running so you do not accidentally score
the wrong served variant.

## Artifact Workbench

Artifact runs ask models to generate practical HTML, Canvas/WebGL, and Python
artifacts. Outputs are saved for human inspection.

```bash
./scripts/setup.sh artifacts
./scripts/model_forge_dgx.py gemma4_26b_a4b artifact base
./scripts/model_forge_dgx.py gemma4_26b_a4b artifact ft
./scripts/model_forge_dgx.py gemma4_26b_a4b artifact abli
```

Run artifact commands with the matching variant server already running.

## Common Overrides

Use these when moving across machines:

```bash
MODEL_FORGE_MODELS_DIR=/data/models ./scripts/model_forge_dgx.py gemma4_26b_a4b serve base
GPU_MEMORY_UTILIZATION=0.80 ./scripts/model_forge_dgx.py gemma4_26b_a4b serve base
MAX_MODEL_LEN=16384 ./scripts/model_forge_dgx.py gemma4_26b_a4b serve base
MODEL_FORGE_EXTERNAL_CONCURRENCY=2 ./scripts/model_forge_dgx.py gemma4_26b_a4b external base ifeval
```

Model downloads can be tuned:

```bash
HF_MAX_WORKERS=16 HF_XET_NUM_CONCURRENT_RANGE_GETS=16 ./scripts/model_forge_dgx.py gemma4_26b_a4b download base
```

## Adding Model Families

Model-family convenience workflows are defined in `configs/model_families/`.
Add a new YAML file to describe:

- variant repo IDs and local directory names
- served model aliases
- serve script and default vLLM settings
- eval and artifact configs
- comparison output path
- external benchmark output path and default tasks

Then run the generic command:

```bash
./scripts/model_forge_dgx.py <family> serve base
./scripts/model_forge_dgx.py <family> full base
./scripts/model_forge_dgx.py <family> external base ifeval
./scripts/model_forge_dgx.py <family> compare
```

## Lower-Level CLI

The Python entrypoints remain available for custom configs and other model
families:

```bash
uv run model-forge-eval --config configs/experiments/qwen35_9b_v0.yaml --dry-run
uv run model-forge-compare --base <base-run> --ft <ft-run>
uv run model-forge-matrix --config <config.yaml> --variant base=<model> --variant ft=<model>
uv run model-forge-external lm-eval --dry-run
```

Use these when building a new model-family workflow before adding a convenience
family registry in `configs/model_families/`.

## Outputs

Evaluation runs write to `results/`. Generated reports write to
`reports/generated/`.

Typical files:

- `manifest.json`
- `scores.csv`
- `responses.jsonl`
- `examples.md`
- `comparison.json`
- `comparison.csv`
- `comparison_report.html`
- `external_run.json`
- captured external benchmark `stdout.txt` and `stderr.txt`
- artifact reports and extracted generated files when running artifact suites

Generated outputs are ignored by git.

## Repository Layout

```text
configs/         Experiment and suite configuration
configs/model_families/
                 Model-family registries for one-command workflows
datasets/        Dataset manifests and metadata
docs/            Evaluation and platform documentation
evals/           Prompt sets and scoring rubrics
models/          Local checkpoint directory conventions
pipelines/       Training and ablation pipeline notes
reports/         Report templates and generated comparison outputs
results/         Machine-readable run outputs
scripts/         Setup, download, serving, and eval wrappers
src/             Python package source
```

## Design Principles

- Prefer short, reproducible commands over bespoke shell sessions.
- Keep raw model outputs for inspection.
- Treat in-repo evals as screening, not final truth.
- Use external benchmark runners where they are stronger.
- Judge ablations by false-refusal reduction without unsafe-overcompliance or normal-use regressions.
- Judge fine-tunes by capability and structure improvements without safety or workflow regressions.
