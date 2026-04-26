# model-forge

model-forge is a reproducible post-training workbench for open models.

It is built for comparing base models, fine-tunes, ablated models, and combined
fine-tuned plus ablated variants under the same prompts, runtime metadata,
scoring rules, and report format.

The core goal is practical: make it clear whether a post-training change made a
model better, worse, or merely different.

## Why This Exists

Open model post-training often improves one behavior while quietly damaging
another. Fine-tunes can improve reasoning structure but increase style drift.
Ablations can reduce refusals but weaken safety boundaries or normal task
quality. model-forge is designed to measure those tradeoffs directly.

The evaluation layer focuses on:

- capability retention across normal coding, extraction, planning, and tool-use tasks
- fine-tune gains in reasoning quality, answer structure, and long-form stability
- ablation gains in reduced false refusals
- safety-boundary regressions and unsafe overcompliance
- artifact-producing tasks such as HTML, Canvas/WebGL, and Python utilities
- reproducible base versus variant comparisons

## Current Scope

The evaluation and comparison system is active. Training and abliteration
pipeline modules are scaffolded so they can converge on the same experiment
configuration and variant matrix.

Starter model families include:

- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.6-27B`
- `google/gemma-4-E4B-it`

## Features

- OpenAI-compatible backend support for local servers such as vLLM.
- Fixed prompt suites for base, fine-tune, ablation, and combined-model testing.
- Variant matrix runs for `base`, `ft`, `abli`, `ft_then_abli`, and `abli_then_ft`.
- Repeated trials for stability checks.
- Rule-based scoring with raw response preservation.
- Artifact extraction for generated HTML, Canvas/WebGL, and Python code.
- Optional Playwright validation for browser-rendered artifacts.
- Python artifact validation through compile, help, and fixture execution checks.
- HTML, JSON, and CSV comparison reports.
- Promotion recommendations for deciding whether a variant should replace a base.
- External benchmark bridge for `lm-eval`, `LightEval`, `Inspect`, and `promptfoo`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional artifact validation:

```bash
pip install -e ".[artifacts]"
python -m playwright install chromium
```

Optional external benchmark adapters:

```bash
pip install -e ".[external]"
```

`promptfoo` is a Node-based tool and should be installed separately if you plan
to use that adapter.

## Quick Start

Run the default Qwen 3.5 9B suite in dry-run mode:

```bash
model-forge-eval \
  --config configs/experiments/qwen35_9b_v0.yaml \
  --dry-run
```

Run against an OpenAI-compatible endpoint:

```bash
export MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1
export MODEL_FORGE_MODEL=Qwen/Qwen3.5-9B
export MODEL_FORGE_VARIANT=base

model-forge-eval \
  --config configs/experiments/qwen35_9b_v0.yaml \
  --output-suffix qwen35_9b_base
```

Run the artifact workbench:

```bash
model-forge-eval \
  --config configs/experiments/qwen35_9b_artifacts_v0.yaml \
  --output-suffix qwen35_9b_artifacts
```

## Variant Matrix

Use the matrix runner when comparing multiple post-training variants under the
same configuration:

```bash
model-forge-matrix \
  --config configs/experiments/qwen35_9b_v0.yaml \
  --variant base=Qwen/Qwen3.5-9B \
  --variant ft=/models/qwen35-ft \
  --variant abli=/models/qwen35-abli \
  --variant ft_then_abli=/models/qwen35-ft-then-abli \
  --variant abli_then_ft=/models/qwen35-abli-then-ft \
  --output-prefix qwen35_9b
```

Each variant records its model alias, runtime metadata, prompt results, response
text, score breakdowns, and aggregate metrics.

## Comparing Runs

Generate a comparison report from completed runs:

```bash
model-forge-compare \
  --base results/qwen35_9b_v0/base/qwen35_9b_base \
  --ft results/qwen35_9b_v0/ft/qwen35_9b_ft \
  --abli results/qwen35_9b_v0/abli/qwen35_9b_abli \
  --ft-then-abli results/qwen35_9b_v0/ft_then_abli/qwen35_9b_ft_then_abli \
  --abli-then-ft results/qwen35_9b_v0/abli_then_ft/qwen35_9b_abli_then_ft \
  --output-dir reports/generated/qwen35_9b_comparison
```

Comparison outputs include:

- `comparison.json`
- `comparison.csv`
- `comparison_report.html`
- aggregate score deltas
- prompt-family deltas
- improvement and regression classifications
- notable failures
- promotion recommendations
- side-by-side artifact links and screenshots when available

## External Benchmarks

model-forge does not try to replace broad benchmark runners. It provides a
bridge so external benchmarks can be run and recorded alongside model-forge
results.

```bash
model-forge-external lm-eval --dry-run
model-forge-external lighteval --dry-run
model-forge-external inspect --dry-run
model-forge-external promptfoo --dry-run
```

External runs write an `external_run.json` record and preserve command output
for reproducibility.

## DGX Spark

DGX Spark workflows are documented in [docs/dgx-spark.md](docs/dgx-spark.md).

Common entrypoints:

```bash
./scripts/dgx_spark_serve_qwen35_9b.sh
./scripts/dgx_spark_smoke_eval_qwen35_9b.sh
./scripts/dgx_spark_full_eval_qwen35_9b.sh
./scripts/dgx_spark_artifact_eval_qwen35_9b.sh
```

Local model paths can be served by mounting the model directory into the vLLM
container and setting `MODEL_FORGE_MODEL` to the advertised model alias.

## Outputs

Evaluation runs write to `results/` by default. Generated reports write to
`reports/generated/`.

Typical run files:

- `manifest.json`
- `scores.json`
- `responses.jsonl`
- `examples.json`
- `artifact_report.html`
- `artifact_validations.json`
- extracted files under `artifacts/`

Generated outputs are ignored by git so experiments can be reproduced locally
without polluting the repository history.

## Repository Layout

```text
configs/         Experiment, backend, and suite configuration
datasets/        Dataset manifests and metadata
docs/            Evaluation and platform documentation
evals/           Prompt sets and scoring rubrics
models/          Local checkpoint directory conventions
pipelines/       Training and ablation pipeline notes and entrypoint plans
reports/         Report templates and generated comparison outputs
results/         Machine-readable run outputs
scripts/         Shell entrypoints for local and DGX Spark workflows
src/             Python package source
```

## Documentation

- [Evaluation strategy](docs/evaluation-strategy.md)
- [DGX Spark workflow](docs/dgx-spark.md)

## Design Principles

- Compare variants against the same prompts, backend settings, and scoring rules.
- Preserve raw outputs so failures can be inspected instead of summarized away.
- Treat ablation success as reduced false refusals without increased unsafe
  overcompliance or normal-use regression.
- Treat fine-tune success as better reasoning quality, more stable structure,
  and less long-form drift without capability loss.
- Use established external benchmark tools where they are strongest, and keep
  model-forge focused on post-training comparison and promotion decisions.
