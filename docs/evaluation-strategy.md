# Evaluation Strategy

model-forge evaluates post-training changes by comparing model variants against the same reproducible workload suite.

The target variants are:

- `base`: original upstream model
- `ft`: fine-tuned model
- `abli`: ablated model
- `ft_then_abli`: fine-tuned model after ablation
- `abli_then_ft`: ablated model after fine-tuning, when supported

The goal is not to optimize for one public leaderboard score. The goal is to determine whether a post-training intervention makes the model more useful for practical local-assistant workloads without causing unacceptable regressions.

## Core Questions

Every model variant should answer these questions:

1. Did useful task performance improve?
2. Did formatting and instruction-following improve or regress?
3. Did benign over-refusal decrease?
4. Did unsafe overcompliance increase?
5. Did normal coding, debugging, and explanation quality regress?
6. Did long-form artifact generation improve?
7. Did serving cost, latency, or throughput change?

A fine-tune that improves style but breaks JSON, code, or safety boundaries is not a clear improvement. An ablation that reduces benign refusals but collapses normal assistant behavior is also not a clear improvement.

## Evaluation Layers

### 1. Fast Regression Suite

This suite is small enough to run frequently.

It covers:

- agentic multi-step planning
- tool-use JSON
- structured extraction
- self-critique
- code/debug help
- benign refusal boundaries
- unsafe overcompliance boundaries
- normal-use regression

Outputs:

- `manifest.json`
- `scores.csv`
- `responses.jsonl`
- `examples.md`

Primary use:

- quick base-vs-variant comparison
- CI-style regression checks
- prompt/template/server sanity checks

### 2. Artifact Workbench Suite

This suite is slower and intended for human inspection plus lightweight automated checks.

It covers:

- single-file HTML dashboards and reports
- Canvas/WebGL visualizations
- utility scripts
- longer code/artifact generation behavior

Outputs:

- `artifact_report.html`
- extracted files in `artifacts/`
- raw responses in `responses.jsonl`
- score summaries in `scores.csv`

Primary use:

- judge creative/code artifact usefulness
- inspect design quality and completeness
- compare base vs fine-tune vs ablation outputs side by side
- detect thinking starvation or output-format collapse on long tasks

### 3. Execution Validation

This is the next planned layer.

HTML artifacts should be opened in a browser runner and checked for:

- console errors
- non-empty DOM
- required visible elements
- screenshot capture
- non-blank Canvas/WebGL pixels

Python artifacts should be checked with:

- syntax compilation
- `--help` execution
- fixture-based functional tests
- expected stdout fields

The purpose is to avoid rewarding artifacts that look plausible in text but fail when executed.

### 4. External Baselines

Public benchmarks are useful as context, not as the sole decision-maker.

Relevant external suites include:

- Open LLM Leaderboard v2 for broad static capability
- IFEval for instruction following
- LiveCodeBench and BigCodeBench for coding
- SWE-bench style tasks for repository-scale patching
- Terminal-Bench for command-line agent work
- OSWorld for computer-use agents
- WebDev Arena, WebGen-Bench, and DesignBench for web/artifact generation

model-forge should be able to report external results alongside internal workload results, but internal workload regressions should block promotion even when public benchmark numbers look good.

## Reproducibility Requirements

Every run should preserve:

- model id or local path
- variant name
- model revision or checkpoint path
- quantization
- serving engine and version
- hardware label
- context length
- max output tokens
- sampling settings
- thinking mode
- prompt suite version
- raw model responses
- extracted artifacts
- latency and token usage

Runs should be comparable only when the prompt suite, serving settings, and sampling settings are held constant.

## Recommended Comparison Matrix

For each candidate model family:

| Variant | Fast suite | Artifact suite | Notes |
| --- | --- | --- | --- |
| `base` | required | required | establishes baseline |
| `ft` | required | required | checks fine-tune usefulness |
| `abli` | required | optional first pass | checks refusal and behavior drift |
| `ft_then_abli` | required | required | main target if ablation preserves fine-tune gains |
| `abli_then_ft` | required | required when available | checks whether fine-tuning restores useful behavior |

If compute is limited, run the fast suite first, then artifact suite only for promising variants.

Run a matrix directly:

```bash
model-forge-matrix \
  --config configs/experiments/qwen35_9b_v0.yaml \
  --variant base=Qwen/Qwen3.5-9B \
  --variant ft=/models/qwen35-ft \
  --variant abli=/models/qwen35-abli \
  --variant ft_then_abli=/models/qwen35-ft-abli \
  --output-prefix qwen35_9b
```

Run repeated trials for sampled long-form tasks:

```bash
MODEL_FORGE_TRIALS=3 ./scripts/dgx_spark_artifact_eval_qwen35_9b.sh
```

## Promotion Criteria

A post-trained variant should not be considered better than base unless it satisfies all of the following:

- equal or better normal-use regression score
- equal or better JSON/schema adherence
- equal or better code/debug workflow score
- lower benign over-refusal, when applicable
- no meaningful increase in unsafe overcompliance
- no recurring thinking starvation or output truncation
- artifact outputs are at least as complete and executable as base

For ablation work, the key distinction is reducing unnecessary refusals while preserving useful assistant behavior. The evaluation should not treat unsafe overcompliance as an improvement.

## Reporting

Each comparison report should include:

- model and serving metadata
- per-bucket score deltas
- latency and throughput deltas
- refusal-boundary examples
- normal-use regression examples
- artifact thumbnails or links
- qualitative notes from human review
- final recommendation: promote, reject, or rerun with adjusted settings

The report should make it easy for another person to reproduce the run from the repository and inspect the raw outputs.

Generate a comparison report:

```bash
model-forge-compare \
  --base results/qwen35_9b_v0/base/qwen35_9b_base \
  --ft results/qwen35_9b_v0/base/qwen35_9b_ft \
  --abli results/qwen35_9b_v0/base/qwen35_9b_abli \
  --ft-then-abli results/qwen35_9b_v0/base/qwen35_9b_ft_then_abli \
  --output-dir reports/generated/qwen35_9b_comparison
```

The report includes:

- score deltas
- improvement/regression classification
- promotion recommendation
- notable failures
- artifact links and screenshots when available

## External Benchmark Bridges

model-forge owns post-training comparison. External benchmark tools should be used for broad benchmark coverage.

Check or run an external tool:

```bash
model-forge-external lm-eval --dry-run
model-forge-external promptfoo --dry-run
```

Install optional Python-backed external tools:

```bash
pip install -e ".[external]"
```

`promptfoo` is a Node.js tool and should be installed with its own documented package manager workflow.

If the external tool is installed, arguments after `--` are passed through to that tool. Each run writes `external_run.json` plus captured stdout/stderr when executed.
