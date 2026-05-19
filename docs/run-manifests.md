# Run Manifests

Model Forge uses canonical run manifests to preserve provenance before, during,
and after experiments. A manifest is not a result by itself. It records the
exact repo, config, command, hardware, environment, and output context for a
planned or completed run.

Default manifests are written under:

```text
reports/generated/manifests/<run_id>/manifest.json
```

That directory is generated output and is ignored by Git. Commit reusable code,
configs, docs, and lightweight report summaries; upload durable run artifacts to
Hugging Face when they should survive local cleanup.

## Write A Planned Manifest

```bash
./forge manifest write \
  --run-type eval \
  --status planned \
  --family gemma4_26b_a4b \
  --variant base \
  --config configs/experiments/gemma4_26b_a4b_v0.yaml \
  --run-output-dir results/gemma4_26b_a4b_v0/base \
  --command './forge eval gemma4_26b_a4b base --internal'
```

Use `planned` before a run, `running` at launch, and `completed` or `failed`
after the attempt. The same schema is used for eval, fine-tune, ablation,
dataset, serving, quantization, publish, and comparison work.

## Add Artifacts Or Metrics

```bash
./forge manifest write \
  --run-type eval \
  --status completed \
  --family gemma4_26b_a4b \
  --variant base \
  --artifact scores_csv=scores.csv \
  --artifact responses_jsonl=responses.jsonl \
  --metric normal_use_regression_pass_rate=1.0 \
  --command './forge eval gemma4_26b_a4b base --internal'
```

Values passed to `--metric` and `--metadata` are parsed as JSON when possible.

## Eval Integration

`model_forge.evals.run_eval` still writes its existing eval-specific
`manifest.json` for compatibility with comparison reports. That file now also
contains a `canonical` block with the shared provenance schema.

## Secret Handling

The manifest writer records only selected operational environment variables and
redacts keys that look like tokens, API keys, passwords, or secrets. Do not pass
secrets through `--metadata`, `--artifact`, or `--note`.

## What To Check

Before publishing or handing off an experiment:

```bash
./forge manifest show reports/generated/manifests/<run_id>/manifest.json
./forge doctor
```

The manifest should identify:

- git commit, branch, and dirty paths
- model family and variant
- config paths and SHA-256 hashes
- command
- hardware profile and GPU metadata
- safe environment snapshot
- output directory, artifacts, metrics, and notes
