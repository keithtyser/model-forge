# Hugging Face Publishing

Model Forge treats Hugging Face as the durable artifact layer and GitHub as the
methodology layer. Use this workflow for models, datasets, reports, and
benchmark evidence that should survive beyond one local machine.

## Commands

Check local Hub auth/config without printing secrets:

```bash
./forge hf status --offline
./forge hf whoami
```

Plan a model release and generate local card/provenance files:

```bash
./forge hf plan-model <family> <variant> --release-class report_only
./forge hf publish-model <family> <variant> --release-class public_quantized_model --dry-run
```

Execute a model upload only after the dry run is unblocked:

```bash
./forge hf publish-model <family> <variant> \
  --release-class public_quantized_model \
  --artifact-path <local-artifact-dir> \
  --validation-state spark_cluster_validated \
  --eval-results <serving-eval-dir-or-scores.csv> \
  --serving-card <serve-bench-summary.json> \
  --quantization-card <quantization-card.json> \
  --promotion-report <promotion-or-gate.json> \
  --source-license-checked \
  --execute
```

`publish-model --execute` defaults to environment tokens only. It will not use a
cached Hugging Face token unless `--token-source cache` is passed explicitly.

Plan a public dataset release through the dataset factory:

```bash
./forge data publish <family> <variant> --source-license-checked --overwrite
```

Audit a prepared dataset or redacted publish bundle before upload:

```bash
./forge hf publish-dataset <dataset_path> --repo-id <user-or-org>/<dataset-repo> --dry-run
```

Plans are written under `reports/generated/hub/<run_id>/` by default:

```text
README.md
hub_publish.json
hub_model_plan.json
```

`publish-model --execute` refuses blocked plans, missing tokens, missing
artifact/evidence files, and public-scan findings before any upload. It uploads
the generated `README.md`, the whitelisted artifact files from the plan,
sanitized supporting evidence under `model-forge-evidence/`, and final
`hub_publish.json` provenance. A blocked dry run returns nonzero so it can be
used in CI. Dataset factory publishing can execute only for non-smoke datasets
and uploads the generated redacted bundle when the release class requires public
redaction; `forge hf publish-dataset` remains a Hub-side dry-run audit.

## Release Classes

Release classes live in `configs/release_classes/`:

- `report_only`: public card and reproducibility metadata, no checkpoint files.
- `public_adapter`: public adapter release with eval evidence and redacted
  outputs.
- `private_research_model`: private checkpoint handoff with license and
  no-secrets checks.
- `public_quantized_model`: public quantized checkpoint with eval, serving, and
  quantization evidence. Public full checkpoints require Spark validation.
- `public_dataset`: public dataset release with card, provenance, and redaction
  gates.

The reusable rule is that the workflow generalizes, but release constants do
not. Add or tune release classes for new artifact types instead of hard-coding
Gemma-specific behavior.

## Gates

Every model plan records `release_gates` in `hub_publish.json`. Common gates:

- generated model card contains source, evidence, reproducibility, and
  limitations sections
- source license and provenance were explicitly checked
- eval results, quantization cards, serving cards, promotion reports, or risk
  reports are present when required
- raw unsafe outputs are excluded from public plans
- planned upload files do not contain secret-like tokens or private absolute
  paths
- model-artifact uploads are blocked when the family variant records
  `promotion.decision: rejected` or includes `hf_upload` in
  `promotion.blocked_actions`
- public full-checkpoint releases are blocked unless the release class allows
  them and validation is at least `spark_single_node_validated`

Do not override these gates in generated JSON. Fix the evidence or choose a
less permissive release class.

Dataset factory publish plans record the same gate shape in `hf_publish_plan.json`.
For `public_dataset`, the factory creates `hf_publish_bundle/` with:

- `README.md`
- `dataset_redacted.jsonl`
- `redaction_report.json`
- manifest, quality, verification, generation, and review reports

`dataset_redacted.jsonl` preserves IDs, skills, provenance, verification,
quality metadata, content hashes, and message lengths while replacing message
text with `<redacted>`. Public plans exclude `accepted.jsonl`, `rejected.jsonl`,
and raw `dataset.jsonl` unless a different release class explicitly allows raw
outputs.

The generic Hub audit path writes `hub_dataset_plan.json` next to the dataset
path, or under `--output-dir` when supplied. It checks:

- dataset path exists
- license is present
- source provenance is present
- PII scan passed when a report is available
- unsafe examples are redacted for public releases unless raw outputs are
  explicitly included
- dataset card has purpose, counts, and provenance sections
- schema and split-size metadata are present
- planned files do not contain private absolute paths or secret-like tokens

Use the factory path to build a Model Forge dataset bundle, then use
`forge hf publish-dataset --dry-run` as the final publication gate before a
dataset-factory upload.

## Secrets And Paths

Use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` only from the environment. The CLI
does not print token values. Model upload execution defaults to env-only tokens;
cached tokens require `--token-source cache` so tests and agent runs do not
publish accidentally. Generated publish plans redact secret-like values and use
external path labels such as
`<external>/model` rather than user-specific paths.

Never commit model weights, tokens, private hostnames, or private local paths.
Public model cards should link back to:

```text
https://github.com/keithtyser/model-forge
```
