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

Plan a public dataset release through the dataset factory:

```bash
./forge data publish <family> <variant> --source-license-checked --overwrite
```

Plans are written under `reports/generated/hub/<run_id>/` by default:

```text
README.md
hub_publish.json
hub_model_plan.json
```

`publish-model` is dry-run only for now. A blocked dry run returns nonzero so it
can be used in CI or before a manual upload. Dataset publishing can execute only
for non-smoke datasets and uploads the generated redacted bundle when the
release class requires public redaction.

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
- public full-checkpoint releases are blocked unless the release class allows
  them and validation is at least `spark_single_node_validated`

Do not override these gates in generated JSON. Fix the evidence or choose a
less permissive release class.

Dataset publish plans record the same gate shape in `hf_publish_plan.json`. For
`public_dataset`, the factory creates `hf_publish_bundle/` with:

- `README.md`
- `dataset_redacted.jsonl`
- `redaction_report.json`
- manifest, quality, verification, generation, and review reports

`dataset_redacted.jsonl` preserves IDs, skills, provenance, verification,
quality metadata, content hashes, and message lengths while replacing message
text with `<redacted>`. Public plans exclude `accepted.jsonl`, `rejected.jsonl`,
and raw `dataset.jsonl` unless a different release class explicitly allows raw
outputs.

## Secrets And Paths

Use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` only from the environment. The CLI
reports token source but never prints token values. Generated publish plans
redact secret-like values and use external path labels such as
`<external>/model` rather than user-specific paths.

Never commit model weights, tokens, private hostnames, or private local paths.
Public model cards should link back to:

```text
https://github.com/keithtyser/model-forge
```
