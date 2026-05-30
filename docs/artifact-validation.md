# Artifact Validation

Artifact validation checks extracted model outputs as executable files, not just
as text. Use it before claiming artifact-generation, coding, HTML, or tool-use
quality improved.

Validate an artifact directory:

```bash
./forge artifacts validate reports/generated/<run>/artifacts/ \
  --run-id <run>_artifact_validation
```

Outputs are written under `reports/generated/artifact_validation/<run_id>/` by
default:

- `artifact_validations.json`: per-file validation details
- `artifact_execution_card.json`: structured summary and metrics
- `artifact_execution_card.md`: human-readable card

HTML validation:

- checks basic document structure and visible text
- opens the artifact in Playwright when available
- captures console/page errors
- checks desktop and mobile DOM rendering
- checks horizontal overflow and text overlap
- writes a desktop screenshot
- checks nonblank canvas/WebGL pixels when canvases are present

Python validation:

- runs `python -m py_compile`
- runs the artifact with `--help`
- optionally runs a fixture input and validates expected stdout

Fixture config example:

```yaml
python:
  validation_fixture:
    kind: responses_jsonl
    args: ["{fixture}"]
    stdout_any: ["workflow"]
```

Then run:

```bash
./forge artifacts validate reports/generated/<run>/artifacts/ \
  --checks-config configs/artifact_validation/python_responses_fixture.yaml \
  --strict
```

Use `--require-browser` for promotion gates so skipped Playwright/browser
validation fails instead of being treated as a smoke-only skip. Use `--strict`
in CI or promotion paths to return nonzero when any artifact fails.

The standalone card validates files that already exist. To prove a model
generated them, connect the card to the eval or serving run manifest that wrote
the artifacts.

