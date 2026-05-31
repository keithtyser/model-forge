# Upstream Pull Requests

Model Forge should not open random upstream pull requests just to satisfy a
roadmap line. Upstream work needs a concrete target, local evidence, and a
small patch that maintainers can reasonably review.

Audit candidate plans:

```bash
./forge upstream audit --config configs/upstream/pr_candidates.yaml
```

Use strict audit only after replacing placeholder targets:

```bash
./forge upstream audit --config configs/upstream/pr_candidates.yaml --strict
```

Write an upstream PR plan:

```bash
./forge upstream plan \
  --config configs/upstream/pr_candidates.yaml \
  --candidate kernel_card_docs_or_example \
  --write-plan
```

Verify a recorded PR and evidence paths:

```bash
./forge upstream verify-pr \
  --config configs/upstream/pr_candidates.yaml \
  --candidate kernel_card_docs_or_example \
  --write-report
```

Use `--offline` only while drafting. Offline reports verify the config and local
evidence paths, but they do not prove the external PR exists.

Plans are written under `reports/generated/upstream_prs/<run>/`:

- `upstream_pr_plan.json`
- `upstream_pr_plan.md`
- `upstream_pr_verification.json`

Completion rule:

- `MF-0808` is not complete until a real external pull request URL is recorded.
- `external_pr_url` must be a GitHub pull request URL such as
  `https://github.com/<owner>/<repo>/pull/<number>`.
- The PR should cite benchmark, profiler, Kernel Card, or serving evidence.
- Opened or merged PR records must point at existing local evidence files.
- The evidence must not contain private paths, private hostnames, unresolved
  placeholders, or secrets.
- Completion requires a non-offline `upstream_pr_verification.json` with
  `verified=true`.
- A docs/example PR is acceptable only when it is grounded in real measured
  artifacts, not generic advice.
