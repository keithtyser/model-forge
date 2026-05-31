# Agent Experiments

Agent experiment plans are lightweight pre-run contracts for AI agents. They do
not replace canonical run manifests. They define the hypothesis, commands,
resource policy, evidence plan, success criteria, rollback path, and handoff
rules before an agent starts work.

## Commands

Print the schema:

```bash
./forge agent schema
```

Validate tracked agent recipes:

```bash
./forge agent audit
```

Write an Agent Run Card for a planned or completed agent plan:

```bash
./forge agent card recipes/agents/gemma4_ft_behavior_edit.yaml \
  --status planned \
  --write-card \
  --update-ledger
```

Create a serving optimization plan without starting a server:

```bash
./forge agent optimize-serving \
  --family gemma4_26b_a4b \
  --variant base \
  --cluster-config configs/clusters/dgx_spark_x2.example.yaml \
  --output recipes/agents/gemma4_base_serving_optimization.yaml
```

Create a quantization optimization plan without loading model weights:

```bash
./forge agent optimize-quantization \
  --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml \
  --variants base,local_ft \
  --output recipes/agents/gemma4_nvfp4_quantization.yaml
```

Create a behavior-edit optimization plan without running ablation:

```bash
./forge agent optimize-behavior-edit \
  --family gemma4_26b_a4b \
  --config configs/abliteration/gemma4_26b_a4b_ft_local_abli.yaml \
  --source-variant local_ft \
  --target-variant ft_local_abli_sota_internal_r7_selected_t34_transfer \
  --backend heretic \
  --output recipes/agents/gemma4_ft_behavior_edit.yaml
```

Create a new plan from the template:

```bash
./forge agent init \
  --experiment-id qwen35_local_abli_plan \
  --title "Qwen 3.5 local ablation plan" \
  --family qwen35_9b \
  --variant base \
  --objective-profile zero_refusal_capability_retention \
  --experiment-type ablation \
  --output recipes/agents/qwen35_local_abli_plan.yaml
```

## Required Shape

Every plan uses `schema_version: model_forge.agent_experiment.v1` and must
include:

- identity: `experiment_id`, `title`, `owner_agent`, `family`, `variant`,
  `objective_profile`, `experiment_type`, and `status`
- hypothesis: the concrete claim to validate or falsify
- `planned_commands`: commands, purpose, whether they start heavy jobs, and
  whether they require explicit execution
- `resource_policy`: memory/disk floors, one-heavy-job-at-a-time policy,
  cluster usage, and checkpoint-or-plan-before-execute requirements
- `evidence_plan`: expected reports, required validation commands, manifest
  requirements, and ledger update requirements
- `success_criteria`, `rollback_plan`, and `handoff`

The validator rejects secret-like command text, unknown families, unknown
variants, missing resource guardrails, and incomplete handoff fields.

## Relationship To Manifests

Use agent experiment plans before work starts. Use run manifests during and
after actual runs:

```bash
./forge manifest write \
  --run-type eval \
  --status planned \
  --family gemma4_26b_a4b \
  --variant base \
  --command './forge eval gemma4_26b_a4b base --internal'
```

For heavy model work, the plan should require a manifest, a guarded launcher,
and evidence in `docs/experiment-ledger.md` before handoff.

## Agent Run Cards

Agent Run Cards summarize a specific agent experiment plan into
`agent_run_card.json` and `agent_run_card.md`. They include the plan identity,
hypothesis, command counts, heavy commands, resource policy, expected evidence,
required validation commands, handoff policy, schema validation findings, and
current Git metadata.

By default, `--write-card` writes under
`reports/generated/agent_runs/<experiment_id>/`. Use `--output-dir` when the
card belongs next to generated run artifacts.

Pass `--update-ledger` to insert or replace a marked block in
`docs/experiment-ledger.md`. Ledger updates are idempotent by
`experiment_id`, so re-running the command refreshes the existing block instead
of appending duplicates. Use `--ledger <path>` for tests or scratch ledgers.
