---
dataset_info:
  config_name: gemma4_26b_a4b_local_ft_v1
  features:
    - name: messages
      dtype: list
    - name: skills
      dtype: list
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
tags:
  - model-forge
  - supervised-fine-tuning
  - eval-adjacent
---

# gemma4_26b_a4b_local_ft_v1

Eval-adjacent dataset factory plan for the next Gemma 4 26B-A4B local fine-tune. It targets the v0 gaps without copying model-forge eval prompts.

## Purpose

This is a Model Forge dataset-factory artifact for `gemma4_26b_a4b` /
`local_ft_v1` under the `capability_sft` objective. It targets
observed fine-tuning gaps without copying held-out model-forge eval prompts.

## Counts

- Accepted rows: 24
- Rejected rows: 0
- Mean quality score: 0.8897

## Skill Counts

- `benign_safety_analysis`: 6
- `checkpoint_selection`: 4
- `config_review`: 3
- `docker_disk_safety`: 2
- `eval_latency_throughput`: 4
- `git_workflow_repair`: 2
- `json_schema_repair`: 2
- `shell_safety`: 2
- `sql_edge_cases`: 3

## Coverage Warnings

- `accepted_count_below_min_target`: accepted rows 24 below configured minimum 500
- `skill_below_min_seed_examples`: eval_latency_throughput has 4 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: docker_disk_safety has 2 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: sql_edge_cases has 3 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: shell_safety has 2 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: config_review has 3 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: json_schema_repair has 2 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: git_workflow_repair has 2 accepted rows, below seed target 5
- `skill_below_min_seed_examples`: checkpoint_selection has 4 accepted rows, below seed target 5

## Provenance

Rows are currently human-seeded and heuristically judged. Future versions can
add teacher-model generation, executable verification, and HF publication using
the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
