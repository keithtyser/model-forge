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

- Accepted rows: 12
- Rejected rows: 0
- Mean quality score: 0.8943

## Skill Counts

- `benign_safety_analysis`: 2
- `checkpoint_selection`: 2
- `config_review`: 1
- `docker_disk_safety`: 1
- `eval_latency_throughput`: 2
- `git_workflow_repair`: 1
- `json_schema_repair`: 1
- `shell_safety`: 1
- `sql_edge_cases`: 2

## Provenance

Rows are currently human-seeded and heuristically judged. Future versions can
add teacher-model generation, executable verification, and HF publication using
the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
