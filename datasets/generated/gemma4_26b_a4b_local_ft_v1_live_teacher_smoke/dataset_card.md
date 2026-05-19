---
dataset_info:
  config_name: gemma4_26b_a4b_local_ft_v1_live_teacher_smoke
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

# gemma4_26b_a4b_local_ft_v1_live_teacher_smoke

Small live-teacher smoke for the local FT v1 dataset factory. This exercises the OpenAI-compatible provider path without starting a training run.

## Purpose

This is a Model Forge dataset-factory artifact for `gemma4_26b_a4b` /
`local_ft_v1` under the `capability_sft` objective. It targets
observed fine-tuning gaps without copying held-out model-forge eval prompts.

## Counts

- Accepted rows: 58
- Rejected rows: 3
- Mean quality score: 0.8965
- Verification passed: 61
- Verification failed: 0
- Seed-only scaffold: false
- Smoke-only scaffold: true

## Skill Counts

- `benign_safety_analysis`: 6
- `checkpoint_selection`: 5
- `config_review`: 9
- `docker_disk_safety`: 5
- `eval_latency_throughput`: 5
- `git_workflow_repair`: 5
- `json_schema_repair`: 9
- `shell_safety`: 9
- `sql_edge_cases`: 7

## Source Counts

- `human_seed`: 37
- `synthetic`: 21

## Generation Method Counts

- `eval_adjacent_generation`: 6
- `evol_instruct`: 5
- `human_seed`: 37
- `instruction_backtranslation`: 4
- `self_instruct`: 6

## Coverage Warnings

- none

## Provenance

Rows include human seeds and any configured synthetic candidates. Synthetic
rows record provider type, generator model, source seed, strategy, and prompt
template hash. Future versions can add executable verification and HF
publication using the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
