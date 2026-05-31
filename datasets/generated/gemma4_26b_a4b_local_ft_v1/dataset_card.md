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

- Accepted rows: 49
- Rejected rows: 0
- Mean quality score: 0.8966
- Verification passed: 49
- Verification failed: 0
- Seed-only scaffold: false
- Smoke-only scaffold: true
- Pack stage: smoke_pack
- Pack stage ready: true

## Skill Counts

- `benign_safety_analysis`: 6
- `checkpoint_selection`: 5
- `config_review`: 5
- `docker_disk_safety`: 6
- `eval_latency_throughput`: 7
- `git_workflow_repair`: 5
- `json_schema_repair`: 5
- `shell_safety`: 5
- `sql_edge_cases`: 7

## Source Counts

- `human_seed`: 37
- `synthetic`: 12

## Generation Method Counts

- `eval_adjacent_generation`: 3
- `evol_instruct`: 3
- `human_seed`: 37
- `instruction_backtranslation`: 3
- `self_instruct`: 3

## Coverage Warnings

- `accepted_count_below_min_target`: accepted rows 49 below configured minimum 500

## Provenance

Rows include human seeds and any configured synthetic candidates. Synthetic
rows record provider type, generator model, source seed, strategy, and prompt
template hash. Future versions can add executable verification and HF
publication using the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
