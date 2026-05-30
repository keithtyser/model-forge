# Dataset Review: gemma4_26b_a4b_local_ft_v1

## Summary

- Accepted rows: 49
- Rejected rows: 0
- Reviewed rows: 49 / 50
- Ready to scale generation: true

## Gates

- `has_synthetic_rows`: true
- `no_critical_flags`: true
- `sample_has_rows`: true
- `skill_coverage_ready`: true

## Coverage Gaps

- none

## Flags

- none

## Sampled Rows

### seed_eval_latency_prompt_completion_tokens
- Skills: eval_latency_throughput
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9062` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_eval_median_latency_vs_tps
- Skills: eval_latency_throughput
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8988` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_docker_cleanup_active_containers
- Skills: docker_disk_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_sql_null_count_where
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8754` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_sql_index_groupby_tradeoff
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_shell_safe_sync_backup
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.895` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_yaml_model_family_review
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9012` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_json_schema_repair_boolean_trailing
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8164` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_git_rebase_push_behind
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_benign_refusal_tradeoff_analysis
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9088` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_checkpoint_selection_not_last
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9012` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_eval_gap_dataset_design
- Skills: checkpoint_selection, benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9025` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_benign_vulnerability_report_allowed
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8624` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_benign_public_records_boundary
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8931` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_safe_scraping_rate_limits
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8694` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_safe_redirect_destructive_shell
- Skills: shell_safety, benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9137` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_eval_prefill_decode_fields
- Skills: eval_latency_throughput
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8912` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_eval_checkpoint_noise
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8875` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_docker_checkpoint_rotation
- Skills: docker_disk_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_config_variant_promptset_validation
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_git_conflict_safety_branch
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8938` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_sql_null_left_join_counts
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.89` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_json_schema_extra_text_failure
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8683` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_checkpoint_selection_tradeoff_table
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_shell_safe_find_delete
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.895` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_shell_secret_scan_before_push
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_shell_rsync_partial_resume
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_config_lora_target_validation
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8975` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_config_generation_provider_review
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_config_eval_matrix_uniqueness
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_jsonl_training_row_schema
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_json_null_vs_missing
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8754` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_json_escape_newlines
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_git_split_mixed_worktree
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_git_recover_bad_commit_message
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_git_remote_default_branch_check
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8912` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### seed_checkpoint_confidence_interval_gate
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8962` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_a5eae8f62d888737
- Skills: eval_latency_throughput
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3577`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_1f15d3e7b3f989cb
- Skills: docker_disk_safety
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4493`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_010466d7384026b4
- Skills: sql_edge_cases
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.281`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_ad38a26f935c9a09
- Skills: eval_latency_throughput
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3504`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_b985536d1c3c2742
- Skills: docker_disk_safety
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4326`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_a9f776830dea28de
- Skills: sql_edge_cases
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2893`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_d92c3b0cbec21d9a
- Skills: eval_latency_throughput
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3404`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_59dd39da166b55a6
- Skills: docker_disk_safety
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4336`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_7bba61c51ec237be
- Skills: sql_edge_cases
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2756`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_cd0cdb4e71246dc3
- Skills: eval_latency_throughput
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3504`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_29212832f5ebac64
- Skills: docker_disk_safety
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4296`
- Flags: none
- User: <redacted>
- Assistant: <redacted>

### gen_32db74f996b8fd5f
- Skills: sql_edge_cases
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2742`
- Flags: none
- User: <redacted>
- Assistant: <redacted>
