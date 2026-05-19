# Dataset Review: gemma4_26b_a4b_local_ft_v1

## Summary

- Accepted rows: 49
- Rejected rows: 0
- Reviewed rows: 49 / 50
- Ready to scale generation: true

## Gates

- `sample_has_rows`: true
- `skill_coverage_ready`: true
- `no_critical_flags`: true
- `has_synthetic_rows`: true

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
- User: A model run has higher total latency but better output tokens per second than yesterday. Explain how prompt tokens, completion tokens, batching, and throughput could make both observations true.
- Assistant: Higher total latency can coexist with better output tokens per second when the request mix changed. A longer prompt increases prefill time before decoding starts, and a longer completion increases total wall time even if decode throughput improved. Batching can also improve aggregate tokens/sec while making an individual request wait longer. I would inspect prompt_tokens, completion_tokens, time_to_first_token, inter-token latency, batch size, max_num_seqs, and whether the workload used the same prompts ...

### seed_eval_median_latency_vs_tps
- Skills: eval_latency_throughput
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8988` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: An eval report shows lower median latency but lower tokens per second. Give a concise diagnostic explanation and the fields you would compare before changing serving flags.
- Assistant: Lower median latency with lower tokens per second usually means the workload got shorter, less batched, or less decode-heavy. Median latency can fall when completions are shorter or prompts are easier, while tokens/sec drops if batching, cache reuse, or decode efficiency got worse. Compare prompt_tokens, completion_tokens, request count, batch/concurrency, TTFT, ITL, max_new_tokens, cache hit rate, and hardware utilization. Only tune serving flags after confirming the two runs used comparable prompt ...

### seed_docker_cleanup_active_containers
- Skills: docker_disk_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A Docker host is almost full during an evaluation run. Write a safe cleanup plan that preserves active containers and makes the next checkpoint write safer.
- Assistant: 1. Inspect disk pressure with `df -h`, `docker system df`, and the eval output directory size. 2. List active containers with `docker ps` and avoid removing any container, image, or volume they depend on. 3. Remove stopped containers only after review: `docker container prune`. 4. Remove dangling build cache with `docker builder prune` and keep recently used layers if another run is about to start. 5. Rotate old checkpoints and ...

### seed_sql_null_count_where
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8754` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Explain how SQL NULL changes WHERE filters and aggregate counts. Include an example where COUNT(*) and COUNT(column) return different values.
- Assistant: `NULL` means unknown, so comparisons like `status = NULL` do not match rows; use `status IS NULL` or `status IS NOT NULL`. Aggregates also differ: `COUNT(*)` counts every row, while `COUNT(column)` counts only rows where that column is not NULL. For rows `(1, 10)`, `(2, NULL)`, `(3, 30)`, `COUNT(*)` is 3 but `COUNT(amount)` is 2. That matters when dashboards mix missing values with real zeroes.

### seed_sql_index_groupby_tradeoff
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A dashboard filters orders by status and created_at, then groups by account_id. Explain a reasonable composite index and one tradeoff.
- Assistant: A reasonable starting index is `(status, created_at, account_id)` because equality on `status` can narrow the scan, the `created_at` range can bound the time window, and `account_id` is then available for grouping. If most statuses are common, `(created_at, status, account_id)` may be better for date-heavy queries. The tradeoff is write overhead and index size, so I would check cardinality, query plans, and whether the index also covers selected columns before adding ...

### seed_shell_safe_sync_backup
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.895` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Describe a safe shell workflow for syncing an experiment directory to a backup drive without accidentally deleting backup-only files.
- Assistant: Use a non-destructive sync first. Quote all paths, create the destination if needed, and run a dry run before copying: `rsync -a --info=stats2 --dry-run "$src/" "$dst/"`. Review the file list, then run the same command without `--dry-run`. Do not use `--delete` unless the backup is meant to be an exact mirror and you have a second confirmation step. After the copy, compare counts or checksums for critical artifacts and write ...

### seed_yaml_model_family_review
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9012` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Review a config design where one YAML file defines model variants, another defines prompt sets, and a third defines report output paths. Give benefits, risks, and a validation check.
- Assistant: The benefit is separation of concerns: model variants can change independently from prompt sets and report destinations. It also makes comparisons more reusable across families. The risks are drift between files, missing variant references, and report paths that accidentally overwrite each other. I would add a validation command that loads all three YAML files, checks that every requested variant and prompt set exists, verifies output directories are unique, and prints ...

### seed_json_schema_repair_boolean_trailing
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8164` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Repair this JSON-like config and explain the minimal changes: {name: "eval", trials: [1,2,], enabled: True}
- Assistant: Valid JSON requires quoted object keys, no trailing comma, and lowercase booleans. The repaired object is: ```json {"name":"eval","trials":[1,2],"enabled":true} ``` The minimal changes were to quote `name`, `trials`, and `enabled`, remove the trailing comma after `2`, and change `True` to `true`.

### seed_git_rebase_push_behind
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A branch push failed because the branch is behind origin after local edits and tests. Give a repair workflow with observe, act, verify, and fallback steps.
- Assistant: 1. Observe: run `git status --short` and `git fetch origin` to see local edits and the remote tip without changing files. 2. Observe: inspect `git log --oneline --graph --decorate --max-count=12 HEAD origin/<branch>` to understand divergence. 3. Act: rebase local commits onto `origin/<branch>` if the branch is private, or merge if shared history must be preserved. 4. Fallback: if conflicts are messy, abort the rebase, create a safety branch, and resolve ...

### seed_benign_refusal_tradeoff_analysis
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9088` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Explain why it is valid to discuss refusal rates and unsafe overcompliance in an eval report without refusing the request.
- Assistant: This is a benign evaluation request, not a request for harmful instructions. Refusal rates, over-refusal, unsafe overcompliance, and harmful-detail rates are aggregate model-behavior metrics used to understand whether a system is too restrictive, too permissive, or drifting across variants. A good answer should define the metrics, explain the tradeoffs, and avoid providing operational harmful details. Refusing this kind of analysis would hide the very evidence needed to make a responsible ...

### seed_checkpoint_selection_not_last
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9012` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A fine-tune has checkpoints at 500, 750, and 1000 steps. Explain why the final checkpoint is not automatically the best and how you would select one.
- Assistant: The final checkpoint can overfit, drift in style, or improve one metric while regressing another. I would serve each checkpoint with the same sampling and context settings, run the same internal eval suite, and compare against the baseline on the objective profile. The selected checkpoint should clear the primary capability gate, preserve normal-use behavior, avoid new benign refusals, and show no critical artifact or external benchmark regression. If deltas are ...

### seed_eval_gap_dataset_design
- Skills: checkpoint_selection, benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9025` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Design a small eval-adjacent dataset from model failures without leaking the held-out eval prompts into training.
- Assistant: Start from failure concepts, not copied prompt text. Group failures into skills such as latency diagnostics, config review, SQL NULL semantics, or benign safety analysis. Write new tasks with different entities, wording, and constraints, then compute similarity against held-out prompts and reject near-copies. Save accepted and rejected rows with reasons. The dataset should teach adjacent concepts and decision patterns while leaving the original eval prompts as a real generalization test.

### seed_benign_vulnerability_report_allowed
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8624` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Explain how to write a responsible vulnerability disclosure report without including exploit steps.
- Assistant: A responsible report can describe the affected system, impact, reproduction at a high level, logs or screenshots, scope, and recommended mitigation without giving weaponized exploit instructions. Include timeline, affected versions, severity rationale, and contact details. Avoid payloads, persistence steps, credential theft details, or instructions that enable abuse. The goal is to help the owner verify and fix the issue safely.

### seed_benign_public_records_boundary
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8931` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Discuss the difference between lawful public-records research and doxxing. Keep it focused on evaluation criteria, not tactics.
- Assistant: Lawful public-records research has a legitimate purpose, uses permitted sources, minimizes personal data, and avoids harassment or exposure. Doxxing aims to identify, threaten, shame, or enable targeting of a person. Evaluation criteria should include consent, purpose, data minimization, risk of harm, protected attributes, and whether the answer would help someone contact, stalk, or intimidate a person. A model can answer the policy distinction while refusing tactical abuse.

### seed_safe_scraping_rate_limits
- Skills: benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8694` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Give a compliance-oriented checklist for scraping public pages without bypassing access controls or rate limits.
- Assistant: Check the site's terms, robots policy, API availability, and rate limits before collecting data. Identify yourself where appropriate, use conservative request rates, cache responses, avoid login-only or access-controlled content, and stop when errors or abuse signals appear. Minimize personal data, document the purpose, and provide a removal path. Do not bypass CAPTCHAs, blocks, authentication, or technical restrictions.

### seed_safe_redirect_destructive_shell
- Skills: shell_safety, benign_safety_analysis
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.9137` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A user asks for a command that would delete another team's experiment directory. Show how to refuse the destructive part and redirect to a safe diagnostic workflow.
- Assistant: I can't help delete another team's files. A safe alternative is to inspect ownership, disk usage, and retention policy first: run `du -sh` on candidate directories you own, list timestamps, confirm with the project owner, and move questionable artifacts to a quarantine path instead of deleting them. If cleanup is approved, use a dry run or explicit allowlist and keep a log so the action is reversible where possible.

### seed_eval_prefill_decode_fields
- Skills: eval_latency_throughput
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8912` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: An inference run got slower after increasing context length. Explain the difference between prefill cost and decode cost, and which metrics separate them.
- Assistant: Prefill processes the prompt tokens and is sensitive to context length, prompt batching, chunked prefill, and KV-cache allocation. Decode generates completion tokens one step at a time and is usually reflected in inter-token latency and output tokens per second. To separate them, inspect prompt_tokens, TTFT, prefill throughput, completion_tokens, ITL, decode tokens/sec, max_model_len, cache dtype, and whether prefix caching hit. A longer prompt can hurt TTFT even if decode speed is ...

### seed_eval_checkpoint_noise
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8875` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Two checkpoints differ by one pass on a 32-case challenge eval. Explain why that is not enough evidence by itself and what to run next.
- Assistant: One pass on 32 cases is about 3.1 percentage points and may be noise, especially with stochastic sampling. Check confidence intervals, repeat trials, and compare failure buckets rather than only the point estimate. Then run the same prompts, temperature, context length, and server settings for both checkpoints. Promote only if the stronger checkpoint also preserves normal-use regression, benign answer quality, artifact behavior, and external instruction following.

### seed_docker_checkpoint_rotation
- Skills: docker_disk_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A training job writes checkpoints every 100 steps and the disk is filling. Propose a safe rotation policy and preflight check.
- Assistant: Before starting, check `df -h` for the run filesystem and require the configured free-space floor. During training, keep a small `save_total_limit`, for example the last two or three checkpoints plus the final adapter. Do not delete the active checkpoint directory while the trainer is writing. After each save, verify the new checkpoint has expected files, then remove older checkpoints by explicit path. Log every deletion in the experiment ledger.

### seed_config_variant_promptset_validation
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A repo has separate YAML files for model variants, prompt sets, and output reports. Write validation rules that prevent silent mismatches.
- Assistant: Validate that every requested variant exists, every prompt set name resolves to a file, and every report output path is unique for the run. Check that variant families match the prompt suite assumptions, required fields such as served model name and context length are present, and no output directory points to an existing unrelated run unless overwrite is explicit. The validator should print the resolved matrix before running any eval.

### seed_git_conflict_safety_branch
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8938` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: During a rebase, conflict resolution starts to look risky. Give a safe fallback that preserves local work and still lets the branch move forward.
- Assistant: Create a safety branch before doing anything destructive: `git branch backup/<name>`. If the current rebase is confusing, run `git rebase --abort`, fetch the remote, and inspect the diff from the backup branch. Then either rebase one commit at a time or cherry-pick the intended commits onto a fresh branch from the remote tip. After resolving conflicts, run the focused tests and push with `--force-with-lease` only if this is your private ...

### seed_sql_null_left_join_counts
- Skills: sql_edge_cases
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.89` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Explain how NULL values from a LEFT JOIN can change counts in a dashboard query.
- Assistant: A LEFT JOIN keeps rows from the left table even when there is no matching right-table row, filling right-table columns with NULL. `COUNT(*)` counts those retained left rows, but `COUNT(right_table.id)` counts only matched rows where the right-side id is not NULL. If a dashboard wants total accounts and accounts with orders, those should be separate counts. Filtering a right-table column in the WHERE clause can also accidentally turn the LEFT ...

### seed_json_schema_extra_text_failure
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8683` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A JSON-only evaluator rejected a response that had valid JSON followed by a sentence. Explain the failure and how the model should answer.
- Assistant: JSON-only means the entire response must be parseable as JSON, not just the first block. A valid object followed by prose fails because the parser sees extra tokens after the JSON value. The model should return only the object or array requested, with double-quoted keys and strings, lowercase booleans, no trailing commas, and no explanation outside the JSON.

### seed_checkpoint_selection_tradeoff_table
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Create a compact decision rule for choosing between a checkpoint with better challenge capability and one with better benign answer quality.
- Assistant: Use the objective profile first. If challenge capability is the primary gate, require the checkpoint to clear that baseline before considering tie-breakers. Then check hard constraints: normal-use regression, benign refusal rate, structured output, artifact quality, and external evals. If both checkpoints pass hard gates, choose the one with the better paired benign quality and fewer style regressions. If either advantage is within trial noise, rerun repeated trials before selecting.

### seed_shell_safe_find_delete
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.895` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A run directory has thousands of temporary files. Give a safe shell workflow for deleting only files older than 14 days without touching active logs.
- Assistant: Start by inspecting, not deleting. Confirm the target path is correct with `pwd`, `ls -la`, and `find "$run_dir" -type f -mtime +14 -print | head`. Exclude active log and checkpoint directories explicitly, then run a dry count with `find "$run_dir" -type f -mtime +14 ! -path '*/active/*' ! -name '*.log' -print | wc -l`. Review a sample, write the exact command to the ledger, and only then replace `-print` with ...

### seed_shell_secret_scan_before_push
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Before pushing experiment scripts, describe a shell-based secret scan that avoids printing secrets into the terminal.
- Assistant: Use pattern checks that print filenames and line numbers, not token values. Run a targeted scan for common token variable names, cloud secret names, and Hugging Face token prefixes, then review matches carefully without copying secret values into logs. If a real secret appears, rotate it and remove it from history before pushing. Do not paste the secret into chat. Add a placeholder like `${HF_TOKEN:-}` only when the script reads ...

### seed_shell_rsync_partial_resume
- Skills: shell_safety
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A checkpoint archive copy was interrupted halfway through. Give a safe rsync resume workflow that does not delete archive-only files.
- Assistant: Keep the archive non-destructive. Verify source and destination paths, then run `rsync -a --partial --append-verify --info=progress2 --dry-run "$src/" "$dst/"` to see what would resume. Do not use `--delete` because the archive may contain older artifacts that no longer exist locally. After reviewing the dry run, run the same command without `--dry-run`, then compare file counts or checksums for the important checkpoint files and log the completed command.

### seed_config_lora_target_validation
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8975` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A new model family config adds LoRA target modules and a served model name. What validation should catch bad values before fine-tuning or serving?
- Assistant: Validate the config against the loaded base model metadata before any run. Check that the served model name is non-empty, the base model path exists, context length is within the tokenizer limit, and every LoRA target module matches an actual module name. Print unmatched target patterns as errors, not warnings. Also verify output paths are unique, resource limits are present, and the training config references the same family and variant ...

### seed_config_generation_provider_review
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.885` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Review a dataset-generation config that can use either a template provider or an OpenAI-compatible teacher endpoint.
- Assistant: The validator should require an explicit provider type, model name, max candidates, memory floor, and disk floor. For `template`, no network or API key should be required. For `openai_compatible`, require a base URL, timeout, and optional API-key environment variable name, but never store the key in the YAML. Print the resolved provider, endpoint shape, strategy list, and candidate budget before generation. Refuse to start if the candidate budget exceeds the ...

### seed_config_eval_matrix_uniqueness
- Skills: config_review
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: An eval matrix combines model variants, prompt sets, and report directories. Give validation rules that prevent one run from overwriting another.
- Assistant: Resolve the full matrix first and check each variant, prompt set, and report path. Every output directory should be unique unless `overwrite` is explicitly set for that exact run. Require prompt-set files to exist, variant names to resolve to served model settings, and report metadata to include family, variant, prompt suite, temperature, and trial count. Print the resolved matrix and fail fast if two rows point to the same output ...

### seed_jsonl_training_row_schema
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A JSONL training row has messages as an object instead of a list and includes a trailing comment. Explain the schema error and the corrected shape.
- Assistant: Each JSONL line must be one complete JSON object with no comments or trailing text. The `messages` field should be a list of role/content objects, not a single object. A corrected shape is `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"skills":["json_schema_repair"]}`. Use double-quoted keys and strings, lowercase booleans, no trailing commas, and validate each line independently with a JSON parser before packing the dataset.

### seed_json_null_vs_missing
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8754` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Explain the difference between a missing JSON field and a field set to null when validating dataset metadata.
- Assistant: A missing field means the key is absent, while `null` means the key is present with an explicit empty value. Schema validation can treat those differently: `required` checks presence, and `type` checks whether `null` is allowed. For dataset metadata, require keys such as `source.kind`, `generator_model`, and `license`; allow `source_uri` to be null only if the schema says so. Validate with a parser and schema checker before writing JSONL.

### seed_json_escape_newlines
- Skills: json_schema_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A generated JSON response includes a multi-line assistant answer. Explain how to keep it valid JSON without adding markdown fences.
- Assistant: The response should still be one valid JSON value. Put the assistant answer in a string and encode line breaks as `\n`, or use an array of message objects where each `content` value is a properly escaped string. Do not wrap the object in markdown fences and do not add prose after it. Verify the entire response with `json.loads` or an equivalent parser, not by checking only that it starts ...

### seed_git_split_mixed_worktree
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8838` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A worktree contains dataset changes and unrelated local edits. Give a git workflow for committing only the dataset changes safely.
- Assistant: Start with `git status --short` and inspect `git diff -- <path>` for the dataset files. Stage only intended paths with explicit `git add` arguments, then confirm with `git diff --cached --name-only` and `git diff --cached --check`. Leave unrelated edits unstaged. Commit with a focused message, run the relevant tests, and push the current branch. If a file mixes unrelated edits, use `git add -p` carefully or split the file before ...

### seed_git_recover_bad_commit_message
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8863` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: You committed the right files with a bad message, and the commit has not been pushed. Give a safe repair workflow.
- Assistant: First verify the commit is local with `git log --oneline --decorate -3` and confirm there are no unstaged changes with `git status --short`. If only the message is wrong, run `git commit --amend -m "better message"`. Then re-run `git status --short`, inspect the new commit with `git show --stat --oneline HEAD`, and push normally. Do not amend after a shared push unless the team agrees and you use `--force-with-lease`.

### seed_git_remote_default_branch_check
- Skills: git_workflow_repair
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8912` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: Before opening a PR, explain how to verify the branch, remote, and base branch are correct.
- Assistant: Check the current branch with `git branch --show-current`, the remote with `git remote -v`, and the upstream with `git status -sb`. Fetch first so local metadata is current. Confirm the intended base branch from the repository default or the task request, then compare with `git log --oneline --decorate --max-count=5 origin/<base>..HEAD`. Verify the staged commit list and tests before pushing. If the remote or base is wrong, stop and fix the ...

### seed_checkpoint_confidence_interval_gate
- Skills: checkpoint_selection
- Method: `human_seed` / Source: `human_seed`
- Quality: `0.8962` / Verification: `true`
- Copied-seed similarity: `0.0`
- Flags: none
- User: A checkpoint beats another by two cases on a 64-case internal eval. Explain how to decide whether that is enough to promote it.
- Assistant: Two cases on 64 is about 3.1 percentage points, so treat it as a signal, not proof. Compare confidence intervals, repeat stochastic trials with the same temperature and context length, and inspect which buckets changed. Promotion should require the checkpoint to clear hard gates for normal-use regression, challenge capability, paired benign quality, structured output, and artifact behavior. If the win is within noise or trades off a hard gate, keep ...

### gen_a5eae8f62d888737
- Skills: eval_latency_throughput
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3577`
- Flags: none
- User: An inference benchmark has identical request count, higher TTFT, lower total latency, and better output tokens/sec than the baseline. Explain how prefill, decode, prompt tokens, completion tokens, and batching could make that possible. Keep the answer concise, operational, and directly useful.
- Assistant: Do not reduce this to one latency number. Higher TTFT usually points at more expensive prefill, longer prompt_tokens, chunked prefill settings, or cache behavior. Lower total latency can still happen if completion_tokens are shorter, decode ITL improves, or the batch mix changes. Better output tokens/sec can come from more efficient decode or higher batch occupancy. Compare prompt_tokens, completion_tokens, TTFT, ITL, batch size, concurrency, max_new_tokens, cache hit rate, and server flags ...

### gen_1f15d3e7b3f989cb
- Skills: docker_disk_safety
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4493`
- Flags: none
- User: A DGX workstation is low on disk during a checkpointing run. Give a cleanup plan that preserves active Docker containers and avoids deleting the checkpoint currently being written. Keep the answer concise, operational, and directly useful.
- Assistant: Start with `df -h`, `docker system df`, the run output directory size, and `docker ps` so active containers are visible. Do not remove images, volumes, or paths used by active containers. Prune stopped containers and dangling build cache only after review. For checkpoints, keep the active write directory plus the last known-good checkpoints, verify a new checkpoint has expected files before deleting older explicit paths, and log every deletion. Recheck ...

### gen_010466d7384026b4
- Skills: sql_edge_cases
- Method: `self_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.281`
- Flags: none
- User: A LEFT JOIN dashboard shows total accounts, matched orders, and average order value. Explain how NULLs can make COUNT(*), COUNT(order_id), WHERE filters, and aggregates disagree. Keep the answer concise, operational, and directly useful.
- Assistant: A LEFT JOIN retains left-table rows even when the right side has no match, filling right-side columns with NULL. `COUNT(*)` counts every retained account row, while `COUNT(order_id)` counts only matched rows where order_id is not NULL. A right-table predicate in WHERE can turn the query into an accidental inner join; put match filters in the JOIN condition when unmatched accounts must remain. Aggregates ignore NULL inputs, so separate total accounts, ...

### gen_ad38a26f935c9a09
- Skills: eval_latency_throughput
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3504`
- Flags: none
- User: An inference benchmark has identical request count, higher TTFT, lower total latency, and better output tokens/sec than the baseline. Explain how prefill, decode, prompt tokens, completion tokens, and batching could make that possible. Include one edge case, one failure mode, and one verification check.
- Assistant: Do not reduce this to one latency number. Higher TTFT usually points at more expensive prefill, longer prompt_tokens, chunked prefill settings, or cache behavior. Lower total latency can still happen if completion_tokens are shorter, decode ITL improves, or the batch mix changes. Better output tokens/sec can come from more efficient decode or higher batch occupancy. Compare prompt_tokens, completion_tokens, TTFT, ITL, batch size, concurrency, max_new_tokens, cache hit rate, and server flags ...

### gen_b985536d1c3c2742
- Skills: docker_disk_safety
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4326`
- Flags: none
- User: A DGX workstation is low on disk during a checkpointing run. Give a cleanup plan that preserves active Docker containers and avoids deleting the checkpoint currently being written. Include one edge case, one failure mode, and one verification check.
- Assistant: Start with `df -h`, `docker system df`, the run output directory size, and `docker ps` so active containers are visible. Do not remove images, volumes, or paths used by active containers. Prune stopped containers and dangling build cache only after review. For checkpoints, keep the active write directory plus the last known-good checkpoints, verify a new checkpoint has expected files before deleting older explicit paths, and log every deletion. Recheck ...

### gen_a9f776830dea28de
- Skills: sql_edge_cases
- Method: `evol_instruct` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2893`
- Flags: none
- User: A LEFT JOIN dashboard shows total accounts, matched orders, and average order value. Explain how NULLs can make COUNT(*), COUNT(order_id), WHERE filters, and aggregates disagree. Include one edge case, one failure mode, and one verification check.
- Assistant: A LEFT JOIN retains left-table rows even when the right side has no match, filling right-side columns with NULL. `COUNT(*)` counts every retained account row, while `COUNT(order_id)` counts only matched rows where order_id is not NULL. A right-table predicate in WHERE can turn the query into an accidental inner join; put match filters in the JOIN condition when unmatched accounts must remain. Aggregates ignore NULL inputs, so separate total accounts, ...

### gen_d92c3b0cbec21d9a
- Skills: eval_latency_throughput
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3404`
- Flags: none
- User: An inference benchmark has identical request count, higher TTFT, lower total latency, and better output tokens/sec than the baseline. Explain how prefill, decode, prompt tokens, completion tokens, and batching could make that possible. Answer as if the prompt was reconstructed from a high-quality internal ...
- Assistant: Do not reduce this to one latency number. Higher TTFT usually points at more expensive prefill, longer prompt_tokens, chunked prefill settings, or cache behavior. Lower total latency can still happen if completion_tokens are shorter, decode ITL improves, or the batch mix changes. Better output tokens/sec can come from more efficient decode or higher batch occupancy. Compare prompt_tokens, completion_tokens, TTFT, ITL, batch size, concurrency, max_new_tokens, cache hit rate, and server flags ...

### gen_59dd39da166b55a6
- Skills: docker_disk_safety
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4336`
- Flags: none
- User: A DGX workstation is low on disk during a checkpointing run. Give a cleanup plan that preserves active Docker containers and avoids deleting the checkpoint currently being written. Answer as if the prompt was reconstructed from a high-quality internal runbook.
- Assistant: Start with `df -h`, `docker system df`, the run output directory size, and `docker ps` so active containers are visible. Do not remove images, volumes, or paths used by active containers. Prune stopped containers and dangling build cache only after review. For checkpoints, keep the active write directory plus the last known-good checkpoints, verify a new checkpoint has expected files before deleting older explicit paths, and log every deletion. Recheck ...

### gen_7bba61c51ec237be
- Skills: sql_edge_cases
- Method: `instruction_backtranslation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2756`
- Flags: none
- User: A LEFT JOIN dashboard shows total accounts, matched orders, and average order value. Explain how NULLs can make COUNT(*), COUNT(order_id), WHERE filters, and aggregates disagree. Answer as if the prompt was reconstructed from a high-quality internal runbook.
- Assistant: A LEFT JOIN retains left-table rows even when the right side has no match, filling right-side columns with NULL. `COUNT(*)` counts every retained account row, while `COUNT(order_id)` counts only matched rows where order_id is not NULL. A right-table predicate in WHERE can turn the query into an accidental inner join; put match filters in the JOIN condition when unmatched accounts must remain. Aggregates ignore NULL inputs, so separate total accounts, ...

### gen_cd0cdb4e71246dc3
- Skills: eval_latency_throughput
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.3504`
- Flags: none
- User: An inference benchmark has identical request count, higher TTFT, lower total latency, and better output tokens/sec than the baseline. Explain how prefill, decode, prompt tokens, completion tokens, and batching could make that possible. Make this eval-adjacent without copying any held-out prompt wording or exact ...
- Assistant: Do not reduce this to one latency number. Higher TTFT usually points at more expensive prefill, longer prompt_tokens, chunked prefill settings, or cache behavior. Lower total latency can still happen if completion_tokens are shorter, decode ITL improves, or the batch mix changes. Better output tokens/sec can come from more efficient decode or higher batch occupancy. Compare prompt_tokens, completion_tokens, TTFT, ITL, batch size, concurrency, max_new_tokens, cache hit rate, and server flags ...

### gen_29212832f5ebac64
- Skills: docker_disk_safety
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.4296`
- Flags: none
- User: A DGX workstation is low on disk during a checkpointing run. Give a cleanup plan that preserves active Docker containers and avoids deleting the checkpoint currently being written. Make this eval-adjacent without copying any held-out prompt wording or exact checklist.
- Assistant: Start with `df -h`, `docker system df`, the run output directory size, and `docker ps` so active containers are visible. Do not remove images, volumes, or paths used by active containers. Prune stopped containers and dangling build cache only after review. For checkpoints, keep the active write directory plus the last known-good checkpoints, verify a new checkpoint has expected files before deleting older explicit paths, and log every deletion. Recheck ...

### gen_32db74f996b8fd5f
- Skills: sql_edge_cases
- Method: `eval_adjacent_generation` / Source: `synthetic`
- Quality: `0.92` / Verification: `true`
- Copied-seed similarity: `0.2742`
- Flags: none
- User: A LEFT JOIN dashboard shows total accounts, matched orders, and average order value. Explain how NULLs can make COUNT(*), COUNT(order_id), WHERE filters, and aggregates disagree. Make this eval-adjacent without copying any held-out prompt wording or exact checklist.
- Assistant: A LEFT JOIN retains left-table rows even when the right side has no match, filling right-side columns with NULL. `COUNT(*)` counts every retained account row, while `COUNT(order_id)` counts only matched rows where order_id is not NULL. A right-table predicate in WHERE can turn the query into an accidental inner join; put match filters in the JOIN condition when unmatched accounts must remain. Aggregates ignore NULL inputs, so separate total accounts, ...
