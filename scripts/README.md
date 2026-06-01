# Scripts

Prefer the `./forge` CLI for normal workflows. Scripts are compatibility
wrappers, operational helpers, or hardware-specific launchers.

## Groups

- `setup.sh`, `download_gemma4_models.sh`: setup and model download helpers.
- `dgx_spark_*`, `gemma4_dgx.sh`, `model_forge_dgx.py`,
  `serve_vllm_dgx_spark.sh`: DGX Spark serving/eval convenience wrappers.
- `serve_teacher_vllm_dgx_spark.sh`: guarded one-server launcher for the small
  local teacher used by dataset generation.
- `run_finetune_spark_container.sh`: guarded CUDA container launcher for full
  fine-tuning on Spark.
- `run_merge_peft_container.sh`: guarded container launcher for PEFT/full
  checkpoint merges on hosts that do not have the local Python ML stack, or
  when the host environment is older than the target model architecture.
- `quantization/gemma4_moe_nvfp4.py`: lower-level ModelOpt helper used by
  `./forge quantize export` for Gemma4 full-MoE NVFP4 checkpoint creation.
- `merge_peft_adapter.py`: PEFT adapter merge helper when live LoRA serving is
  unsupported or inconvenient. Direct merge mode supports `--lora-scale` for
  bounded ablation-strength sweeps from an existing adapter.
- `scale_lora_adapter.py`: creates small scaled copies of a LoRA adapter for
  live-serving ablation sweeps without writing a full checkpoint.
- `publish_hf_artifact.py`: Hugging Face upload helper for completed models,
  datasets, or durable report bundles.
- `rescore_internal_eval.py`: re-score an existing internal eval run from its
  saved `responses.jsonl` after rubric/check changes, without querying the
  model again.
- `model_forge_watchdog.py`: optional host-side emergency brake for long jobs.
- `mock_openai_server.py`: local test server for OpenAI-compatible eval paths.

## Rules

- Do not bypass resource guardrails for training or large eval jobs.
- Keep scripts portable: derive repo paths from the script location and use
  environment variables for machine-specific paths.
- Never commit literal tokens. Read `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or
  provider API keys from the environment at runtime.
- Keep one large model server or training process active at a time.
- If a script becomes the recommended user path, expose it through `./forge` or
  document why it remains a lower-level helper.
- When changing internal eval rubrics, use `rescore_internal_eval.py` on
  affected saved runs before interpreting old candidates as regressions.
