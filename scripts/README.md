# Scripts

Prefer the `./forge` CLI for normal workflows. Scripts are compatibility
wrappers, operational helpers, or hardware-specific launchers.

## Groups

- `setup.sh`, `download_gemma4_models.sh`: setup and model download helpers.
- `dgx_spark_*`, `gemma4_dgx.sh`, `model_forge_dgx.py`,
  `serve_vllm_dgx_spark.sh`: DGX Spark serving/eval convenience wrappers.
- `run_finetune_spark_container.sh`: guarded CUDA container launcher for full
  fine-tuning on Spark.
- `merge_peft_adapter.py`: PEFT adapter merge helper when live LoRA serving is
  unsupported or inconvenient.
- `publish_hf_artifact.py`: Hugging Face upload helper for completed models,
  datasets, or durable report bundles.
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
