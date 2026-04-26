# Running model-forge on DGX Spark

This repo is set up to evaluate against any OpenAI-compatible local endpoint. For DGX Spark, the cleanest first path is:

1. serve the model with vLLM
2. point model-forge at that endpoint
3. run the base eval

## Recommended first target

Start with:
- model: `Qwen/Qwen3.5-9B`
- runner: vLLM
- eval config: `configs/experiments/qwen35_9b_v0.yaml`

That is the right first move because it is current, strong enough to be interesting, and small enough to iterate on without turning the first run into infrastructure drama.

## 1. Create the environment

```bash
cd model-forge
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install vllm
```

## 2. Start the OpenAI-compatible server

```bash
./scripts/serve_vllm_dgx_spark.sh Qwen/Qwen3.5-9B
```

Useful overrides:

```bash
MAX_MODEL_LEN=16384 GPU_MEMORY_UTILIZATION=0.90 ./scripts/serve_vllm_dgx_spark.sh Qwen/Qwen3.5-9B
```

If you want to use a different served model name, set it at eval time with `MODEL_FORGE_MODEL`.

## 3. Smoke test the endpoint

```bash
curl http://127.0.0.1:8000/v1/models
```

If that fails, do not run the eval yet. Fix serving first.

## 4. Run the evaluation

Full run:

```bash
source .venv/bin/activate
./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml qwen35_9b_dgx_spark
```

Quick smoke run:

```bash
source .venv/bin/activate
MODEL_FORGE_MAX_CASES=4 ./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml smoke
```

## 5. Optional metadata overrides

These values are recorded in the manifest if you set them:

```bash
export MODEL_FORGE_HARDWARE_LABEL="DGX Spark"
export MODEL_FORGE_QUANT="fp16"
export MODEL_FORGE_CONTEXT_LENGTH="32768"
```

## Output location

Results are written under the configured output directory with the run suffix appended:

```text
results/qwen35_9b_v0/base/<run-name>/
```

You will get:
- `manifest.json`
- `scores.csv`
- `responses.jsonl`
- `examples.md`

## Environment overrides

The harness supports these runtime overrides:
- `MODEL_FORGE_BASE_URL`
- `MODEL_FORGE_MODEL`
- `MODEL_FORGE_API_KEY`
- `MODEL_FORGE_API_KEY_ENV`
- `MODEL_FORGE_TEMPERATURE`
- `MODEL_FORGE_MAX_TOKENS`
- `MODEL_FORGE_TIMEOUT_SECONDS`
- `MODEL_FORGE_EXTRA_BODY`
- `MODEL_FORGE_HARDWARE_LABEL`
- `MODEL_FORGE_QUANT`
- `MODEL_FORGE_CONTEXT_LENGTH`
- `MODEL_FORGE_MAX_CASES`

## Strong recommendation

Do the first run in two stages:

1. smoke run with `MODEL_FORGE_MAX_CASES=4`
2. full base eval

That catches dumb serving issues fast and keeps the first iteration tight.
