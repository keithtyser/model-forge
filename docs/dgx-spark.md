# Running model-forge on DGX Spark

Use `spark-vllm-docker` for DGX Spark. It is the right default, not vanilla host-level vLLM.

Reference:
- https://github.com/eugr/spark-vllm-docker

Why this is the better path:
- built specifically for DGX Spark
- supports single-node and multi-node Spark setups
- already handles the annoying container and cluster details
- keeps the serving side closer to what serious Spark users are actually doing

model-forge only needs one thing from the serving layer:
- an OpenAI-compatible endpoint

So the clean setup is:
1. serve the model with `spark-vllm-docker`
2. point model-forge at that endpoint
3. run the eval

## Recommended first target

Start with:
- model: `Qwen/Qwen3.5-9B`
- serving stack: `spark-vllm-docker`
- eval config: `configs/experiments/qwen35_9b_v0.yaml`

That is current enough to matter and small enough to keep the first run from turning into infrastructure bullshit.

## 1. Clone both repos

```bash
git clone https://github.com/keithtyser/model-forge.git
git clone https://github.com/eugr/spark-vllm-docker.git
```

## 2. Set up model-forge

```bash
cd model-forge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Build the DGX Spark vLLM container

If you want a one-command wrapper from this repo:

```bash
./scripts/dgx_spark_build_vllm.sh
```

Equivalent manual command:

```bash
cd ../spark-vllm-docker
./build-and-copy.sh
```

That is the single-node path. If you later run multi-node Spark cluster inference, use their cluster instructions instead of inventing your own mess.

## 4. Launch the model server on a single DGX Spark

Preferred wrapper:

```bash
./scripts/dgx_spark_serve_qwen35_9b.sh
```

Equivalent manual command:

```bash
cd ../spark-vllm-docker
./launch-cluster.sh --solo exec \
  vllm serve \
    Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768
```

If memory is tighter than expected, back off fast instead of pretending:

```bash
./launch-cluster.sh --solo exec \
  vllm serve \
    Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384
```

## 5. Smoke test the endpoint

```bash
curl http://127.0.0.1:8000/v1/models
```

If that fails, stop. Fix serving first.

## 6. Run a quick smoke eval

Preferred wrapper:

```bash
./scripts/dgx_spark_smoke_eval_qwen35_9b.sh
```

Equivalent manual command:

```bash
cd ../model-forge
source .venv/bin/activate
MODEL_FORGE_MAX_CASES=4 \
MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1 \
MODEL_FORGE_MODEL=Qwen/Qwen3.5-9B \
./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml smoke
```

## 7. Run the full base eval

Preferred wrapper:

```bash
./scripts/dgx_spark_full_eval_qwen35_9b.sh
```

Equivalent manual command:

```bash
cd ../model-forge
source .venv/bin/activate
MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1 \
MODEL_FORGE_MODEL=Qwen/Qwen3.5-9B \
MODEL_FORGE_HARDWARE_LABEL="DGX Spark" \
MODEL_FORGE_QUANT="fp16" \
MODEL_FORGE_CONTEXT_LENGTH="32768" \
./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml qwen35_9b_dgx_spark
```

## Output location

Results are written under:

```text
results/qwen35_9b_v0/base/<run-name>/
```

You will get:
- `manifest.json`
- `scores.csv`
- `responses.jsonl`
- `examples.md`

## Runtime overrides supported by model-forge

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
1. smoke run with 4 cases
2. full base eval

That catches dumb serving failures immediately and keeps the first iteration honest.
