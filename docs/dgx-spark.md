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

That is current enough to matter and small enough to keep the first run focused on serving and evaluation rather than infrastructure debugging.

## Gemma 4 26B-A4B target

Gemma 4 26B-A4B is a good second target for DGX Spark because it is a MoE model with roughly 26B total parameters and roughly 4B active parameters per token. Use it to compare:

- base: `google/gemma-4-26B-A4B-it`
- fine-tune: `Jackrong/Gemopus-4-26B-A4B-it`
- ablation: `huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated`

For fair benchmark comparisons, start with the HF safetensors/BF16 models on the same vLLM stack. Quantized GGUF or NVFP4 runs are useful for throughput testing, but they add a quantization variable and should be compared separately.

The simplest workflow is the convenience wrapper:

```bash
./scripts/gemma4_dgx.sh serve base
./scripts/gemma4_dgx.sh smoke base
./scripts/gemma4_dgx.sh full base

./scripts/gemma4_dgx.sh serve ft
./scripts/gemma4_dgx.sh smoke ft
./scripts/gemma4_dgx.sh full ft

./scripts/gemma4_dgx.sh serve abli
./scripts/gemma4_dgx.sh smoke abli
./scripts/gemma4_dgx.sh full abli

./scripts/gemma4_dgx.sh compare
```

Run `serve` in one terminal. Run `smoke`, `full`, or `artifact` in another terminal while that server is running. Stop the server with `Ctrl-C` before switching variants.

Serve the base model:

```bash
./scripts/dgx_spark_serve_gemma4_26b_a4b.sh
```

The Gemma wrapper defaults to `GPU_MEMORY_UTILIZATION=0.85` because DGX Spark usually exposes less free CUDA memory than the full unified-memory total at server startup. If startup still fails during memory reservation, retry with `GPU_MEMORY_UTILIZATION=0.80` or reduce `MAX_MODEL_LEN` to `16384`.

When serving a local model path, the wrapper automatically mounts `MODEL_FORGE_MODELS_DIR` into the vLLM container. The default is `$HOME/models`.

Serve Jackrong's fine-tune:

```bash
MODEL_FORGE_MODEL=Jackrong/Gemopus-4-26B-A4B-it \
MODEL_FORGE_SERVED_MODEL_NAME=Jackrong/Gemopus-4-26B-A4B-it \
./scripts/dgx_spark_serve_gemma4_26b_a4b.sh
```

Serve Huihui's ablated model:

```bash
MODEL_FORGE_MODEL=huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated \
MODEL_FORGE_SERVED_MODEL_NAME=huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated \
./scripts/dgx_spark_serve_gemma4_26b_a4b.sh
```

Then run the matching eval wrapper in another terminal:

```bash
./scripts/dgx_spark_smoke_eval_gemma4_26b_a4b.sh
./scripts/dgx_spark_full_eval_gemma4_26b_a4b.sh
./scripts/dgx_spark_artifact_eval_gemma4_26b_a4b.sh
```

For non-base variants, set the same model alias and variant name used by the server:

```bash
MODEL_FORGE_VARIANT=ft \
MODEL_FORGE_MODEL=Jackrong/Gemopus-4-26B-A4B-it \
./scripts/dgx_spark_full_eval_gemma4_26b_a4b.sh

MODEL_FORGE_VARIANT=abli \
MODEL_FORGE_MODEL=huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated \
./scripts/dgx_spark_full_eval_gemma4_26b_a4b.sh
```

Use a distinct run name when running variants back-to-back through the lower-level runner:

```bash
MODEL_FORGE_VARIANT=ft \
MODEL_FORGE_MODEL=Jackrong/Gemopus-4-26B-A4B-it \
./scripts/run_dgx_spark_eval.sh configs/experiments/gemma4_26b_a4b_v0.yaml gemma4_26b_a4b_ft
```

Compare completed Gemma 4 runs:

```bash
model-forge-compare \
  --base results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_base_dgx_spark \
  --ft results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_ft_dgx_spark \
  --abli results/gemma4_26b_a4b_v0/base/gemma4_26b_a4b_abli_dgx_spark \
  --output-dir reports/generated/gemma4_26b_a4b_comparison
```

Expected result shape:

- Jackrong should show stronger workflow, structure, reasoning-style, or artifact quality without critical regressions.
- Huihui should show lower benign-refusal rate while keeping normal-use pass rate and workflow success close to base.
- Any unsafe-overcompliance increase should be treated as a regression, even if false refusals improve.

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

## 8. Run the artifact workbench eval

The artifact suite is a longer, human-inspection-oriented run inspired by practical base-vs-fine-tune workbench evals. It asks the model to generate web pages, Canvas/WebGL artifacts, and utility scripts, then saves extracted files under `artifacts/` with an `artifact_report.html` index.

Quick one-case check:

```bash
MODEL_FORGE_MAX_CASES=1 ./scripts/dgx_spark_artifact_eval_qwen35_9b.sh
```

Full artifact run:

```bash
./scripts/dgx_spark_artifact_eval_qwen35_9b.sh
```

On DGX Spark with Qwen/Qwen3.5-9B BF16, expect long runtime for this suite. A single HTML artifact can take several minutes at roughly 10-12 generated tokens/sec.

For sampled long-form/artifact comparisons, run repeated trials:

```bash
MODEL_FORGE_TRIALS=3 ./scripts/dgx_spark_artifact_eval_qwen35_9b.sh
```

For non-base variants, set `MODEL_FORGE_VARIANT` and a distinct output suffix:

```bash
MODEL_FORGE_VARIANT=ft \
MODEL_FORGE_MODEL=/path/to/fine-tuned-model \
./scripts/run_dgx_spark_eval.sh configs/experiments/qwen35_9b_v0.yaml qwen35_9b_dgx_spark_ft
```

Compare finished runs:

```bash
model-forge-compare \
  --base results/qwen35_9b_v0/base/qwen35_9b_dgx_spark \
  --ft results/qwen35_9b_v0/base/qwen35_9b_dgx_spark_ft \
  --output-dir reports/generated/qwen35_9b_comparison
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

Artifact runs also write:
- `artifact_report.html`
- `artifacts/<case>.html`
- `artifacts/<case>.py`
- `artifact_validations.json`

Basic artifact validation runs with the standard Python dependencies. Browser validation for HTML/Canvas/WebGL artifacts is enabled when Playwright and browser binaries are installed:

```bash
pip install -e ".[artifacts]"
python -m playwright install chromium
```

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

That catches serving failures immediately and keeps the first iteration honest.
