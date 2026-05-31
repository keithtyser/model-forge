# DGX Spark vLLM Serving Recipe Upstream Draft

Target: `https://github.com/vllm-project/vllm`

Candidate id: `dgx_spark_vllm_serving_recipe`

Purpose: open a small vLLM documentation PR that adds a DGX Spark / GB10
serving benchmark recipe grounded in Model Forge BF16 and NVFP4 serving
benchmark evidence.

## Files

- `vllm_dgx_spark_gb10_serving_recipe.patch`: patch against vLLM commit
  `3fd9d2d35714e80b4cb3fcd3c408a0398fa2525f`
- `pr_body.md`: draft PR body for the upstream contribution

## Evidence

- `reports/generated/serve_bench/gemma4_base_bf16_core_r3_20260530/summary.json`
- `reports/generated/serve_bench/gemma4_base_bf16_core_r3_20260530/serving_card.md`
- `reports/generated/serve_bench/gemma4_base_nvfp4_mlp_fullram_smoke16_core_r3_metrics_20260530/summary.json`
- `reports/generated/serve_bench/gemma4_base_nvfp4_mlp_fullram_smoke16_core_r3_metrics_20260530/serving_card.md`

## Handoff

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3fd9d2d35714e80b4cb3fcd3c408a0398fa2525f
git checkout -b docs/dgx-spark-gb10-serving-recipe
git apply /path/to/model-forge/docs/upstream/dgx_spark_vllm_serving_recipe/vllm_dgx_spark_gb10_serving_recipe.patch
```

After opening the PR, update `configs/upstream/pr_candidates.yaml`:

```yaml
status: opened
external_pr_url: https://github.com/vllm-project/vllm/pull/<number>
```

Then run:

```bash
./forge upstream verify-pr \
  --config configs/upstream/pr_candidates.yaml \
  --candidate dgx_spark_vllm_serving_recipe \
  --write-report
```

Only mark `MF-0808` complete after non-offline verification returns
`verified=true`.
