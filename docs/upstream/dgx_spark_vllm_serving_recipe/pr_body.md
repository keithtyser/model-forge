## Summary

- Add a DGX Spark / GB10 serving benchmark recipe to the benchmarking docs.
- Link the recipe from the benchmarking overview and docs nav.
- Include a compact report format for comparing BF16 and quantized serving runs on the same node.

## Motivation

DGX Spark / GB10 users need a small, reproducible starting point for validating
OpenAI-compatible vLLM serving configs without treating one-off token/sec numbers
as general claims. This recipe documents a conservative single-node workflow:
start one server, keep resource headroom, run `vllm bench serve`, and report
latency/throughput plus request/error counts.

The example table is grounded in local Model Forge evidence on a repeatable
27-request workload:

- BF16: median TTFT 0.128 s, median ITL 0.043 s, median output tok/s 22.76
- Quantized: median TTFT 0.126 s, median ITL 0.039 s, median output tok/s 25.10

The doc explicitly labels these as workload-specific example numbers, not
guaranteed DGX Spark performance.

## Testing

Docs-only change. Locally checked that:

- the new page is linked from `docs/benchmarking/README.md`
- the new page is listed in `docs/.nav.yml`
- the patch applies against vLLM commit `3fd9d2d35714e80b4cb3fcd3c408a0398fa2525f`

## Evidence

Model Forge source evidence:

- `reports/generated/serve_bench/gemma4_base_bf16_core_r3_20260530/summary.json`
- `reports/generated/serve_bench/gemma4_base_bf16_core_r3_20260530/serving_card.md`
- `reports/generated/serve_bench/gemma4_base_nvfp4_mlp_fullram_smoke16_core_r3_metrics_20260530/summary.json`
- `reports/generated/serve_bench/gemma4_base_nvfp4_mlp_fullram_smoke16_core_r3_metrics_20260530/serving_card.md`
