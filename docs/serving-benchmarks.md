# Serving Benchmarks

`./forge bench serve` measures an already-running OpenAI-compatible endpoint. It
does not start vLLM, Docker, torchrun, Ray, or any training job.

Use it after a server is running and before publishing serving claims:

```bash
./forge bench serve \
  --family gemma4_26b_a4b \
  --variant base
```

Dry-run first when wiring a new endpoint or cluster:

```bash
./forge bench serve \
  --family gemma4_26b_a4b \
  --variant base \
  --dry-run
```

Generic model aliases are supported without a model-family config:

```bash
MODEL_FORGE_BASE_URL=http://127.0.0.1:8000/v1 \
./forge bench serve --model served/model-name
```

Expand a bounded serving sweep:

```bash
./forge bench sweep doctor --config configs/sweeps/dgx_spark_vllm_baseline.yaml --strict
./forge bench sweep plan \
  --config configs/sweeps/dgx_spark_vllm_baseline.yaml \
  --family gemma4_26b_a4b \
  --variant base
```

For a two-node DGX Spark setup, pass the env-backed cluster inventory after
setting private node values outside Git:

```bash
./forge bench sweep plan \
  --config configs/sweeps/dgx_spark_vllm_baseline.yaml \
  --family gemma4_26b_a4b \
  --variant base \
  --cluster-config configs/clusters/dgx_spark_x2.example.yaml
```

## Config

The default smoke config is:

```text
configs/serving/serve_bench_smoke.yaml
```

The broader reusable workload config is:

```text
configs/serving/serve_bench_core.yaml
```

The config defines:

- OpenAI-compatible endpoint defaults and env overrides
- streaming mode for time-to-first-token measurement
- deterministic sampling defaults
- a serial request mix loaded from `configs/serving/workloads/*.yaml`
- output root under `reports/generated/serve_bench/`

The MVP intentionally supports `concurrency: 1`. Concurrency sweeps belong in a
serving sweep/workload layer so results remain comparable and resource-safe.

Current workload definitions:

```text
short_chat
long_prefill
decode_heavy
structured_json
artifact_generation
reused_prefix
refusal_primary
long_context_retrieval
```

Each workload file declares its purpose, metric focus, default sampling, and one
or more requests. Benchmark manifests record both the benchmark config and the
workload definition files.

The first serving sweep config is:

```text
configs/sweeps/dgx_spark_vllm_baseline.yaml
```

It defines DGX Spark vLLM startup env cases, hypotheses, resource policy,
quality gates, and the matching `bench serve` command shape. It does not start a
server because most vLLM settings are startup-time flags and should be tested
one server configuration at a time.

## Outputs

Each run writes:

```text
requests.jsonl
summary.json
serving_card.md
manifest.json
```

Metrics include total latency, time to first token when streaming is enabled,
inter-token latency when token counts are available, output tokens/sec, decode
tokens/sec, and token counts returned by the backend.

`serving_card.md` is the human-facing report card for a run. It includes:

- model, family, variant, endpoint shape, and run manifest id
- hardware profile and recorded GPU count
- benchmark config and workload definition files
- TTFT, first-chunk, ITL, latency, output-tok/sec, decode-tok/sec, and request-throughput summaries
- per-workload metric rows
- artifact links and promotion gates
- explicit caveats for missing memory, cache-hit, truncation, quality, or behavior evidence

## Interpretation

Serving benchmarks are operational evidence, not quality evidence. A faster run
can still be a worse model or a worse serving configuration if output quality,
format following, refusal/capability behavior, or long-context behavior
regresses. Run sampled evals under the same serving config before treating a
latency or throughput result as a promotion signal.
