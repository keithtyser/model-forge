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

## Config

The default smoke config is:

```text
configs/serving/serve_bench_smoke.yaml
```

The config defines:

- OpenAI-compatible endpoint defaults and env overrides
- streaming mode for time-to-first-token measurement
- deterministic sampling defaults
- a small serial request mix
- output root under `reports/generated/serve_bench/`

The MVP intentionally supports `concurrency: 1`. Concurrency sweeps belong in a
serving sweep/workload layer so results remain comparable and resource-safe.

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

## Interpretation

Serving benchmarks are operational evidence, not quality evidence. A faster run
can still be a worse model or a worse serving configuration if output quality,
format following, refusal/capability behavior, or long-context behavior
regresses. Run sampled evals under the same serving config before treating a
latency or throughput result as a promotion signal.
