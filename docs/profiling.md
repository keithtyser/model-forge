# Profiling

Model Forge profiling commands are planning-first. They generate profiler
commands and expected output paths, but they do not start model servers or run
profilers unless an operator explicitly runs the generated command script.

## Nsight

Validate the default Nsight profile config:

```bash
./forge profile nsight doctor \
  --config configs/profiling/nsight_serving_smoke.yaml
```

Write a profile plan:

```bash
./forge profile nsight plan \
  --config configs/profiling/nsight_serving_smoke.yaml \
  --run-id gemma4_base_serving_smoke \
  --write-plan
```

Override the profiled command when you already have a matching server running:

```bash
./forge profile nsight plan \
  --config configs/profiling/nsight_serving_smoke.yaml \
  --command './forge bench serve --config configs/serving/serve_bench_core.yaml --family gemma4_26b_a4b --variant base' \
  --write-plan
```

The generated plan writes:

- `nsight_profile_plan.json`
- `profile_commands.sh`
- expected `profile_nsys.nsys-rep`
- expected `profile_ncu.ncu-rep`

Summarize the profile artifacts after running the generated command script:

```bash
./forge profile nsight summarize \
  --plan reports/generated/profiles/nsight/gemma4_base_serving_smoke/nsight_profile_plan.json \
  --write-summary
```

The summary writes:

- `profile_summary.json`
- `profile_summary.md`

The default config requires an already-running OpenAI-compatible endpoint,
uses the small serving benchmark as the target command, and records a
one-profile-at-a-time resource policy.

## Rules

- Run one profiler at a time.
- Start with the smoke workload before profiling long or high-concurrency runs.
- Keep profiler outputs under `reports/generated/profiles/` or another ignored
  runtime directory.
- Treat `profile_summary.md` as an artifact inventory and triage aid. It does
  not replace Nsight's own kernel timeline, statistics, or exported reports.
- Pair profile traces with the serving card, eval evidence, and exact command
  plan before making performance claims.
