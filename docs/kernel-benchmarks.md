# Kernel Benchmarks

Kernel microbenchmarks are narrow diagnostics for profiler follow-up work. They
do not replace serving benchmarks, evals, or Nsight evidence.

Run an RMSNorm dry run without importing Torch:

```bash
./forge bench kernel rmsnorm --dry-run --json
./forge bench kernel rope --dry-run --json
```

Run a small benchmark and write artifacts:

```bash
./forge bench kernel rmsnorm \
  --device auto \
  --dtype bfloat16 \
  --batch 1 \
  --seq-len 1024 \
  --hidden-size 4096 \
  --write
```

Run a small RoPE benchmark and write artifacts:

```bash
./forge bench kernel rope \
  --device auto \
  --dtype bfloat16 \
  --batch 1 \
  --seq-len 1024 \
  --heads 16 \
  --head-dim 128 \
  --write
```

Outputs are written under `reports/generated/kernel_benchmarks/<run>/`:

- `summary.json`
- `kernel_card.md`

The initial RMSNorm benchmark compares `torch.nn.functional.rms_norm` against a
manual reference implementation, records correctness tolerance, p50/p95 latency,
and approximate effective bandwidth. Treat it as a baseline harness for future
Triton/CUDA work, not as an optimized kernel claim.

The initial RoPE benchmark compares an interleaved Torch reference against a
complex-number Torch candidate and records the same correctness and latency
fields. It is meant to anchor future fused or backend-specific RoPE work.

Promotion rules:

- Pair each kernel card with an Nsight profile summary from a real serving run.
- Include correctness evidence before discussing speed.
- Tie the measured shape to an actual model bottleneck.
- Do not claim end-to-end gains from a microbenchmark alone.
