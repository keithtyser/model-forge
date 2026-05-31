# Kernel Benchmarks

Kernel microbenchmarks are narrow diagnostics for profiler follow-up work. They
do not replace serving benchmarks, evals, or Nsight evidence.

Run an RMSNorm dry run without importing Torch:

```bash
./forge bench kernel rmsnorm --dry-run --json
./forge bench kernel rope --dry-run --json
./forge bench kernel dequant --dry-run --json
./forge bench kernel kv-layout --dry-run --json
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

Run a small NVFP4 E2M1 dequantization proxy and write artifacts:

```bash
./forge bench kernel dequant \
  --format nvfp4-e2m1 \
  --device auto \
  --output-dtype bfloat16 \
  --num-elements 1048576 \
  --block-size 16 \
  --write
```

Run a small KV-cache layout benchmark and write artifacts:

```bash
./forge bench kernel kv-layout \
  --device auto \
  --dtype bfloat16 \
  --batch 1 \
  --seq-len 4096 \
  --heads 16 \
  --head-dim 128 \
  --page-size 16 \
  --write
```

Outputs are written under `reports/generated/kernel_benchmarks/<run>/`:

- `summary.json`
- `kernel_card.json`
- `kernel_card.md`

Regenerate or enrich a Kernel Card from an existing summary:

```bash
./forge bench kernel card \
  --summary reports/generated/kernel_benchmarks/<run>/summary.json \
  --profile-summary reports/generated/profiles/nsight/<run>/profile_summary.json \
  --write-card
```

`--profile-summary` is optional. Use it when a benchmark has matching Nsight
artifact inventory or exported profiler evidence.

The initial RMSNorm benchmark compares `torch.nn.functional.rms_norm` against a
manual reference implementation, records correctness tolerance, p50/p95 latency,
and approximate effective bandwidth. Treat it as a baseline harness for future
Triton/CUDA work, not as an optimized kernel claim.

The initial RoPE benchmark compares an interleaved Torch reference against a
complex-number Torch candidate and records the same correctness and latency
fields. It is meant to anchor future fused or backend-specific RoPE work.

The initial dequant benchmark uses an NVFP4 E2M1 proxy with packed nibbles, a
local scale per 16 values, and a global scale. NVIDIA documents NVFP4 as using
E2M1 values with a local E4M3 scaling factor every 16 values plus a global FP32
scale. This benchmark keeps the scales as FP32 for portability and explicitly
does not claim native Blackwell Tensor Core behavior.

The initial KV-layout benchmark compares contiguous KV-cache reads with a
paged/gathered proxy layout. It is intended to quantify layout and gather/copy
overhead before testing a real serving backend's PagedAttention or KV-cache
implementation.

Promotion rules:

- Pair each kernel card with an Nsight profile summary from a real serving run.
- Include correctness evidence before discussing speed.
- Tie the measured shape to an actual model bottleneck.
- Do not claim end-to-end gains from a microbenchmark alone.
