# Cluster Support

Model Forge cluster support is intentionally generic. A two-node DGX Spark setup
is one example of the same inventory and launcher model used for any local,
SSH, Docker, torchrun, Ray, Slurm, vLLM, or future backend.

The abstraction is:

```text
hardware profile + cluster inventory + launcher backend + workload command
```

Do not commit private hostnames, IP addresses, usernames, tokens, or absolute
machine-specific paths. Use environment-backed fields in cluster configs.

## Files

```text
configs/hardware/dgx_spark.yaml
configs/clusters/local.example.yaml
configs/clusters/dgx_spark_x2.example.yaml
```

Copy an example to a private config before using it:

```bash
cp configs/clusters/dgx_spark_x2.example.yaml configs/clusters/my_cluster.yaml
```

Set private values through environment variables:

```bash
export MODEL_FORGE_NODE0_HOST=<coordinator-host>
export MODEL_FORGE_NODE1_HOST=<worker-host>
export MODEL_FORGE_NODE0_USER=<ssh-user>
export MODEL_FORGE_NODE1_USER=<ssh-user>
export MODEL_FORGE_CLUSTER_WORK_DIR=<repo-path-on-each-node>
export MODEL_FORGE_RDZV_ENDPOINT=<coordinator-host>:29500
```

Keep that private config untracked unless it contains only placeholder values.

## Validate

Non-strict mode is useful for reviewing public examples:

```bash
./forge cluster doctor --config configs/clusters/dgx_spark_x2.example.yaml
```

Strict mode is the preflight gate before a real run:

```bash
./forge cluster doctor --config configs/clusters/my_cluster.yaml --strict
```

Strict mode fails when required env-backed host, user, or work-dir values are
missing.

## Sync And Health

Before any real multi-node training, serving, quantization, ablation, or eval
run, sync the repo to worker nodes and probe every node:

```bash
./forge cluster sync \
  --config configs/clusters/my_cluster.yaml \
  --execute

./forge cluster health \
  --config configs/clusters/my_cluster.yaml

./forge cluster runtime \
  --config configs/clusters/my_cluster.yaml \
  --image nemotron-runner:latest
```

`cluster sync` uses `rsync` over SSH and skips only local caches and the root
`runs/` directory by default. It does not commit private hostnames or paths; the
inventory still resolves them from env vars or an untracked private config.

`cluster health` probes all nodes in parallel and writes JSON evidence under
`reports/generated/cluster/`. It checks that the repo exists, `forge` exists,
Git branch/head/status are visible, `nvidia-smi` responds, and RAM/disk
headroom is available. Treat a failed health probe as a hard stop for heavy
jobs.

`cluster runtime` runs a bounded Docker GPU/Python probe on every node. It uses
`--gpus all`, `--cpus=1`, an 8 GB memory cap, and a low PID limit, then verifies
that the selected image can import the expected post-training libraries and see
CUDA devices. Treat a failed runtime probe as a hard stop for distributed
training, vLLM serving, ModelOpt quantization, or ablation jobs that rely on
that container image.

## Plan

Plans are dry-run only. They show node inventory, total declared GPU/RAM,
resource policy, required env vars, job lock, preflight command, and the launcher
command shape.

```bash
./forge cluster plan \
  --config configs/clusters/my_cluster.yaml \
  --workload train \
  --launcher torchrun \
  --command './forge finetune --config configs/finetuning/gemma4_26b_a4b_local_ft_v1_dryrun.yaml run --execute'
```

The planner does not execute SSH, Docker, vLLM, torchrun, or Ray commands. It
exists so humans and agents can check the shape before starting a heavyweight
job.

## Resource Contract

Cluster jobs should remain tenants on every node:

- one cluster-wide large job at a time
- job lock required
- CPU quota around 80% per node
- memory max around 85% per node
- start only when memory and disk floors pass
- stop/checkpoint if runtime memory headroom falls below the configured floor
- use high parallelism only where the workload is designed for it

For DGX Spark x2, the public example declares two 128 GB nodes for roughly 256 GB
total RAM. It does not assume hostnames or network layout.

## Next Integrations

The current cluster layer is a safe planning and validation layer. Real
distributed serving/training execution should be added behind explicit launcher
backends only after:

- single-node smoke serving passes
- `./forge cluster doctor --strict` passes
- run manifests include the cluster config and launcher plan
- the workload has checkpointing, cleanup, and failure handling
