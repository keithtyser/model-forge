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

./forge cluster torchrun-smoke \
  --config configs/clusters/my_cluster.yaml \
  --image nemotron-runner:latest \
  --nccl-socket-ifname <distributed-network-interface>
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

`cluster torchrun-smoke` launches a bounded Docker `torchrun` job on every
configured node and performs a CUDA/NCCL all-reduce. It uses static
`--master-addr`/`--master-port` launch arguments derived from
`MODEL_FORGE_RDZV_ENDPOINT`, host networking, `--gpus all`, a 2 CPU cap, a
16 GB memory cap, and the same systemd resource policy used by other guarded
cluster probes. Treat this as the preflight gate before distributed fine-tuning,
distributed quantization, or any benchmark claim that says a workload used both
Spark nodes.

## Serving On Spark

Model serving is model-family driven. `./forge serve <family> <variant>` loads
the serving script from `configs/model_families/<family>.yaml`.

For Qwen-family models on DGX Spark, the generic launcher is:

```text
scripts/dgx_spark_serve_qwen.sh
```

It runs solo by default. To use a two-node Spark cluster, set env-backed values
outside Git:

```bash
export MODEL_FORGE_SPARK_CLUSTER=1
export MODEL_FORGE_SPARK_CLUSTER_NODES=<coordinator-ip>,<worker-ip>
export MODEL_FORGE_SPARK_ETH_IF=<direct-link-interface>
export MODEL_FORGE_TENSOR_PARALLEL_SIZE=2
export MODEL_FORGE_MODELS_DIR=<same-model-root-on-each-node>
```

Then start the configured variant:

```bash
./forge serve qwen36_27b base
```

The launcher passes `--tensor-parallel-size 2` and
`--distributed-executor-backend ray` when cluster mode is enabled. It also
mounts `MODEL_FORGE_MODELS_DIR` into the vLLM container, supports LoRA adapter
serving, and keeps private node identities in environment variables. If a
worker has no Hugging Face egress, download on the coordinator first, then
sync the completed checkpoint directory to the same model root on the worker
before serving:

```bash
./forge cluster model-sync \
  --config configs/clusters/my_cluster.yaml \
  --source <coordinator-models-dir>/<model-dir> \
  --execute
```

`model-sync` resolves each worker's `models_dir` from the cluster config or
environment, skips the local coordinator, and copies the model directory with
`rsync --partial` so interrupted transfers can resume.

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

The public configs default to `systemd-run --user --scope` because unprivileged
remote system scopes can require interactive authentication. Set
`resource_policy.systemd_user_scope: false` only on machines where system scopes
are explicitly available for non-interactive workload launchers.

For DGX Spark x2, the public example declares two 128 GB nodes for roughly 256 GB
total RAM. It does not assume hostnames or network layout.

## Handoff

Generated JSON evidence lives under `reports/generated/cluster/` and stays out
of Git. If a run depends on a private inventory, record the public config path,
private env variable names, command shape, and artifact locations in the
experiment ledger without committing private hostnames, IPs, tokens, or model
weights.
