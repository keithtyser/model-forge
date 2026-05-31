from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table


REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_DIR / "configs" / "clusters" / "local.example.yaml"
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
IP_ADDRESS_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
ALLOWED_LAUNCHERS = {"local", "ssh", "docker", "torchrun", "ray", "vllm", "slurm"}
WORKLOADS = {"serve", "train", "eval", "ablate", "data", "quantize", "publish", "generic"}
SYNC_EXCLUDES = (
    "/.venv/",
    "__pycache__/",
    "/.pytest_cache/",
    "/.mypy_cache/",
    "/runs/",
)

console = Console()


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None
    node: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except (OSError, ValueError):
        try:
            return str(path.relative_to(REPO_DIR))
        except ValueError:
            return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def load_cluster_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    config = load_yaml(config_path)
    config["_path"] = display_path(config_path)
    return config, config_path


def load_hardware_profile(cluster: Mapping[str, Any]) -> dict[str, Any]:
    hardware_path = cluster.get("hardware_profile")
    if not hardware_path:
        return {}
    path = resolve_repo_path(str(hardware_path))
    if not path.exists():
        return {"_path": display_path(path), "_error": "hardware_profile path does not exist"}
    profile = load_yaml(path)
    profile["_path"] = display_path(path)
    return profile


def env_value(env: Mapping[str, str], key: str | None) -> str | None:
    if not key:
        return None
    value = env.get(key)
    return value if value not in {None, ""} else None


def resolve_env_backed_value(raw: Mapping[str, Any], key: str, env: Mapping[str, str]) -> tuple[str | None, str | None]:
    env_key = raw.get(f"{key}_env")
    if env_key:
        return env_value(env, str(env_key)), str(env_key)
    value = raw.get(key)
    return str(value) if value not in {None, ""} else None, None


def node_ssh_port(node: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str | None, str | None]:
    return resolve_env_backed_value(node, "ssh_port", env)


def node_host(node: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str | None, str | None]:
    return resolve_env_backed_value(node, "host", env)


def node_user(node: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str | None, str | None]:
    return resolve_env_backed_value(node, "user", env)


def node_work_dir(node: Mapping[str, Any], cluster: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str | None, str | None]:
    value, env_key = resolve_env_backed_value(node, "work_dir", env)
    if value or env_key:
        return value, env_key
    return resolve_env_backed_value(cluster.get("paths", {}), "work_dir", env)


def node_models_dir(node: Mapping[str, Any], cluster: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str | None, str | None]:
    value, env_key = resolve_env_backed_value(node, "models_dir", env)
    if value or env_key:
        return value, env_key
    return resolve_env_backed_value(cluster.get("paths", {}), "models_dir", env)


def total_declared_memory_gb(cluster: Mapping[str, Any], hardware: Mapping[str, Any]) -> int | float:
    hardware_default = (hardware.get("node_defaults") or {}).get("memory_total_gb", 0)
    total: int | float = 0
    for node in cluster.get("nodes", []):
        if isinstance(node, dict):
            total += node.get("memory_total_gb", hardware_default) or 0
    return total


def total_declared_gpus(cluster: Mapping[str, Any], hardware: Mapping[str, Any]) -> int:
    hardware_default = int((hardware.get("node_defaults") or {}).get("gpu_count", 0) or 0)
    total = 0
    for node in cluster.get("nodes", []):
        if isinstance(node, dict):
            total += int(node.get("gpu_count", hardware_default) or 0)
    return total


def is_example_config(path: Path, cluster: Mapping[str, Any]) -> bool:
    return path.name.endswith(".example.yaml") or str(cluster.get("id", "")).endswith("_example")


def check_no_secret_literals(obj: Any, config_path: str, findings: list[Finding], node: str | None = None) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            check_no_secret_literals(value, config_path, findings, node=node if key != "name" else str(value))
    elif isinstance(obj, list):
        for item in obj:
            check_no_secret_literals(item, config_path, findings, node=node)
    elif isinstance(obj, str) and SECRET_PATTERN.search(obj):
        findings.append(Finding("error", "secret_literal", "secret-like literal found in cluster config", config_path, node))


def audit_cluster(
    cluster: Mapping[str, Any],
    config_path: Path,
    hardware: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    strict: bool = False,
) -> list[Finding]:
    env = env or os.environ
    hardware = hardware or load_hardware_profile(cluster)
    findings: list[Finding] = []
    config_display = display_path(config_path)
    check_no_secret_literals(cluster, config_display, findings)

    if not cluster.get("id"):
        findings.append(Finding("error", "schema", "cluster id is required", config_display))
    nodes = cluster.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        findings.append(Finding("error", "schema", "nodes must be a non-empty list", config_display))
        return findings

    supported = set((cluster.get("launchers") or {}).get("supported") or [])
    if not supported:
        supported = {str((cluster.get("launchers") or {}).get("default") or "local")}
    unknown_launchers = sorted(supported - ALLOWED_LAUNCHERS)
    if unknown_launchers:
        findings.append(Finding("error", "launcher", f"unsupported launchers: {', '.join(unknown_launchers)}", config_display))

    coordinators = [node for node in nodes if isinstance(node, dict) and node.get("role") == "coordinator"]
    if len(coordinators) != 1:
        findings.append(Finding("error", "schema", "exactly one node must have role=coordinator", config_display))

    names = []
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            findings.append(Finding("error", "schema", "node entries must be mappings", config_display))
            continue
        name = str(raw_node.get("name", ""))
        names.append(name)
        if not name:
            findings.append(Finding("error", "node", "node name is required", config_display))
        role = raw_node.get("role")
        if role not in {"coordinator", "worker"}:
            findings.append(Finding("error", "node", "node role must be coordinator or worker", config_display, name))
        launcher = raw_node.get("launcher") or (cluster.get("launchers") or {}).get("default") or "local"
        if launcher not in ALLOWED_LAUNCHERS:
            findings.append(Finding("error", "launcher", f"unsupported node launcher {launcher!r}", config_display, name))

        host, host_env = node_host(raw_node, env)
        if not raw_node.get("host") and not raw_node.get("host_env"):
            findings.append(Finding("error", "node", "node must define host or host_env", config_display, name))
        if raw_node.get("host") and raw_node.get("host_env"):
            findings.append(Finding("error", "node", "node must not define both host and host_env", config_display, name))
        if host_env and not host:
            severity = "error" if strict else "warning"
            findings.append(Finding(severity, "env", f"environment variable {host_env} is not set", config_display, name))
        if raw_node.get("host") and IP_ADDRESS_PATTERN.match(str(raw_node["host"])) and is_example_config(config_path, cluster):
            findings.append(Finding("error", "portability", "example configs must not commit literal IP addresses", config_display, name))

        user, user_env = node_user(raw_node, env)
        if user_env and not user and strict:
            findings.append(Finding("error", "env", f"environment variable {user_env} is not set", config_display, name))

        work_dir, work_env = node_work_dir(raw_node, cluster, env)
        if work_env and not work_dir and strict:
            findings.append(Finding("error", "env", f"environment variable {work_env} is not set", config_display, name))
        raw_work_dir = raw_node.get("work_dir")
        if raw_work_dir is None and not raw_node.get("work_dir_env"):
            raw_work_dir = (cluster.get("paths") or {}).get("work_dir")
        if raw_work_dir and PRIVATE_PATH_PATTERN.search(str(raw_work_dir)) and is_example_config(config_path, cluster):
            findings.append(Finding("error", "portability", "example configs must not commit machine-specific absolute paths", config_display, name))

        memory_fraction = raw_node.get("memory_max_fraction", (cluster.get("resource_policy", {}).get("per_node") or {}).get("memory_max_fraction"))
        if memory_fraction is not None and float(memory_fraction) > 0.90:
            findings.append(Finding("warning", "resource_policy", "memory_max_fraction above 0.90 can starve SSH/control plane", config_display, name))

    if len(names) != len(set(names)):
        findings.append(Finding("error", "schema", "node names must be unique", config_display))

    policy = cluster.get("resource_policy") or {}
    if policy.get("max_concurrent_large_jobs", 1) != 1:
        findings.append(Finding("warning", "resource_policy", "large jobs should default to one cluster-wide job", config_display))
    if not policy.get("require_job_lock", False):
        findings.append(Finding("warning", "resource_policy", "cluster job lock is not required", config_display))

    distributed = cluster.get("distributed") or {}
    if int(distributed.get("nnodes", len(nodes)) or len(nodes)) != len(nodes):
        findings.append(Finding("warning", "distributed", "distributed.nnodes does not match node count", config_display))
    if strict and len(nodes) > 1 and distributed.get("rdzv_endpoint_env") and not env_value(env, str(distributed["rdzv_endpoint_env"])):
        findings.append(
            Finding(
                "error",
                "env",
                f"environment variable {distributed['rdzv_endpoint_env']} is not set",
                config_display,
            )
        )

    if hardware.get("_error"):
        findings.append(Finding("error", "hardware", str(hardware["_error"]), config_display))

    return findings


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("cluster doctor: OK")
        return
    table = Table(title="Cluster Doctor")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Node")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity.upper(), finding.check, finding.node or "", finding.message)
    console.print(table)


def node_plan(node: Mapping[str, Any], cluster: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    host, host_env = node_host(node, env)
    user, user_env = node_user(node, env)
    ssh_port, ssh_port_env = node_ssh_port(node, env)
    work_dir, work_dir_env = node_work_dir(node, cluster, env)
    models_dir, models_dir_env = node_models_dir(node, cluster, env)
    launcher = node.get("launcher") or (cluster.get("launchers") or {}).get("default") or "local"
    display_host = host if host is not None else f"${host_env}" if host_env else None
    display_user = user if user is not None else f"${user_env}" if user_env else None
    display_work_dir = work_dir if work_dir is not None else f"${work_dir_env}" if work_dir_env else None
    display_models_dir = models_dir if models_dir is not None else f"${models_dir_env}" if models_dir_env else None
    return {
        "name": node.get("name"),
        "role": node.get("role"),
        "launcher": launcher,
        "host": display_host,
        "host_env": host_env,
        "user": display_user,
        "user_env": user_env,
        "ssh_port": ssh_port if ssh_port is not None else f"${ssh_port_env}" if ssh_port_env else None,
        "ssh_port_env": ssh_port_env,
        "work_dir": display_work_dir,
        "work_dir_env": work_dir_env,
        "models_dir": display_models_dir,
        "models_dir_env": models_dir_env,
        "gpu_count": node.get("gpu_count"),
        "memory_total_gb": node.get("memory_total_gb"),
        "cpu_quota": node.get("cpu_quota") or (cluster.get("resource_policy", {}).get("per_node") or {}).get("cpu_quota"),
        "memory_max_fraction": node.get("memory_max_fraction") or (cluster.get("resource_policy", {}).get("per_node") or {}).get("memory_max_fraction"),
    }


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def guarded_command(command: str, resource_policy: Mapping[str, Any], job_lock: str) -> str:
    per_node = resource_policy.get("per_node", {}) if isinstance(resource_policy.get("per_node"), dict) else {}
    cpu_quota = per_node.get("cpu_quota", resource_policy.get("cpu_quota", "80%"))
    memory_fraction = float(per_node.get("memory_max_fraction", resource_policy.get("memory_max_fraction", 0.85)))
    io_weight = per_node.get("io_weight", resource_policy.get("io_weight", 100))
    nice = per_node.get("nice", resource_policy.get("nice", 10))
    user_scope = bool(resource_policy.get("systemd_user_scope", True))
    systemd_scope = "systemd-run --user --scope" if user_scope else "systemd-run --scope"
    return (
        f"flock {shlex.quote(str(job_lock))} "
        f"{systemd_scope} -p CPUQuota={cpu_quota} "
        f"-p MemoryMax={int(memory_fraction * 100)}% "
        f"-p IOWeight={io_weight} "
        f"nice -n {nice} {command}"
    )


def is_local_host(host: str | None) -> bool:
    if host in {None, "", "localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return host in {socket.gethostname(), socket.getfqdn()}
    except OSError:
        return False


def ssh_target(node: Mapping[str, Any]) -> str:
    host = str(node.get("host") or "")
    user = str(node.get("user") or "")
    if not host:
        raise ValueError(f"node {node.get('name')} has no resolved host")
    return f"{user}@{host}" if user and not is_local_host(host) else host


def ssh_prefix(node: Mapping[str, Any]) -> list[str]:
    prefix = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    port = node.get("ssh_port")
    if port and not str(port).startswith("$"):
        prefix.extend(["-p", str(port)])
    prefix.append(ssh_target(node))
    return prefix


def run_node_shell(node: Mapping[str, Any], command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    host = str(node.get("host") or "")
    if is_local_host(host):
        return subprocess.run(
            command,
            cwd=REPO_DIR,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    return subprocess.run(
        [*ssh_prefix(node), command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def health_command(work_dir: str) -> str:
    script = r"""
import json
import os
import shutil
import subprocess
from pathlib import Path

def run(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8, check=False)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": result.returncode == 0, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode}

work_dir = Path(os.environ["MODEL_FORGE_HEALTH_WORK_DIR"]).expanduser()
disk_root = work_dir if work_dir.exists() else work_dir.parent
disk = shutil.disk_usage(str(disk_root))
memory = {}
try:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in meminfo.splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(raw.strip().split()[0]) * 1024
except OSError:
    pass

data = {
    "hostname": run(["hostname"])["stdout"],
    "work_dir": str(work_dir),
    "work_dir_exists": work_dir.exists(),
    "forge_exists": (work_dir / "forge").exists(),
    "git_branch": run(["git", "branch", "--show-current"], cwd=work_dir) if work_dir.exists() else None,
    "git_head": run(["git", "rev-parse", "--short", "HEAD"], cwd=work_dir) if work_dir.exists() else None,
    "git_status": run(["git", "status", "-sb"], cwd=work_dir) if work_dir.exists() else None,
    "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]),
    "memory": memory,
    "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
}
print(json.dumps(data, sort_keys=True))
"""
    return f"MODEL_FORGE_HEALTH_WORK_DIR={shlex.quote(work_dir)} python3 -c {shlex.quote(script)}"


def default_cluster_output(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_DIR / "reports" / "generated" / "cluster" / f"{prefix}_{stamp}.json"


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_cluster_health(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    env: Mapping[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    env = env or os.environ
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]

    def probe(node: Mapping[str, Any]) -> dict[str, Any]:
        work_dir = str(node.get("work_dir") or "")
        result = run_node_shell(node, health_command(work_dir), timeout=timeout)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {}
        return {
            "name": node.get("name"),
            "role": node.get("role"),
            "host": node.get("host"),
            "returncode": result.returncode,
            "ok": result.returncode == 0 and bool(payload.get("work_dir_exists")) and bool(payload.get("forge_exists")),
            "payload": payload,
            "stderr": result.stderr.strip(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as executor:
        results = list(executor.map(probe, nodes))

    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "nodes": results,
        "ok": all(node["ok"] for node in results),
    }


def docker_gpu_runtime_command(image: str) -> str:
    script = r"""
import importlib.util
import json
import sys

mods = ["torch", "transformers", "accelerate", "trl", "peft", "bitsandbytes", "vllm", "modelopt"]
data = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "modules": {name: bool(importlib.util.find_spec(name)) for name in mods},
}
try:
    import torch
    data["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
except Exception as exc:
    data["torch_error"] = repr(exc)
print(json.dumps(data, sort_keys=True))
"""
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--cpus=1",
        "--memory=8g",
        "--memory-swap=8g",
        "--pids-limit=512",
        "--entrypoint",
        "python3",
        image,
        "-c",
        script,
    ]
    return shell_join(command)


def torchrun_smoke_script() -> str:
    return r"""
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
world_size = dist.get_world_size()
tensor = torch.tensor([float(rank + 1)], device="cuda")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = world_size * (world_size + 1) / 2
ok = abs(float(tensor.item()) - expected) < 1e-5
dist.barrier()
print(json.dumps({
    "hostname": socket.gethostname(),
    "rank": rank,
    "local_rank": local_rank,
    "world_size": world_size,
    "cuda_device": torch.cuda.get_device_name(local_rank),
    "all_reduce_sum": float(tensor.item()),
    "expected_sum": expected,
    "ok": ok,
}, sort_keys=True), flush=True)
dist.destroy_process_group()
if not ok:
    raise SystemExit(2)
"""


def docker_torchrun_smoke_command(
    image: str,
    *,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    rdzv_endpoint: str,
    nccl_socket_ifname: str | None = None,
) -> str:
    master_addr, _, master_port = rdzv_endpoint.rpartition(":")
    if not master_addr or not master_port:
        raise ValueError(f"rdzv_endpoint must be host:port, got {rdzv_endpoint!r}")
    bash = "\n".join(
        [
            "set -euo pipefail",
            f"cat > /tmp/model_forge_torchrun_smoke.py <<'PY'\n{torchrun_smoke_script()}\nPY",
            shell_join(
                [
                    "python3",
                    "-m",
                    "torch.distributed.run",
                    f"--nnodes={nnodes}",
                    f"--nproc-per-node={nproc_per_node}",
                    f"--node-rank={node_rank}",
                    f"--master-addr={master_addr}",
                    f"--master-port={master_port}",
                    "--max-restarts=0",
                    "/tmp/model_forge_torchrun_smoke.py",
                ]
            ),
        ]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--cpus=2",
        "--memory=16g",
        "--memory-swap=16g",
        "--pids-limit=1024",
        "-e",
        "NCCL_DEBUG=WARN",
        "-e",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
    ]
    if nccl_socket_ifname:
        command.extend(["-e", f"NCCL_SOCKET_IFNAME={nccl_socket_ifname}"])
    command.extend(["--entrypoint", "bash", image, "-lc", bash])
    return shell_join(command)


def collect_cluster_runtime(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    image: str,
    env: Mapping[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    env = env or os.environ
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]
    command = docker_gpu_runtime_command(image)

    def probe(node: Mapping[str, Any]) -> dict[str, Any]:
        result = run_node_shell(node, command, timeout=timeout)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {}
        torch_info = payload.get("torch") if isinstance(payload, dict) else {}
        ok = (
            result.returncode == 0
            and isinstance(torch_info, dict)
            and bool(torch_info.get("cuda_available"))
            and int(torch_info.get("device_count") or 0) >= 1
        )
        return {
            "name": node.get("name"),
            "role": node.get("role"),
            "host": node.get("host"),
            "returncode": result.returncode,
            "ok": ok,
            "payload": payload,
            "stderr": result.stderr.strip(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as executor:
        results = list(executor.map(probe, nodes))

    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "image": image,
        "mode": "docker_gpu_python",
        "nodes": results,
        "ok": all(node["ok"] for node in results),
    }


def json_lines(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def collect_torchrun_smoke(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    image: str,
    env: Mapping[str, str] | None = None,
    timeout: int = 180,
    nccl_socket_ifname: str | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]
    distributed = cluster.get("distributed") or {}
    nnodes = int(distributed.get("nnodes", len(nodes)) or len(nodes))
    nproc_per_node = int(distributed.get("nproc_per_node", 1) or 1)
    endpoint_env = str(distributed.get("rdzv_endpoint_env") or "MODEL_FORGE_RDZV_ENDPOINT")
    rdzv_endpoint = str(distributed.get("rdzv_endpoint") or env_value(env, endpoint_env) or "")
    resource_policy = cluster.get("resource_policy") or {}
    job_lock = str((cluster.get("paths") or {}).get("job_lock") or "runs/locks/model-forge-cluster.lock")
    if not rdzv_endpoint:
        return {
            "created_at": utc_timestamp(),
            "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
            "image": image,
            "ok": False,
            "error": f"rendezvous endpoint is missing; set {endpoint_env}",
        }

    def probe(index_and_node: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        node_rank, node = index_and_node
        work_dir = str(node.get("work_dir") or ".")
        lock_path = job_lock if Path(job_lock).is_absolute() else str(Path(work_dir) / job_lock)
        docker_command = docker_torchrun_smoke_command(
            image,
            node_rank=node_rank,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            rdzv_endpoint=rdzv_endpoint,
            nccl_socket_ifname=nccl_socket_ifname,
        )
        command = shell_join(["mkdir", "-p", str(Path(lock_path).parent)])
        command += " && "
        command += guarded_command(docker_command, resource_policy, lock_path)
        result = run_node_shell(node, command, timeout=timeout)
        records = json_lines(result.stdout)
        ok = result.returncode == 0 and len(records) == nproc_per_node and all(bool(record.get("ok")) for record in records)
        return {
            "name": node.get("name"),
            "role": node.get("role"),
            "host": node.get("host"),
            "node_rank": node_rank,
            "returncode": result.returncode,
            "ok": ok,
            "records": records,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as executor:
        results = list(executor.map(probe, enumerate(nodes)))
    observed_world = sum(len(node["records"]) for node in results)
    expected_world = nnodes * nproc_per_node

    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "image": image,
        "mode": "docker_torchrun_nccl_all_reduce",
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "expected_world_size": expected_world,
        "observed_world_size": observed_world,
        "nccl_socket_ifname": nccl_socket_ifname,
        "nodes": results,
        "ok": observed_world == expected_world and all(node["ok"] for node in results),
    }


def build_sync_plan(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    env: Mapping[str, str] | None = None,
    delete: bool = False,
) -> dict[str, Any]:
    env = env or os.environ
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]
    actions: list[dict[str, Any]] = []
    for node in nodes:
        host = str(node.get("host") or "")
        work_dir = str(node.get("work_dir") or "")
        if not work_dir:
            actions.append({"node": node.get("name"), "skip": False, "error": "work_dir is not resolved"})
            continue
        if is_local_host(host):
            actions.append({"node": node.get("name"), "skip": True, "reason": "local coordinator"})
            continue
        target = f"{ssh_target(node)}:{work_dir.rstrip('/')}/"
        mkdir_command = [*ssh_prefix(node), shell_join(["mkdir", "-p", work_dir])]
        rsync_command = ["rsync", "-az"]
        if delete:
            rsync_command.append("--delete")
        for pattern in SYNC_EXCLUDES:
            rsync_command.extend(["--exclude", pattern])
        rsync_command.extend(["./", target])
        actions.append(
            {
                "node": node.get("name"),
                "host": host,
                "work_dir": work_dir,
                "skip": False,
                "mkdir_command": shell_join(mkdir_command),
                "rsync_command": shell_join(rsync_command),
            }
        )
    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "delete": delete,
        "excludes": list(SYNC_EXCLUDES),
        "actions": actions,
    }


def build_model_sync_plan(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    source: Path,
    env: Mapping[str, str] | None = None,
    target_name: str | None = None,
    delete: bool = False,
) -> dict[str, Any]:
    env = env or os.environ
    source = source.expanduser()
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]
    actions: list[dict[str, Any]] = []
    model_name = target_name or source.name
    if not source.exists() or not source.is_dir():
        actions.append({"node": "coordinator", "skip": False, "error": f"source model dir does not exist: {source}"})
    for node in nodes:
        host = str(node.get("host") or "")
        models_dir = str(node.get("models_dir") or "")
        if not models_dir or models_dir.startswith("$"):
            actions.append({"node": node.get("name"), "skip": False, "error": "models_dir is not resolved"})
            continue
        if is_local_host(host):
            actions.append({"node": node.get("name"), "skip": True, "reason": "local coordinator"})
            continue
        target_dir = str(Path(models_dir) / model_name)
        target = f"{ssh_target(node)}:{target_dir.rstrip('/')}/"
        mkdir_command = [*ssh_prefix(node), shell_join(["mkdir", "-p", target_dir])]
        rsync_command = ["rsync", "-az", "--partial"]
        if delete:
            rsync_command.append("--delete")
        rsync_command.extend([f"{str(source).rstrip('/')}/", target])
        actions.append(
            {
                "node": node.get("name"),
                "host": host,
                "models_dir": models_dir,
                "source": str(source),
                "target_dir": target_dir,
                "skip": False,
                "mkdir_command": shell_join(mkdir_command),
                "rsync_command": shell_join(rsync_command),
            }
        )
    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "delete": delete,
        "source": str(source),
        "target_name": model_name,
        "actions": actions,
    }


def execute_sync_plan(plan: dict[str, Any], timeout: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for action in plan["actions"]:
        if action.get("skip") or action.get("error"):
            results.append({**action, "ok": not action.get("error")})
            continue
        mkdir = subprocess.run(
            shlex.split(action["mkdir_command"]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if mkdir.returncode != 0:
            results.append({**action, "ok": False, "step": "mkdir", "stderr": mkdir.stderr.strip()})
            continue
        rsync = subprocess.run(
            shlex.split(action["rsync_command"]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        results.append(
            {
                **action,
                "ok": rsync.returncode == 0,
                "step": "rsync",
                "stdout": rsync.stdout.strip(),
                "stderr": rsync.stderr.strip(),
                "returncode": rsync.returncode,
            }
        )
    return {**plan, "executed": True, "actions": results, "ok": all(action.get("ok") for action in results)}


def build_launcher_plan(
    cluster: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config_path: Path,
    workload: str,
    command: str | None,
    launcher: str | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    if workload not in WORKLOADS:
        raise ValueError(f"unsupported workload {workload!r}")
    selected_launcher = launcher or (cluster.get("launchers") or {}).get("default") or "local"
    nodes = [node_plan(node, cluster, env) for node in cluster.get("nodes", []) if isinstance(node, dict)]
    coordinator = next((node for node in nodes if node.get("role") == "coordinator"), nodes[0] if nodes else {})
    distributed = cluster.get("distributed") or {}
    resource_policy = cluster.get("resource_policy") or {}
    paths = cluster.get("paths") or {}
    user_command = command or "<workload command>"
    job_lock = paths.get("job_lock", "runs/locks/model-forge-cluster.lock")

    preflight = [
        shell_join(["mkdir", "-p", str(Path(job_lock).parent)]),
        shell_join(["./forge", "cluster", "doctor", "--config", display_path(config_path), "--strict"]),
    ]

    if selected_launcher == "torchrun":
        nnodes = int(distributed.get("nnodes", len(nodes)) or len(nodes))
        nproc = int(distributed.get("nproc_per_node", coordinator.get("gpu_count") or 1) or 1)
        endpoint_env = distributed.get("rdzv_endpoint_env", "MODEL_FORGE_RDZV_ENDPOINT")
        endpoint = distributed.get("rdzv_endpoint") or env_value(env, str(endpoint_env)) or f"${endpoint_env}"
        torchrun_command = shell_join([
            "torchrun",
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc}",
            f"--rdzv-backend={distributed.get('rdzv_backend', 'c10d')}",
            f"--rdzv-endpoint={endpoint}",
            *shlex.split(user_command),
        ])
        launch_command = guarded_command(torchrun_command, resource_policy, str(job_lock))
    elif selected_launcher == "ssh":
        host = coordinator.get("host") or "<coordinator-host>"
        launch_command = f"ssh {shlex.quote(str(host))} {shlex.quote(guarded_command(user_command, resource_policy, str(job_lock)))}"
    elif selected_launcher == "local":
        launch_command = guarded_command(user_command, resource_policy, str(job_lock))
    else:
        launch_command = f"{selected_launcher} launcher is configured; backend-specific execution is intentionally dry-run only"

    return {
        "created_at": utc_timestamp(),
        "cluster": {
            "id": cluster.get("id"),
            "config": display_path(config_path),
            "hardware_profile": cluster.get("hardware_profile"),
            "node_count": len(nodes),
            "total_declared_gpus": total_declared_gpus(cluster, hardware),
            "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        },
        "workload": workload,
        "launcher": selected_launcher,
        "nodes": nodes,
        "resource_policy": resource_policy,
        "environment_contract": {
            "required_host_env": [node.get("host_env") for node in nodes if node.get("host_env")],
            "required_user_env": [node.get("user_env") for node in nodes if node.get("user_env")],
            "job_lock": job_lock,
        },
        "preflight": preflight,
        "execution_plan": {
            "dry_run_only": True,
            "coordinator": coordinator.get("name"),
            "command": user_command,
            "launcher_command": launch_command,
            "notes": [
                "This plan does not execute cluster commands.",
                "Run doctor with --strict and verify one large job is active cluster-wide before launching.",
                "Use backend-specific distributed serving/training docs before enabling multi-node execution.",
            ],
        },
    }


def render_plan(plan: Mapping[str, Any]) -> None:
    cluster = plan["cluster"]
    console.print(f"Cluster plan: {cluster['id']} ({cluster['node_count']} node(s), {cluster['total_declared_memory_gb']} GB declared RAM)")
    node_table = Table(title="Nodes")
    node_table.add_column("Name")
    node_table.add_column("Role")
    node_table.add_column("Launcher")
    node_table.add_column("Host")
    node_table.add_column("GPU")
    node_table.add_column("RAM GB")
    for node in plan["nodes"]:
        node_table.add_row(
            str(node.get("name")),
            str(node.get("role")),
            str(node.get("launcher")),
            str(node.get("host") or ""),
            str(node.get("gpu_count") or ""),
            str(node.get("memory_total_gb") or ""),
        )
    console.print(node_table)
    console.print("Preflight:")
    for command in plan["preflight"]:
        console.print(f"  {command}")
    console.print("Launcher command:")
    console.print(f"  {plan['execution_plan']['launcher_command']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and validate generic model-forge cluster inventories")
    subparsers = parser.add_subparsers(dest="action", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Validate a cluster inventory")
    doctor_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor_parser.add_argument("--strict", action="store_true", help="Treat missing env-backed node values as errors")
    doctor_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="Render a dry-run cluster launch plan")
    plan_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan_parser.add_argument("--workload", default="generic", choices=sorted(WORKLOADS))
    plan_parser.add_argument("--launcher", choices=sorted(ALLOWED_LAUNCHERS))
    plan_parser.add_argument("--command", dest="workload_command", help="Workload command to place behind cluster guardrails")
    plan_parser.add_argument("--json", action="store_true")

    health_parser = subparsers.add_parser("health", help="Probe every cluster node for repo, GPU, RAM, and disk readiness")
    health_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    health_parser.add_argument("--timeout", type=int, default=30)
    health_parser.add_argument("--output", type=Path, help="Write JSON health evidence")
    health_parser.add_argument("--json", action="store_true")

    runtime_parser = subparsers.add_parser("runtime", help="Probe bounded GPU container runtime on every cluster node")
    runtime_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    runtime_parser.add_argument("--image", default="nemotron-runner:latest")
    runtime_parser.add_argument("--timeout", type=int, default=120)
    runtime_parser.add_argument("--output", type=Path, help="Write JSON runtime evidence")
    runtime_parser.add_argument("--json", action="store_true")

    smoke_parser = subparsers.add_parser("torchrun-smoke", help="Run bounded two-node Docker torchrun/NCCL all-reduce smoke")
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--image", default="nemotron-runner:latest")
    smoke_parser.add_argument("--timeout", type=int, default=180)
    smoke_parser.add_argument("--nccl-socket-ifname", help="Optional NCCL_SOCKET_IFNAME override, e.g. a direct-link interface")
    smoke_parser.add_argument("--output", type=Path, help="Write JSON smoke evidence")
    smoke_parser.add_argument("--json", action="store_true")

    sync_parser = subparsers.add_parser("sync", help="Plan or execute env-backed repo sync to worker nodes")
    sync_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sync_parser.add_argument("--execute", action="store_true", help="Run mkdir/rsync commands")
    sync_parser.add_argument("--delete", action="store_true", help="Pass --delete to rsync")
    sync_parser.add_argument("--timeout", type=int, default=120)
    sync_parser.add_argument("--output", type=Path, help="Write JSON sync evidence")
    sync_parser.add_argument("--json", action="store_true")

    model_sync_parser = subparsers.add_parser("model-sync", help="Plan or execute model directory sync to worker nodes")
    model_sync_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    model_sync_parser.add_argument("--source", type=Path, required=True, help="Local coordinator model directory to copy")
    model_sync_parser.add_argument("--target-name", help="Directory name under each worker models_dir; defaults to source basename")
    model_sync_parser.add_argument("--execute", action="store_true", help="Run mkdir/rsync commands")
    model_sync_parser.add_argument("--delete", action="store_true", help="Pass --delete to rsync")
    model_sync_parser.add_argument("--timeout", type=int, default=3600)
    model_sync_parser.add_argument("--output", type=Path, help="Write JSON sync evidence")
    model_sync_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    cluster, config_path = load_cluster_config(args.config)
    hardware = load_hardware_profile(cluster)

    if args.action == "doctor":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2, sort_keys=True) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.action == "health":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=True)
        if findings:
            data = {
                "created_at": utc_timestamp(),
                "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
                "ok": False,
                "doctor_findings": [asdict(finding) for finding in findings],
            }
            if args.output:
                write_json(args.output if args.output.is_absolute() else REPO_DIR / args.output, data)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        health = collect_cluster_health(cluster, hardware, config_path, timeout=args.timeout)
        output = args.output if args.output else default_cluster_output("health")
        output_path = output if output.is_absolute() else REPO_DIR / output
        write_json(output_path, health)
        health["output"] = display_path(output_path)
        if args.json:
            print(json.dumps(health, indent=2, sort_keys=True) + "\n")
        else:
            console.print(f"cluster health: {'OK' if health['ok'] else 'FAILED'}")
            console.print(f"evidence: {display_path(output_path)}")
        raise SystemExit(0 if health["ok"] else 1)

    if args.action == "runtime":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=True)
        if findings:
            data = {
                "created_at": utc_timestamp(),
                "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
                "ok": False,
                "doctor_findings": [asdict(finding) for finding in findings],
            }
            if args.output:
                write_json(args.output if args.output.is_absolute() else REPO_DIR / args.output, data)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        runtime = collect_cluster_runtime(cluster, hardware, config_path, image=args.image, timeout=args.timeout)
        output = args.output if args.output else default_cluster_output("runtime")
        output_path = output if output.is_absolute() else REPO_DIR / output
        write_json(output_path, runtime)
        runtime["output"] = display_path(output_path)
        if args.json:
            print(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        else:
            console.print(f"cluster runtime: {'OK' if runtime['ok'] else 'FAILED'}")
            console.print(f"image: {args.image}")
            console.print(f"evidence: {display_path(output_path)}")
        raise SystemExit(0 if runtime["ok"] else 1)

    if args.action == "torchrun-smoke":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=True)
        if findings:
            data = {
                "created_at": utc_timestamp(),
                "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
                "ok": False,
                "doctor_findings": [asdict(finding) for finding in findings],
            }
            if args.output:
                write_json(args.output if args.output.is_absolute() else REPO_DIR / args.output, data)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        smoke = collect_torchrun_smoke(
            cluster,
            hardware,
            config_path,
            image=args.image,
            timeout=args.timeout,
            nccl_socket_ifname=args.nccl_socket_ifname,
        )
        output = args.output if args.output else default_cluster_output("torchrun_smoke")
        output_path = output if output.is_absolute() else REPO_DIR / output
        write_json(output_path, smoke)
        smoke["output"] = display_path(output_path)
        if args.json:
            print(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
        else:
            console.print(f"cluster torchrun smoke: {'OK' if smoke['ok'] else 'FAILED'}")
            console.print(f"image: {args.image}")
            console.print(f"evidence: {display_path(output_path)}")
        raise SystemExit(0 if smoke["ok"] else 1)

    if args.action == "sync":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=True)
        if findings:
            data = {
                "created_at": utc_timestamp(),
                "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
                "ok": False,
                "doctor_findings": [asdict(finding) for finding in findings],
            }
            if args.output:
                write_json(args.output if args.output.is_absolute() else REPO_DIR / args.output, data)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        sync = build_sync_plan(cluster, hardware, config_path, delete=args.delete)
        if args.execute:
            sync = execute_sync_plan(sync, timeout=args.timeout)
        else:
            sync["executed"] = False
            sync["ok"] = all(not action.get("error") for action in sync["actions"])
        output = args.output if args.output else default_cluster_output("sync")
        output_path = output if output.is_absolute() else REPO_DIR / output
        write_json(output_path, sync)
        sync["output"] = display_path(output_path)
        if args.json:
            print(json.dumps(sync, indent=2, sort_keys=True) + "\n")
        else:
            console.print(f"cluster sync: {'OK' if sync['ok'] else 'FAILED'} ({'executed' if args.execute else 'plan only'})")
            console.print(f"evidence: {display_path(output_path)}")
            for action in sync["actions"]:
                if action.get("skip"):
                    console.print(f"  {action['node']}: skipped ({action.get('reason')})")
                elif action.get("error"):
                    console.print(f"  {action['node']}: ERROR {action['error']}")
                else:
                    console.print(f"  {action['node']}: {action.get('rsync_command')}")
        raise SystemExit(0 if sync["ok"] else 1)

    if args.action == "model-sync":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=True)
        if findings:
            data = {
                "created_at": utc_timestamp(),
                "cluster": {"id": cluster.get("id"), "config": display_path(config_path)},
                "ok": False,
                "doctor_findings": [asdict(finding) for finding in findings],
            }
            if args.output:
                write_json(args.output if args.output.is_absolute() else REPO_DIR / args.output, data)
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        sync = build_model_sync_plan(
            cluster,
            hardware,
            config_path,
            source=args.source,
            target_name=args.target_name,
            delete=args.delete,
        )
        if args.execute:
            sync = execute_sync_plan(sync, timeout=args.timeout)
        else:
            sync["executed"] = False
            sync["ok"] = all(action.get("skip") or not action.get("error") for action in sync["actions"])
        output = args.output if args.output else default_cluster_output("model_sync")
        output_path = output if output.is_absolute() else REPO_DIR / output
        write_json(output_path, sync)
        sync["output"] = display_path(output_path)
        if args.json:
            print(json.dumps(sync, indent=2, sort_keys=True) + "\n")
        else:
            console.print(f"cluster model-sync: {'OK' if sync['ok'] else 'FAILED'} ({'executed' if args.execute else 'plan only'})")
            console.print(f"evidence: {display_path(output_path)}")
            for action in sync["actions"]:
                if action.get("skip"):
                    console.print(f"  {action['node']}: skipped ({action.get('reason')})")
                elif action.get("error"):
                    console.print(f"  {action['node']}: ERROR {action['error']}")
                else:
                    console.print(f"  {action['node']}: {action.get('rsync_command')}")
        raise SystemExit(0 if sync["ok"] else 1)

    if args.action == "plan":
        findings = audit_cluster(cluster, config_path, hardware=hardware, strict=False)
        plan = build_launcher_plan(
            cluster,
            hardware,
            config_path,
            workload=args.workload,
            command=args.workload_command,
            launcher=args.launcher,
        )
        plan["doctor_findings"] = [asdict(finding) for finding in findings]
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
            non_error_findings = [finding for finding in findings if finding.severity != "error"]
            if non_error_findings:
                console.print()
                render_findings(non_error_findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)


if __name__ == "__main__":
    main()
