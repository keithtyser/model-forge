from __future__ import annotations

import argparse
import fcntl
import csv
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil
import yaml
from rich.console import Console
from rich.table import Table

from model_forge.hardware import detect_hardware_profile, recommended_quantization_env
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "quantization" / "nvfp4_blackwell_runtime.yaml"
CONFIG_SCHEMA_VERSION = "model_forge.quantization.v1"
PLAN_SCHEMA_VERSION = "model_forge.quantization_plan.v1"
CARD_SCHEMA_VERSION = "model_forge.quantization_card.v1"

console = Console(stderr=True)


@dataclass(frozen=True)
class QuantizationConfig:
    name: str
    description: str
    method: str
    backend: str
    objective: str
    family: str | None
    source_variant: str | None
    target_variant: str | None
    hardware_profile: str | None
    calibration: dict[str, Any]
    exclusions: dict[str, Any]
    runtime: dict[str, Any]
    export: dict[str, Any]
    matrix: dict[str, Any]
    outputs: dict[str, Any]
    evals: dict[str, Any]
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class QuantizationSource:
    family: str | None
    variant: str | None
    model_id: str
    served_model_name: str
    local_path: Path | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def load_quantization_config(path: Path) -> QuantizationConfig:
    raw = load_yaml(path)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"{display_path(path)} must use schema_version {CONFIG_SCHEMA_VERSION}")
    return QuantizationConfig(
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        method=str(raw.get("method") or "nvfp4_runtime"),
        backend=str(raw.get("backend") or "vllm_blackwell"),
        objective=str(raw.get("objective") or "quantized_quality_retention"),
        family=str(raw.get("family")) if raw.get("family") else None,
        source_variant=str(raw.get("source_variant")) if raw.get("source_variant") else None,
        target_variant=str(raw.get("target_variant")) if raw.get("target_variant") else None,
        hardware_profile=str(raw.get("hardware_profile")) if raw.get("hardware_profile") else None,
        calibration=dict(raw.get("calibration") or {}),
        exclusions=dict(raw.get("exclusions") or {}),
        runtime=dict(raw.get("runtime") or {}),
        export=dict(raw.get("export") or {}),
        matrix=dict(raw.get("matrix") or {}),
        outputs=dict(raw.get("outputs") or {}),
        evals=dict(raw.get("evals") or {}),
        raw_config=raw,
    )


def load_family(name: str) -> dict[str, Any]:
    path = REPO_DIR / "configs" / "model_families" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown family {name!r}; expected {display_path(path)}")
    return load_yaml(path)


def models_dir(family_config: Mapping[str, Any], env: Mapping[str, str]) -> Path:
    env_name = str(family_config.get("models_dir_env") or "MODEL_FORGE_MODELS_DIR")
    raw = env.get(env_name) or family_config.get("default_models_dir") or "~/models"
    return Path(str(raw)).expanduser()


def resolve_family_variant(family: str, variant: str, env: Mapping[str, str]) -> QuantizationSource:
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    if variant not in variants:
        raise ValueError(f"unknown variant {variant!r} for family {family!r}; valid: {', '.join(sorted(variants))}")
    raw_variant = variants[variant]
    local_dir = str(raw_variant.get("merged_local_dir") or raw_variant.get("local_dir") or "")
    local_path = None
    if local_dir:
        local_path = Path(local_dir).expanduser()
        if not local_path.is_absolute():
            local_path = models_dir(family_config, env) / local_path
    model_id = str(raw_variant.get("repo_id") or raw_variant.get("served_model_name") or local_dir)
    served_model_name = str(raw_variant.get("served_model_name") or model_id)
    if not model_id:
        raise ValueError(f"variant {family}/{variant} must define repo_id, served_model_name, or local_dir")
    return QuantizationSource(
        family=family,
        variant=variant,
        model_id=model_id,
        served_model_name=served_model_name,
        local_path=local_path,
    )


def resolve_source(config: QuantizationConfig, family: str | None, variant: str | None, env: Mapping[str, str]) -> QuantizationSource:
    resolved_family = family or config.family
    resolved_variant = variant or config.source_variant
    if resolved_family and resolved_variant:
        return resolve_family_variant(resolved_family, resolved_variant, env)

    model_id = str(config.runtime.get("model_id") or config.raw_config.get("source_model_id") or "")
    if not model_id:
        raise ValueError("quantize plan needs either family+variant or runtime.model_id")
    return QuantizationSource(
        family=resolved_family,
        variant=resolved_variant,
        model_id=model_id,
        served_model_name=str(config.runtime.get("served_model_name") or model_id),
        local_path=Path(str(config.runtime["local_path"])).expanduser() if config.runtime.get("local_path") else None,
    )


def comma_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def render_template(value: str, context: Mapping[str, Any]) -> str:
    try:
        return value.format(**context)
    except KeyError as exc:
        raise ValueError(f"unknown template key {exc.args[0]!r} in {value!r}") from exc


def plan_run_id(config: QuantizationConfig, source: QuantizationSource, now: datetime | None = None) -> str:
    now = now or utc_now()
    source_name = source.family or source.model_id.replace("/", "_")
    source_variant = source.variant or "runtime"
    raw = "_".join([config.name, source_name, source_variant, config.method, config.backend, now.strftime("%Y%m%dT%H%M%SZ")])
    return sanitize_run_id(raw)


def method_notes(method: str, backend: str) -> list[str]:
    notes = []
    if method == "nvfp4_runtime":
        notes.append("This imports or serves an already-quantized NVFP4 checkpoint; it does not create a new checkpoint.")
        notes.append("Promotion still requires source-vs-candidate serving, sampled quality, and behavior-preservation evidence.")
    elif method == "nvfp4":
        notes.append("NVFP4 checkpoint creation must record calibration data, sensitive-module handling, backend versions, and export format.")
        notes.append("Promotion requires comparing each quantized variant against the same unquantized source variant.")
    elif method == "fp8_runtime":
        notes.append("Runtime FP8 is a serving configuration; compare it against an otherwise identical BF16/auto endpoint.")
    else:
        notes.append("Custom quantization method; treat generated commands as a reproducibility contract.")
    if "blackwell" in backend:
        notes.append("Blackwell/Spark runs should prefer native FP4-capable containers and record the actual vLLM backend that loaded.")
    return notes


def host_models_root(env: Mapping[str, str]) -> Path:
    return Path(env.get("MODEL_FORGE_MODELS_DIR", "~/models")).expanduser()


def host_hf_cache(env: Mapping[str, str]) -> Path:
    return Path(env.get("HF_HOME", "~/cache/huggingface")).expanduser()


def container_model_path(source: QuantizationSource, env: Mapping[str, str]) -> str:
    if not source.local_path:
        return source.model_id
    models_root = host_models_root(env).resolve()
    local_path = source.local_path.expanduser().resolve()
    try:
        relative = local_path.relative_to(models_root)
    except ValueError:
        return str(local_path)
    return "/models/" + str(relative)


def modelopt_export_path(config: QuantizationConfig, source: QuantizationSource, output_dir: Path) -> Path:
    explicit = config.export.get("output_subdir")
    target = config.target_variant or f"{source.variant or 'runtime'}_{config.method}_{config.backend}"
    subdir = render_template(
        str(explicit or target),
        {
            "source_variant": source.variant or "runtime",
            "target_variant": target,
            "method": config.method,
            "backend": config.backend,
        },
    )
    return output_dir / sanitize_run_id(subdir)


def build_modelopt_export_command(
    config: QuantizationConfig,
    source: QuantizationSource,
    *,
    output_dir: Path,
    run_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    if config.backend != "modelopt":
        raise ValueError(f"export currently supports backend=modelopt, got {config.backend!r}")
    if config.method not in {"nvfp4", "fp8", "fp8_w8a8"}:
        raise ValueError(f"export currently supports nvfp4/fp8 ModelOpt recipes, got {config.method!r}")

    export = dict(config.export)
    docker = dict(export.get("docker") or {})
    ptq = dict(export.get("ptq") or {})
    dataset = env.get("MODEL_FORGE_QUANT_CALIB_DATASET") or str(
        ptq.get("dataset") or config.calibration.get("dataset") or "cnn_dailymail,wikipedia"
    )
    calib_size = env.get("MODEL_FORGE_QUANT_CALIB_SIZE") or str(ptq.get("calib_size") or config.calibration.get("samples") or "64,64")
    calib_seq = env.get("MODEL_FORGE_QUANT_CALIB_SEQ") or str(ptq.get("calib_seq") or config.calibration.get("seq_len") or "2048")
    output_path = modelopt_export_path(config, source, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_root = host_models_root(env)
    hf_cache = host_hf_cache(env)
    container_source = container_model_path(source, env)
    container_output_root = "/workspace/output_models"
    container_output = f"{container_output_root}/{output_path.name}"
    image = str(export.get("image") or "model-forge-modelopt-nvfp4:0.43.0")
    container_name = sanitize_run_id(f"model_forge_quantize_{run_id}")
    qformat = str(ptq.get("qformat") or ("fp8" if config.method.startswith("fp8") else "nvfp4"))
    batch_size = int(ptq.get("batch_size") or config.calibration.get("batch_size") or 1)
    tensor_parallel = int(ptq.get("tensor_parallel") or 1)
    pipeline_parallel = int(ptq.get("pipeline_parallel") or 1)
    gpu_max_mem_percentage = float(ptq.get("gpu_max_mem_percentage") or 0.7)
    kv_cache_qformat = str(ptq.get("kv_cache_qformat") or "fp8_cast")

    hf_ptq = [
        "python3",
        "/opt/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py",
        "--pyt_ckpt_path",
        container_source,
        "--export_path",
        container_output,
        "--sparsity_fmt",
        str(ptq.get("sparsity_fmt") or "dense"),
        "--qformat",
        qformat,
        "--calib_size",
        calib_size,
        "--batch_size",
        str(batch_size),
        "--calib_seq",
        calib_seq,
        "--dataset",
        dataset,
        "--inference_tensor_parallel",
        str(tensor_parallel),
        "--inference_pipeline_parallel",
        str(pipeline_parallel),
        "--gpu_max_mem_percentage",
        str(gpu_max_mem_percentage),
        "--kv_cache_qformat",
        kv_cache_qformat,
        "--skip_generate",
    ]
    if bool(ptq.get("trust_remote_code", True)):
        hf_ptq.append("--trust_remote_code")
    if bool(ptq.get("low_memory_mode", True)):
        hf_ptq.append("--low_memory_mode")
    if bool(ptq.get("use_seq_device_map", True)):
        hf_ptq.append("--use_seq_device_map")
    if ptq.get("moe_calib_experts_ratio") is not None:
        hf_ptq.extend(["--moe_calib_experts_ratio", str(ptq["moe_calib_experts_ratio"])])
    if ptq.get("attn_implementation"):
        hf_ptq.extend(["--attn_implementation", str(ptq["attn_implementation"])])
    if not bool(ptq.get("verbose", True)):
        hf_ptq.append("--no-verbose")

    docker_run = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        str(docker.get("gpus") or "all"),
        "--ipc=host",
        "--name",
        container_name,
        "--cpus",
        str(docker.get("cpus") or 8),
        "--memory",
        f"{docker.get('memory_gb', 100)}g",
        "--memory-swap",
        f"{docker.get('memory_swap_gb', docker.get('memory_gb', 100))}g",
        "--shm-size",
        f"{docker.get('shm_size_gb', 32)}g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-v",
        f"{models_root}:/models:ro",
        "-v",
        f"{hf_cache}:/root/.cache/huggingface",
        "-v",
        f"{output_path.parent}:{container_output_root}",
        "-e",
        "HF_TOKEN",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
        "-e",
        f"OMP_NUM_THREADS={docker.get('omp_num_threads', docker.get('cpus', 8))}",
        image,
        *hf_ptq,
    ]
    shell_command = ["nice", "-n", str(docker.get("nice", 10)), *docker_run]
    systemd_scope = dict(export.get("systemd_scope") or {})
    if truthy(systemd_scope.get("enabled"), True):
        scope_command = ["systemd-run", "--scope"]
        if truthy(systemd_scope.get("user"), False):
            scope_command = ["systemd-run", "--user", "--scope"]
        shell_command = [
            *scope_command,
            "-p",
            f"CPUQuota={systemd_scope.get('CPUQuota', '80%')}",
            "-p",
            f"MemoryMax={systemd_scope.get('MemoryMax', '85%')}",
            "-p",
            f"IOWeight={systemd_scope.get('IOWeight', 100)}",
            *shell_command,
        ]
    return {
        "schema_version": "model_forge.quantization_export.v1",
        "created_at": utc_now().isoformat(),
        "run_id": run_id,
        "source": {
            "family": source.family,
            "variant": source.variant,
            "model_id": source.model_id,
            "local_path": display_path(source.local_path) if source.local_path else None,
            "container_path": container_source,
        },
        "target": {
            "variant": config.target_variant or f"{source.variant or 'runtime'}_{config.method}_{config.backend}",
            "host_output_path": display_path(output_path),
            "container_output_path": container_output,
            "served_model_name": str(config.runtime.get("served_model_name") or f"{source.served_model_name}-{config.method}"),
        },
        "method": config.method,
        "backend": config.backend,
        "image": image,
        "calibration": {
            "dataset": dataset,
            "calib_size": calib_size,
            "calib_seq": calib_seq,
            "batch_size": batch_size,
            "kv_cache_qformat": kv_cache_qformat,
            "moe_calib_experts_ratio": ptq.get("moe_calib_experts_ratio"),
        },
        "resource_policy": {
            "start_if_memory_available_above_fraction": float(export.get("start_if_memory_available_above_fraction", 0.05)),
            "stop_if_memory_available_below_fraction": float(export.get("stop_if_memory_available_below_fraction", 0.10)),
            "watchdog_poll_seconds": float(export.get("watchdog_poll_seconds", 5)),
            "require_disk_free_fraction": float(export.get("require_disk_free_fraction", 0.15)),
            "lock_path": display_path(REPO_DIR / "reports" / "generated" / ".locks" / "quantization_export.lock"),
            "systemd_scope": {
                "enabled": truthy(systemd_scope.get("enabled"), True),
                "user": truthy(systemd_scope.get("user"), False),
                "CPUQuota": str(systemd_scope.get("CPUQuota", "80%")),
                "MemoryMax": str(systemd_scope.get("MemoryMax", "85%")),
                "IOWeight": int(systemd_scope.get("IOWeight", 100)),
            },
            "docker": redact_value(docker),
        },
        "command": shell_command,
        "command_display": shlex.join(shell_command),
    }


def guard_export(export_plan: Mapping[str, Any]) -> None:
    policy = export_plan.get("resource_policy") or {}
    memory_floor = float(policy.get("start_if_memory_available_above_fraction", 0.05))
    disk_floor = float(policy.get("require_disk_free_fraction", 0.15))
    mem = psutil.virtual_memory()
    if mem.available / mem.total < memory_floor:
        raise RuntimeError(
            f"not enough free memory to start quantization: available={mem.available / mem.total:.3f}, floor={memory_floor:.3f}"
        )
    output_path = resolve_repo_path((export_plan.get("target") or {})["host_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_path.parent)
    if disk.free / disk.total < disk_floor:
        raise RuntimeError(f"not enough free disk for quantization: free={disk.free / disk.total:.3f}, floor={disk_floor:.3f}")


def memory_available_fraction() -> float:
    mem = psutil.virtual_memory()
    return float(mem.available / mem.total)


def stop_export_container(export_plan: Mapping[str, Any]) -> None:
    command = [str(item) for item in export_plan.get("command") or []]
    try:
        name_index = command.index("--name") + 1
        container_name = command[name_index]
    except (ValueError, IndexError):
        return
    subprocess.run(["docker", "stop", "--time", "15", container_name], cwd=REPO_DIR, check=False)


def terminate_process(process: subprocess.Popen[Any], export_plan: Mapping[str, Any]) -> None:
    stop_export_container(export_plan)
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


@contextmanager
def export_execution_lock(export_plan: Mapping[str, Any]) -> Any:
    policy = export_plan.get("resource_policy") or {}
    lock_path = resolve_repo_path(policy.get("lock_path") or "reports/generated/.locks/quantization_export.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another quantization export is already running; lock={display_path(lock_path)}") from exc
        handle.write(f"pid={os.getpid()}\nrun_id={export_plan.get('run_id')}\nstarted_at={utc_now().isoformat()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_export_plan(export_plan: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "quantization_export_plan.json"
    path.write_text(json.dumps(redact_value(export_plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def execute_export(export_plan: Mapping[str, Any]) -> int:
    with export_execution_lock(export_plan):
        guard_export(export_plan)
        command = [str(item) for item in export_plan["command"]]
        policy = export_plan.get("resource_policy") or {}
        stop_floor = float(policy.get("stop_if_memory_available_below_fraction", policy.get("start_if_memory_available_above_fraction", 0.10)))
        poll_seconds = float(policy.get("watchdog_poll_seconds", 5))
        process = subprocess.Popen(command, cwd=REPO_DIR)
        while True:
            return_code = process.poll()
            if return_code is not None:
                return int(return_code)
            available_fraction = memory_available_fraction()
            if available_fraction < stop_floor:
                console.print(
                    f"[red]quantization watchdog stopping job: available memory {available_fraction:.4f} below floor {stop_floor:.4f}[/red]"
                )
                terminate_process(process, export_plan)
                return 137
            time.sleep(poll_seconds)


def build_runtime_command(config: QuantizationConfig, source: QuantizationSource) -> list[str]:
    runtime = dict(config.runtime)
    launch = dict(runtime.get("launch") or {})
    spark_dir_env = str(launch.get("spark_vllm_dir_env", "MODEL_FORGE_SPARK_VLLM_DOCKER"))
    nodes_env = str(launch.get("nodes_env", "MODEL_FORGE_SPARK_CLUSTER_NODES"))
    spark_dir = "${" + spark_dir_env + ":?/path/to/spark-vllm-docker}"
    nodes = "${" + nodes_env + ":?comma-separated Spark nodes}"
    container = str(launch.get("container") or runtime.get("container") or "vllm-node-tf5")
    name = str(launch.get("container_name") or f"model_forge_{sanitize_run_id(config.name)}")
    port = int(runtime.get("port", 8000))
    gpu_memory = float(runtime.get("gpu_memory_utilization", 0.7))
    max_model_len = int(runtime.get("max_model_len", 8192))
    tensor_parallel = int(runtime.get("tensor_parallel", 2))
    max_num_seqs = int(runtime.get("max_num_seqs", 2))
    max_num_batched_tokens = int(runtime.get("max_num_batched_tokens", 4096))
    no_ray = bool(launch.get("no_ray", False))
    non_privileged = bool(launch.get("non_privileged", True))

    launcher = [
        "./launch-cluster.sh",
        "-t",
        container,
        "-n",
        nodes,
        "-d",
        "--name",
        name,
    ]
    if no_ray:
        launcher.append("--no-ray")
    if non_privileged:
        launcher.extend(
            [
                "--non-privileged",
                "--mem-limit-gb",
                str(launch.get("mem_limit_gb", 110)),
                "--mem-swap-limit-gb",
                str(launch.get("mem_swap_limit_gb", 120)),
                "--shm-size-gb",
                str(launch.get("shm_size_gb", 32)),
            ]
        )
    for item in launch.get("extra_launcher_args") or []:
        launcher.append(str(item))

    serve = [
        "exec",
        "vllm",
        "serve",
        source.model_id,
        "--served-model-name",
        str(runtime.get("served_model_name") or source.served_model_name),
        "--host",
        str(runtime.get("host", "0.0.0.0")),
        "--port",
        str(port),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(gpu_memory),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--tensor-parallel-size",
        str(tensor_parallel),
    ]
    for key, value in (runtime.get("vllm_flags") or {}).items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                serve.append(flag)
        else:
            serve.extend([flag, str(value)])

    return ["cd", spark_dir, "&&", *launcher, *serve]


def build_plan(
    config: QuantizationConfig,
    *,
    config_path: Path,
    family: str | None,
    variant: str | None,
    output_dir: str | Path | None,
    run_id: str | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    source = resolve_source(config, family, variant, env)
    actual_run_id = run_id or plan_run_id(config, source)
    output_root = resolve_repo_path(output_dir or config.outputs.get("reports_dir") or "reports/generated/quantization")
    plan_output_dir = output_root / actual_run_id
    profile = detect_hardware_profile(env)
    quant_env = recommended_quantization_env(env)
    exclusions = dict(config.exclusions)
    keep_patterns = comma_list(quant_env.get("MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS"))
    if keep_patterns:
        modules = list(exclusions.get("modules") or [])
        modules.extend(pattern for pattern in keep_patterns if pattern not in modules)
        exclusions["modules"] = modules

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "run_id": actual_run_id,
        "dry_run_plan_only": True,
        "config": display_path(config_path),
        "name": config.name,
        "description": config.description,
        "objective": config.objective,
        "source": {
            "family": source.family,
            "variant": source.variant,
            "model_id": source.model_id,
            "served_model_name": source.served_model_name,
            "local_path": display_path(source.local_path) if source.local_path else None,
            "local_path_exists": source.local_path.exists() if source.local_path else None,
        },
        "target": {
            "variant": config.target_variant or f"{source.variant or 'runtime'}_{config.method}",
            "served_model_name": str(config.runtime.get("served_model_name") or source.served_model_name),
            "checkpoint_written_by_this_plan": config.method not in {"nvfp4_runtime", "fp8_runtime"},
        },
        "quantization": {
            "method": config.method,
            "backend": config.backend,
            "calibration": config.calibration,
            "exclusions": exclusions,
            "hardware_profile": config.hardware_profile or profile.name,
            "recommended_env": quant_env,
            "notes": method_notes(config.method, config.backend),
        },
        "runtime": redact_value(config.runtime),
        "hardware": {
            "profile": profile.name,
            "label": profile.label,
            "gpus": [{"name": gpu.name, "memory_total_mb": gpu.memory_total_mb} for gpu in profile.gpus],
            "notes": list(profile.notes),
        },
        "resource_policy": {
            "max_concurrent_large_jobs": 1,
            "require_job_lock": True,
            "start_if_memory_available_above_fraction": 0.05,
            "stop_if_memory_available_below_fraction": 0.05,
            "require_disk_free_fraction": 0.15,
            "systemd_scope": {"CPUQuota": "80%", "MemoryMax": "85%", "IOWeight": 100, "nice": 10},
        },
        "outputs": {
            "output_dir": display_path(plan_output_dir),
            "plan_json": "quantization_plan.json",
            "plan_md": "quantization_plan.md",
        },
        "launch_command": build_runtime_command(config, source) if config.runtime else [],
        "validation_gates": list(config.evals.get("required") or []),
        "execution_contract": {
            "starts_heavy_job": False,
            "loads_model": False,
            "writes_checkpoint": config.method not in {"nvfp4_runtime", "fp8_runtime"},
            "notes": [
                "Plan generation does not load a model.",
                "Run one large model server at a time and stop it before switching candidates.",
                "A quantization feature is complete only after the candidate endpoint has serving and behavior evidence.",
            ],
        },
    }
    return plan


def write_plan_card(path: Path, plan: Mapping[str, Any]) -> None:
    quant = plan.get("quantization") or {}
    source = plan.get("source") or {}
    target = plan.get("target") or {}
    launch = " ".join(str(part) for part in plan.get("launch_command") or [])
    lines = [
        f"# Quantization Plan: {plan.get('name')}",
        "",
        "## Identity",
        "",
        f"- Objective: `{plan.get('objective')}`",
        f"- Source model: `{source.get('model_id')}`",
        f"- Source family/variant: `{source.get('family') or 'n/a'}` / `{source.get('variant') or 'n/a'}`",
        f"- Target variant: `{target.get('variant')}`",
        f"- Method: `{quant.get('method')}`",
        f"- Backend: `{quant.get('backend')}`",
        f"- Output directory: `{(plan.get('outputs') or {}).get('output_dir')}`",
        "",
        "## Launch Command",
        "",
        "```bash",
        launch or "# no runtime command configured",
        "```",
        "",
        "## Calibration",
        "",
        f"- Dataset: `{(quant.get('calibration') or {}).get('dataset') or 'n/a'}`",
        f"- Samples: `{(quant.get('calibration') or {}).get('samples') or 'n/a'}`",
        f"- Sequence length: `{(quant.get('calibration') or {}).get('seq_len') or 'n/a'}`",
        "",
        "## Higher-Precision Keeps",
        "",
        f"- `{', '.join((quant.get('exclusions') or {}).get('modules') or []) or 'n/a'}`",
        "",
        "## Required Evidence",
        "",
    ]
    lines.extend(f"- `{item}`" for item in plan.get("validation_gates") or [])
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in quant.get("notes") or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_outputs(plan: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str((plan.get("outputs") or {})["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantization_plan.json").write_text(
        json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plan_card(output_dir / "quantization_plan.md", plan)
    return output_dir


def stats_value(summary: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = summary
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, (int, float)) else None


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {display_path(path)}")
    return data


def scores_csv_path(path: Path) -> Path:
    if path.is_dir():
        return path / "scores.csv"
    return path


def load_scores(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    with scores_csv_path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bucket = str(row["bucket"])
            metric = str(row["metric"])
            value = row.get("value")
            scores.setdefault(bucket, {})[metric] = {
                "value": float(value) if value not in {None, ""} else None,
                "count": int(float(row["count"])) if row.get("count") else None,
                "ci_low": float(row["ci_low"]) if row.get("ci_low") else None,
                "ci_high": float(row["ci_high"]) if row.get("ci_high") else None,
            }
    return scores


DEFAULT_SCORE_METRICS = [
    ("normal_use_regression", "normal_use_regression_pass_rate"),
    ("capability_preservation_challenge", "normal_use_regression_pass_rate"),
    ("agentic_tool_use_json", "schema_adherence"),
    ("agentic_tool_use_json", "workflow_success"),
    ("refusal_paired_boundary", "benign_answer_quality_rate"),
    ("refusal_paired_boundary", "benign_refusal_rate"),
    ("refusal_paired_boundary", "unsafe_overcompliance_rate"),
    ("unsafe_overcompliance", "unsafe_overcompliance_rate"),
    ("overall", "latency_seconds_median"),
]


def metric_pairs(config: QuantizationConfig) -> list[tuple[str, str]]:
    raw = config.evals.get("card_metrics") or []
    pairs = []
    for item in raw:
        if isinstance(item, str) and "." in item:
            bucket, metric = item.split(".", 1)
            pairs.append((bucket, metric))
        elif isinstance(item, Mapping):
            pairs.append((str(item["bucket"]), str(item["metric"])))
    return pairs or DEFAULT_SCORE_METRICS


def score_value(scores: Mapping[str, Mapping[str, Mapping[str, Any]]], bucket: str, metric: str) -> float | None:
    raw = ((scores.get(bucket) or {}).get(metric) or {}).get("value")
    return float(raw) if isinstance(raw, (int, float)) else None


def build_card(
    config: QuantizationConfig,
    *,
    config_path: Path,
    source_serving_summary: Path | None,
    candidate_serving_summary: Path,
    source_serving_eval: Path | None,
    candidate_serving_eval: Path | None,
    output_dir: Path,
    run_id: str,
    candidate_only_smoke: bool = False,
) -> dict[str, Any]:
    source_summary = load_json(source_serving_summary) if source_serving_summary else {}
    candidate_summary = load_json(candidate_serving_summary)
    source_scores = load_scores(source_serving_eval) if source_serving_eval else {}
    candidate_scores = load_scores(candidate_serving_eval) if candidate_serving_eval else {}
    serving_metrics = {
        "success_rate": ("success_rate",),
        "throughput_req_per_s": ("request_throughput_per_second_serial_estimate",),
        "total_latency_p50": ("metrics", "total_latency_seconds", "p50"),
        "total_latency_p95": ("metrics", "total_latency_seconds", "p95"),
        "ttft_p50": ("metrics", "time_to_first_chunk_seconds", "p50"),
        "ttft_p95": ("metrics", "time_to_first_chunk_seconds", "p95"),
        "system_available_fraction_min": ("memory", "system", "available_fraction", "min"),
        "gpu_utilization_p50": ("memory", "gpu", "utilization_gpu_percent", "p50"),
    }
    serving = {}
    for name, path in serving_metrics.items():
        source_value = stats_value(source_summary, path)
        candidate_value = stats_value(candidate_summary, path)
        serving[name] = {
            "source": source_value,
            "candidate": candidate_value,
            "delta": None if source_value is None or candidate_value is None else round(candidate_value - source_value, 6),
        }

    sampled = {}
    for bucket, metric in metric_pairs(config):
        source_value = score_value(source_scores, bucket, metric)
        candidate_value = score_value(candidate_scores, bucket, metric)
        sampled[f"{bucket}.{metric}"] = {
            "source": source_value,
            "candidate": candidate_value,
            "delta": None if source_value is None or candidate_value is None else round(candidate_value - source_value, 6),
            "source_count": ((source_scores.get(bucket) or {}).get(metric) or {}).get("count"),
            "candidate_count": ((candidate_scores.get(bucket) or {}).get(metric) or {}).get("count"),
        }

    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "run_id": run_id,
        "config": display_path(config_path),
        "objective": config.objective,
        "method": config.method,
        "backend": config.backend,
        "candidate_only_smoke": candidate_only_smoke,
        "source": {
            "model": source_summary.get("model"),
            "serving_summary": display_path(source_serving_summary) if source_serving_summary else None,
            "serving_eval": display_path(source_serving_eval) if source_serving_eval else None,
        },
        "candidate": {
            "model": candidate_summary.get("model"),
            "serving_summary": display_path(candidate_serving_summary),
            "serving_eval": display_path(candidate_serving_eval) if candidate_serving_eval else None,
        },
        "serving_deltas": serving,
        "sampled_eval_deltas": sampled,
        "output_dir": display_path(output_dir),
        "notes": [
            "Quantization cards compare evidence from already completed runs; they do not start a server.",
            "Use matching workload configs, sampling, hardware, and model source when interpreting deltas.",
            "Candidate-only smoke cards prove loader and endpoint viability only; they do not prove quantized quality retention.",
            "For ablated models, lower harmful-refusal rates may be objective progress only if capability and benign quality hold.",
        ],
    }


def write_card_markdown(path: Path, card: Mapping[str, Any]) -> None:
    source = card.get("source") or {}
    candidate = card.get("candidate") or {}
    lines = [
        f"# Quantization Card: {card.get('run_id')}",
        "",
        "## Identity",
        "",
        f"- Objective: `{card.get('objective')}`",
        f"- Method: `{card.get('method')}`",
        f"- Backend: `{card.get('backend')}`",
        f"- Candidate-only smoke: `{card.get('candidate_only_smoke')}`",
        f"- Source model: `{source.get('model')}`",
        f"- Candidate model: `{candidate.get('model')}`",
        "",
        "## Serving Deltas",
        "",
        "| Metric | Source | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in (card.get("serving_deltas") or {}).items():
        lines.append(f"| {metric} | {values.get('source')} | {values.get('candidate')} | {values.get('delta')} |")
    lines.extend(["", "## Sampled Eval Deltas", "", "| Metric | Source | Candidate | Delta | n source | n candidate |", "|---|---:|---:|---:|---:|---:|"])
    for metric, values in (card.get("sampled_eval_deltas") or {}).items():
        lines.append(
            f"| {metric} | {values.get('source')} | {values.get('candidate')} | {values.get('delta')} | "
            f"{values.get('source_count')} | {values.get('candidate_count')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in card.get("notes") or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_card_outputs(card: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(card["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantization_card.json").write_text(
        json.dumps(redact_value(card), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_card_markdown(output_dir / "quantization_card.md", card)
    return output_dir


def render_plan(plan: Mapping[str, Any]) -> None:
    source = plan.get("source") or {}
    target = plan.get("target") or {}
    quant = plan.get("quantization") or {}
    table = Table(title="Quantization Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in [
        ("source_model", source.get("model_id")),
        ("target_variant", target.get("variant")),
        ("method", quant.get("method")),
        ("backend", quant.get("backend")),
        ("hardware", (plan.get("hardware") or {}).get("label")),
        ("output_dir", (plan.get("outputs") or {}).get("output_dir")),
    ]:
        table.add_row(key, str(value))
    console.print(table)


def render_card(card: Mapping[str, Any]) -> None:
    table = Table(title="Quantization Card")
    table.add_column("Metric")
    table.add_column("Source")
    table.add_column("Candidate")
    table.add_column("Delta")
    for metric, values in (card.get("serving_deltas") or {}).items():
        table.add_row(metric, str(values.get("source")), str(values.get("candidate")), str(values.get("delta")))
    console.print(table)


def add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("family", nargs="?", help="Optional model family")
    parser.add_argument("variant", nargs="?", help="Optional source variant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Plan artifact output root")
    parser.add_argument("--run-id", help="Stable output subdirectory name")
    parser.add_argument("--write-plan", action="store_true", help="Write quantization_plan.json and quantization_plan.md")
    parser.add_argument("--json", action="store_true", help="Print JSON plan")


def add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("family", nargs="?", help="Model family; defaults to config family")
    parser.add_argument("variant", nargs="?", help="Source variant; defaults to config source_variant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target-variant", help="Target variant label for export metadata")
    parser.add_argument("--output-dir", type=Path, help="Quantized checkpoint output root")
    parser.add_argument("--run-id", help="Stable export run id")
    parser.add_argument("--write-plan", action="store_true", help="Write quantization_export_plan.json")
    parser.add_argument("--execute", action="store_true", help="Run the generated export command")
    parser.add_argument("--json", action="store_true", help="Print JSON export plan")


def matrix_entries(config: QuantizationConfig) -> list[dict[str, Any]]:
    entries = config.matrix.get("variants") or []
    if not isinstance(entries, list):
        raise ValueError("matrix.variants must be a list")
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def filter_matrix_entries(entries: list[dict[str, Any]], variants: str | None) -> list[dict[str, Any]]:
    wanted = set(comma_list(variants))
    if not wanted:
        return entries
    filtered = [entry for entry in entries if str(entry.get("source_variant") or "") in wanted]
    missing = sorted(wanted - {str(entry.get("source_variant") or "") for entry in filtered})
    if missing:
        raise ValueError(f"matrix variants not found: {', '.join(missing)}")
    return filtered


def matrix_workers(config: QuantizationConfig, env: Mapping[str, str] | None = None) -> list[str]:
    env = env or os.environ
    env_name = str(config.matrix.get("workers_env") or "MODEL_FORGE_QUANT_WORKERS")
    raw = env.get(env_name) or config.matrix.get("workers") or "local"
    workers = comma_list(str(raw))
    return workers or ["local"]


def annotate_matrix_worker(plan: dict[str, Any], worker: str, index: int) -> dict[str, Any]:
    plan["execution"] = {
        "worker": worker,
        "worker_index": index,
        "runner": "local" if worker == "local" else "ssh",
        "requires_paths_on_worker": [
            str((plan.get("source") or {}).get("local_path")),
            str(Path(str((plan.get("target") or {}).get("host_output_path", ""))).parent),
            str(host_hf_cache(os.environ)),
        ],
        "notes": [
            "ModelOpt export is one heavy process per worker.",
            "Use MODEL_FORGE_QUANT_WORKERS=local,<ssh-host> to distribute variant exports across a Spark cluster.",
            "Do not run multiple exports on the same Spark node.",
        ],
    }
    if worker != "local":
        remote_workdir = os.environ.get("MODEL_FORGE_REMOTE_WORKDIR", str(REPO_DIR))
        plan["execution"]["remote_workdir"] = remote_workdir
        plan["execution"]["remote_command_display"] = f"ssh {shlex.quote(worker)} {shlex.quote('cd ' + remote_workdir + ' && ' + plan['command_display'])}"
    return plan


def matrix_run_id(config: QuantizationConfig, entry: Mapping[str, Any]) -> str:
    source_variant = str(entry.get("source_variant") or config.source_variant or "runtime")
    target_variant = str(entry.get("target_variant") or f"{source_variant}_{config.method}_{config.backend}")
    return sanitize_run_id(f"{config.name}_{target_variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and report model quantization workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Resolve a quantization runtime or checkpoint-creation plan")
    add_plan_args(plan_parser)

    export_parser = subparsers.add_parser("export", help="Export a self-quantized checkpoint from a source variant")
    add_export_args(export_parser)

    matrix_parser = subparsers.add_parser("matrix-plan", help="Print export plans for all variants in config matrix")
    matrix_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix_parser.add_argument("--variants", help="Comma-separated source variants to include from the matrix")
    matrix_parser.add_argument("--output-dir", type=Path, help="Quantized checkpoint output root")
    matrix_parser.add_argument("--write-plan", action="store_true", help="Write matrix export plans under the output root")
    matrix_parser.add_argument("--json", action="store_true", help="Print JSON matrix")

    card_parser = subparsers.add_parser("card", help="Write a source-vs-quantized serving and sampled-eval report")
    card_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    card_parser.add_argument("--source-serving-summary", type=Path)
    card_parser.add_argument("--candidate-serving-summary", type=Path, required=True)
    card_parser.add_argument("--source-serving-eval", type=Path)
    card_parser.add_argument("--candidate-serving-eval", type=Path)
    card_parser.add_argument(
        "--candidate-only-smoke",
        action="store_true",
        help="Allow a candidate-only loader/serving smoke card when no matching source baseline exists",
    )
    card_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    card_parser.add_argument("--run-id", required=True)
    card_parser.add_argument("--write-card", action="store_true", help="Write quantization_card.json and quantization_card.md")
    card_parser.add_argument("--json", action="store_true", help="Print JSON card")

    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_quantization_config(config_path)

    if args.command == "plan":
        plan = build_plan(
            config,
            config_path=config_path,
            family=args.family,
            variant=args.variant,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        if args.write_plan:
            write_plan_outputs(plan)
        if args.json:
            print(json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
        return

    if args.command == "export":
        source = resolve_source(config, args.family, args.variant, os.environ)
        actual_run_id = sanitize_run_id(args.run_id or f"{config.name}_{args.target_variant or source.variant or 'runtime'}")
        output_root = resolve_repo_path(args.output_dir or config.export.get("output_root") or config.outputs.get("models_dir") or "~/models/model-forge-quantized")
        export_config = config
        if args.target_variant:
            export_config = QuantizationConfig(
                name=config.name,
                description=config.description,
                method=config.method,
                backend=config.backend,
                objective=config.objective,
                family=config.family,
                source_variant=config.source_variant,
                target_variant=sanitize_run_id(args.target_variant),
                hardware_profile=config.hardware_profile,
                calibration=config.calibration,
                exclusions=config.exclusions,
                runtime=config.runtime,
                export=config.export,
                matrix=config.matrix,
                outputs=config.outputs,
                evals=config.evals,
                raw_config=config.raw_config,
            )
        export_plan = build_modelopt_export_command(
            export_config,
            source,
            output_dir=output_root,
            run_id=actual_run_id,
        )
        if args.write_plan:
            write_export_plan(export_plan, output_root / actual_run_id)
        if args.json:
            print(json.dumps(redact_value(export_plan), indent=2, sort_keys=True) + "\n")
        else:
            console.print(export_plan["command_display"])
        if args.execute:
            raise SystemExit(execute_export(export_plan))
        return

    if args.command == "matrix-plan":
        output_root = resolve_repo_path(args.output_dir or config.export.get("output_root") or config.outputs.get("models_dir") or "~/models/model-forge-quantized")
        plans = []
        workers = matrix_workers(config)
        for entry in filter_matrix_entries(matrix_entries(config), args.variants):
            index = len(plans)
            family = str(entry.get("family") or config.family or "")
            variant = str(entry.get("source_variant") or "")
            source = resolve_source(config, family or None, variant or None, os.environ)
            target_variant = str(entry.get("target_variant") or f"{source.variant}_{config.method}_{config.backend}")
            variant_config = QuantizationConfig(
                name=config.name,
                description=config.description,
                method=config.method,
                backend=config.backend,
                objective=config.objective,
                family=config.family,
                source_variant=source.variant,
                target_variant=target_variant,
                hardware_profile=config.hardware_profile,
                calibration=config.calibration,
                exclusions=config.exclusions,
                runtime=config.runtime,
                export={**config.export, **dict(entry.get("export") or {})},
                matrix=config.matrix,
                outputs=config.outputs,
                evals=config.evals,
                raw_config=config.raw_config,
            )
            run_id = matrix_run_id(config, {"source_variant": source.variant, "target_variant": target_variant})
            plan = build_modelopt_export_command(variant_config, source, output_dir=output_root, run_id=run_id)
            annotate_matrix_worker(plan, workers[index % len(workers)], index % len(workers))
            plan["baseline_eval"] = entry.get("baseline_eval")
            plan["baseline_artifact_eval"] = entry.get("baseline_artifact_eval")
            plans.append(plan)
            if args.write_plan:
                write_export_plan(plan, output_root / run_id)
        matrix = {
            "schema_version": "model_forge.quantization_matrix.v1",
            "created_at": utc_now().isoformat(),
            "config": display_path(config_path),
            "variant_count": len(plans),
            "workers": workers,
            "plans": plans,
        }
        if args.json:
            print(json.dumps(redact_value(matrix), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="Quantization Matrix")
            table.add_column("Source")
            table.add_column("Target")
            table.add_column("Worker")
            table.add_column("Output")
            for plan in plans:
                source = plan.get("source") or {}
                target = plan.get("target") or {}
                execution = plan.get("execution") or {}
                table.add_row(str(source.get("variant")), str(target.get("variant")), str(execution.get("worker")), str(target.get("host_output_path")))
            console.print(table)
        return

    if not args.source_serving_summary and not args.candidate_only_smoke:
        raise SystemExit("--source-serving-summary is required unless --candidate-only-smoke is set")
    if args.source_serving_eval and not args.source_serving_summary:
        raise SystemExit("--source-serving-eval requires --source-serving-summary")

    output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
    card = build_card(
        config,
        config_path=config_path,
        source_serving_summary=resolve_repo_path(args.source_serving_summary) if args.source_serving_summary else None,
        candidate_serving_summary=resolve_repo_path(args.candidate_serving_summary),
        source_serving_eval=resolve_repo_path(args.source_serving_eval) if args.source_serving_eval else None,
        candidate_serving_eval=resolve_repo_path(args.candidate_serving_eval) if args.candidate_serving_eval else None,
        output_dir=output_dir,
        run_id=sanitize_run_id(args.run_id),
        candidate_only_smoke=bool(args.candidate_only_smoke),
    )
    if args.write_card:
        write_card_outputs(card)
    if args.json:
        print(json.dumps(redact_value(card), indent=2, sort_keys=True) + "\n")
    else:
        render_card(card)


if __name__ == "__main__":
    main()
