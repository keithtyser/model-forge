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
from model_forge.variants.checkpoint_audit import build_checkpoint_audit
from model_forge.variants.tokenizer_audit import compare_records, live_round_trip, tokenizer_record


DEFAULT_CONFIG = REPO_DIR / "configs" / "quantization" / "nvfp4_blackwell_runtime.yaml"
CONFIG_SCHEMA_VERSION = "model_forge.quantization.v1"
PLAN_SCHEMA_VERSION = "model_forge.quantization_plan.v1"
CARD_SCHEMA_VERSION = "model_forge.quantization_card.v1"
CALIBRATION_MANIFEST_SCHEMA_VERSION = "model_forge.quantization_calibration_manifest.v1"
FP8_KV_REPORT_SCHEMA_VERSION = "model_forge.fp8_kv_behavior_report.v1"
BEHAVIOR_REPORT_SCHEMA_VERSION = "model_forge.quantization_behavior_preservation_report.v1"
TOKENIZER_REPORT_SCHEMA_VERSION = "model_forge.quantization_tokenizer_preservation_report.v1"
SENSITIVITY_REPORT_SCHEMA_VERSION = "model_forge.quantization_sensitivity_report.v1"
NVFP4_GATE_SCHEMA_VERSION = "model_forge.nvfp4_evidence_gate.v1"
MODELOPT_RUNTIME_COMPAT_SCHEMA_VERSION = "model_forge.modelopt_runtime_compat_report.v1"
VLLM_MODELOPT_QUANT_ALGOS = {
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
    "FP8_PB_WO",
    "NVFP4",
    "W4A16_NVFP4",
    "MXFP8",
    "MIXED_PRECISION",
}

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
    promotion: dict[str, Any] | None = None


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
        promotion=dict(raw_variant.get("promotion") or {}),
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
        promotion={},
    )


def promotion_blocks_action(promotion: Mapping[str, Any] | None, action: str) -> tuple[bool, str]:
    if not promotion:
        return False, ""
    decision = str(promotion.get("decision") or "").strip().lower()
    blocked_actions = {str(item).strip().lower() for item in promotion.get("blocked_actions") or []}
    blocked = decision == "rejected" or action in blocked_actions or "all" in blocked_actions
    if not blocked:
        return False, ""
    reason = str(promotion.get("reason") or f"promotion decision is {decision or 'blocked'}")
    evidence = promotion.get("evidence")
    evidence_note = ""
    if isinstance(evidence, list) and evidence:
        evidence_note = f"; evidence={', '.join(str(item) for item in evidence[:3])}"
    return True, f"{reason}{evidence_note}"


def source_promotion(source: QuantizationSource) -> dict[str, Any]:
    return dict(source.promotion or {})


def comma_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return comma_list(value)
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                items.extend(comma_list(item))
            elif item is not None:
                items.append(str(item))
        return items
    return [str(value)]


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


def target_variant_label(config: QuantizationConfig, source: QuantizationSource) -> str:
    raw = config.target_variant or f"{source.variant or 'runtime'}_{config.method}_{config.backend}"
    return sanitize_run_id(
        render_template(
            raw,
            {
                "source_variant": source.variant or "runtime",
                "source_family": source.family or "generic",
                "method": config.method,
                "backend": config.backend,
            },
        )
    )


def sanitize_relative_subpath(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"output subdir must be a safe relative path, got {value!r}")
    parts = [sanitize_run_id(part) for part in path.parts if part not in {"", "."}]
    return Path(*parts) if parts else Path("output")


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
    elif method == "fp8_w8a8":
        notes.append("FP8 W8A8 checkpoint creation must record calibration data, backend versions, exported loader format, and source deltas.")
        notes.append("Promotion requires source-vs-FP8 W8A8 serving and sampled behavior evidence for the same family/variant.")
    elif method.startswith("gguf"):
        notes.append("GGUF export must preserve tokenizer/chat-template metadata and prove llama.cpp load plus benchmark evidence.")
        notes.append("Promotion requires comparing the GGUF candidate against the same source variant with behavior and tokenizer reports.")
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
    target = target_variant_label(config, source)
    subdir = render_template(
        str(explicit or target),
        {
            "source_variant": source.variant or "runtime",
            "source_family": source.family or "generic",
            "target_variant": target,
            "method": config.method,
            "backend": config.backend,
        },
    )
    return output_dir / sanitize_relative_subpath(subdir)


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
    recipe = str(ptq.get("recipe") or "")
    strategy = str(ptq.get("strategy") or export.get("strategy") or "hf_ptq")
    batch_size = int(ptq.get("batch_size") or config.calibration.get("batch_size") or 1)
    tensor_parallel = int(ptq.get("tensor_parallel") or 1)
    pipeline_parallel = int(ptq.get("pipeline_parallel") or 1)
    gpu_max_mem_percentage = float(ptq.get("gpu_max_mem_percentage") or 0.7)
    kv_cache_qformat = str(ptq.get("kv_cache_qformat") or "fp8_cast")
    disable_patterns = string_list(ptq.get("disable_patterns") or ptq.get("keep_bf16_patterns"))

    repo_mount: list[str] = []
    if strategy in {"gemma4_moe_modelopt", "qwen_text_modelopt"}:
        default_script = (
            "scripts/quantization/qwen_text_modelopt.py"
            if strategy == "qwen_text_modelopt"
            else "scripts/quantization/gemma4_moe_nvfp4.py"
        )
        script = Path(str(ptq.get("script") or export.get("script") or default_script))
        script_host_path = resolve_repo_path(script)
        if not script_host_path.exists():
            raise ValueError(f"{strategy} quantization script not found: {display_path(script_host_path)}")
        repo_mount = ["-v", f"{REPO_DIR}:/workspace/model-forge:ro"]
        container_command = [
            "python3",
            f"/workspace/model-forge/{display_path(script_host_path)}",
            "--model",
            container_source,
            "--output",
            container_output,
            "--qformat",
            qformat,
            "--dataset",
            dataset,
            "--calib-samples",
            calib_size,
            "--calib-seq-len",
            calib_seq,
            "--batch-size",
            str(batch_size),
            "--device",
            str(ptq.get("device") or "cuda:0"),
            "--device-map",
            str(ptq.get("device_map") or "auto"),
        ]
        if strategy == "gemma4_moe_modelopt":
            container_command.extend(["--max-shard-size-gb", str(ptq.get("max_shard_size_gb") or 8)])
        if strategy == "qwen_text_modelopt" and bool(ptq.get("keep_text_input", False)):
            container_command.append("--keep-text-input")
        if strategy == "qwen_text_modelopt" and bool(ptq.get("reject_meta_tensors", False)):
            container_command.append("--reject-meta-tensors")
        if strategy == "qwen_text_modelopt":
            for pattern in disable_patterns:
                container_command.extend(["--disable-pattern", pattern])
        if bool(ptq.get("trust_remote_code", True)):
            container_command.append("--trust-remote-code")
    elif strategy == "hf_ptq":
        container_command = [
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
            container_command.append("--trust_remote_code")
        if recipe:
            container_command.extend(["--recipe", recipe])
        if bool(ptq.get("low_memory_mode", True)):
            container_command.append("--low_memory_mode")
        if bool(ptq.get("use_seq_device_map", True)):
            container_command.append("--use_seq_device_map")
        if ptq.get("moe_calib_experts_ratio") is not None:
            container_command.extend(["--moe_calib_experts_ratio", str(ptq["moe_calib_experts_ratio"])])
        if ptq.get("attn_implementation"):
            container_command.extend(["--attn_implementation", str(ptq["attn_implementation"])])
        if not bool(ptq.get("verbose", True)):
            container_command.append("--no-verbose")
    else:
        raise ValueError(f"unknown ModelOpt export strategy {strategy!r}")

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
        *repo_mount,
        "-e",
        "HF_TOKEN",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
        "-e",
        f"OMP_NUM_THREADS={docker.get('omp_num_threads', docker.get('cpus', 8))}",
        image,
        *container_command,
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
            "promotion": source_promotion(source),
        },
        "target": {
            "variant": target_variant_label(config, source),
            "host_output_path": display_path(output_path),
            "container_output_path": container_output,
            "served_model_name": str(config.runtime.get("served_model_name") or f"{source.served_model_name}-{config.method}"),
        },
        "method": config.method,
        "backend": config.backend,
        "strategy": strategy,
        "image": image,
        "calibration": {
            "dataset": dataset,
            "calib_size": calib_size,
            "calib_seq": calib_seq,
            "batch_size": batch_size,
            "kv_cache_qformat": kv_cache_qformat,
            "recipe": recipe or None,
            "moe_calib_experts_ratio": ptq.get("moe_calib_experts_ratio"),
            "disable_patterns": disable_patterns,
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


def gguf_output_paths(config: QuantizationConfig, source: QuantizationSource, output_dir: Path) -> tuple[Path, Path]:
    conversion = dict(config.raw_config.get("conversion") or {})
    target = target_variant_label(config, source)
    output_subdir = str((config.export or {}).get("output_subdir") or "{source_family}/{target_variant}")
    root = output_dir / sanitize_relative_subpath(
        render_template(
            output_subdir,
            {
                "source_variant": source.variant or "runtime",
                "source_family": source.family or "generic",
                "target_variant": target,
                "method": config.method,
                "backend": config.backend,
            },
        )
    )
    filename = render_template(
        str(conversion.get("output_filename") or "{target_variant}.gguf"),
        {
            "source_variant": source.variant or "runtime",
            "source_family": source.family or "generic",
            "target_variant": target,
            "method": config.method,
            "backend": config.backend,
        },
    )
    quantized = root / sanitize_relative_subpath(filename)
    intermediate_name = render_template(
        str(conversion.get("intermediate_filename") or "{target_variant}.f16.gguf"),
        {
            "source_variant": source.variant or "runtime",
            "source_family": source.family or "generic",
            "target_variant": target,
            "method": config.method,
            "backend": config.backend,
        },
    )
    intermediate = root / sanitize_relative_subpath(intermediate_name)
    return intermediate, quantized


def build_gguf_export_command(
    config: QuantizationConfig,
    source: QuantizationSource,
    *,
    output_dir: Path,
    run_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    if config.backend not in {"llama_cpp", "llama-cpp"}:
        raise ValueError(f"GGUF export requires backend=llama_cpp, got {config.backend!r}")
    conversion = dict(config.raw_config.get("conversion") or {})
    export = dict(config.export or {})
    intermediate, quantized = gguf_output_paths(config, source, output_dir)
    quantized.parent.mkdir(parents=True, exist_ok=True)
    llama_cpp_dir = str(env.get("MODEL_FORGE_LLAMA_CPP_DIR") or conversion.get("llama_cpp_dir") or "${MODEL_FORGE_LLAMA_CPP_DIR:?/path/to/llama.cpp}")
    convert_script = str(conversion.get("convert_script") or "convert_hf_to_gguf.py")
    quantize_binary = str(conversion.get("quantize_binary") or "./llama-quantize")
    llama_cli = str(conversion.get("llama_cli") or "./llama-cli")
    llama_bench = str(conversion.get("llama_bench") or "./llama-bench")
    quant_type = str(conversion.get("quant_type") or "Q4_K_M")
    outtype = str(conversion.get("intermediate_outtype") or "f16")
    source_path = display_path(source.local_path) if source.local_path else source.model_id
    convert_command = [
        "python3",
        convert_script,
        shlex.quote(source_path),
        "--outfile",
        shlex.quote(display_path(intermediate)),
        "--outtype",
        shlex.quote(outtype),
    ]
    if truthy(conversion.get("trust_remote_code"), True):
        convert_command.append("--trust-remote-code")
    quantize_command = [
        shlex.quote(quantize_binary),
        shlex.quote(display_path(intermediate)),
        shlex.quote(display_path(quantized)),
        shlex.quote(quant_type),
    ]
    load_command = [
        shlex.quote(str(conversion.get("load_binary") or llama_cli)),
        "-m",
        shlex.quote(display_path(quantized)),
        "-p",
        shlex.quote(str(conversion.get("load_prompt") or "ping")),
        "-n",
        str(int(conversion.get("load_tokens") or 32)),
    ]
    bench_command = [
        shlex.quote(llama_bench),
        "-m",
        shlex.quote(display_path(quantized)),
    ]
    shell_steps = [
        "cd " + shlex.quote(llama_cpp_dir),
        " ".join(convert_command),
        " ".join(quantize_command),
        " ".join(load_command),
        " ".join(bench_command),
    ]
    shell_script = " && ".join(shell_steps)
    command = ["nice", "-n", str((export.get("docker") or {}).get("nice", export.get("nice", 10))), "bash", "-lc", shell_script]
    systemd_scope = dict(export.get("systemd_scope") or {})
    if truthy(systemd_scope.get("enabled"), True):
        scope_command = ["systemd-run", "--scope"]
        if truthy(systemd_scope.get("user"), True):
            scope_command = ["systemd-run", "--user", "--scope"]
        command = [
            *scope_command,
            "-p",
            f"CPUQuota={systemd_scope.get('CPUQuota', '80%')}",
            "-p",
            f"MemoryMax={systemd_scope.get('MemoryMax', '85%')}",
            "-p",
            f"IOWeight={systemd_scope.get('IOWeight', 100)}",
            *command,
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
            "promotion": source_promotion(source),
        },
        "target": {
            "variant": target_variant_label(config, source),
            "host_output_path": display_path(quantized),
            "intermediate_output_path": display_path(intermediate),
            "served_model_name": str(config.runtime.get("served_model_name") or f"{source.served_model_name}-{config.method}"),
        },
        "method": config.method,
        "backend": config.backend,
        "strategy": "llama_cpp_gguf",
        "conversion": {
            "llama_cpp_dir": llama_cpp_dir,
            "convert_script": convert_script,
            "quantize_binary": quantize_binary,
            "quant_type": quant_type,
            "intermediate_outtype": outtype,
            "preserve_tokenizer": truthy(conversion.get("preserve_tokenizer"), True),
            "preserve_chat_template": truthy(conversion.get("preserve_chat_template"), True),
        },
        "resource_policy": {
            "start_if_memory_available_above_fraction": float(export.get("start_if_memory_available_above_fraction", 0.05)),
            "stop_if_memory_available_below_fraction": float(export.get("stop_if_memory_available_below_fraction", 0.05)),
            "require_disk_free_fraction": float(export.get("require_disk_free_fraction", 0.15)),
            "systemd_scope": {
                "enabled": truthy(systemd_scope.get("enabled"), True),
                "user": truthy(systemd_scope.get("user"), True),
                "CPUQuota": str(systemd_scope.get("CPUQuota", "80%")),
                "MemoryMax": str(systemd_scope.get("MemoryMax", "85%")),
                "IOWeight": int(systemd_scope.get("IOWeight", 100)),
            },
        },
        "command": command,
        "command_display": shlex.join(command),
        "validation_gates": [
            "llama.cpp conversion command completes",
            "llama-quantize command completes",
            "llama-cli load probe completes",
            "llama-bench completes",
            "tokenizer-report passes against the source tokenizer",
            "behavior-report passes against the source serving/eval evidence",
        ],
    }


def build_export_command(
    config: QuantizationConfig,
    source: QuantizationSource,
    *,
    output_dir: Path,
    run_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if config.backend in {"llama_cpp", "llama-cpp"} or config.method.startswith("gguf"):
        return build_gguf_export_command(config, source, output_dir=output_dir, run_id=run_id, env=env)
    return build_modelopt_export_command(config, source, output_dir=output_dir, run_id=run_id, env=env)


def calibration_dataset_entries(raw_dataset: str, *, optional_gated_dataset: str | None = None) -> list[dict[str, Any]]:
    selected = comma_list(raw_dataset)
    gated = set(comma_list(optional_gated_dataset))
    entries = []
    for index, dataset in enumerate(selected):
        entries.append(
            {
                "name": dataset,
                "order": index,
                "role": "primary" if index == 0 else "supplemental",
                "access": "gated" if dataset in gated else "public_or_local",
                "checksum": None,
                "revision": None,
            }
        )
    return entries


def build_calibration_manifest(
    config: QuantizationConfig,
    *,
    config_path: Path,
    family: str | None,
    variant: str | None,
    output_dir: str | Path | None,
    run_id: str | None,
    dataset: str | None = None,
    samples: str | None = None,
    seq_len: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    source = resolve_source(config, family, variant, env)
    calibration = dict(config.calibration)
    ptq = dict((config.export or {}).get("ptq") or {})
    resolved_dataset = dataset or env.get("MODEL_FORGE_QUANT_CALIB_DATASET") or str(ptq.get("dataset") or calibration.get("dataset") or "")
    resolved_samples = samples or env.get("MODEL_FORGE_QUANT_CALIB_SIZE") or str(ptq.get("calib_size") or calibration.get("samples") or "")
    resolved_seq_len = seq_len or env.get("MODEL_FORGE_QUANT_CALIB_SEQ") or str(ptq.get("calib_seq") or calibration.get("seq_len") or "")
    batch_size = int(ptq.get("batch_size") or calibration.get("batch_size") or 1)
    actual_run_id = sanitize_run_id(run_id or f"{config.name}_{source.variant or 'runtime'}_calibration_manifest")
    output_root = resolve_repo_path(output_dir or config.outputs.get("reports_dir") or "reports/generated/quantization")
    manifest_dir = output_root / actual_run_id
    return redact_value(
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": actual_run_id,
            "config": display_path(config_path),
            "name": config.name,
            "objective": config.objective,
            "method": config.method,
            "backend": config.backend,
            "source": {
                "family": source.family,
                "variant": source.variant,
                "model_id": source.model_id,
                "local_path": display_path(source.local_path) if source.local_path else None,
                "local_path_exists": source.local_path.exists() if source.local_path else None,
                "promotion": source_promotion(source),
            },
            "target": {
                "variant": target_variant_label(config, source),
            },
            "calibration": {
                "dataset": resolved_dataset,
                "datasets": calibration_dataset_entries(
                    resolved_dataset,
                    optional_gated_dataset=str(calibration.get("optional_gated_dataset") or ""),
                ),
                "samples": resolved_samples,
                "seq_len": resolved_seq_len,
                "batch_size": batch_size,
                "smoke_samples": calibration.get("smoke_samples"),
                "smoke_seq_len": calibration.get("smoke_seq_len"),
                "production_samples": calibration.get("production_samples"),
                "production_seq_len": calibration.get("production_seq_len"),
                "selection_source": {
                    "dataset": "argument" if dataset else ("environment" if env.get("MODEL_FORGE_QUANT_CALIB_DATASET") else "config"),
                    "samples": "argument" if samples else ("environment" if env.get("MODEL_FORGE_QUANT_CALIB_SIZE") else "config"),
                    "seq_len": "argument" if seq_len else ("environment" if env.get("MODEL_FORGE_QUANT_CALIB_SEQ") else "config"),
                },
                "notes": calibration.get("notes"),
            },
            "exclusions": config.exclusions,
            "output_dir": display_path(manifest_dir),
            "promotion_requirements": [
                "Attach dataset revisions/checksums when calibration rows are materialized.",
                "Record the exact export command and quantized checkpoint path.",
                "Compare the quantized checkpoint against the matching unquantized source variant.",
                "Attach serving metrics and sampled quality/behavior evidence before promotion.",
            ],
        }
    )


def write_calibration_manifest_outputs(manifest: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(manifest["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calibration_manifest.json"
    json_path.write_text(json.dumps(redact_value(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calibration = manifest.get("calibration") or {}
    source = manifest.get("source") or {}
    lines = [
        f"# Quantization Calibration Manifest: {manifest.get('run_id')}",
        "",
        f"- Source: `{source.get('family') or 'n/a'}` / `{source.get('variant') or 'n/a'}`",
        f"- Method/backend: `{manifest.get('method')}` / `{manifest.get('backend')}`",
        f"- Dataset: `{calibration.get('dataset')}`",
        f"- Samples: `{calibration.get('samples')}`",
        f"- Sequence length: `{calibration.get('seq_len')}`",
        f"- Batch size: `{calibration.get('batch_size')}`",
        "",
        "## Datasets",
        "",
        "| Order | Name | Role | Access | Revision | Checksum |",
        "|---:|---|---|---|---|---|",
    ]
    for item in calibration.get("datasets") or []:
        lines.append(
            f"| {item.get('order')} | {item.get('name')} | {item.get('role')} | {item.get('access')} | "
            f"{item.get('revision')} | {item.get('checksum')} |"
        )
    lines.extend(["", "## Promotion Requirements", ""])
    lines.extend(f"- {item}" for item in manifest.get("promotion_requirements") or [])
    (output_dir / "calibration_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def guard_export(export_plan: Mapping[str, Any]) -> None:
    source = export_plan.get("source") or {}
    blocked, reason = promotion_blocks_action(source.get("promotion"), "quantization_export")
    if blocked:
        source_label = "/".join(str(item) for item in (source.get("family"), source.get("variant")) if item)
        raise RuntimeError(
            f"source variant {source_label or source.get('model_id') or '<unknown>'} is blocked by variant promotion metadata: {reason}"
        )
    source_family = str(source.get("family") or "")
    source_variant = str(source.get("variant") or "")
    if source_family and source_variant:
        checkpoint_audit = build_checkpoint_audit(source_family, variant=source_variant, strict=True)
        if not checkpoint_audit["passed"]:
            findings = checkpoint_audit.get("findings") or []
            summary = "; ".join(
                f"{item.get('variant')} {item.get('check')}: {item.get('message')}" for item in findings[:5]
            )
            raise RuntimeError(f"source checkpoint audit failed before quantization export: {summary}")
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
    subprocess.run(["docker", "stop", "--timeout", "5", container_name], cwd=REPO_DIR, check=False)
    subprocess.run(["docker", "kill", container_name], cwd=REPO_DIR, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


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

    runtime_env = dict(runtime.get("env") or {})
    env_prefix = [f"{key}={value}" for key, value in sorted(runtime_env.items())]

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

    return ["cd", spark_dir, "&&", *env_prefix, *launcher, *serve]


def runtime_source(config: QuantizationConfig, source: QuantizationSource) -> QuantizationSource:
    runtime = dict(config.runtime)
    model_id = str(runtime.get("model_id") or source.model_id)
    served_model_name = str(runtime.get("served_model_name") or source.served_model_name)
    local_path = Path(str(runtime["local_path"])).expanduser() if runtime.get("local_path") else None
    return QuantizationSource(
        family=source.family,
        variant=target_variant_label(config, source) if config.target_variant else source.variant,
        model_id=model_id,
        served_model_name=served_model_name,
        local_path=local_path,
    )


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
    candidate = runtime_source(config, source) if config.runtime else source
    actual_run_id = run_id or plan_run_id(config, source)
    output_root = resolve_repo_path(output_dir or config.outputs.get("reports_dir") or "reports/generated/quantization")
    plan_output_dir = output_root / actual_run_id
    profile = detect_hardware_profile(env)
    quant_env = recommended_quantization_env(env)
    exclusions = dict(config.exclusions)
    keep_patterns = comma_list(quant_env.get("MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS"))
    if keep_patterns and truthy(exclusions.get("apply_recommended_keep_patterns"), True):
        modules = list(exclusions.get("modules") or [])
        modules.extend(pattern for pattern in keep_patterns if pattern not in modules)
        exclusions["modules"] = modules
    export_policy = dict(config.export or {})
    export_systemd_scope = dict(export_policy.get("systemd_scope") or {})
    export_docker = dict(export_policy.get("docker") or {})
    start_memory_floor = float(export_policy.get("start_if_memory_available_above_fraction", 0.05))
    stop_memory_floor = float(export_policy.get("stop_if_memory_available_below_fraction", start_memory_floor))

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
            "promotion": source_promotion(source),
        },
        "target": {
            "variant": target_variant_label(config, source),
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
            "start_if_memory_available_above_fraction": start_memory_floor,
            "stop_if_memory_available_below_fraction": stop_memory_floor,
            "require_disk_free_fraction": float(export_policy.get("require_disk_free_fraction", 0.15)),
            "systemd_scope": {
                "CPUQuota": str(export_systemd_scope.get("CPUQuota", "80%")),
                "MemoryMax": str(export_systemd_scope.get("MemoryMax", "85%")),
                "IOWeight": int(export_systemd_scope.get("IOWeight", 100)),
                "nice": int(export_docker.get("nice", 10)),
            },
        },
        "outputs": {
            "output_dir": display_path(plan_output_dir),
            "plan_json": "quantization_plan.json",
            "plan_md": "quantization_plan.md",
        },
        "launch_command": build_runtime_command(config, candidate) if config.runtime else [],
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


def load_optional_json(path: Path | None) -> dict[str, Any]:
    return load_json(resolve_repo_path(path)) if path else {}


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
        "output_tokens_per_second_p50": ("metrics", "output_tokens_per_second", "p50"),
        "output_tokens_per_second_p95": ("metrics", "output_tokens_per_second", "p95"),
        "decode_tokens_per_second_p50": ("metrics", "decode_tokens_per_second", "p50"),
        "total_tokens_per_second_p50": ("metrics", "total_tokens_per_second", "p50"),
        "decode_heavy_output_tokens_per_second_p50": (
            "by_category",
            "decode_heavy",
            "metrics",
            "output_tokens_per_second",
            "p50",
        ),
        "decode_heavy_decode_tokens_per_second_p50": (
            "by_category",
            "decode_heavy",
            "metrics",
            "decode_tokens_per_second",
            "p50",
        ),
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


def status_check(name: str, passed: bool, message: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if required else "missing"),
        "required": required,
        "message": message,
    }


def build_fp8_kv_report(
    config: QuantizationConfig,
    *,
    config_path: Path,
    source_serving_summary: Path,
    candidate_serving_summary: Path,
    source_serving_eval: Path,
    candidate_serving_eval: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    card = build_card(
        config,
        config_path=config_path,
        source_serving_summary=source_serving_summary,
        candidate_serving_summary=candidate_serving_summary,
        source_serving_eval=source_serving_eval,
        candidate_serving_eval=candidate_serving_eval,
        output_dir=output_dir,
        run_id=run_id,
    )
    runtime_flags = dict((config.runtime or {}).get("vllm_flags") or {})
    kv_cache_dtype = str(runtime_flags.get("kv_cache_dtype") or runtime_flags.get("kv-cache-dtype") or "")
    serving_deltas = card["serving_deltas"]
    sampled = card["sampled_eval_deltas"]
    normal = sampled.get("normal_use_regression.normal_use_regression_pass_rate") or {}
    schema = sampled.get("agentic_tool_use_json.schema_adherence") or {}
    workflow = sampled.get("agentic_tool_use_json.workflow_success") or {}

    def retained(values: Mapping[str, Any], tolerance: float) -> bool:
        source = values.get("source")
        candidate = values.get("candidate")
        if source is None or candidate is None:
            return False
        return float(candidate) >= float(source) - tolerance

    checks = [
        status_check("kv_cache_dtype_fp8", "fp8" in kv_cache_dtype, f"kv_cache_dtype={kv_cache_dtype or 'unset'}"),
        status_check(
            "candidate_success_rate_complete",
            (serving_deltas.get("success_rate") or {}).get("candidate") == 1.0,
            f"candidate_success_rate={(serving_deltas.get('success_rate') or {}).get('candidate')}",
        ),
        status_check(
            "normal_use_retained",
            retained(normal, 0.03),
            f"source={normal.get('source')} candidate={normal.get('candidate')} tolerance=0.03",
        ),
        status_check(
            "schema_adherence_retained",
            retained(schema, 0.03),
            f"source={schema.get('source')} candidate={schema.get('candidate')} tolerance=0.03",
        ),
        status_check(
            "workflow_success_retained",
            retained(workflow, 0.03),
            f"source={workflow.get('source')} candidate={workflow.get('candidate')} tolerance=0.03",
        ),
    ]
    return redact_value(
        {
            "schema_version": FP8_KV_REPORT_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "config": display_path(config_path),
            "objective": config.objective,
            "method": config.method,
            "backend": config.backend,
            "kv_cache_dtype": kv_cache_dtype or None,
            "source": card["source"],
            "candidate": card["candidate"],
            "serving_deltas": serving_deltas,
            "sampled_eval_deltas": sampled,
            "checks": checks,
            "behavior_ready": all(check["status"] == "pass" for check in checks if check["required"]),
            "output_dir": display_path(output_dir),
            "notes": [
                "FP8 KV reports compare completed source and candidate endpoint evidence.",
                "This report does not start a server and does not prove checkpoint quantization.",
                "Promotion still requires the broader quantization card and serving evidence gate.",
            ],
        }
    )


BEHAVIOR_RETENTION_THRESHOLDS = {
    "normal_use_regression.normal_use_regression_pass_rate": -0.03,
    "capability_preservation_challenge.normal_use_regression_pass_rate": -0.05,
    "agentic_tool_use_json.schema_adherence": -0.03,
    "agentic_tool_use_json.workflow_success": -0.03,
    "refusal_paired_boundary.benign_answer_quality_rate": -0.03,
}


def behavior_metric_check(metric: str, values: Mapping[str, Any]) -> dict[str, Any]:
    threshold = BEHAVIOR_RETENTION_THRESHOLDS.get(metric)
    source = values.get("source")
    candidate = values.get("candidate")
    delta = values.get("delta")
    required = threshold is not None
    if not required:
        return status_check(metric, True, f"reported source={source} candidate={candidate} delta={delta}", required=False)
    passed = isinstance(delta, (int, float)) and float(delta) >= threshold
    return status_check(
        metric,
        passed,
        f"source={source} candidate={candidate} delta={delta} minimum_delta={threshold}",
    )


def build_behavior_report(
    config: QuantizationConfig,
    *,
    config_path: Path,
    source_serving_summary: Path,
    candidate_serving_summary: Path,
    source_serving_eval: Path,
    candidate_serving_eval: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    card = build_card(
        config,
        config_path=config_path,
        source_serving_summary=source_serving_summary,
        candidate_serving_summary=candidate_serving_summary,
        source_serving_eval=source_serving_eval,
        candidate_serving_eval=candidate_serving_eval,
        output_dir=output_dir,
        run_id=run_id,
    )
    serving = card["serving_deltas"]
    sampled = card["sampled_eval_deltas"]
    checks = [
        status_check(
            "candidate_success_rate_complete",
            (serving.get("success_rate") or {}).get("candidate") == 1.0,
            f"candidate_success_rate={(serving.get('success_rate') or {}).get('candidate')}",
        ),
    ]
    checks.extend(behavior_metric_check(metric, values) for metric, values in sampled.items())
    return redact_value(
        {
            "schema_version": BEHAVIOR_REPORT_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "config": display_path(config_path),
            "objective": config.objective,
            "method": config.method,
            "backend": config.backend,
            "source": card["source"],
            "candidate": card["candidate"],
            "sampled_eval_deltas": sampled,
            "serving_deltas": serving,
            "checks": checks,
            "behavior_preserved": all(check["status"] == "pass" for check in checks if check["required"]),
            "output_dir": display_path(output_dir),
            "notes": [
                "Behavior preservation is evaluated against quantized_quality_retention tolerances.",
                "Risk metrics such as unsafe overcompliance are reported but not treated as retention failures here.",
                "Promotion also requires serving evidence, tokenizer integrity, quantization card, and release-class gates.",
            ],
        }
    )


def write_behavior_report_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "behavior_preservation_report.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Quantization Behavior Preservation Report: {report.get('run_id')}",
        "",
        f"- Behavior preserved: `{str(report.get('behavior_preserved')).lower()}`",
        f"- Method/backend: `{report.get('method')}` / `{report.get('backend')}`",
        "",
        "## Checks",
        "",
        "| Status | Required | Check | Message |",
        "|---|---|---|---|",
    ]
    for check in report.get("checks") or []:
        lines.append(
            f"| {str(check.get('status')).upper()} | {'yes' if check.get('required') else 'no'} | "
            f"{check.get('name')} | {check.get('message')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "behavior_preservation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def build_tokenizer_report(
    *,
    source_tokenizer_dir: Path,
    candidate_tokenizer_dir: Path,
    output_dir: Path,
    run_id: str,
    source_variant: str = "source",
    candidate_variant: str = "candidate",
    load_tokenizer: bool = False,
    strict: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    source_path = resolve_repo_path(source_tokenizer_dir)
    candidate_path = resolve_repo_path(candidate_tokenizer_dir)
    source = tokenizer_record(source_path)
    candidate = tokenizer_record(candidate_path)
    findings = []
    if not source.get("exists"):
        findings.append({"level": "error" if strict else "warning", "variant": source_variant, "check": "source_tokenizer_dir", "message": "source tokenizer dir is not present"})
    if not candidate.get("exists"):
        findings.append({"level": "error" if strict else "warning", "variant": candidate_variant, "check": "candidate_tokenizer_dir", "message": "candidate tokenizer dir is not present"})
    if source.get("exists") and candidate.get("exists"):
        findings.extend(finding.__dict__ for finding in compare_records(candidate_variant, candidate, source_variant, source))
    if load_tokenizer:
        if source.get("exists"):
            source["round_trip"] = live_round_trip(source_path, trust_remote_code=trust_remote_code)
            if source["round_trip"].get("status") != "passed":
                findings.append(
                    {
                        "level": "error" if strict else "warning",
                        "variant": source_variant,
                        "check": "round_trip",
                        "message": str(source["round_trip"].get("reason") or source["round_trip"].get("status")),
                    }
                )
        if candidate.get("exists"):
            candidate["round_trip"] = live_round_trip(candidate_path, trust_remote_code=trust_remote_code)
            if candidate["round_trip"].get("status") != "passed":
                findings.append(
                    {
                        "level": "error" if strict else "warning",
                        "variant": candidate_variant,
                        "check": "round_trip",
                        "message": str(candidate["round_trip"].get("reason") or candidate["round_trip"].get("status")),
                    }
                )
    errors = [finding for finding in findings if finding.get("level") == "error"]
    return redact_value(
        {
            "schema_version": TOKENIZER_REPORT_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "source_variant": source_variant,
            "candidate_variant": candidate_variant,
            "source": source,
            "candidate": candidate,
            "metadata_only": not load_tokenizer,
            "strict": strict,
            "findings": findings,
            "passed": not errors,
            "output_dir": display_path(output_dir),
            "notes": [
                "Use this report for quantized or GGUF export directories before they are promoted.",
                "Configured family variants can also use ./forge variants tokenizer-audit.",
                "Promotion should use strict mode and a live tokenizer round trip when the tokenizer can be loaded locally.",
            ],
        }
    )


def write_tokenizer_report_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer_preservation_report.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Quantization Tokenizer Preservation Report: {report.get('run_id')}",
        "",
        f"- Passed: `{str(report.get('passed')).lower()}`",
        f"- Source: `{report.get('source_variant')}`",
        f"- Candidate: `{report.get('candidate_variant')}`",
        f"- Metadata only: `{str(report.get('metadata_only')).lower()}`",
        "",
        "## Findings",
        "",
    ]
    if report.get("findings"):
        lines.extend(f"- {item.get('level')} `{item.get('variant')}` {item.get('check')}: {item.get('message')}" for item in report["findings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "tokenizer_preservation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def maybe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def modelopt_metadata_from_candidate(candidate_dir: Path) -> dict[str, Any]:
    candidate_path = resolve_repo_path(candidate_dir)
    hf_quant_config_path = candidate_path / "hf_quant_config.json"
    config_path = candidate_path / "config.json"
    hf_quant_config = maybe_load_json(hf_quant_config_path)
    config = maybe_load_json(config_path)
    hf_quantization = hf_quant_config.get("quantization") if isinstance(hf_quant_config.get("quantization"), Mapping) else {}
    config_quantization = config.get("quantization_config") if isinstance(config.get("quantization_config"), Mapping) else {}
    producer = {}
    for source in (hf_quant_config, config_quantization):
        raw_producer = source.get("producer") if isinstance(source, Mapping) else None
        if isinstance(raw_producer, Mapping):
            producer = dict(raw_producer)
            break
    quant_algo_values = {
        "hf_quant_config": hf_quantization.get("quant_algo") if isinstance(hf_quantization, Mapping) else None,
        "config": config_quantization.get("quant_algo") if isinstance(config_quantization, Mapping) else None,
    }
    quant_method_values = {
        "hf_quant_config": hf_quantization.get("quant_method") if isinstance(hf_quantization, Mapping) else None,
        "config": config_quantization.get("quant_method") if isinstance(config_quantization, Mapping) else None,
    }
    resolved_quant_algo = next((str(value) for value in quant_algo_values.values() if value), None)
    resolved_quant_method = next((str(value) for value in quant_method_values.values() if value), None)
    return {
        "candidate_dir": display_path(candidate_path),
        "candidate_dir_exists": candidate_path.exists(),
        "files": {
            "hf_quant_config": display_path(hf_quant_config_path) if hf_quant_config_path.exists() else None,
            "config": display_path(config_path) if config_path.exists() else None,
        },
        "producer": producer,
        "quant_algo": resolved_quant_algo,
        "quant_method": resolved_quant_method,
        "quant_algo_values": {key: value for key, value in quant_algo_values.items() if value},
        "quant_method_values": {key: value for key, value in quant_method_values.items() if value},
        "exclude_module_count": len(hf_quantization.get("exclude_modules") or config_quantization.get("ignore") or []),
    }


def build_modelopt_runtime_compat_report(
    *,
    candidate_dir: Path,
    runtime: str,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    runtime_name = str(runtime).lower()
    metadata = modelopt_metadata_from_candidate(candidate_dir)
    quant_algo = metadata.get("quant_algo")
    quant_method = metadata.get("quant_method")
    producer = metadata.get("producer") or {}
    quant_algo_values = set((metadata.get("quant_algo_values") or {}).values())
    is_modelopt = quant_method == "modelopt" or producer.get("name") == "modelopt"
    supported_algos = sorted(VLLM_MODELOPT_QUANT_ALGOS) if runtime_name == "vllm" else []
    runtime_supported = runtime_name == "vllm" and quant_algo in VLLM_MODELOPT_QUANT_ALGOS
    checks = [
        status_check(
            "candidate_dir_exists",
            bool(metadata.get("candidate_dir_exists")),
            f"candidate_dir={metadata.get('candidate_dir')}",
        ),
        status_check(
            "modelopt_metadata_found",
            bool((metadata.get("files") or {}).get("hf_quant_config") or (metadata.get("files") or {}).get("config")),
            f"files={metadata.get('files')}",
        ),
        status_check(
            "quant_algo_present",
            bool(quant_algo),
            f"quant_algo={quant_algo}",
        ),
        status_check(
            "quant_algo_metadata_consistent",
            len(quant_algo_values) <= 1,
            f"quant_algo_values={metadata.get('quant_algo_values')}",
        ),
        status_check(
            "quant_method_modelopt",
            is_modelopt,
            f"quant_method={quant_method} producer={producer}",
        ),
        status_check(
            "runtime_supported",
            runtime_name == "vllm",
            f"runtime={runtime_name}",
        ),
        status_check(
            "runtime_supports_quant_algo",
            runtime_supported,
            f"runtime={runtime_name} quant_algo={quant_algo} supported={supported_algos}",
        ),
    ]
    passed = all(check["status"] == "pass" for check in checks if check["required"])
    return redact_value(
        {
            "schema_version": MODELOPT_RUNTIME_COMPAT_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "runtime": runtime_name,
            "candidate": metadata,
            "supported_quant_algos": supported_algos,
            "checks": checks,
            "passed": passed,
            "output_dir": display_path(output_dir),
            "notes": [
                "Run this after a ModelOpt export and before launching a serving runtime.",
                "This report catches metadata/runtime mismatches that checkpoint and tokenizer audits cannot detect.",
                "Passing this report does not prove generation quality; serving eval and behavior-preservation gates are still required.",
            ],
        }
    )


def write_modelopt_runtime_compat_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modelopt_runtime_compat_report.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate = report.get("candidate") or {}
    lines = [
        f"# ModelOpt Runtime Compatibility Report: {report.get('run_id')}",
        "",
        f"- Passed: `{str(report.get('passed')).lower()}`",
        f"- Runtime: `{report.get('runtime')}`",
        f"- Quant algo: `{candidate.get('quant_algo')}`",
        f"- Quant method: `{candidate.get('quant_method')}`",
        "",
        "## Checks",
        "",
        "| Status | Required | Check | Message |",
        "|---|---|---|---|",
    ]
    for check in report.get("checks") or []:
        lines.append(
            f"| {str(check.get('status')).upper()} | {'yes' if check.get('required') else 'no'} | "
            f"{check.get('name')} | {check.get('message')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "modelopt_runtime_compat_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def candidate_arg(raw: str) -> dict[str, str]:
    values = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"candidate entries must be key=value pairs, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    required = {"name", "component", "summary", "eval"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"candidate entry missing keys: {', '.join(missing)}")
    return values


def sensitivity_candidate_summary(
    config: QuantizationConfig,
    *,
    config_path: Path,
    baseline_summary: Path,
    baseline_eval: Path,
    candidate: Mapping[str, str],
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    card = build_card(
        config,
        config_path=config_path,
        source_serving_summary=baseline_summary,
        candidate_serving_summary=resolve_repo_path(candidate["summary"]),
        source_serving_eval=baseline_eval,
        candidate_serving_eval=resolve_repo_path(candidate["eval"]),
        output_dir=output_dir,
        run_id=run_id,
    )
    checks = [behavior_metric_check(metric, values) for metric, values in card["sampled_eval_deltas"].items()]
    behavior_preserved = all(check["status"] == "pass" for check in checks if check["required"])
    throughput = card["serving_deltas"].get("output_tokens_per_second_p50") or {}
    decode_heavy = card["serving_deltas"].get("decode_heavy_output_tokens_per_second_p50") or {}
    latency = card["serving_deltas"].get("total_latency_p50") or {}
    return {
        "name": candidate["name"],
        "component": candidate["component"],
        "policy": candidate.get("policy"),
        "precision": candidate.get("precision"),
        "serving_summary": display_path(resolve_repo_path(candidate["summary"])),
        "serving_eval": display_path(resolve_repo_path(candidate["eval"])),
        "behavior_preserved": behavior_preserved,
        "required_behavior_checks": checks,
        "throughput_delta": throughput.get("delta"),
        "decode_heavy_throughput_delta": decode_heavy.get("delta"),
        "latency_delta": latency.get("delta"),
        "serving_deltas": card["serving_deltas"],
        "sampled_eval_deltas": card["sampled_eval_deltas"],
    }


def sensitivity_sort_key(item: Mapping[str, Any]) -> tuple[int, float, float]:
    preserved = 1 if item.get("behavior_preserved") else 0
    decode_delta = item.get("decode_heavy_throughput_delta")
    throughput_delta = item.get("throughput_delta")
    return (
        preserved,
        float(decode_delta) if isinstance(decode_delta, (int, float)) else float("-inf"),
        float(throughput_delta) if isinstance(throughput_delta, (int, float)) else float("-inf"),
    )


def build_sensitivity_report(
    config: QuantizationConfig,
    *,
    config_path: Path,
    baseline_serving_summary: Path,
    baseline_serving_eval: Path,
    candidates: list[Mapping[str, str]],
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    baseline_summary = resolve_repo_path(baseline_serving_summary)
    baseline_eval = resolve_repo_path(baseline_serving_eval)
    candidate_reports = [
        sensitivity_candidate_summary(
            config,
            config_path=config_path,
            baseline_summary=baseline_summary,
            baseline_eval=baseline_eval,
            candidate=candidate,
            output_dir=output_dir,
            run_id=f"{run_id}_{candidate['name']}",
        )
        for candidate in candidates
    ]
    ranked = sorted(candidate_reports, key=sensitivity_sort_key, reverse=True)
    return redact_value(
        {
            "schema_version": SENSITIVITY_REPORT_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "config": display_path(config_path),
            "objective": config.objective,
            "method": config.method,
            "backend": config.backend,
            "baseline": {
                "serving_summary": display_path(baseline_summary),
                "serving_eval": display_path(baseline_eval),
            },
            "candidate_count": len(candidate_reports),
            "candidates": candidate_reports,
            "ranking": [
                {
                    "rank": index + 1,
                    "name": item["name"],
                    "component": item["component"],
                    "behavior_preserved": item["behavior_preserved"],
                    "decode_heavy_throughput_delta": item["decode_heavy_throughput_delta"],
                    "throughput_delta": item["throughput_delta"],
                    "latency_delta": item["latency_delta"],
                }
                for index, item in enumerate(ranked)
            ],
            "recommended_candidate": ranked[0]["name"] if ranked else None,
            "output_dir": display_path(output_dir),
            "notes": [
                "Sensitivity reports rank completed candidate runs; they do not run quantization or serving jobs.",
                "A candidate must preserve required behavior checks before throughput deltas are considered promotion evidence.",
                "Use this to compare policies such as all-linear, MLP-only, attention-only, experts-only, or keep-router-BF16.",
            ],
        }
    )


def write_sensitivity_report_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sensitivity_report.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Quantization Sensitivity Report: {report.get('run_id')}",
        "",
        f"- Candidate count: `{report.get('candidate_count')}`",
        f"- Recommended candidate: `{report.get('recommended_candidate')}`",
        "",
        "## Ranking",
        "",
        "| Rank | Name | Component | Behavior preserved | Decode-heavy tok/s delta | Output tok/s delta | Latency delta |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for item in report.get("ranking") or []:
        lines.append(
            f"| {item.get('rank')} | {item.get('name')} | {item.get('component')} | {item.get('behavior_preserved')} | "
            f"{item.get('decode_heavy_throughput_delta')} | {item.get('throughput_delta')} | {item.get('latency_delta')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def nested_float(data: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value = stats_value(data, path)
    return value


def positive_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if numeric > 0 else None


def speedup_from_delta(card_data: Mapping[str, Any], key: str) -> float | None:
    serving_deltas = card_data.get("serving_deltas") or {}
    values = serving_deltas.get(key) or serving_deltas.get(f"{key}_p50") or {}
    if not isinstance(values, Mapping):
        return None
    source = positive_float(values.get("source"))
    candidate = positive_float(values.get("candidate"))
    if source is None or candidate is None:
        return None
    return candidate / source


def nvfp4_gate_config(card_data: Mapping[str, Any]) -> dict[str, Any]:
    config_path = card_data.get("config")
    if not config_path:
        return {}
    try:
        config = load_quantization_config(resolve_repo_path(str(config_path)))
    except Exception:
        return {}
    gates = config.raw_config.get("gates") or {}
    nvfp4 = gates.get("nvfp4") if isinstance(gates, Mapping) else {}
    return dict(nvfp4) if isinstance(nvfp4, Mapping) else {}


def nvfp4_gate_targets(card_data: Mapping[str, Any], min_output_tps: float | None) -> dict[str, Any]:
    gate_config = nvfp4_gate_config(card_data)
    output_speedup = speedup_from_delta(card_data, "output_tokens_per_second")
    decode_heavy_output_speedup = speedup_from_delta(card_data, "decode_heavy_output_tokens_per_second")
    if min_output_tps is not None:
        absolute_tps = float(min_output_tps)
        absolute_source = "cli"
    elif gate_config.get("min_output_tokens_per_second") is not None:
        absolute_tps = float(gate_config["min_output_tokens_per_second"])
        absolute_source = "config"
    elif gate_config.get("min_output_speedup") is not None or gate_config.get("min_decode_heavy_output_speedup") is not None:
        absolute_tps = None
        absolute_source = "not_required"
    else:
        absolute_tps = 45.0
        absolute_source = "default"
    return {
        "min_output_tokens_per_second": absolute_tps,
        "min_output_tokens_per_second_source": absolute_source,
        "min_output_speedup": float(gate_config["min_output_speedup"]) if gate_config.get("min_output_speedup") is not None else None,
        "min_decode_heavy_output_speedup": (
            float(gate_config["min_decode_heavy_output_speedup"])
            if gate_config.get("min_decode_heavy_output_speedup") is not None
            else None
        ),
        "output_tokens_per_second_speedup": output_speedup,
        "decode_heavy_output_tokens_per_second_speedup": decode_heavy_output_speedup,
    }


def build_nvfp4_gate_report(
    *,
    export_plan: Path,
    serving_summary: Path,
    serving_eval: Path,
    quantization_card: Path,
    behavior_report: Path,
    tokenizer_report: Path,
    output_dir: Path,
    run_id: str,
    min_output_tps: float | None = None,
) -> dict[str, Any]:
    export_data = load_json(resolve_repo_path(export_plan))
    serving_data = load_json(resolve_repo_path(serving_summary))
    card_data = load_json(resolve_repo_path(quantization_card))
    behavior_data = load_json(resolve_repo_path(behavior_report))
    tokenizer_data = load_json(resolve_repo_path(tokenizer_report))
    eval_scores_path = scores_csv_path(resolve_repo_path(serving_eval))
    output_tps = nested_float(serving_data, ("metrics", "output_tokens_per_second", "p50"))
    decode_heavy_tps = nested_float(serving_data, ("by_category", "decode_heavy", "metrics", "output_tokens_per_second", "p50"))
    command_display = str(export_data.get("command_display") or "")
    targets = nvfp4_gate_targets(card_data, min_output_tps)
    checks = [
        status_check("export_plan_schema", export_data.get("schema_version") == "model_forge.quantization_export.v1", f"schema={export_data.get('schema_version')}"),
        status_check("export_method_nvfp4", export_data.get("method") == "nvfp4", f"method={export_data.get('method')}"),
        status_check("export_backend_modelopt", export_data.get("backend") == "modelopt", f"backend={export_data.get('backend')}"),
        status_check("export_command_modelopt_nvfp4", "--qformat nvfp4" in command_display or "--qformat nvfp4" in " ".join(map(str, export_data.get("command") or [])), "command includes qformat nvfp4"),
        status_check("serving_success_rate_complete", serving_data.get("success_rate") == 1.0, f"success_rate={serving_data.get('success_rate')}"),
        status_check("serving_eval_scores_exist", eval_scores_path.exists(), f"scores={display_path(eval_scores_path)}"),
        status_check("quantization_card_schema", card_data.get("schema_version") == CARD_SCHEMA_VERSION, f"schema={card_data.get('schema_version')}"),
        status_check("behavior_report_schema", behavior_data.get("schema_version") == BEHAVIOR_REPORT_SCHEMA_VERSION, f"schema={behavior_data.get('schema_version')}"),
        status_check("behavior_preserved", behavior_data.get("behavior_preserved") is True, f"behavior_preserved={behavior_data.get('behavior_preserved')}"),
        status_check("tokenizer_report_schema", tokenizer_data.get("schema_version") == TOKENIZER_REPORT_SCHEMA_VERSION, f"schema={tokenizer_data.get('schema_version')}"),
        status_check("tokenizer_preserved", tokenizer_data.get("passed") is True, f"passed={tokenizer_data.get('passed')}"),
    ]
    absolute_tps = targets["min_output_tokens_per_second"]
    if absolute_tps is not None:
        checks.insert(
            6,
            status_check(
                "output_tps_target_met",
                any(isinstance(value, (int, float)) and float(value) >= absolute_tps for value in (output_tps, decode_heavy_tps)),
                (
                    f"output_tps={output_tps} decode_heavy_output_tps={decode_heavy_tps} "
                    f"min={absolute_tps} source={targets['min_output_tokens_per_second_source']}"
                ),
            ),
        )
    if targets["min_output_speedup"] is not None:
        checks.insert(
            7,
            status_check(
                "output_tps_speedup_target_met",
                isinstance(targets["output_tokens_per_second_speedup"], (int, float))
                and float(targets["output_tokens_per_second_speedup"]) >= float(targets["min_output_speedup"]),
                (
                    f"output_speedup={targets['output_tokens_per_second_speedup']} "
                    f"min={targets['min_output_speedup']}"
                ),
            ),
        )
    if targets["min_decode_heavy_output_speedup"] is not None:
        checks.insert(
            8,
            status_check(
                "decode_heavy_output_tps_speedup_target_met",
                isinstance(targets["decode_heavy_output_tokens_per_second_speedup"], (int, float))
                and float(targets["decode_heavy_output_tokens_per_second_speedup"]) >= float(targets["min_decode_heavy_output_speedup"]),
                (
                    f"decode_heavy_output_speedup={targets['decode_heavy_output_tokens_per_second_speedup']} "
                    f"min={targets['min_decode_heavy_output_speedup']}"
                ),
            ),
        )
    ready = all(check["status"] == "pass" for check in checks if check["required"])
    return redact_value(
        {
            "schema_version": NVFP4_GATE_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": run_id,
            "artifact_paths": {
                "export_plan": display_path(resolve_repo_path(export_plan)),
                "serving_summary": display_path(resolve_repo_path(serving_summary)),
                "serving_eval": display_path(resolve_repo_path(serving_eval)),
                "quantization_card": display_path(resolve_repo_path(quantization_card)),
                "behavior_report": display_path(resolve_repo_path(behavior_report)),
                "tokenizer_report": display_path(resolve_repo_path(tokenizer_report)),
            },
            "metrics": {
                "output_tokens_per_second_p50": output_tps,
                "decode_heavy_output_tokens_per_second_p50": decode_heavy_tps,
                **targets,
            },
            "checks": checks,
            "nvfp4_ready": ready,
            "output_dir": display_path(output_dir),
            "notes": [
                "This gate consumes completed artifacts; it does not run export, serving, or eval jobs.",
                "Blackwell NVFP4 promotion requires export evidence, serving throughput, behavior preservation, tokenizer preservation, and a quantization card.",
                "Configs may declare source-relative throughput gates when absolute tok/s targets are not portable across model families.",
                "For Gemma 4 MoE on DGX Spark, the near-term absolute target is roughly 45-60 output tok/s on decode-heavy workloads.",
            ],
        }
    )


def write_nvfp4_gate_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nvfp4_evidence_gate.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# NVFP4 Evidence Gate: {report.get('run_id')}",
        "",
        f"- Ready: `{str(report.get('nvfp4_ready')).lower()}`",
        f"- Output tok/s p50: `{(report.get('metrics') or {}).get('output_tokens_per_second_p50')}`",
        f"- Decode-heavy output tok/s p50: `{(report.get('metrics') or {}).get('decode_heavy_output_tokens_per_second_p50')}`",
        "",
        "## Checks",
        "",
        "| Status | Required | Check | Message |",
        "|---|---|---|---|",
    ]
    for check in report.get("checks") or []:
        lines.append(
            f"| {str(check.get('status')).upper()} | {'yes' if check.get('required') else 'no'} | "
            f"{check.get('name')} | {check.get('message')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "nvfp4_evidence_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def write_fp8_kv_report_outputs(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fp8_kv_behavior_report.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# FP8 KV Behavior Report: {report.get('run_id')}",
        "",
        f"- Behavior ready: `{str(report.get('behavior_ready')).lower()}`",
        f"- KV cache dtype: `{report.get('kv_cache_dtype')}`",
        f"- Method/backend: `{report.get('method')}` / `{report.get('backend')}`",
        "",
        "## Checks",
        "",
        "| Status | Required | Check | Message |",
        "|---|---|---|---|",
    ]
    for check in report.get("checks") or []:
        lines.append(
            f"| {str(check.get('status')).upper()} | {'yes' if check.get('required') else 'no'} | "
            f"{check.get('name')} | {check.get('message')} |"
        )
    lines.extend(["", "## Serving Deltas", "", "| Metric | Source | Candidate | Delta |", "|---|---:|---:|---:|"])
    for metric, values in (report.get("serving_deltas") or {}).items():
        lines.append(f"| {metric} | {values.get('source')} | {values.get('candidate')} | {values.get('delta')} |")
    (output_dir / "fp8_kv_behavior_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


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


def add_calibration_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("family", nargs="?", help="Optional model family")
    parser.add_argument("variant", nargs="?", help="Optional source variant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="Override calibration dataset list")
    parser.add_argument("--samples", help="Override calibration sample count or comma-separated per-dataset counts")
    parser.add_argument("--seq-len", help="Override calibration sequence length or comma-separated per-dataset lengths")
    parser.add_argument("--output-dir", type=Path, help="Manifest artifact output root")
    parser.add_argument("--run-id", help="Stable output subdirectory name")
    parser.add_argument("--write-manifest", action="store_true", help="Write calibration_manifest.json and calibration_manifest.md")
    parser.add_argument("--json", action="store_true", help="Print JSON manifest")


def matrix_entries(config: QuantizationConfig) -> list[dict[str, Any]]:
    entries = config.matrix.get("variants") or []
    if not isinstance(entries, list):
        raise ValueError("matrix.variants must be a list")
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def matrix_entry_names(entry: Mapping[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            entry.get("name"),
            entry.get("source_variant"),
            entry.get("target_variant"),
        )
        if value
    }


def filter_matrix_entries(entries: list[dict[str, Any]], variants: str | None) -> list[dict[str, Any]]:
    wanted = set(comma_list(variants))
    if not wanted:
        return entries
    filtered = [entry for entry in entries if matrix_entry_names(entry) & wanted]
    matched = set().union(*(matrix_entry_names(entry) for entry in filtered)) if filtered else set()
    missing = sorted(wanted - matched)
    if missing:
        raise ValueError(f"matrix entries not found: {', '.join(missing)}")
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

    calibration_parser = subparsers.add_parser("calibration-manifest", help="Resolve exact calibration inputs for a quantization run")
    add_calibration_manifest_args(calibration_parser)

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

    fp8_kv_parser = subparsers.add_parser("fp8-kv-report", help="Write an FP8 KV source-vs-candidate behavior report")
    fp8_kv_parser.add_argument("--config", type=Path, required=True)
    fp8_kv_parser.add_argument("--source-serving-summary", type=Path, required=True)
    fp8_kv_parser.add_argument("--candidate-serving-summary", type=Path, required=True)
    fp8_kv_parser.add_argument("--source-serving-eval", type=Path, required=True)
    fp8_kv_parser.add_argument("--candidate-serving-eval", type=Path, required=True)
    fp8_kv_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    fp8_kv_parser.add_argument("--run-id", required=True)
    fp8_kv_parser.add_argument("--write-report", action="store_true", help="Write fp8_kv_behavior_report.json and .md")
    fp8_kv_parser.add_argument("--json", action="store_true", help="Print JSON report")

    behavior_parser = subparsers.add_parser("behavior-report", help="Write a quantization behavior-preservation report")
    behavior_parser.add_argument("--config", type=Path, required=True)
    behavior_parser.add_argument("--source-serving-summary", type=Path, required=True)
    behavior_parser.add_argument("--candidate-serving-summary", type=Path, required=True)
    behavior_parser.add_argument("--source-serving-eval", type=Path, required=True)
    behavior_parser.add_argument("--candidate-serving-eval", type=Path, required=True)
    behavior_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    behavior_parser.add_argument("--run-id", required=True)
    behavior_parser.add_argument("--write-report", action="store_true", help="Write behavior_preservation_report.json and .md")
    behavior_parser.add_argument("--json", action="store_true", help="Print JSON report")

    tokenizer_parser = subparsers.add_parser("tokenizer-report", help="Check tokenizer/chat-template preservation for a quantized or GGUF export")
    tokenizer_parser.add_argument("--source-tokenizer-dir", type=Path, required=True)
    tokenizer_parser.add_argument("--candidate-tokenizer-dir", type=Path, required=True)
    tokenizer_parser.add_argument("--source-variant", default="source")
    tokenizer_parser.add_argument("--candidate-variant", default="candidate")
    tokenizer_parser.add_argument("--load-tokenizer", action="store_true", help="Run a live AutoTokenizer chat-template round trip")
    tokenizer_parser.add_argument("--strict", action="store_true", help="Treat missing dirs or skipped round trips as errors")
    tokenizer_parser.add_argument("--trust-remote-code", action="store_true")
    tokenizer_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    tokenizer_parser.add_argument("--run-id", required=True)
    tokenizer_parser.add_argument("--write-report", action="store_true", help="Write tokenizer_preservation_report.json and .md")
    tokenizer_parser.add_argument("--json", action="store_true", help="Print JSON report")

    modelopt_compat_parser = subparsers.add_parser("modelopt-compat-report", help="Check ModelOpt artifact metadata against a serving runtime")
    modelopt_compat_parser.add_argument("--candidate-dir", type=Path, required=True)
    modelopt_compat_parser.add_argument("--runtime", default="vllm", help="Serving runtime to check against")
    modelopt_compat_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    modelopt_compat_parser.add_argument("--run-id", required=True)
    modelopt_compat_parser.add_argument("--write-report", action="store_true", help="Write modelopt_runtime_compat_report.json and .md")
    modelopt_compat_parser.add_argument("--json", action="store_true", help="Print JSON report")

    sensitivity_parser = subparsers.add_parser("sensitivity-report", help="Rank layer/component quantization candidates from completed evidence")
    sensitivity_parser.add_argument("--config", type=Path, required=True)
    sensitivity_parser.add_argument("--baseline-serving-summary", type=Path, required=True)
    sensitivity_parser.add_argument("--baseline-serving-eval", type=Path, required=True)
    sensitivity_parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate key-value list: name=<id>,component=<component>,summary=<summary.json>,eval=<eval-dir>[,policy=<text>,precision=<text>]",
    )
    sensitivity_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    sensitivity_parser.add_argument("--run-id", required=True)
    sensitivity_parser.add_argument("--write-report", action="store_true", help="Write sensitivity_report.json and .md")
    sensitivity_parser.add_argument("--json", action="store_true", help="Print JSON report")

    nvfp4_gate_parser = subparsers.add_parser("nvfp4-gate", help="Validate completed Blackwell NVFP4 promotion evidence")
    nvfp4_gate_parser.add_argument("--export-plan", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--serving-summary", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--serving-eval", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--quantization-card", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--behavior-report", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--tokenizer-report", type=Path, required=True)
    nvfp4_gate_parser.add_argument("--min-output-tps", type=float, default=None)
    nvfp4_gate_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "quantization")
    nvfp4_gate_parser.add_argument("--run-id", required=True)
    nvfp4_gate_parser.add_argument("--write-gate", action="store_true", help="Write nvfp4_evidence_gate.json and .md")
    nvfp4_gate_parser.add_argument("--json", action="store_true", help="Print JSON report")

    args = parser.parse_args()
    config_path = resolve_repo_path(args.config) if hasattr(args, "config") else DEFAULT_CONFIG
    config = load_quantization_config(config_path) if hasattr(args, "config") else None

    if args.command == "tokenizer-report":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_tokenizer_report(
            source_tokenizer_dir=args.source_tokenizer_dir,
            candidate_tokenizer_dir=args.candidate_tokenizer_dir,
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
            source_variant=args.source_variant,
            candidate_variant=args.candidate_variant,
            load_tokenizer=bool(args.load_tokenizer),
            strict=bool(args.strict),
            trust_remote_code=bool(args.trust_remote_code),
        )
        if args.write_report:
            write_tokenizer_report_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="Quantization Tokenizer Preservation Report")
            table.add_column("Field")
            table.add_column("Value")
            table.add_row("passed", str(report["passed"]))
            table.add_row("findings", str(len(report.get("findings") or [])))
            table.add_row("metadata_only", str(report["metadata_only"]))
            console.print(table)
        return

    if args.command == "modelopt-compat-report":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_modelopt_runtime_compat_report(
            candidate_dir=args.candidate_dir,
            runtime=args.runtime,
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
        )
        if args.write_report:
            write_modelopt_runtime_compat_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="ModelOpt Runtime Compatibility Report")
            table.add_column("Check")
            table.add_column("Status")
            table.add_column("Message")
            for check in report.get("checks") or []:
                table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
            console.print(table)
            console.print(f"Compatible: {report.get('passed')}")
        raise SystemExit(0 if report.get("passed") else 1)

    if args.command == "sensitivity-report":
        if not args.candidate:
            raise SystemExit("--candidate is required at least once")
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_sensitivity_report(
            config,
            config_path=config_path,
            baseline_serving_summary=resolve_repo_path(args.baseline_serving_summary),
            baseline_serving_eval=resolve_repo_path(args.baseline_serving_eval),
            candidates=[candidate_arg(raw) for raw in args.candidate],
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
        )
        if args.write_report:
            write_sensitivity_report_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="Quantization Sensitivity Report")
            table.add_column("Rank")
            table.add_column("Name")
            table.add_column("Component")
            table.add_column("Behavior")
            table.add_column("Decode-heavy delta")
            for item in report.get("ranking") or []:
                table.add_row(
                    str(item.get("rank")),
                    str(item.get("name")),
                    str(item.get("component")),
                    str(item.get("behavior_preserved")),
                    str(item.get("decode_heavy_throughput_delta")),
                )
            console.print(table)
        return

    if args.command == "nvfp4-gate":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_nvfp4_gate_report(
            export_plan=args.export_plan,
            serving_summary=args.serving_summary,
            serving_eval=args.serving_eval,
            quantization_card=args.quantization_card,
            behavior_report=args.behavior_report,
            tokenizer_report=args.tokenizer_report,
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
            min_output_tps=float(args.min_output_tps) if args.min_output_tps is not None else None,
        )
        if args.write_gate:
            write_nvfp4_gate_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="NVFP4 Evidence Gate")
            table.add_column("Check")
            table.add_column("Status")
            table.add_column("Message")
            for check in report.get("checks") or []:
                table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
            console.print(table)
            console.print(f"NVFP4 ready: {report.get('nvfp4_ready')}")
        raise SystemExit(0 if report.get("nvfp4_ready") else 1)

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
        export_plan = build_export_command(
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
            try:
                raise SystemExit(execute_export(export_plan))
            except RuntimeError as exc:
                console.print(f"[red]quantization export failed preflight: {exc}[/red]")
                raise SystemExit(1) from None
        return

    if args.command == "calibration-manifest":
        manifest = build_calibration_manifest(
            config,
            config_path=config_path,
            family=args.family,
            variant=args.variant,
            output_dir=args.output_dir,
            run_id=args.run_id,
            dataset=args.dataset,
            samples=args.samples,
            seq_len=args.seq_len,
        )
        if args.write_manifest:
            write_calibration_manifest_outputs(manifest)
        if args.json:
            print(json.dumps(redact_value(manifest), indent=2, sort_keys=True) + "\n")
        else:
            calibration = manifest.get("calibration") or {}
            table = Table(title="Calibration Manifest")
            table.add_column("Field")
            table.add_column("Value")
            for key in ["dataset", "samples", "seq_len", "batch_size"]:
                table.add_row(key, str(calibration.get(key)))
            console.print(table)
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
                calibration=deep_merge_mappings(config.calibration, entry.get("calibration") if isinstance(entry.get("calibration"), Mapping) else None),
                exclusions=deep_merge_mappings(config.exclusions, entry.get("exclusions") if isinstance(entry.get("exclusions"), Mapping) else None),
                runtime=deep_merge_mappings(config.runtime, entry.get("runtime") if isinstance(entry.get("runtime"), Mapping) else None),
                export=deep_merge_mappings(config.export, entry.get("export") if isinstance(entry.get("export"), Mapping) else None),
                matrix=config.matrix,
                outputs=config.outputs,
                evals=config.evals,
                raw_config=config.raw_config,
            )
            run_id = matrix_run_id(config, {"source_variant": source.variant, "target_variant": target_variant})
            plan = build_export_command(variant_config, source, output_dir=output_root, run_id=run_id)
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

    if args.command == "fp8-kv-report":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_fp8_kv_report(
            config,
            config_path=config_path,
            source_serving_summary=resolve_repo_path(args.source_serving_summary),
            candidate_serving_summary=resolve_repo_path(args.candidate_serving_summary),
            source_serving_eval=resolve_repo_path(args.source_serving_eval),
            candidate_serving_eval=resolve_repo_path(args.candidate_serving_eval),
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
        )
        if args.write_report:
            write_fp8_kv_report_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="FP8 KV Behavior Report")
            table.add_column("Check")
            table.add_column("Status")
            table.add_column("Message")
            for check in report.get("checks") or []:
                table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
            console.print(table)
        return

    if args.command == "behavior-report":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_behavior_report(
            config,
            config_path=config_path,
            source_serving_summary=resolve_repo_path(args.source_serving_summary),
            candidate_serving_summary=resolve_repo_path(args.candidate_serving_summary),
            source_serving_eval=resolve_repo_path(args.source_serving_eval),
            candidate_serving_eval=resolve_repo_path(args.candidate_serving_eval),
            output_dir=output_dir,
            run_id=sanitize_run_id(args.run_id),
        )
        if args.write_report:
            write_behavior_report_outputs(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            table = Table(title="Quantization Behavior Preservation Report")
            table.add_column("Check")
            table.add_column("Status")
            table.add_column("Message")
            for check in report.get("checks") or []:
                table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
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
