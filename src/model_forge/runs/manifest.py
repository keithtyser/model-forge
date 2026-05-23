from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil
import yaml
from rich.console import Console
from rich.table import Table

from model_forge.hardware import detect_hardware_profile


REPO_DIR = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "model_forge.run_manifest.v1"
DEFAULT_OUTPUT_DIR = REPO_DIR / "reports" / "generated" / "manifests"
RUN_TYPES = {
    "eval",
    "finetune",
    "ablation",
    "data",
    "serving",
    "quantization",
    "publish",
    "compare",
    "generic",
}
STATUSES = {"planned", "running", "completed", "failed", "skipped"}
SAFE_ENV_PREFIXES = ("MODEL_FORGE_", "VLLM_", "CUDA_", "NVIDIA_", "PYTORCH_", "TORCH_")
SAFE_ENV_KEYS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
}
SECRET_KEY_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "AUTH", "CREDENTIAL")
SECRET_VALUE_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
)

console = Console()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_repo_path(value: str | Path, repo_dir: Path = REPO_DIR) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return repo_dir / path


def display_path(path: str | Path, repo_dir: Path = REPO_DIR) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(repo_dir.resolve()))
    except ValueError:
        return str(path)
    except OSError:
        try:
            return str(path.relative_to(repo_dir))
        except ValueError:
            return str(path)


def run_git(args: list[str], repo_dir: Path = REPO_DIR) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata(repo_dir: Path = REPO_DIR) -> dict[str, Any]:
    status = run_git(["status", "--porcelain"], repo_dir) or ""
    remote = run_git(["remote", "get-url", "origin"], repo_dir)
    return {
        "commit": run_git(["rev-parse", "HEAD"], repo_dir),
        "branch": run_git(["branch", "--show-current"], repo_dir),
        "dirty": bool(status.strip()),
        "dirty_paths": sorted(porcelain_path(line) for line in status.splitlines() if porcelain_path(line)),
        "remote_origin": remote,
    }


def porcelain_path(line: str) -> str:
    if len(line) >= 4 and line[2] == " ":
        return line[3:]
    if len(line) >= 3:
        return line[2:].strip()
    return ""


def redact_env_value(key: str, value: str) -> str:
    upper = key.upper()
    if any(marker in upper for marker in SECRET_KEY_MARKERS):
        return "<redacted>"
    redacted = value
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def redact_value(value: Any, key: str = "") -> Any:
    upper = key.upper()
    if any(marker in upper for marker in SECRET_KEY_MARKERS):
        return "<redacted>"
    if isinstance(value, str):
        return redact_env_value(key, value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(child_key): redact_value(child_value, str(child_key)) for child_key, child_value in value.items()}
    return value


def safe_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    snapshot: dict[str, str] = {}
    for key, value in sorted(env.items()):
        if key in SAFE_ENV_KEYS or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            snapshot[key] = redact_env_value(key, str(value))
    return snapshot


def file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def config_entries(config_paths: list[str | Path], repo_dir: Path = REPO_DIR) -> list[dict[str, Any]]:
    entries = []
    for raw_path in config_paths:
        path = resolve_repo_path(raw_path, repo_dir)
        exists = path.exists()
        entries.append({
            "path": display_path(path, repo_dir),
            "exists": exists,
            "sha256": file_sha256(path) if exists and path.is_file() else None,
        })
    return entries


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("KEY must not be empty")
    value = value.strip()
    if value == "":
        return key, ""
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def key_value_mapping(items: list[str] | None) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for item in items or []:
        key, value = parse_key_value(item)
        mapping[key] = value
    return mapping


def sanitize_run_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "run"


def default_run_id(
    run_type: str,
    family: str | None,
    variant: str | None,
    now: datetime,
) -> str:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    parts = [family or "unknown_family", variant or "unknown_variant", run_type, stamp]
    return sanitize_run_id("_".join(parts))


def hardware_snapshot(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    profile = detect_hardware_profile(env or os.environ)
    return {
        "profile": profile.name,
        "label": profile.label,
        "gpus": [asdict(gpu) for gpu in profile.gpus],
        "notes": list(profile.notes),
    }


def system_snapshot(output_dir: Path | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk_path = output_dir or REPO_DIR
    probe_path = disk_path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        disk = psutil.disk_usage(str(probe_path))
    except OSError:
        disk = psutil.disk_usage(str(REPO_DIR))
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "available_fraction": round(memory.available / memory.total, 4) if memory.total else None,
        },
        "disk": {
            "path": display_path(disk_path),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "free_fraction": round(disk.free / disk.total, 4) if disk.total else None,
        },
    }


def load_family_variant_context(family: str | None, variant: str | None, repo_dir: Path = REPO_DIR) -> dict[str, Any]:
    if not family:
        return {}
    family_config = repo_dir / "configs" / "model_families" / f"{family}.yaml"
    context: dict[str, Any] = {
        "family_config": display_path(family_config, repo_dir),
        "family_config_exists": family_config.exists(),
    }
    if not family_config.exists():
        return context
    data = yaml.safe_load(family_config.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        context["family_name"] = data.get("name") or family
        if variant:
            raw_variant = data.get("variants", {}).get(variant, {})
            if isinstance(raw_variant, dict):
                context["variant_config"] = {
                    key: raw_variant.get(key)
                    for key in (
                        "repo_id",
                        "revision",
                        "local_dir",
                        "served_model_name",
                        "adapter",
                        "base_variant",
                        "serve_strategy",
                        "lora_rank",
                    )
                    if key in raw_variant
                }
    return context


def build_canonical_manifest(
    *,
    run_type: str,
    status: str = "planned",
    family: str | None = None,
    variant: str | None = None,
    objective_profile: str | None = None,
    command: list[str] | str | None = None,
    config_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    repo_dir: Path = REPO_DIR,
) -> dict[str, Any]:
    if run_type not in RUN_TYPES:
        raise ValueError(f"unsupported run_type {run_type!r}; expected one of {sorted(RUN_TYPES)}")
    if status not in STATUSES:
        raise ValueError(f"unsupported status {status!r}; expected one of {sorted(STATUSES)}")
    now = now or utc_now()
    output_path = resolve_repo_path(output_dir, repo_dir) if output_dir else None
    command_argv = redact_value(shlex.split(command) if isinstance(command, str) else list(command or []))
    artifacts = redact_value(dict(artifacts or {}))
    metrics = redact_value(dict(metrics or {}))
    metadata = redact_value(dict(metadata or {}))
    notes = redact_value(list(notes or []))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or default_run_id(run_type, family, variant, now),
        "run_type": run_type,
        "status": status,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "identity": {
            "family": family,
            "variant": variant,
            "objective_profile": objective_profile,
        },
        "source": load_family_variant_context(family, variant, repo_dir),
        "git": git_metadata(repo_dir),
        "command": {
            "argv": command_argv,
            "display": shlex.join(command_argv) if command_argv else None,
            "cwd": display_path(repo_dir, repo_dir),
        },
        "configs": config_entries(config_paths or [], repo_dir),
        "hardware": hardware_snapshot(env),
        "system": system_snapshot(output_path),
        "environment": safe_environment(env),
        "outputs": {
            "output_dir": display_path(output_path, repo_dir) if output_path else None,
            "artifacts": artifacts,
            "metrics": metrics,
        },
        "metadata": metadata,
        "notes": notes,
    }


def manifest_output_path(
    manifest: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_dir: Path = REPO_DIR,
) -> Path:
    run_id = sanitize_run_id(str(manifest["run_id"]))
    return resolve_repo_path(output_dir, repo_dir) / run_id / "manifest.json"


def write_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_dir: Path = REPO_DIR,
) -> Path:
    path = manifest_output_path(manifest, output_dir, repo_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_manifest_summary(manifest: Mapping[str, Any], path: Path | None = None) -> None:
    table = Table(title="Run Manifest")
    table.add_column("Field")
    table.add_column("Value")
    identity = manifest.get("identity", {})
    table.add_row("Run ID", str(manifest.get("run_id", "")))
    table.add_row("Type", str(manifest.get("run_type", "")))
    table.add_row("Status", str(manifest.get("status", "")))
    table.add_row("Family", str(identity.get("family") or ""))
    table.add_row("Variant", str(identity.get("variant") or ""))
    table.add_row("Git", str(manifest.get("git", {}).get("commit") or ""))
    table.add_row("Output", str(manifest.get("outputs", {}).get("output_dir") or ""))
    if path:
        table.add_row("Manifest", display_path(path))
    console.print(table)


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected manifest object in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or inspect canonical model-forge run manifests")
    subparsers = parser.add_subparsers(dest="action", required=True)

    write_parser = subparsers.add_parser("write", help="Write a planned/running/completed run manifest")
    write_parser.add_argument("--run-type", required=True, choices=sorted(RUN_TYPES))
    write_parser.add_argument("--status", default="planned", choices=sorted(STATUSES))
    write_parser.add_argument("--family")
    write_parser.add_argument("--variant")
    write_parser.add_argument("--objective-profile")
    write_parser.add_argument("--config", action="append", default=[], help="Config path used by the run")
    write_parser.add_argument("--artifact", action="append", default=[], help="Artifact mapping KEY=PATH")
    write_parser.add_argument("--metric", action="append", default=[], help="Metric mapping KEY=JSON_VALUE")
    write_parser.add_argument("--metadata", action="append", default=[], help="Metadata mapping KEY=JSON_VALUE")
    write_parser.add_argument("--note", action="append", default=[])
    write_parser.add_argument("--run-id")
    write_parser.add_argument("--command", dest="run_command", help="Command being planned or summarized")
    write_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where manifest is written")
    write_parser.add_argument("--run-output-dir", help="Run output directory recorded inside the manifest")
    write_parser.add_argument("--json", action="store_true", help="Print manifest JSON instead of a summary")

    show_parser = subparsers.add_parser("show", help="Show a manifest")
    show_parser.add_argument("path", type=Path)
    show_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.action == "write":
        command = shlex.split(args.run_command) if args.run_command else []
        manifest = build_canonical_manifest(
            run_type=args.run_type,
            status=args.status,
            family=args.family,
            variant=args.variant,
            objective_profile=args.objective_profile,
            command=command,
            config_paths=args.config,
            output_dir=args.run_output_dir,
            artifacts=key_value_mapping(args.artifact),
            metrics=key_value_mapping(args.metric),
            metadata=key_value_mapping(args.metadata),
            notes=args.note,
            run_id=args.run_id,
        )
        path = write_manifest(manifest, args.output_dir)
        if args.json:
            print(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            render_manifest_summary(manifest, path)
        return

    if args.action == "show":
        manifest = read_manifest(args.path)
        if args.json:
            print(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            render_manifest_summary(manifest, args.path)


if __name__ == "__main__":
    main()
