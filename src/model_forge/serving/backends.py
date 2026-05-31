from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "serving" / "backends" / "sglang_openai.yaml"
DEFAULT_ARCHITECTURE_CONFIG = REPO_DIR / "configs" / "serving" / "architectures" / "distributed_kv_placeholder.yaml"
SUPPORTED_ENGINES = {"sglang", "tensorrt_llm"}
SCHEMA_VERSION = "model_forge.serving_backend.v1"
PLAN_SCHEMA_VERSION = "model_forge.serving_backend_plan.v1"
ARCHITECTURE_SCHEMA_VERSION = "model_forge.serving_architecture.v1"
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")

console = Console(stderr=True)


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None


def engine_label(engine: Any) -> str:
    labels = {"sglang": "SGLang", "tensorrt_llm": "TensorRT-LLM"}
    return labels.get(str(engine), str(engine))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_repo_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(resolved)}")
    return data, resolved


def env_value(env: Mapping[str, str], key: str | None) -> str | None:
    if not key:
        return None
    value = env.get(key)
    return value if value not in {None, ""} else None


def scan_value(value: Any, findings: list[Finding], *, path: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            scan_value(child, findings, path=path)
    elif isinstance(value, list):
        for child in value:
            scan_value(child, findings, path=path)
    elif isinstance(value, str):
        if SECRET_PATTERN.search(value):
            findings.append(Finding("error", "secret_literal", "secret-like literal found", path))
        if PRIVATE_PATH_PATTERN.search(value):
            findings.append(Finding("error", "private_path", "private absolute path found", path))


def audit_config(config: Mapping[str, Any], config_path: Path, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    scan_value(config, findings, path=config_display)
    if config.get("schema_version") != SCHEMA_VERSION:
        findings.append(Finding("error", "schema", f"schema_version must be {SCHEMA_VERSION}", config_display))
    if config.get("engine") not in SUPPORTED_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_ENGINES))
        findings.append(Finding("error", "engine", f"engine must be one of: {supported}", config_display))
    entrypoint = config.get("entrypoint") or {}
    if not isinstance(entrypoint.get("command"), list) or not entrypoint["command"]:
        findings.append(Finding("error", "entrypoint", "entrypoint.command must be a non-empty list", config_display))
    network = config.get("network") or {}
    if int(network.get("port", 0)) <= 0:
        findings.append(Finding("error", "network", "network.port must be positive", config_display))
    for key in ("smoke_config", "core_config"):
        raw = (config.get("benchmarks") or {}).get(key)
        if not raw:
            findings.append(Finding("error" if strict else "warning", "benchmarks", f"{key} is missing", config_display))
        elif not resolve_repo_path(str(raw)).exists():
            findings.append(Finding("error", "benchmarks", f"{key} does not exist: {raw}", config_display))
    policy = config.get("resource_policy") or {}
    if int(policy.get("max_concurrent_servers", 1)) != 1:
        findings.append(Finding("warning", "resource_policy", "serving backend should default to one server at a time", config_display))
    if not policy.get("prefer_systemd_scope", False):
        findings.append(Finding("warning", "resource_policy", "prefer_systemd_scope should be true for Spark safety", config_display))
    return findings


def audit_architecture_config(config: Mapping[str, Any], config_path: Path, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    scan_value(config, findings, path=config_display)
    if config.get("schema_version") != ARCHITECTURE_SCHEMA_VERSION:
        findings.append(Finding("error", "schema", f"schema_version must be {ARCHITECTURE_SCHEMA_VERSION}", config_display))
    if not config.get("name"):
        findings.append(Finding("error", "schema", "architecture name is required", config_display))
    components = config.get("components")
    if not isinstance(components, list) or not components:
        findings.append(Finding("error", "components", "components must be a non-empty list", config_display))
    else:
        component_ids = []
        for component in components:
            if not isinstance(component, dict):
                findings.append(Finding("error", "components", "component entries must be mappings", config_display))
                continue
            component_ids.append(str(component.get("id") or ""))
            if not component.get("role"):
                component_id = str(component.get("id") or "<unknown>")
                findings.append(Finding("error", "components", f"component role is required: {component_id}", config_display))
        if len(component_ids) != len(set(component_ids)):
            findings.append(Finding("error", "components", "component ids must be unique", config_display))
    for key in ("promotion_blockers", "validation_gates", "open_questions"):
        value = config.get(key)
        severity = "error" if strict else "warning"
        if not isinstance(value, list) or not value:
            findings.append(Finding(severity, "architecture", f"{key} must be a non-empty list", config_display))
    return findings


def load_family(family: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    if not family:
        return None, None
    path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not path.exists():
        raise ValueError(f"unknown model family: {family}")
    data, resolved = load_yaml(path)
    return data, resolved


def resolve_family_variant(
    family: str | None,
    variant: str | None,
    *,
    env: Mapping[str, str],
) -> tuple[str | None, str | None, str | None, list[str]]:
    family_config, _ = load_family(family)
    if not family_config:
        return None, None, None, []
    variant_name = variant or "base"
    raw_variant = (family_config.get("variants") or {}).get(variant_name)
    if not isinstance(raw_variant, dict):
        raise ValueError(f"family {family!r} has no variant {variant_name!r}")
    env_name = str(family_config.get("models_dir_env") or "MODEL_FORGE_MODELS_DIR")
    explicit_models_dir = env_value(env, env_name)
    local_dir = raw_variant.get("local_dir")
    repo_id = raw_variant.get("repo_id")
    served_name = raw_variant.get("served_model_name") or repo_id or local_dir
    model_path = None
    evidence = [f"configs/model_families/{family}.yaml"]
    if local_dir and explicit_models_dir:
        models_dir = Path(explicit_models_dir).expanduser()
        local_path = Path(str(local_dir)).expanduser()
        model_path = str(local_path if local_path.is_absolute() else models_dir / local_path)
    elif repo_id and not str(served_name or "").startswith("local/"):
        model_path = str(repo_id)
    elif local_dir:
        model_path = f"${{{env_name}}}/{local_dir}"
    elif repo_id:
        model_path = str(repo_id)
    return model_path, str(served_name) if served_name else None, variant_name, evidence


def shell_assignments(assignments: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in assignments.items() if value not in {None, ""})


def build_sglang_launch_command(
    config: Mapping[str, Any], *, model_path: str, served_model_name: str, env: Mapping[str, str]
) -> list[str]:
    command = [str(part) for part in (config.get("entrypoint") or {}).get("command", [])]
    network = config.get("network") or {}
    runtime = config.get("runtime") or {}
    command.extend(["--model-path", model_path])
    command.extend(["--host", str(network.get("host", "0.0.0.0"))])
    command.extend(["--port", str(network.get("port", 8000))])
    command.extend(["--served-model-name", served_model_name])
    if runtime.get("dtype"):
        command.extend(["--dtype", str(runtime["dtype"])])
    if runtime.get("context_length"):
        command.extend(["--context-length", str(runtime["context_length"])])
    tp = env_value(env, str(runtime.get("tensor_parallel_size_env") or ""))
    if tp:
        command.extend(["--tp", tp])
    dp = env_value(env, str(runtime.get("data_parallel_size_env") or ""))
    if dp:
        command.extend(["--dp", dp])
    extra = env_value(env, str(runtime.get("extra_args_env") or ""))
    if extra:
        command.extend(shlex.split(extra))
    return command


def build_trtllm_launch_command(config: Mapping[str, Any], *, model_path: str, env: Mapping[str, str]) -> list[str]:
    command = [str(part) for part in (config.get("entrypoint") or {}).get("command", [])]
    network = config.get("network") or {}
    runtime = config.get("runtime") or {}
    command.extend(["--host", str(network.get("host", "0.0.0.0"))])
    command.extend(["--port", str(network.get("port", 8000))])
    if runtime.get("backend"):
        command.extend(["--backend", str(runtime["backend"])])
    if runtime.get("max_seq_len"):
        command.extend(["--max_seq_len", str(runtime["max_seq_len"])])
    tokenizer = env_value(env, str(runtime.get("tokenizer_env") or ""))
    if tokenizer:
        command.extend(["--tokenizer", tokenizer])
    for env_key, flag in (
        ("tensor_parallel_size_env", "--tp_size"),
        ("pipeline_parallel_size_env", "--pp_size"),
        ("expert_parallel_size_env", "--ep_size"),
    ):
        value = env_value(env, str(runtime.get(env_key) or ""))
        if value:
            command.extend([flag, value])
    extra = env_value(env, str(runtime.get("extra_args_env") or ""))
    if extra:
        command.extend(shlex.split(extra))
    command.append(model_path)
    return command


def build_launch_command(config: Mapping[str, Any], *, model_path: str, served_model_name: str, env: Mapping[str, str]) -> list[str]:
    engine = str(config.get("engine") or "")
    if engine == "sglang":
        return build_sglang_launch_command(config, model_path=model_path, served_model_name=served_model_name, env=env)
    if engine == "tensorrt_llm":
        return build_trtllm_launch_command(config, model_path=model_path, env=env)
    raise ValueError(f"unsupported serving engine: {engine}")


def build_plan(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    family: str | None = None,
    variant: str | None = None,
    model_path: str | None = None,
    served_model_name: str | None = None,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    resolved_model_path, resolved_served_name, resolved_variant, family_evidence = resolve_family_variant(family, variant, env=env)
    model_cfg = config.get("model") or {}
    final_model_path = model_path or env_value(env, str(model_cfg.get("model_path_env") or "")) or resolved_model_path
    final_served_name = served_model_name or env_value(env, str(model_cfg.get("served_model_name_env") or "")) or resolved_served_name or final_model_path
    if not final_model_path:
        raise ValueError("serving backend plan needs --model-path, MODEL_FORGE_MODEL_PATH, or --family/--variant")
    if not final_served_name:
        raise ValueError("serving backend plan needs --served-model-name, MODEL_FORGE_MODEL, or --family/--variant")
    network = config.get("network") or {}
    base_url = f"http://127.0.0.1:{int(network.get('port', 8000))}/v1"
    candidate_run_id = sanitize_run_id(run_id or f"{config.get('name', 'serving_backend')}_{family or 'model'}_{resolved_variant or variant or 'manual'}")
    output_dir = REPO_DIR / "reports" / "generated" / "serving_backends" / candidate_run_id
    launch_command = build_launch_command(config, model_path=str(final_model_path), served_model_name=str(final_served_name), env=env)
    benchmark_config = (config.get("benchmarks") or {}).get("smoke_config", "configs/serving/serve_bench_smoke.yaml")
    bench_command = [
        "./forge",
        "bench",
        "serve",
        "--config",
        str(benchmark_config),
        "--model",
        str(final_served_name),
        "--base-url",
        base_url,
        "--run-id",
        f"{candidate_run_id}_smoke",
    ]
    env_assignments = {
        "MODEL_FORGE_MODEL_PATH": final_model_path,
        "MODEL_FORGE_MODEL": final_served_name,
        "MODEL_FORGE_BASE_URL": base_url,
    }
    return redact_value(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "run_id": candidate_run_id,
            "source_config": display_path(config_path),
            "output_dir": display_path(output_dir),
            "engine": config.get("engine"),
            "model": {
                "family": family,
                "variant": resolved_variant or variant,
                "model_path": str(final_model_path),
                "served_model_name": str(final_served_name),
            },
            "network": {
                "base_url": base_url,
                "host": network.get("host", "0.0.0.0"),
                "port": int(network.get("port", 8000)),
            },
            "resource_policy": dict(config.get("resource_policy") or {}),
            "launch": {
                "command": launch_command,
                "shell": f"{shell_assignments(env_assignments)} {' '.join(shlex.quote(part) for part in launch_command)}",
                "execute_by_default": False,
            },
            "benchmarks": {
                "smoke_command": bench_command,
                "smoke_shell": " ".join(shlex.quote(part) for part in bench_command),
                "core_config": (config.get("benchmarks") or {}).get("core_config"),
            },
            "research_basis": list(config.get("research_basis") or []),
            "evidence_sources": [display_path(config_path), *family_evidence],
            "outputs": {
                "plan_json": display_path(output_dir / "serving_backend_plan.json"),
                "plan_md": display_path(output_dir / "serving_backend_plan.md"),
            },
            "notes": [
                f"This plan does not start {engine_label(config.get('engine'))}.",
                "Start only one large model server at a time.",
                "Run the smoke serving benchmark before comparing engines.",
            ],
        }
    )


def render_plan_markdown(plan: Mapping[str, Any]) -> str:
    model = plan.get("model") or {}
    launch = plan.get("launch") or {}
    benchmarks = plan.get("benchmarks") or {}
    lines = [
        f"# Serving Backend Plan: {plan.get('run_id')}",
        "",
        "## Engine",
        "",
        f"- Engine: `{plan.get('engine')}`",
        f"- Model path: `{model.get('model_path')}`",
        f"- Served model name: `{model.get('served_model_name')}`",
        f"- Base URL: `{(plan.get('network') or {}).get('base_url')}`",
        "",
        "## Launch",
        "",
        "```bash",
        str(launch.get("shell")),
        "```",
        "",
        "## Smoke Benchmark",
        "",
        "```bash",
        str(benchmarks.get("smoke_shell")),
        "```",
        "",
        "## Resource Policy",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in (plan.get("resource_policy") or {}).items())
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in plan.get("notes") or [])
    lines.append("")
    return "\n".join(lines)


def write_plan(plan: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "serving_backend_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "serving_backend_plan.md").write_text(render_plan_markdown(plan), encoding="utf-8")
    return plan_path


def render_findings(findings: list[Finding]) -> None:
    table = Table(title="Serving Backend Audit")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity, finding.check, finding.message)
    console.print(table)


def render_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title=f"Serving Backend Plan: {plan.get('run_id')}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("engine", str(plan.get("engine")))
    table.add_row("model", str((plan.get("model") or {}).get("served_model_name")))
    table.add_row("base_url", str((plan.get("network") or {}).get("base_url")))
    table.add_row("output", str(plan.get("output_dir")))
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan serving backend launch and benchmark commands")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true")
    architecture = sub.add_parser("architecture-doctor")
    architecture.add_argument("--config", type=Path, default=DEFAULT_ARCHITECTURE_CONFIG)
    architecture.add_argument("--strict", action="store_true")
    architecture.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--family")
    plan.add_argument("--variant")
    plan.add_argument("--model-path")
    plan.add_argument("--served-model-name")
    plan.add_argument("--run-id")
    plan.add_argument("--write-plan", action="store_true")
    plan.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_yaml(args.config)
    if args.command == "doctor":
        findings = audit_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.command == "architecture-doctor":
        findings = audit_architecture_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.command == "plan":
        plan = build_plan(
            config,
            config_path,
            family=args.family,
            variant=args.variant,
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            run_id=args.run_id,
        )
        if args.write_plan:
            write_plan(plan)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)


if __name__ == "__main__":
    main()
