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

from model_forge.benchmarks.serve import load_config as load_serve_config
from model_forge.cluster.cli import (
    audit_cluster,
    load_cluster_config,
    load_hardware_profile,
    total_declared_gpus,
    total_declared_memory_gb,
)
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "sweeps" / "dgx_spark_vllm_baseline.yaml"
SCHEMA_VERSION = "model_forge.serving_sweep_plan.v1"
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")
IP_ADDRESS_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

console = Console(stderr=True)


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None
    case: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def load_sweep_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    return load_yaml(config_path), config_path


def as_env_map(raw: Mapping[str, Any] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if value is None:
            continue
        env[str(key)] = str(value)
    return env


def check_no_secret_literals(obj: Any, findings: list[Finding], path: str, case: str | None = None) -> None:
    if isinstance(obj, dict):
        next_case = str(obj.get("id", case)) if obj.get("id") is not None else case
        for value in obj.values():
            check_no_secret_literals(value, findings, path, next_case)
    elif isinstance(obj, list):
        for item in obj:
            check_no_secret_literals(item, findings, path, case)
    elif isinstance(obj, str) and SECRET_PATTERN.search(obj):
        findings.append(Finding("error", "secret_literal", "secret-like literal found in sweep config", path, case))


def audit_sweep_config(config: Mapping[str, Any], config_path: Path, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    check_no_secret_literals(config, findings, config_display)

    if config.get("schema_version") != "model_forge.serving_sweep.v1":
        findings.append(Finding("error", "schema", "schema_version must be model_forge.serving_sweep.v1", config_display))
    if not config.get("name"):
        findings.append(Finding("error", "schema", "sweep name is required", config_display))

    for key in ("hardware_profile", "benchmark_config"):
        raw_path = config.get(key)
        if not raw_path:
            findings.append(Finding("error", "schema", f"{key} is required", config_display))
            continue
        path = resolve_repo_path(str(raw_path))
        if not path.exists():
            findings.append(Finding("error", "path", f"{key} path does not exist: {raw_path}", config_display))
        if PRIVATE_PATH_PATTERN.search(str(raw_path)):
            findings.append(Finding("error", "portability", f"{key} must not be a machine-specific absolute path", config_display))

    backend = config.get("backend") or {}
    literal_base_url = backend.get("base_url")
    if literal_base_url and IP_ADDRESS_PATTERN.match(str(literal_base_url).removeprefix("http://").removeprefix("https://").split(":")[0]):
        findings.append(Finding("warning", "portability", "prefer env-backed base URLs for shared sweep configs", config_display))

    policy = config.get("resource_policy") or {}
    if policy.get("max_concurrent_servers", 1) != 1:
        findings.append(Finding("warning", "resource_policy", "serving sweeps should default to one server at a time", config_display))
    if policy.get("max_concurrent_benchmarks", 1) != 1:
        findings.append(Finding("warning", "resource_policy", "benchmark sweeps should default to serial benchmark runs", config_display))
    if not policy.get("require_job_lock", False):
        findings.append(Finding("warning", "resource_policy", "serving sweep job lock is not required", config_display))

    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        findings.append(Finding("error", "schema", "cases must be a non-empty list", config_display))
        return findings
    ids: list[str] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            findings.append(Finding("error", "schema", "case entries must be mappings", config_display))
            continue
        case_id = str(raw_case.get("id", ""))
        ids.append(case_id)
        if not case_id:
            findings.append(Finding("error", "case", "case id is required", config_display))
        if not raw_case.get("hypothesis"):
            severity = "error" if strict else "warning"
            findings.append(Finding(severity, "case", "case hypothesis is missing", config_display, case_id))
        if "env" in raw_case and not isinstance(raw_case["env"], dict):
            findings.append(Finding("error", "case", "case env must be a mapping", config_display, case_id))
    if len(ids) != len(set(ids)):
        findings.append(Finding("error", "case", "case ids must be unique", config_display))
    return findings


def env_value(env: Mapping[str, str], key: str | None) -> str | None:
    if not key:
        return None
    value = env.get(key)
    return value if value not in {None, ""} else None


def resolve_model_and_base_url(
    config: Mapping[str, Any],
    *,
    family: str | None,
    variant: str | None,
    model: str | None,
    base_url: str | None,
    env: Mapping[str, str],
) -> tuple[str | None, str]:
    backend = config.get("backend") or {}
    benchmark_config = resolve_repo_path(str(config["benchmark_config"]))
    serve_config = load_serve_config(
        benchmark_config,
        family=family,
        variant=variant,
        model=model,
        base_url=base_url,
        limit=1,
        env=env,
    )
    resolved_base_url = (
        base_url
        or env_value(env, str(backend.get("base_url_env") or ""))
        or str(backend.get("default_base_url") or serve_config.base_url)
    )
    return serve_config.model, resolved_base_url.rstrip("/")


def case_run_id(sweep_name: str, case_id: str, family: str | None, variant: str | None, model: str | None) -> str:
    model_part = family or model or "model"
    variant_part = variant or "variant"
    return sanitize_run_id(f"{sweep_name}_{model_part}_{variant_part}_{case_id}")


def build_case_command(
    config: Mapping[str, Any],
    case_id: str,
    *,
    family: str | None,
    variant: str | None,
    model: str | None,
    base_url: str,
) -> list[str]:
    workload = config.get("workload") or {}
    benchmark_config = str(config["benchmark_config"])
    output_dir = str(Path(str(config.get("output_dir", "reports/generated/serve_sweeps"))) / case_id)
    command = list(workload.get("benchmark_command") or ["./forge", "bench", "serve"])
    command.extend(["--config", benchmark_config, "--base-url", base_url, "--output-dir", output_dir])
    command.extend(["--run-id", case_run_id(str(config["name"]), case_id, family, variant, model)])
    if family:
        command.extend(["--family", family, "--variant", variant or "base"])
    else:
        command.extend(["--model", model or "${MODEL_FORGE_MODEL}"])
    repetitions = workload.get("repetitions")
    if repetitions is not None:
        command.extend(["--repetitions", str(repetitions)])
    if workload.get("limit") is not None:
        command.extend(["--limit", str(workload["limit"])])
    if workload.get("stream") is False:
        command.append("--no-stream")
    return [str(part) for part in command]


def cluster_summary(cluster_config: str | Path | None, env: Mapping[str, str]) -> dict[str, Any] | None:
    if not cluster_config:
        return None
    cluster, path = load_cluster_config(cluster_config)
    hardware = load_hardware_profile(cluster)
    findings = audit_cluster(cluster, path, hardware=hardware, env=env, strict=True)
    nodes = []
    for node in cluster.get("nodes", []):
        if not isinstance(node, dict):
            continue
        host_env = node.get("host_env")
        user_env = node.get("user_env")
        nodes.append({
            "name": node.get("name"),
            "role": node.get("role"),
            "host": env_value(env, str(host_env)) or f"${host_env}" if host_env else node.get("host"),
            "user": env_value(env, str(user_env)) or f"${user_env}" if user_env else node.get("user"),
            "gpu_count": node.get("gpu_count"),
            "memory_total_gb": node.get("memory_total_gb"),
        })
    return {
        "config": display_path(path),
        "id": cluster.get("id"),
        "node_count": len(cluster.get("nodes", [])),
        "total_declared_gpus": total_declared_gpus(cluster, hardware),
        "total_declared_memory_gb": total_declared_memory_gb(cluster, hardware),
        "serving": cluster.get("serving") or {},
        "doctor_findings": [asdict(finding) for finding in findings],
        "nodes": nodes,
    }


def build_sweep_plan(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    family: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    cluster_config: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    resolved_model, resolved_base_url = resolve_model_and_base_url(
        config,
        family=family,
        variant=variant,
        model=model,
        base_url=base_url,
        env=env,
    )
    fixed_env = as_env_map(config.get("fixed_env") or {})
    cases = []
    for raw_case in config.get("cases", []):
        case_id = str(raw_case["id"])
        server_env = dict(fixed_env)
        server_env.update(as_env_map(raw_case.get("env") or {}))
        bench_env = {
            "MODEL_FORGE_BASE_URL": resolved_base_url,
            "MODEL_FORGE_MODEL": resolved_model,
        }
        command = build_case_command(
            config,
            case_id,
            family=family,
            variant=variant,
            model=resolved_model,
            base_url=resolved_base_url,
        )
        cases.append({
            "id": case_id,
            "description": raw_case.get("description", ""),
            "hypothesis": raw_case.get("hypothesis", ""),
            "server_env": server_env,
            "bench_env": bench_env,
            "bench_command": shlex.join(command),
            "output_dir": display_path(Path(str(config.get("output_dir", "reports/generated/serve_sweeps"))) / case_id),
            "requires_server_restart": True,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "sweep": {
            "name": config.get("name"),
            "config": display_path(config_path),
            "description": config.get("description", ""),
            "hardware_profile": config.get("hardware_profile"),
            "benchmark_config": config.get("benchmark_config"),
            "output_dir": config.get("output_dir"),
        },
        "target": {
            "family": family,
            "variant": variant,
            "model": resolved_model,
            "base_url": resolved_base_url,
        },
        "resource_policy": config.get("resource_policy") or {},
        "cluster": cluster_summary(cluster_config, env),
        "cases": cases,
        "quality_gate": config.get("quality_gate") or {},
        "execution_contract": {
            "dry_run_plan_only": True,
            "starts_server": False,
            "runs_benchmark": False,
            "server_restart_required_between_cases": True,
            "notes": [
                "Start exactly one vLLM serving configuration per case, then run that case's bench_command.",
                "For multi-node DGX Spark, use the env-backed cluster inventory and backend-specific vLLM launcher.",
                "Serving metrics are not promotion evidence without sampled quality and behavior evals under the same config.",
            ],
        },
    }


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("serving sweep doctor: OK")
        return
    table = Table(title="Serving Sweep Doctor")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Case")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity.upper(), finding.check, finding.case or "", finding.message)
    console.print(table)


def render_plan(plan: Mapping[str, Any]) -> None:
    sweep = plan["sweep"]
    target = plan["target"]
    cluster = plan.get("cluster") or {}
    console.print(f"Serving sweep: {sweep['name']} -> {target['model']}")
    if cluster:
        console.print(
            f"Cluster: {cluster.get('id')} ({cluster.get('node_count')} nodes, "
            f"{cluster.get('total_declared_gpus')} GPUs, {cluster.get('total_declared_memory_gb')} GB declared RAM)"
        )
    table = Table(title="Sweep Cases")
    table.add_column("Case")
    table.add_column("Changed Env")
    table.add_column("Bench Command")
    fixed = set((plan.get("cases") or [{}])[0].get("server_env", {}).keys()) if plan.get("cases") else set()
    for case in plan["cases"]:
        changed = {
            key: value
            for key, value in case["server_env"].items()
            if key not in fixed or any(other["server_env"].get(key) != value for other in plan["cases"])
        }
        table.add_row(case["id"], json.dumps(changed, sort_keys=True), case["bench_command"])
    console.print(table)


def write_plan(plan: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    root = resolve_repo_path(output_dir or plan["sweep"].get("output_dir") or "reports/generated/serve_sweeps")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sweep_plan.json"
    path.write_text(json.dumps(redact_value(dict(plan)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path = root / "bench_commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Start the matching vLLM server configuration before each command.",
    ]
    for case in plan["cases"]:
        lines.append("")
        lines.append(f"# Case: {case['id']}")
        for key, value in sorted(case["server_env"].items()):
            lines.append(f"# server env {key}={shlex.quote(str(value))}")
        lines.append(case["bench_command"])
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand serving sweep configs")
    subparsers = parser.add_subparsers(dest="action", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate a serving sweep config")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="Render a dry-run serving sweep plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--family")
    plan.add_argument("--variant", default="base")
    plan.add_argument("--model")
    plan.add_argument("--base-url")
    plan.add_argument("--cluster-config", type=Path)
    plan.add_argument("--write-plan", action="store_true")
    plan.add_argument("--output-dir", type=Path)
    plan.add_argument("--json", action="store_true")

    args = parser.parse_args()
    config, config_path = load_sweep_config(args.config)

    if args.action == "doctor":
        findings = audit_sweep_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2, sort_keys=True) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.action == "plan":
        findings = audit_sweep_config(config, config_path, strict=True)
        if any(finding.severity == "error" for finding in findings):
            if args.json:
                print(json.dumps({"doctor_findings": [asdict(finding) for finding in findings]}, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        sweep_plan = build_sweep_plan(
            config,
            config_path,
            family=args.family,
            variant=args.variant if args.family else None,
            model=args.model,
            base_url=args.base_url,
            cluster_config=args.cluster_config,
        )
        if args.write_plan:
            path = write_plan(sweep_plan, args.output_dir)
            sweep_plan["written_plan"] = display_path(path)
        if args.json:
            print(json.dumps(redact_value(sweep_plan), indent=2, sort_keys=True) + "\n")
        else:
            render_plan(sweep_plan)


if __name__ == "__main__":
    main()
