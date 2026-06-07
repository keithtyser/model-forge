from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.diagnostics import Finding, severity_exit_code
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "profiling" / "nsight_serving_smoke.yaml"
SCHEMA_VERSION = "model_forge.nsight_profile.v1"
PLAN_SCHEMA_VERSION = "model_forge.nsight_profile_plan.v1"
SUMMARY_SCHEMA_VERSION = "model_forge.profile_summary.v1"
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")

console = Console()


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
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    return load_yaml(config_path), config_path


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {display_path(path)}")
    return data


def check_no_secret_literals(value: Any, findings: list[Finding], path: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            check_no_secret_literals(child, findings, path)
    elif isinstance(value, list):
        for child in value:
            check_no_secret_literals(child, findings, path)
    elif isinstance(value, str) and SECRET_PATTERN.search(value):
        findings.append(Finding("error", "secret_literal", "secret-like literal found in Nsight profile config", path))


def audit_config(config: Mapping[str, Any], config_path: Path, *, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    check_no_secret_literals(config, findings, config_display)
    if config.get("schema_version") != SCHEMA_VERSION:
        findings.append(Finding("error", "schema", f"schema_version must be {SCHEMA_VERSION}", config_display))
    if not config.get("name"):
        findings.append(Finding("error", "schema", "name is required", config_display))

    command = config.get("command")
    if not isinstance(command, list) or not command:
        findings.append(Finding("error", "schema", "command must be a non-empty list", config_display))
    else:
        joined = " ".join(str(part) for part in command)
        if SECRET_PATTERN.search(joined):
            findings.append(Finding("error", "secret_literal", "command contains a secret-like literal", config_display))

    output_root = str((config.get("outputs") or {}).get("root") or "")
    if output_root and PRIVATE_PATH_PATTERN.search(output_root):
        findings.append(Finding("error", "portability", "outputs.root must not be a machine-specific absolute path", config_display))

    tools = config.get("tools") or {}
    enabled = [name for name, raw in tools.items() if isinstance(raw, Mapping) and raw.get("enabled", True)]
    if not enabled:
        findings.append(Finding("error", "schema", "at least one Nsight tool must be enabled", config_display))
    for name in enabled:
        if name not in {"nsys", "ncu"}:
            findings.append(Finding("error", "schema", f"unsupported Nsight tool: {name}", config_display))
        elif shutil.which(name) is None:
            severity = "error" if strict else "warning"
            findings.append(Finding(severity, "tool_available", f"{name} is not available on PATH", config_display))

    policy = config.get("resource_policy") or {}
    if int(policy.get("max_concurrent_profiles", 1)) != 1:
        findings.append(Finding("warning", "resource_policy", "Nsight profiling should run one profile at a time", config_display))
    if not policy.get("require_existing_server", True):
        findings.append(Finding("warning", "resource_policy", "profile configs should default to an already-running server", config_display))
    return findings


def output_path(output_root: Path, run_id: str, suffix: str) -> Path:
    return output_root / run_id / suffix


def build_nsys_command(tool: Mapping[str, Any], target_command: list[str], output: Path) -> list[str]:
    trace = ",".join(str(item) for item in tool.get("trace", ["cuda", "nvtx", "osrt"]))
    command = [
        "nsys",
        "profile",
        "--trace",
        trace,
        "--sample",
        str(tool.get("sample", "cpu")),
        "--force-overwrite",
        str(tool.get("force_overwrite", True)).lower(),
        "--output",
        str(output.with_suffix("")),
    ]
    if tool.get("capture_range"):
        command.extend(["--capture-range", str(tool["capture_range"])])
    command.extend(["--", *target_command])
    return command


def build_ncu_command(tool: Mapping[str, Any], target_command: list[str], output: Path) -> list[str]:
    command = [
        "ncu",
        "--target-processes",
        str(tool.get("target_processes", "all")),
        "--force-overwrite",
        "--export",
        str(output.with_suffix("")),
    ]
    if tool.get("set"):
        command.extend(["--set", str(tool["set"])])
    for metric in tool.get("metrics") or []:
        command.extend(["--metrics", str(metric)])
    command.extend(["--", *target_command])
    return command


def build_plan(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    run_id: str | None = None,
    command: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    actual_run_id = sanitize_run_id(run_id or str(config.get("name") or "nsight_profile"))
    target_command = shlex.split(command) if command else [str(part) for part in config.get("command") or []]
    output_root = resolve_repo_path(output_root or (config.get("outputs") or {}).get("root") or "reports/generated/profiles/nsight")
    tools = config.get("tools") or {}
    profiles: list[dict[str, Any]] = []
    if (tools.get("nsys") or {}).get("enabled", True):
        path = output_path(output_root, actual_run_id, "profile_nsys.nsys-rep")
        profiles.append(
            {
                "tool": "nsys",
                "available": shutil.which("nsys") is not None,
                "output": display_path(path),
                "command": build_nsys_command(tools.get("nsys") or {}, target_command, path),
            }
        )
    if (tools.get("ncu") or {}).get("enabled", False):
        path = output_path(output_root, actual_run_id, "profile_ncu.ncu-rep")
        profiles.append(
            {
                "tool": "ncu",
                "available": shutil.which("ncu") is not None,
                "output": display_path(path),
                "command": build_ncu_command(tools.get("ncu") or {}, target_command, path),
            }
        )
    return redact_value(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "run_id": actual_run_id,
            "config": display_path(config_path),
            "profile": {
                "name": config.get("name"),
                "description": config.get("description"),
                "kind": config.get("kind", "serving"),
                "target_command": target_command,
            },
            "resource_policy": dict(config.get("resource_policy") or {}),
            "profiles": profiles,
            "outputs": {
                "root": display_path(output_root),
                "plan": display_path(output_root / actual_run_id / "nsight_profile_plan.json"),
                "commands": display_path(output_root / actual_run_id / "profile_commands.sh"),
            },
            "execution_contract": {
                "dry_run_by_default": True,
                "starts_server": False,
                "requires_existing_server": bool((config.get("resource_policy") or {}).get("require_existing_server", True)),
                "run_one_profile_at_a_time": True,
            },
        }
    )


def render_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title="Nsight Profile Plan")
    table.add_column("Tool")
    table.add_column("Available")
    table.add_column("Output")
    for profile in plan.get("profiles") or []:
        table.add_row(str(profile["tool"]), str(profile["available"]), str(profile["output"]))
    console.print(table)
    console.print(f"Run id: {plan['run_id']}")
    console.print("Target: " + shlex.join(plan["profile"]["target_command"]))


def write_plan(plan: Mapping[str, Any], output_root: str | Path | None = None) -> Path:
    root = resolve_repo_path(output_root or (plan.get("outputs") or {}).get("root") or "reports/generated/profiles/nsight")
    run_dir = root / str(plan["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "nsight_profile_plan.json"
    commands_path = run_dir / "profile_commands.sh"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for profile in plan.get("profiles") or []:
        lines.append("# " + str(profile["tool"]))
        lines.append(shlex.join(str(part) for part in profile["command"]))
        lines.append("")
    commands_path.write_text("\n".join(lines), encoding="utf-8")
    commands_path.chmod(0o755)
    return plan_path


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    exists = resolved.exists()
    return {
        "path": display_path(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists and resolved.is_file() else None,
    }


def build_summary(plan: Mapping[str, Any], *, plan_path: Path | None = None, extra_artifacts: list[str] | None = None) -> dict[str, Any]:
    profiles = list(plan.get("profiles") or [])
    profile_artifacts = [artifact_record(str(profile.get("output") or "")) for profile in profiles if profile.get("output")]
    extra_records = [artifact_record(path) for path in extra_artifacts or []]
    present = [record for record in [*profile_artifacts, *extra_records] if record["exists"]]
    missing = [record for record in [*profile_artifacts, *extra_records] if not record["exists"]]
    return redact_value(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "source_plan": display_path(plan_path) if plan_path else None,
            "run_id": plan.get("run_id"),
            "profile": dict(plan.get("profile") or {}),
            "execution_contract": dict(plan.get("execution_contract") or {}),
            "profile_artifacts": profile_artifacts,
            "extra_artifacts": extra_records,
            "summary": {
                "expected_profile_artifacts": len(profile_artifacts),
                "present_profile_artifacts": len([record for record in profile_artifacts if record["exists"]]),
                "missing_profile_artifacts": len([record for record in profile_artifacts if not record["exists"]]),
                "present_total_artifacts": len(present),
                "missing_total_artifacts": len(missing),
                "total_present_size_bytes": sum(int(record["size_bytes"] or 0) for record in present),
                "tools": sorted({str(profile.get("tool")) for profile in profiles if profile.get("tool")}),
            },
            "notes": [
                "This summary records profiler artifact presence and sizes.",
                "Use Nsight Systems/Compute reports or exported stats for kernel-level interpretation.",
            ],
        }
    )


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("summary") or {}
    lines = [
        f"# Profile Summary: {summary.get('run_id')}",
        "",
        "## Source",
        "",
        f"- Plan: `{summary.get('source_plan')}`",
        f"- Target command: `{shlex.join((summary.get('profile') or {}).get('target_command') or [])}`",
        f"- Tools: `{', '.join(metrics.get('tools') or [])}`",
        "",
        "## Artifact Status",
        "",
        f"- Expected profile artifacts: {metrics.get('expected_profile_artifacts')}",
        f"- Present profile artifacts: {metrics.get('present_profile_artifacts')}",
        f"- Missing profile artifacts: {metrics.get('missing_profile_artifacts')}",
        f"- Total present artifact bytes: {metrics.get('total_present_size_bytes')}",
        "",
        "## Profile Artifacts",
        "",
    ]
    for record in summary.get("profile_artifacts") or []:
        status = "present" if record.get("exists") else "missing"
        lines.append(f"- `{record.get('path')}`: {status}, bytes={record.get('size_bytes')}")
    extra = summary.get("extra_artifacts") or []
    if extra:
        lines.extend(["", "## Extra Artifacts", ""])
        for record in extra:
            status = "present" if record.get("exists") else "missing"
            lines.append(f"- `{record.get('path')}`: {status}, bytes={record.get('size_bytes')}")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary.get("notes") or [])
    lines.append("")
    return "\n".join(lines)


def write_summary(summary: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    if output_dir:
        root = resolve_repo_path(output_dir)
    elif summary.get("source_plan"):
        root = resolve_repo_path(str(summary["source_plan"])).parent
    else:
        root = REPO_DIR / "reports" / "generated" / "profiles" / "nsight" / str(summary.get("run_id") or "profile_summary")
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "profile_summary.json"
    md_path = root / "profile_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    return json_path


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("[green]nsight profile config OK[/green]")
        return
    table = Table(title="Nsight Profile Findings")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity, finding.check, finding.message)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Nsight profiling runs around Model Forge commands")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Validate an Nsight profiling config")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true")

    plan_parser = sub.add_parser("plan", help="Build Nsight profile command plan")
    plan_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan_parser.add_argument("--run-id")
    plan_parser.add_argument("--command", dest="profile_command", help="Override profiled command")
    plan_parser.add_argument("--output-root", type=Path)
    plan_parser.add_argument("--write-plan", action="store_true")
    plan_parser.add_argument("--json", action="store_true")

    summary_parser = sub.add_parser("summarize", help="Summarize expected and present Nsight profile artifacts")
    summary_parser.add_argument("--plan", type=Path, required=True)
    summary_parser.add_argument("--artifact", action="append", default=[], help="Additional artifact path to include")
    summary_parser.add_argument("--output-dir", type=Path)
    summary_parser.add_argument("--write-summary", action="store_true")
    summary_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "doctor":
        config, config_path = load_config(args.config)
        findings = audit_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(severity_exit_code(findings))

    if args.command == "plan":
        config, config_path = load_config(args.config)
        plan = build_plan(config, config_path, run_id=args.run_id, command=args.profile_command, output_root=args.output_root)
        if args.write_plan:
            write_plan(plan, args.output_root)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
        return

    if args.command == "summarize":
        plan_path = resolve_repo_path(args.plan)
        plan = load_json(plan_path)
        summary = build_summary(plan, plan_path=plan_path, extra_artifacts=args.artifact)
        if args.write_summary:
            write_summary(summary, args.output_dir)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        else:
            print(render_summary_markdown(summary))


if __name__ == "__main__":
    main()
