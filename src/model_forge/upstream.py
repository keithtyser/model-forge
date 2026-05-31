from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "upstream" / "pr_candidates.yaml"
SCHEMA_VERSION = "model_forge.upstream_pr_candidates.v1"
PLAN_SCHEMA_VERSION = "model_forge.upstream_pr_plan.v1"
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")

console = Console(stderr=True)


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None
    candidate: str | None = None


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


def scan_value(value: Any, findings: list[Finding], *, path: str, candidate: str | None = None) -> None:
    if isinstance(value, Mapping):
        next_candidate = str(value.get("id", candidate)) if value.get("id") is not None else candidate
        for child in value.values():
            scan_value(child, findings, path=path, candidate=next_candidate)
    elif isinstance(value, list):
        for item in value:
            scan_value(item, findings, path=path, candidate=candidate)
    elif isinstance(value, str):
        if SECRET_PATTERN.search(value):
            findings.append(Finding("error", "secret_literal", "secret-like literal found", path, candidate))
        if PRIVATE_PATH_PATTERN.search(value):
            findings.append(Finding("error", "private_path", "private absolute path found", path, candidate))


def audit_config(config: Mapping[str, Any], config_path: Path, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    scan_value(config, findings, path=config_display)
    if config.get("schema_version") != SCHEMA_VERSION:
        findings.append(Finding("error", "schema", f"schema_version must be {SCHEMA_VERSION}", config_display))
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        findings.append(Finding("error", "schema", "candidates must be a non-empty list", config_display))
        return findings
    ids: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            findings.append(Finding("error", "candidate", "candidate entries must be mappings", config_display))
            continue
        candidate_id = str(item.get("id") or "")
        ids.append(candidate_id)
        for key in ("id", "target_project", "target_url", "hypothesis", "contribution_type", "next_action"):
            if not item.get(key):
                severity = "error" if strict else "warning"
                findings.append(Finding(severity, "candidate", f"{key} is required", config_display, candidate_id))
        if item.get("status") not in {"candidate", "drafting", "opened", "merged", "rejected", "blocked"}:
            findings.append(Finding("error", "candidate", "status must be candidate/drafting/opened/merged/rejected/blocked", config_display, candidate_id))
        if str(item.get("target_url", "")).startswith("https://github.com/<"):
            findings.append(Finding("warning", "target", "target_url is still a placeholder", config_display, candidate_id))
        if item.get("status") in {"opened", "merged"} and not item.get("external_pr_url"):
            findings.append(Finding("error", "evidence", "opened/merged candidates must record external_pr_url", config_display, candidate_id))
    if len(ids) != len(set(ids)):
        findings.append(Finding("error", "candidate", "candidate ids must be unique", config_display))
    return findings


def candidate_by_id(config: Mapping[str, Any], candidate_id: str | None) -> dict[str, Any]:
    candidates = [item for item in config.get("candidates", []) if isinstance(item, dict)]
    if candidate_id is None:
        if len(candidates) != 1:
            raise ValueError("--candidate is required when config has more than one candidate")
        return dict(candidates[0])
    for item in candidates:
        if str(item.get("id")) == candidate_id:
            return dict(item)
    raise ValueError(f"unknown upstream candidate: {candidate_id}")


def build_plan(config: Mapping[str, Any], config_path: Path, *, candidate_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    candidate = candidate_by_id(config, candidate_id)
    candidate_run_id = sanitize_run_id(run_id or f"upstream_{candidate['id']}")
    output_dir = REPO_DIR / "reports" / "generated" / "upstream_prs" / candidate_run_id
    return redact_value(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "run_id": candidate_run_id,
            "source_config": display_path(config_path),
            "output_dir": display_path(output_dir),
            "policy": dict(config.get("policy") or {}),
            "candidate": candidate,
            "evidence_requirements": {
                "external_pr_url_required_for_completion": True,
                "benchmark_or_profile_required": True,
                "private_path_and_sensitive_literal_scan_required": True,
            },
            "outputs": {
                "plan_json": display_path(output_dir / "upstream_pr_plan.json"),
                "plan_md": display_path(output_dir / "upstream_pr_plan.md"),
            },
            "completion_rule": "MF-0808 remains incomplete until external_pr_url points to a real opened upstream pull request.",
        }
    )


def render_plan_markdown(plan: Mapping[str, Any]) -> str:
    candidate = plan.get("candidate") or {}
    lines = [
        f"# Upstream PR Plan: {plan.get('run_id')}",
        "",
        "## Target",
        "",
        f"- Project: {candidate.get('target_project')}",
        f"- URL: {candidate.get('target_url')}",
        f"- Status: `{candidate.get('status')}`",
        f"- Contribution type: `{candidate.get('contribution_type')}`",
        "",
        "## Hypothesis",
        "",
        str(candidate.get("hypothesis") or ""),
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- `{path}`" for path in candidate.get("local_evidence") or [])
    lines.extend(["", "## Required Validation", ""])
    lines.extend(f"- `{command}`" for command in candidate.get("required_validation") or [])
    lines.extend(
        [
            "",
            "## Completion Rule",
            "",
            f"- {plan.get('completion_rule')}",
            "",
            "## Next Action",
            "",
            f"- {candidate.get('next_action')}",
            "",
        ]
    )
    return "\n".join(lines)


def write_plan(plan: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "upstream_pr_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "upstream_pr_plan.md").write_text(render_plan_markdown(plan), encoding="utf-8")
    return plan_path


def render_findings(findings: list[Finding]) -> None:
    table = Table(title="Upstream PR Candidate Audit")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Candidate")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity, finding.check, finding.candidate or "", finding.message)
    console.print(table)


def render_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title=f"Upstream PR Plan: {plan.get('run_id')}")
    table.add_column("Field")
    table.add_column("Value")
    candidate = plan.get("candidate") or {}
    table.add_row("candidate", str(candidate.get("id")))
    table.add_row("target", str(candidate.get("target_url")))
    table.add_row("status", str(candidate.get("status")))
    table.add_row("output", str(plan.get("output_dir")))
    table.add_row("completion", str(plan.get("completion_rule")))
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan upstream pull requests with evidence requirements")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--candidate")
    plan.add_argument("--run-id")
    plan.add_argument("--write-plan", action="store_true")
    plan.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_yaml(args.config)
    if args.command == "audit":
        findings = audit_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.command == "plan":
        plan = build_plan(config, config_path, candidate_id=args.candidate, run_id=args.run_id)
        if args.write_plan:
            write_plan(plan)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)


if __name__ == "__main__":
    main()
