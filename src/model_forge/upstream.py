from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.diagnostics import severity_exit_code
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "upstream" / "pr_candidates.yaml"
SCHEMA_VERSION = "model_forge.upstream_pr_candidates.v1"
PLAN_SCHEMA_VERSION = "model_forge.upstream_pr_plan.v1"
VERIFICATION_SCHEMA_VERSION = "model_forge.upstream_pr_verification.v1"
APPLY_DRAFT_SCHEMA_VERSION = "model_forge.upstream_pr_apply_draft.v1"
SECRET_PATTERN = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
PRIVATE_PATH_PATTERN = re.compile(r"^/(home|Users)/[^/]+/")
GITHUB_PR_URL_PATTERN = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/pull/[0-9]+/?$")
GITHUB_PR_PARSE_PATTERN = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/(?P<number>[0-9]+)/?$")
PLACEHOLDER_PATTERN = re.compile(r"<[^>]+>")

console = Console(stderr=True)


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None
    candidate: str | None = None


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    status: str
    message: str


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


def has_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(PLACEHOLDER_PATTERN.search(value))


def check_status(name: str, passed: bool, message: str) -> VerificationCheck:
    return VerificationCheck(name=name, status="pass" if passed else "fail", message=message)


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
        target_url = str(item.get("target_url", ""))
        if has_placeholder(target_url) or target_url.startswith("https://github.com/<"):
            severity = "error" if strict else "warning"
            findings.append(Finding(severity, "target", "target_url is still a placeholder", config_display, candidate_id))
        external_pr_url = str(item.get("external_pr_url") or "")
        if item.get("status") in {"opened", "merged"}:
            if not external_pr_url:
                findings.append(Finding("error", "evidence", "opened/merged candidates must record external_pr_url", config_display, candidate_id))
            elif not GITHUB_PR_URL_PATTERN.match(external_pr_url):
                findings.append(Finding("error", "evidence", "external_pr_url must be a GitHub pull request URL", config_display, candidate_id))
            evidence_paths = item.get("local_evidence") or []
            if not isinstance(evidence_paths, list) or not evidence_paths:
                findings.append(Finding("error", "evidence", "opened/merged candidates must record local_evidence", config_display, candidate_id))
            for raw_path in evidence_paths if isinstance(evidence_paths, list) else []:
                evidence = str(raw_path)
                if has_placeholder(evidence):
                    findings.append(Finding("error", "evidence", f"local_evidence contains unresolved placeholder: {evidence}", config_display, candidate_id))
                    continue
                if not resolve_repo_path(evidence).exists():
                    findings.append(Finding("error", "evidence", f"local_evidence path does not exist: {evidence}", config_display, candidate_id))
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


def parse_github_pr_url(url: str) -> dict[str, str] | None:
    match = GITHUB_PR_PARSE_PATTERN.match(url)
    if not match:
        return None
    return match.groupdict()


def fetch_github_pr_metadata(url: str, *, timeout: int = 20) -> tuple[dict[str, Any] | None, str | None]:
    parsed = parse_github_pr_url(url)
    if not parsed:
        return None, "external_pr_url is not a GitHub pull request URL"
    api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}/pulls/{parsed['number']}"
    request = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json", "User-Agent": "model-forge-upstream-verifier"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"GitHub API returned HTTP {exc.code}"
    except Exception as exc:
        return None, f"GitHub API request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "GitHub API response was not an object"
    return payload, None


def evidence_checks(paths: list[Any]) -> tuple[list[VerificationCheck], list[str]]:
    checks: list[VerificationCheck] = []
    resolved_paths: list[str] = []
    checks.append(check_status("local_evidence_present", bool(paths), "candidate records local_evidence paths"))
    for index, raw_path in enumerate(paths):
        evidence = str(raw_path)
        if has_placeholder(evidence):
            checks.append(check_status(f"local_evidence_{index}_no_placeholder", False, f"unresolved placeholder: {evidence}"))
            continue
        path = resolve_repo_path(evidence)
        exists = path.exists()
        checks.append(check_status(f"local_evidence_{index}_exists", exists, display_path(path)))
        if exists:
            resolved_paths.append(display_path(path))
            findings: list[Finding] = []
            scan_value(path.read_text(encoding="utf-8", errors="replace") if path.is_file() and path.stat().st_size <= 2_000_000 else display_path(path), findings, path=display_path(path))
            checks.append(check_status(f"local_evidence_{index}_no_sensitive_literals", not findings, "; ".join(f.message for f in findings) or "no private paths or secret-like literals found"))
    return checks, resolved_paths


def build_verification(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    candidate_id: str | None = None,
    offline: bool = False,
    require_merged: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    candidate = candidate_by_id(config, candidate_id)
    candidate_run_id = sanitize_run_id(run_id or f"upstream_verify_{candidate['id']}")
    output_dir = REPO_DIR / "reports" / "generated" / "upstream_prs" / candidate_run_id
    external_pr_url = str(candidate.get("external_pr_url") or "")
    checks = [
        check_status("candidate_status_opened_or_merged", candidate.get("status") in {"opened", "merged"}, f"status={candidate.get('status')}"),
        check_status("target_url_concrete", bool(candidate.get("target_url")) and not has_placeholder(candidate.get("target_url")), str(candidate.get("target_url") or "")),
        check_status("external_pr_url_present", bool(external_pr_url), external_pr_url or "missing external_pr_url"),
        check_status("external_pr_url_shape", bool(parse_github_pr_url(external_pr_url)), external_pr_url or "missing external_pr_url"),
    ]
    evidence, evidence_paths = evidence_checks(candidate.get("local_evidence") or [])
    checks.extend(evidence)
    remote: dict[str, Any] | None = None
    remote_error: str | None = None
    if offline:
        checks.append(VerificationCheck("github_pr_remote_verified", "skip", "offline mode skipped GitHub API verification"))
    elif external_pr_url:
        remote, remote_error = fetch_github_pr_metadata(external_pr_url)
        checks.append(check_status("github_pr_remote_verified", remote is not None, remote_error or "GitHub API returned pull request metadata"))
        if remote is not None:
            checks.append(check_status("github_pr_state_open", remote.get("state") == "open" or bool(remote.get("merged_at")), f"state={remote.get('state')} merged_at={remote.get('merged_at')}"))
            checks.append(check_status("github_pr_not_draft", not bool(remote.get("draft")), f"draft={remote.get('draft')}"))
            if require_merged or candidate.get("status") == "merged":
                checks.append(check_status("github_pr_merged", bool(remote.get("merged_at")), f"merged_at={remote.get('merged_at')}"))
    failures = [check for check in checks if check.status == "fail" and check.name != "github_pr_remote_verified"]
    if not offline:
        failures.extend(check for check in checks if check.name == "github_pr_remote_verified" and check.status == "fail")
    verification = redact_value({
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "run_id": candidate_run_id,
        "source_config": display_path(config_path),
        "output_dir": display_path(output_dir),
        "candidate_id": candidate.get("id"),
        "status": candidate.get("status"),
        "target_url": candidate.get("target_url"),
        "external_pr_url": external_pr_url,
        "offline": offline,
        "require_merged": require_merged,
        "local_evidence": evidence_paths,
        "remote": {
            "available": remote is not None,
            "error": remote_error,
            "state": remote.get("state") if remote else None,
            "draft": remote.get("draft") if remote else None,
            "merged_at": remote.get("merged_at") if remote else None,
            "html_url": remote.get("html_url") if remote else None,
            "title": remote.get("title") if remote else None,
        },
        "checks": [asdict(check) for check in checks],
        "verified": not failures,
        "blocked_until": [f"{check.name}: {check.message}" for check in failures],
        "completion_rule": "MF-0808 can be marked complete only when this verification is non-offline and verified=true for a real external PR.",
    })
    return verification


def patch_paths_for_candidate(candidate: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in candidate.get("local_evidence") or []:
        path = resolve_repo_path(str(raw_path))
        if path.suffix == ".patch":
            paths.append(path)
    return paths


def run_git_apply_check(target_worktree: Path, patch_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=target_worktree,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode == 0, result.stdout.strip()


def run_git_apply(target_worktree: Path, patch_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=target_worktree,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode == 0, result.stdout.strip()


def is_git_worktree(path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def build_apply_draft(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    candidate_id: str | None,
    target_worktree: str | Path,
    branch: str | None = None,
    apply: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    candidate = candidate_by_id(config, candidate_id)
    candidate_run_id = sanitize_run_id(run_id or f"upstream_apply_{candidate['id']}")
    output_dir = REPO_DIR / "reports" / "generated" / "upstream_prs" / candidate_run_id
    worktree = Path(str(target_worktree)).expanduser().resolve()
    branch_name = sanitize_run_id(branch or f"model-forge-{candidate['id']}")
    patch_paths = patch_paths_for_candidate(candidate)
    checks: list[VerificationCheck] = []
    checks.append(check_status("target_worktree_exists", worktree.exists() and worktree.is_dir(), str(worktree)))
    checks.append(check_status("target_worktree_has_git", worktree.exists() and worktree.is_dir() and is_git_worktree(worktree), str(worktree)))
    checks.append(check_status("candidate_has_patch", bool(patch_paths), ", ".join(display_path(path) for path in patch_paths) or "no .patch in local_evidence"))
    patch_results: list[dict[str, Any]] = []
    applied = False
    if checks[0].status == "pass" and checks[1].status == "pass":
        for patch_path in patch_paths:
            ok, output = run_git_apply_check(worktree, patch_path)
            checks.append(check_status(f"patch_{patch_path.name}_applies", ok, output or "git apply --check passed"))
            patch_results.append(
                {
                    "patch": display_path(patch_path),
                    "apply_check_passed": ok,
                    "apply_check_output": output,
                }
            )
        if apply and patch_paths and all(item["apply_check_passed"] for item in patch_results):
            for patch_path in patch_paths:
                ok, output = run_git_apply(worktree, patch_path)
                checks.append(check_status(f"patch_{patch_path.name}_applied", ok, output or "git apply passed"))
                if not ok:
                    break
            applied = all(check.status == "pass" for check in checks if check.name.endswith("_applied"))
    elif apply:
        checks.append(check_status("patch_apply_skipped", False, "target worktree must exist and be a git checkout"))
    failures = [check for check in checks if check.status == "fail"]
    draft = redact_value(
        {
            "schema_version": APPLY_DRAFT_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "run_id": candidate_run_id,
            "source_config": display_path(config_path),
            "output_dir": display_path(output_dir),
            "candidate_id": candidate.get("id"),
            "target_project": candidate.get("target_project"),
            "target_url": candidate.get("target_url"),
            "target_worktree": display_path(worktree),
            "branch": branch_name,
            "apply_requested": apply,
            "applied": applied,
            "patches": patch_results,
            "checks": [asdict(check) for check in checks],
            "ready": not failures,
            "blocked_until": [f"{check.name}: {check.message}" for check in failures],
            "handoff_commands": [
                f"cd {worktree}",
                f"git switch -c {branch_name}",
                *[f"git apply {patch}" for patch in (display_path(path) for path in patch_paths)],
                "git diff --check",
                "git status --short",
                "git add docs/benchmarking/dgx_spark_gb10.md docs/benchmarking/README.md docs/.nav.yml",
                'git commit -m "docs: add DGX Spark GB10 serving recipe"',
                f"git push <your-fork-remote> {branch_name}",
                "open a pull request against the upstream default branch",
            ],
            "completion_rule": "After the external PR is opened, record external_pr_url in the upstream candidate and run non-offline verify-pr.",
        }
    )
    return draft


def write_apply_draft(draft: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(draft["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "upstream_pr_apply_draft.json"
    path.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_verification(verification: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(verification["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "upstream_pr_verification.json"
    path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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


def render_verification(verification: Mapping[str, Any]) -> None:
    table = Table(title=f"Upstream PR Verification: {verification.get('candidate_id')}")
    table.add_column("Field")
    table.add_column("Value")
    for key in ("status", "target_url", "external_pr_url", "offline", "verified"):
        table.add_row(key, str(verification.get(key)))
    console.print(table)
    check_table = Table(title="Verification Checks")
    check_table.add_column("Check")
    check_table.add_column("Status")
    check_table.add_column("Message")
    for check in verification.get("checks") or []:
        check_table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
    console.print(check_table)


def render_apply_draft(draft: Mapping[str, Any]) -> None:
    table = Table(title=f"Upstream PR Draft Apply: {draft.get('candidate_id')}")
    table.add_column("Field")
    table.add_column("Value")
    for key in ("target_url", "target_worktree", "branch", "apply_requested", "applied", "ready"):
        table.add_row(key, str(draft.get(key)))
    console.print(table)
    check_table = Table(title="Apply Checks")
    check_table.add_column("Check")
    check_table.add_column("Status")
    check_table.add_column("Message")
    for check in draft.get("checks") or []:
        check_table.add_row(str(check.get("name")), str(check.get("status")), str(check.get("message")))
    console.print(check_table)


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
    verify = sub.add_parser("verify-pr", help="Verify recorded upstream PR URL and local evidence")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--candidate")
    verify.add_argument("--offline", action="store_true", help="Skip GitHub API lookup and verify only recorded fields/evidence")
    verify.add_argument("--require-merged", action="store_true")
    verify.add_argument("--run-id")
    verify.add_argument("--write-report", action="store_true")
    verify.add_argument("--json", action="store_true")
    apply_draft = sub.add_parser("apply-draft", help="Validate or apply a prepared upstream patch to a local upstream checkout")
    apply_draft.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    apply_draft.add_argument("--candidate")
    apply_draft.add_argument("--target-worktree", required=True, help="Local clone of the upstream repository")
    apply_draft.add_argument("--branch", help="Suggested upstream contribution branch")
    apply_draft.add_argument("--apply", action="store_true", help="Apply the candidate patch after git apply --check passes")
    apply_draft.add_argument("--run-id")
    apply_draft.add_argument("--write-report", action="store_true")
    apply_draft.add_argument("--json", action="store_true")
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
        raise SystemExit(severity_exit_code(findings))

    if args.command == "plan":
        plan = build_plan(config, config_path, candidate_id=args.candidate, run_id=args.run_id)
        if args.write_plan:
            write_plan(plan)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
        return

    if args.command == "verify-pr":
        verification = build_verification(
            config,
            config_path,
            candidate_id=args.candidate,
            offline=args.offline,
            require_merged=args.require_merged,
            run_id=args.run_id,
        )
        if args.write_report:
            write_verification(verification)
        if args.json:
            print(json.dumps(verification, indent=2, sort_keys=True) + "\n")
        else:
            render_verification(verification)
        raise SystemExit(0 if verification.get("verified") else 1)

    if args.command == "apply-draft":
        draft = build_apply_draft(
            config,
            config_path,
            candidate_id=args.candidate,
            target_worktree=args.target_worktree,
            branch=args.branch,
            apply=args.apply,
            run_id=args.run_id,
        )
        if args.write_report:
            write_apply_draft(draft)
        if args.json:
            print(json.dumps(draft, indent=2, sort_keys=True) + "\n")
        else:
            render_apply_draft(draft)
        raise SystemExit(0 if draft.get("ready") else 1)


if __name__ == "__main__":
    main()
