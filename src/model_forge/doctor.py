from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from model_forge.objectives import audit_profiles
from model_forge.roadmap import (
    audit_roadmap_command_examples,
    audit_roadmap_items,
    parse_roadmap_command_examples,
    parse_roadmap_items,
)
from model_forge.variants.manifest import validate_variant_node


REPO_DIR = Path(__file__).resolve().parents[2]

ARCHIVE_PREFIXES = (
    "docs/roadmaps/",
)
ALLOWED_GENERATED_DATASET_PREFIXES = (
    "datasets/generated/gemma4_26b_a4b_local_ft_v1/",
    "datasets/generated/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke/",
)
REQUIRED_FILES = (
    "README.md",
    "AGENTS.md",
    "docs/status.md",
    "docs/artifact-retention.md",
    "docs/experiment-ledger.md",
    "docs/roadmap-status-audit.md",
    "docs/roadmaps/README.md",
    "configs/README.md",
    "scripts/README.md",
    "recipes/README.md",
)
SECRET_PATTERNS = (
    ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
)
MACHINE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + r"/" + r"home/[A-Za-z0-9._-]+"
    + r"|/Users/[A-Za-z0-9._-]+"
    + r")(?:/|$)"
)
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    check: str
    message: str
    path: str | None = None
    line: int | None = None


def git_lines(args: list[str], repo_dir: Path = REPO_DIR) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def tracked_files(repo_dir: Path = REPO_DIR) -> list[str]:
    return git_lines(["ls-files"], repo_dir)


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_archived(path: str) -> bool:
    return path.startswith(ARCHIVE_PREFIXES)


def iter_text_lines(repo_dir: Path, files: Iterable[str]) -> Iterable[tuple[str, int, str]]:
    for relative in files:
        text = read_text(repo_dir / relative)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            yield relative, line_number, line


def check_required_files(repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings = []
    for relative in REQUIRED_FILES:
        if not (repo_dir / relative).exists():
            findings.append(Finding("required_files", "required handoff file is missing", relative))
    return findings


def check_tracked_ignored(repo_dir: Path = REPO_DIR) -> list[Finding]:
    ignored = git_lines(["ls-files", "-ci", "--exclude-standard"], repo_dir)
    return [
        Finding("tracked_ignored", "tracked file matches .gitignore; move it or update ignore rules", path)
        for path in ignored
    ]


def check_secret_literals(files: list[str], repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings = []
    for path, line_number, line in iter_text_lines(repo_dir, files):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(Finding("secret_literals", f"literal secret-like value matched {name}", path, line_number))
    return findings


def check_machine_paths(files: list[str], repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings = []
    for path, line_number, line in iter_text_lines(repo_dir, files):
        if is_archived(path):
            continue
        if MACHINE_PATH_PATTERN.search(line):
            findings.append(
                Finding(
                    "machine_paths",
                    "machine-specific absolute path found; use repo-relative paths, ~/models, or env overrides",
                    path,
                    line_number,
                )
            )
    return findings


def check_generated_dataset_policy(files: list[str]) -> list[Finding]:
    findings = []
    for path in files:
        if not path.startswith("datasets/generated/"):
            continue
        if path == "datasets/generated/.gitkeep":
            continue
        if path.startswith(ALLOWED_GENERATED_DATASET_PREFIXES):
            continue
        findings.append(
            Finding(
                "generated_dataset_policy",
                "generated dataset output is tracked without an explicit allowlist entry",
                path,
            )
        )
    return findings


def check_large_tracked_files(files: list[str], repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings = []
    for path in files:
        if is_archived(path):
            continue
        full_path = repo_dir / path
        try:
            size = full_path.stat().st_size
        except OSError:
            continue
        if size > MAX_TRACKED_FILE_BYTES:
            findings.append(
                Finding(
                    "large_tracked_files",
                    f"tracked file is {size} bytes; large artifacts belong in ignored dirs or Hugging Face",
                    path,
                )
            )
    return findings


def check_objective_profiles() -> list[Finding]:
    _, errors = audit_profiles()
    return [
        Finding(
            "objective_profiles",
            f"{error['field']}: {error['message']}",
            f"configs/objectives/{error['profile']}.yaml",
        )
        for error in errors
    ]


def check_roadmap_status() -> list[Finding]:
    items = parse_roadmap_items()
    errors = audit_roadmap_items(items)
    return [
        Finding(
            "roadmap_status",
            f"{error.field}: {error.message}",
            "docs/roadmaps/model_forge_sota_roadmap_2026-05-18_with_huggingface.md",
            error.line,
        )
        for error in errors
    ]


def check_roadmap_cli_drift() -> list[Finding]:
    examples = parse_roadmap_command_examples()
    errors = audit_roadmap_command_examples(examples)
    return [
        Finding(
            "roadmap_cli_drift",
            error.message,
            "docs/roadmaps/model_forge_sota_roadmap_2026-05-18_with_huggingface.md",
            error.line,
        )
        for error in errors
    ]


def check_tracked_variant_nodes(files: list[str], repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        if not path.endswith("variant_node.json"):
            continue
        try:
            data = json.loads((repo_dir / path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding("variant_node", f"could not parse variant node: {exc}", path))
            continue
        for error in validate_variant_node(data):
            findings.append(Finding("variant_node", error, path))
    return findings


def run_checks(repo_dir: Path = REPO_DIR) -> list[Finding]:
    files = tracked_files(repo_dir)
    findings: list[Finding] = []
    findings.extend(check_required_files(repo_dir))
    findings.extend(check_tracked_ignored(repo_dir))
    findings.extend(check_secret_literals(files, repo_dir))
    findings.extend(check_machine_paths(files, repo_dir))
    findings.extend(check_generated_dataset_policy(files))
    findings.extend(check_large_tracked_files(files, repo_dir))
    findings.extend(check_objective_profiles())
    findings.extend(check_roadmap_status())
    findings.extend(check_roadmap_cli_drift())
    findings.extend(check_tracked_variant_nodes(files, repo_dir))
    return findings


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        print("model-forge doctor: OK")
        return
    print(f"model-forge doctor: {len(findings)} issue(s) found", file=sys.stderr)
    for finding in findings:
        location = finding.path or "<repo>"
        if finding.line is not None:
            location = f"{location}:{finding.line}"
        print(f"- [{finding.check}] {location}: {finding.message}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-forge repository hygiene checks")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable findings")
    args = parser.parse_args()

    findings = run_checks(REPO_DIR)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
    else:
        render_findings(findings)
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
