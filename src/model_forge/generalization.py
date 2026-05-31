from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml

from model_forge.runs.manifest import REPO_DIR, display_path


SCAN_PATHS = (
    Path("forge"),
    Path("src/model_forge"),
    Path("scripts"),
)
SKIP_DIR_NAMES = {"__pycache__"}
TEXT_SUFFIXES = {"", ".py", ".sh"}
FAMILY_CASE_PATTERN = r"^\s*(?:{family})\)\s*$"
CONFIG_DEFAULT_PATTERN = r"--config[^\n]+(?:default=|:=)[^\n]*{family}"


@dataclass(frozen=True)
class Finding:
    check: str
    message: str
    path: str
    line: int


def load_family_ids(repo_dir: Path = REPO_DIR) -> list[str]:
    family_dir = repo_dir / "configs" / "model_families"
    ids: list[str] = []
    for path in sorted(family_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and data.get("name"):
            ids.append(str(data["name"]))
        else:
            ids.append(path.stem)
    return sorted(set(ids))


def iter_scan_files(repo_dir: Path = REPO_DIR) -> Iterable[Path]:
    for relative in SCAN_PATHS:
        path = repo_dir / relative
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for child in sorted(path.rglob("*")):
            if any(part in SKIP_DIR_NAMES for part in child.parts):
                continue
            if child.is_file() and child.suffix in TEXT_SUFFIXES:
                yield child


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


def audit_family_control_flow(repo_dir: Path = REPO_DIR) -> list[Finding]:
    family_ids = load_family_ids(repo_dir)
    case_patterns = {
        family: re.compile(FAMILY_CASE_PATTERN.format(family=re.escape(family)))
        for family in family_ids
    }
    default_patterns = {
        family: re.compile(CONFIG_DEFAULT_PATTERN.format(family=re.escape(family)))
        for family in family_ids
    }
    findings: list[Finding] = []
    for path in iter_scan_files(repo_dir):
        text = read_text(path)
        if text is None:
            continue
        relative = display_path(path.relative_to(repo_dir))
        for line_number, line in enumerate(text.splitlines(), start=1):
            for family, pattern in case_patterns.items():
                if pattern.search(line):
                    findings.append(
                        Finding(
                            "family_control_flow",
                            f"family {family!r} is hard-coded as a control-flow branch; use config discovery instead",
                            relative,
                            line_number,
                        )
                    )
            for family, pattern in default_patterns.items():
                if pattern.search(line):
                    findings.append(
                        Finding(
                            "family_config_default",
                            f"family {family!r} is hard-coded as a default config; require --config or discover by family",
                            relative,
                            line_number,
                        )
                    )
    return findings


def run_audit(repo_dir: Path = REPO_DIR) -> list[Finding]:
    return audit_family_control_flow(repo_dir)


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        print("model-forge generalization audit: OK")
        return
    print(f"model-forge generalization audit: {len(findings)} issue(s) found")
    for finding in findings:
        print(f"- [{finding.check}] {finding.path}:{finding.line}: {finding.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model-family generalization assumptions")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Check common code for family-specific control flow")
    audit.add_argument("--json", action="store_true", help="Emit machine-readable findings")
    args = parser.parse_args()

    findings = run_audit(REPO_DIR)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
    else:
        render_findings(findings)
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
