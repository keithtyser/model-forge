from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from model_forge.objectives import IMPLEMENTATION_STATUSES, VALIDATION_STATES


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROADMAP = REPO_DIR / "docs" / "roadmaps" / "model_forge_sota_roadmap_2026-05-18_with_huggingface.md"
DEFAULT_OUTPUT = REPO_DIR / "docs" / "roadmap-status-audit.md"

ITEM_RE = re.compile(r"^(?P<id>MF-\d{4})\s+(?P<body>.+)$")
IMPLEMENTATION_RE = re.compile(r"\s+implementation_status=(?P<status>[a-z_]+)")
VALIDATION_RE = re.compile(r"\s+validation_state=(?P<state>[a-z_]+)")
SECTION_RE = re.compile(r"^###\s+(?P<section>.+)$")
FORGE_TOKEN = "./forge"
TARGET_MARKERS = (
    "target cli",
    "target command",
    "target interface",
    "target api",
    "planned cli",
    "planned command",
    "planned api",
)

console = Console()


@dataclass(frozen=True)
class RoadmapItem:
    item_id: str
    title: str
    section: str
    implementation_status: str | None
    validation_state: str | None
    line: int


@dataclass(frozen=True)
class RoadmapFinding:
    item_id: str
    line: int
    field: str
    message: str


@dataclass(frozen=True)
class RoadmapCommandExample:
    line: int
    command: str
    key: str
    target_marked: bool
    implemented: bool


@dataclass(frozen=True)
class RoadmapCommandFinding:
    line: int
    command: str
    key: str
    message: str


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except (OSError, ValueError):
        return str(path)


def item_title(body: str) -> str:
    title = IMPLEMENTATION_RE.split(body, maxsplit=1)[0]
    title = VALIDATION_RE.split(title, maxsplit=1)[0]
    return title.strip()


def parse_roadmap_items(path: Path = DEFAULT_ROADMAP) -> list[RoadmapItem]:
    items: list[RoadmapItem] = []
    section = "Unsectioned"
    in_prioritized_backlog = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("## 16. Prioritized backlog"):
            in_prioritized_backlog = True
            continue
        if in_prioritized_backlog and line.startswith("## 17. "):
            break
        section_match = SECTION_RE.match(line)
        if in_prioritized_backlog and section_match:
            section = section_match.group("section")
            continue
        match = ITEM_RE.match(line)
        if not match:
            continue
        body = match.group("body")
        implementation_match = IMPLEMENTATION_RE.search(body)
        validation_match = VALIDATION_RE.search(body)
        items.append(
            RoadmapItem(
                item_id=match.group("id"),
                title=item_title(body).rstrip("."),
                section=section,
                implementation_status=implementation_match.group("status") if implementation_match else None,
                validation_state=validation_match.group("state") if validation_match else None,
                line=line_number,
            )
        )
    return items


def audit_roadmap_items(items: list[RoadmapItem]) -> list[RoadmapFinding]:
    findings: list[RoadmapFinding] = []
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            findings.append(RoadmapFinding(item.item_id, item.line, "id", "duplicate roadmap item id"))
        seen.add(item.item_id)
        if not item.implementation_status:
            findings.append(RoadmapFinding(item.item_id, item.line, "implementation_status", "missing implementation_status"))
        elif item.implementation_status not in IMPLEMENTATION_STATUSES:
            findings.append(
                RoadmapFinding(
                    item.item_id,
                    item.line,
                    "implementation_status",
                    f"unsupported value {item.implementation_status!r}",
                )
            )
        if not item.validation_state:
            findings.append(RoadmapFinding(item.item_id, item.line, "validation_state", "missing validation_state"))
        elif item.validation_state not in VALIDATION_STATES:
            findings.append(
                RoadmapFinding(
                    item.item_id,
                    item.line,
                    "validation_state",
                    f"unsupported value {item.validation_state!r}",
                )
            )
    return findings


def normalize_token(token: str) -> str:
    value = token.strip().strip("`'\",;:()[]{}|")
    if value != FORGE_TOKEN:
        value = value.rstrip(".")
        value = value.strip("`'\",;:()[]{}|")
    return value


def command_tokens(raw_line: str) -> list[str] | None:
    if FORGE_TOKEN not in raw_line:
        return None
    fragment = raw_line[raw_line.index(FORGE_TOKEN) :]
    raw_tokens = fragment.replace("\\", " ").split()
    raw_lookahead = [token.strip("`'\".,;:()[]{}|") for token in raw_tokens[1:3]]
    if any("..." in token for token in raw_tokens[1:3]):
        return None
    if any(token.startswith("<") for token in raw_lookahead):
        return None
    fragment = fragment.replace("\\", " ")
    tokens = [normalize_token(token) for token in fragment.split()]
    tokens = [token for token in tokens if token]
    if not tokens or tokens[0] != FORGE_TOKEN:
        return None
    return tokens


def parse_help_surface(help_text: str) -> dict[str, set[str] | None]:
    surface: dict[str, set[str] | None] = {}
    for raw_line in help_text.splitlines():
        line = raw_line.strip()
        if line == "Examples:":
            break
        if not line.startswith("./forge "):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        command = normalize_token(tokens[1])
        if not command or command in {"help", "-h", "--help"}:
            continue
        subcommands: set[str] | None = None
        if len(tokens) >= 3:
            match = re.fullmatch(r"\[([A-Za-z0-9_|-]+)\]", tokens[2])
            if match:
                subcommands = set(match.group(1).split("|"))
        existing = surface.get(command)
        if existing is None and command in surface:
            continue
        if subcommands is None:
            surface[command] = None
        elif existing is None:
            surface[command] = subcommands
        else:
            existing.update(subcommands)
    return surface


def current_help_surface(repo_dir: Path = REPO_DIR) -> dict[str, set[str] | None]:
    result = subprocess.run(
        ["./forge", "--help"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return parse_help_surface(result.stdout)


def command_key(tokens: list[str], surface: dict[str, set[str] | None]) -> tuple[str, bool]:
    if len(tokens) < 2:
        return FORGE_TOKEN, False
    command = tokens[1]
    if command in {"help", "-h", "--help"}:
        return command, True
    subcommands = surface.get(command)
    if command not in surface:
        return command, False
    if subcommands is None:
        return command, True
    if len(tokens) < 3:
        return command, False
    subcommand = tokens[2]
    return f"{command} {subcommand}", subcommand in subcommands


def context_is_target_marked(lines: list[str]) -> bool:
    context = " ".join(line.strip().lower() for line in lines)
    return any(marker in context for marker in TARGET_MARKERS)


def parse_roadmap_command_examples(
    path: Path = DEFAULT_ROADMAP,
    surface: dict[str, set[str] | None] | None = None,
) -> list[RoadmapCommandExample]:
    surface = current_help_surface() if surface is None else surface
    examples: list[RoadmapCommandExample] = []
    context_window: list[str] = []
    code_context: list[str] = []
    in_code_block = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_context = []
            else:
                in_code_block = True
                code_context = context_window[-8:]
            context_window.append(raw_line)
            context_window = context_window[-8:]
            continue

        tokens = command_tokens(raw_line)
        if tokens is not None:
            context = (code_context if in_code_block else context_window[-8:]) + [raw_line]
            key, implemented = command_key(tokens, surface)
            examples.append(
                RoadmapCommandExample(
                    line=line_number,
                    command=" ".join(tokens),
                    key=key,
                    target_marked=context_is_target_marked(context),
                    implemented=implemented,
                )
            )

        if stripped:
            context_window.append(raw_line)
            context_window = context_window[-8:]
    return examples


def audit_roadmap_command_examples(examples: list[RoadmapCommandExample]) -> list[RoadmapCommandFinding]:
    findings: list[RoadmapCommandFinding] = []
    for example in examples:
        if example.implemented or example.target_marked:
            continue
        findings.append(
            RoadmapCommandFinding(
                example.line,
                example.command,
                example.key,
                "command is not exposed by ./forge --help and is not marked target/planned",
            )
        )
    return findings


def summarize(items: list[RoadmapItem], findings: list[RoadmapFinding], roadmap_path: Path) -> dict[str, Any]:
    return {
        "roadmap": display_path(roadmap_path),
        "item_count": len(items),
        "finding_count": len(findings),
        "implementation_counts": dict(sorted(Counter(item.implementation_status or "<missing>" for item in items).items())),
        "validation_counts": dict(sorted(Counter(item.validation_state or "<missing>" for item in items).items())),
        "items": [asdict(item) for item in items],
        "findings": [asdict(finding) for finding in findings],
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Roadmap Status Audit",
        "",
        "This file is generated from the prioritized backlog in",
        f"`{summary['roadmap']}`.",
        "",
        "## Summary",
        "",
        f"- Items: {summary['item_count']}",
        f"- Findings: {summary['finding_count']}",
        "",
        "## Implementation Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in summary["implementation_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Validation State", "", "| State | Count |", "|---|---:|"])
    for state, count in summary["validation_counts"].items():
        lines.append(f"| {state} | {count} |")
    lines.extend(["", "## Findings", ""])
    if summary["findings"]:
        lines.extend(["| Item | Line | Field | Message |", "|---|---:|---|---|"])
        for finding in summary["findings"]:
            lines.append(f"| {finding['item_id']} | {finding['line']} | {finding['field']} | {finding['message']} |")
    else:
        lines.append("No status audit findings.")
    lines.extend(["", "## Items", "", "| Item | Section | Implementation | Validation | Title |", "|---|---|---|---|---|"])
    for item in summary["items"]:
        lines.append(
            f"| {item['item_id']} | {item['section']} | {item['implementation_status']} | "
            f"{item['validation_state']} | {item['title']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Roadmap Status Audit")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in summary["implementation_counts"].items():
        table.add_row(f"implementation:{status}", str(count))
    for state, count in summary["validation_counts"].items():
        table.add_row(f"validation:{state}", str(count))
    console.print(table)
    if summary["findings"]:
        finding_table = Table(title="Findings")
        finding_table.add_column("Item")
        finding_table.add_column("Line", justify="right")
        finding_table.add_column("Field")
        finding_table.add_column("Message")
        for finding in summary["findings"]:
            finding_table.add_row(finding["item_id"], str(finding["line"]), finding["field"], finding["message"])
        console.print(finding_table)


def summarize_cli_drift(
    examples: list[RoadmapCommandExample],
    findings: list[RoadmapCommandFinding],
    roadmap_path: Path,
) -> dict[str, Any]:
    return {
        "roadmap": display_path(roadmap_path),
        "example_count": len(examples),
        "finding_count": len(findings),
        "implemented_count": sum(1 for example in examples if example.implemented),
        "target_marked_count": sum(1 for example in examples if example.target_marked),
        "examples": [asdict(example) for example in examples],
        "findings": [asdict(finding) for finding in findings],
    }


def render_cli_drift(summary: dict[str, Any]) -> None:
    table = Table(title="Roadmap CLI Drift")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("examples", str(summary["example_count"]))
    table.add_row("implemented", str(summary["implemented_count"]))
    table.add_row("target_marked", str(summary["target_marked_count"]))
    table.add_row("findings", str(summary["finding_count"]))
    console.print(table)
    if summary["findings"]:
        finding_table = Table(title="Findings")
        finding_table.add_column("Line", justify="right")
        finding_table.add_column("Key")
        finding_table.add_column("Command")
        finding_table.add_column("Message")
        for finding in summary["findings"]:
            finding_table.add_row(
                str(finding["line"]),
                finding["key"],
                finding["command"],
                finding["message"],
            )
        console.print(finding_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Model Forge roadmap backlog status fields")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit implementation_status and validation_state on MF backlog items")
    audit_parser.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument("--write-doc", action="store_true", help="Write docs/roadmap-status-audit.md")
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    drift_parser = subparsers.add_parser(
        "cli-drift",
        help="Audit roadmap ./forge examples against implemented help or target/planned markers",
    )
    drift_parser.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    drift_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    roadmap_path = args.roadmap if args.roadmap.is_absolute() else REPO_DIR / args.roadmap
    if args.command == "cli-drift":
        examples = parse_roadmap_command_examples(roadmap_path)
        findings = audit_roadmap_command_examples(examples)
        summary = summarize_cli_drift(examples, findings, roadmap_path)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            render_cli_drift(summary)
        raise SystemExit(1 if findings else 0)

    items = parse_roadmap_items(roadmap_path)
    findings = audit_roadmap_items(items)
    summary = summarize(items, findings, roadmap_path)
    if args.write_doc:
        output_path = args.output if args.output.is_absolute() else REPO_DIR / args.output
        write_markdown_report(output_path, summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        render_summary(summary)
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
