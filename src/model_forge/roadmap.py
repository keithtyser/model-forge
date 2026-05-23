from __future__ import annotations

import argparse
import json
import re
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Model Forge roadmap backlog status fields")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit implementation_status and validation_state on MF backlog items")
    audit_parser.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument("--write-doc", action="store_true", help="Write docs/roadmap-status-audit.md")
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    roadmap_path = args.roadmap if args.roadmap.is_absolute() else REPO_DIR / args.roadmap
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
