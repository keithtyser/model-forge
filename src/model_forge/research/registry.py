from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from rich.console import Console
from rich.table import Table


REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_DIR / "configs" / "research_registry.yaml"
DEFAULT_SOTA_DOC = REPO_DIR / "docs" / "research" / "sota-2026-05-18.md"
DEFAULT_WATCH_CONFIG = REPO_DIR / "configs" / "research_watch" / "advanced_serving.yaml"
OBJECTIVES_DIR = REPO_DIR / "configs" / "objectives"

REQUIRED_ENTRY_FIELDS = (
    "title",
    "kind",
    "year",
    "status",
    "url",
    "areas",
    "claims",
    "implementation_hooks",
    "eval_hooks",
    "limitations",
)
ALLOWED_STATUSES = {
    "implemented_baseline",
    "tracked_basis",
    "tracked_sota",
    "evaluation_standard",
    "serving_standard",
}

console = Console()


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    check: str
    message: str
    path: str
    entry_id: str | None = None


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = resolve_repo_path(path)
    registry = load_yaml(registry_path)
    entries = registry.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"research registry entries must be a mapping: {registry_path}")
    normalized: dict[str, dict[str, Any]] = {}
    for entry_id, raw in entries.items():
        if not isinstance(raw, dict):
            raise ValueError(f"research registry entry must be a mapping: {entry_id}")
        entry = dict(raw)
        entry.setdefault("id", str(entry_id))
        normalized[str(entry_id)] = entry
    registry["entries"] = normalized
    registry["path"] = display_path(registry_path)
    return registry


def sorted_entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    entries = registry.get("entries", {})
    return [entries[entry_id] for entry_id in sorted(entries)]


def filter_entries(
    entries: Iterable[dict[str, Any]],
    area: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for entry in entries:
        areas = [str(item) for item in entry.get("areas", [])]
        if area and area not in areas:
            continue
        if status and str(entry.get("status", "")) != status:
            continue
        filtered.append(entry)
    return filtered


def objective_research_references(objectives_dir: Path = OBJECTIVES_DIR) -> list[tuple[Path, str]]:
    references: list[tuple[Path, str]] = []
    if not objectives_dir.exists():
        return references
    for path in sorted(objectives_dir.glob("*.yaml")):
        data = load_yaml(path)
        raw_refs = data.get("research_basis", [])
        if not isinstance(raw_refs, list):
            references.append((path, "<invalid-list>"))
            continue
        for item in raw_refs:
            references.append((path, str(item)))
    return references


def _required_list(entry: dict[str, Any], field: str) -> bool:
    value = entry.get(field)
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def audit_registry(
    registry_path: str | Path = DEFAULT_REGISTRY,
    sota_doc_path: str | Path = DEFAULT_SOTA_DOC,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    registry_full_path = resolve_repo_path(registry_path)
    sota_full_path = resolve_repo_path(sota_doc_path)
    try:
        registry = load_registry(registry_full_path)
    except Exception as exc:
        return [
            AuditFinding(
                severity="error",
                check="registry_load",
                message=str(exc),
                path=display_path(registry_full_path),
            )
        ]

    for field in ("version", "snapshot_date", "last_verified", "entries"):
        if field not in registry:
            findings.append(
                AuditFinding("error", "registry_schema", f"missing top-level field {field!r}", registry["path"])
            )

    entries = registry.get("entries", {})
    if not entries:
        findings.append(AuditFinding("error", "registry_schema", "registry has no entries", registry["path"]))

    for entry_id, entry in entries.items():
        for field in REQUIRED_ENTRY_FIELDS:
            if field not in entry:
                findings.append(
                    AuditFinding("error", "entry_schema", f"missing field {field!r}", registry["path"], entry_id)
                )
        if entry.get("id") != entry_id:
            findings.append(
                AuditFinding("error", "entry_id", "entry id must match mapping key", registry["path"], entry_id)
            )
        if entry.get("status") not in ALLOWED_STATUSES:
            findings.append(
                AuditFinding(
                    "error",
                    "entry_status",
                    f"unsupported status {entry.get('status')!r}",
                    registry["path"],
                    entry_id,
                )
            )
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            findings.append(
                AuditFinding("error", "entry_url", "url must be an http(s) URL", registry["path"], entry_id)
            )
        for list_field in ("areas", "claims", "implementation_hooks", "eval_hooks", "limitations"):
            if not _required_list(entry, list_field):
                findings.append(
                    AuditFinding(
                        "error",
                        "entry_schema",
                        f"{list_field!r} must be a non-empty list of strings",
                        registry["path"],
                        entry_id,
                    )
                )

    known_ids = set(entries)
    for objective_path, reference_id in objective_research_references():
        if reference_id == "<invalid-list>":
            findings.append(
                AuditFinding(
                    "error",
                    "objective_research_basis",
                    "research_basis must be a list",
                    display_path(objective_path),
                )
            )
        elif reference_id not in known_ids:
            findings.append(
                AuditFinding(
                    "error",
                    "objective_research_basis",
                    f"unknown research_basis id {reference_id!r}",
                    display_path(objective_path),
                    reference_id,
                )
            )

    if not sota_full_path.exists():
        findings.append(
            AuditFinding("error", "sota_doc", "SOTA research snapshot is missing", display_path(sota_full_path))
        )
    else:
        text = sota_full_path.read_text(encoding="utf-8")
        for entry_id in sorted(known_ids):
            if entry_id not in text:
                findings.append(
                    AuditFinding(
                        "warning",
                        "sota_doc",
                        "entry id is not mentioned in the SOTA snapshot",
                        display_path(sota_full_path),
                        entry_id,
                    )
                )

    return findings


def audit_watch_config(path: str | Path = DEFAULT_WATCH_CONFIG, registry_path: str | Path = DEFAULT_REGISTRY) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    watch_path = resolve_repo_path(path)
    registry = load_registry(registry_path)
    known_ids = set(registry.get("entries", {}))
    try:
        config = load_yaml(watch_path)
    except Exception as exc:
        return [AuditFinding("error", "watch_load", str(exc), display_path(watch_path))]

    if config.get("schema_version") != "model_forge.research_watch.v1":
        findings.append(AuditFinding("error", "watch_schema", "schema_version must be model_forge.research_watch.v1", display_path(watch_path)))
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        findings.append(AuditFinding("error", "watch_schema", "entries must be a non-empty list", display_path(watch_path)))
        return findings
    ids: list[str] = []
    for raw in entries:
        if not isinstance(raw, dict):
            findings.append(AuditFinding("error", "watch_entry", "watch entries must be mappings", display_path(watch_path)))
            continue
        entry_id = str(raw.get("id") or "")
        ids.append(entry_id)
        registry_entry = str(raw.get("registry_entry") or "")
        if not entry_id:
            findings.append(AuditFinding("error", "watch_entry", "watch entry id is required", display_path(watch_path)))
        if registry_entry not in known_ids:
            findings.append(AuditFinding("error", "watch_registry", f"unknown registry_entry {registry_entry!r}", display_path(watch_path), entry_id))
        watch_urls = raw.get("watch_urls")
        if not isinstance(watch_urls, list) or not watch_urls:
            findings.append(AuditFinding("error", "watch_urls", "watch_urls must be a non-empty list", display_path(watch_path), entry_id))
        else:
            for url in watch_urls:
                if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                    findings.append(AuditFinding("error", "watch_urls", "watch_urls must contain http(s) URLs", display_path(watch_path), entry_id))
        for field in ("adoption_hooks", "promotion_blockers"):
            value = raw.get(field)
            if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                findings.append(AuditFinding("error", "watch_schema", f"{field} must be a non-empty list of strings", display_path(watch_path), entry_id))
    if len(ids) != len(set(ids)):
        findings.append(AuditFinding("error", "watch_entry", "watch entry ids must be unique", display_path(watch_path)))
    return findings


def render_entry(entry: dict[str, Any]) -> None:
    console.print(f"[bold]{entry['id']}[/bold]")
    console.print(f"Title: {entry.get('title')}")
    console.print(f"Kind: {entry.get('kind')}  Year: {entry.get('year')}  Status: {entry.get('status')}")
    console.print(f"URL: {entry.get('url')}")
    console.print(f"Areas: {', '.join(entry.get('areas', []))}")
    for section in ("claims", "implementation_hooks", "eval_hooks", "limitations"):
        console.print(f"\n[bold]{section.replace('_', ' ').title()}[/bold]")
        for item in entry.get(section, []):
            console.print(f"- {item}")


def render_list(entries: list[dict[str, Any]]) -> None:
    table = Table(title="Research Registry")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Year", justify="right")
    table.add_column("Areas")
    table.add_column("Title")
    for entry in entries:
        table.add_row(
            str(entry.get("id", "")),
            str(entry.get("status", "")),
            str(entry.get("year", "")),
            ", ".join(str(area) for area in entry.get("areas", [])),
            str(entry.get("title", "")),
        )
    console.print(table)


def render_audit(findings: list[AuditFinding]) -> None:
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    if not findings:
        console.print("research registry audit: OK")
        return
    table = Table(title=f"Research Registry Audit: {len(errors)} error(s), {len(warnings)} warning(s)")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Path")
    table.add_column("Entry")
    table.add_column("Message")
    for finding in findings:
        table.add_row(
            finding.severity.upper(),
            finding.check,
            finding.path,
            finding.entry_id or "",
            finding.message,
        )
    console.print(table)


def render_watch(findings: list[AuditFinding]) -> None:
    if not findings:
        console.print("research watch audit: OK")
        return
    render_audit(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and audit the model-forge research registry")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registry entries")
    list_parser.add_argument("--area", help="Filter by area")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    show_parser = subparsers.add_parser("show", help="Show one registry entry")
    show_parser.add_argument("entry_id")
    show_parser.add_argument("--json", action="store_true", help="Emit JSON")

    audit_parser = subparsers.add_parser("audit", help="Validate registry consistency")
    audit_parser.add_argument("--json", action="store_true", help="Emit JSON")
    audit_parser.add_argument("--sota-doc", type=Path, default=DEFAULT_SOTA_DOC)

    watch_parser = subparsers.add_parser("watch", help="Validate research-watch hooks")
    watch_parser.add_argument("--config", type=Path, default=DEFAULT_WATCH_CONFIG)
    watch_parser.add_argument("--json", action="store_true", help="Emit JSON")

    args = parser.parse_args()
    registry = load_registry(args.registry)

    if args.command == "list":
        entries = filter_entries(sorted_entries(registry), area=args.area, status=args.status)
        if args.json:
            print(json.dumps(entries, indent=2, sort_keys=True) + "\n")
        else:
            render_list(entries)
        return

    if args.command == "show":
        entries = registry["entries"]
        if args.entry_id not in entries:
            valid = ", ".join(sorted(entries))
            raise SystemExit(f"unknown research entry {args.entry_id!r}; valid ids: {valid}")
        if args.json:
            print(json.dumps(entries[args.entry_id], indent=2, sort_keys=True) + "\n")
        else:
            render_entry(entries[args.entry_id])
        return

    if args.command == "audit":
        findings = audit_registry(args.registry, args.sota_doc)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2, sort_keys=True) + "\n")
        else:
            render_audit(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)

    if args.command == "watch":
        findings = audit_watch_config(args.config, args.registry)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2, sort_keys=True) + "\n")
        else:
            render_watch(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)


if __name__ == "__main__":
    main()
