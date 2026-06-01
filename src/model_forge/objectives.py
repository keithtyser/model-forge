from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.research.registry import load_registry


REPO_DIR = Path(__file__).resolve().parents[2]
OBJECTIVES_DIR = REPO_DIR / "configs" / "objectives"
SCHEMA_VERSION = "model_forge.objective_profile.v1"

IMPLEMENTATION_STATUSES = {
    "not_started",
    "scaffolded",
    "implemented",
    "wired_to_cli",
    "tested",
}

VALIDATION_STATES = {
    "planned",
    "smoke_validated",
    "spark_single_node_validated",
    "spark_cluster_validated",
    "generalizable",
}

COMPARISON_PROFILE_KEYS = {
    "lower_is_better",
    "higher_is_better",
    "risk_metrics",
    "critical_regression_metrics",
    "capability_metrics",
}

console = Console()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_objective_profile(path: Path) -> dict[str, Any]:
    profile = load_yaml(path)
    profile["_path"] = str(path.relative_to(REPO_DIR) if path.is_absolute() and path.is_relative_to(REPO_DIR) else path)
    return profile


def load_objective_profiles(root: Path = OBJECTIVES_DIR) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.yaml")):
        profile = load_objective_profile(path)
        objective_id = str(profile.get("id") or path.stem)
        profiles[objective_id] = profile
    return profiles


def metric_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    raise TypeError(f"expected metric list or comma-separated string, got {type(raw).__name__}")


def comparison_profile_from_objective(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = profile.get("comparison_profile") or {}
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {
        "description": str(profile.get("description") or ""),
    }
    primary_goal = profile.get("primary_goal")
    if isinstance(primary_goal, Mapping):
        out["primary_goal"] = dict(primary_goal)
    has_metrics = False
    for key in COMPARISON_PROFILE_KEYS:
        values = set(metric_list(raw.get(key)))
        out[key] = values
        has_metrics = has_metrics or bool(values)
    return out if has_metrics else None


def load_comparison_objective_profiles(root: Path = OBJECTIVES_DIR) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for objective_id, profile in load_objective_profiles(root).items():
        comparison_profile = comparison_profile_from_objective(profile)
        if comparison_profile:
            profiles[objective_id] = comparison_profile
    return profiles


def audit_profile(profile: Mapping[str, Any], *, known_research_ids: set[str]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    objective_id = str(profile.get("id") or "")
    path = str(profile.get("_path") or f"configs/objectives/{objective_id}.yaml")

    def add(field: str, message: str) -> None:
        errors.append({"profile": objective_id or path, "field": field, "message": message})

    if profile.get("schema_version") != SCHEMA_VERSION:
        add("schema_version", f"must be {SCHEMA_VERSION}")
    if not objective_id:
        add("id", "required")
    elif Path(path).stem != objective_id:
        add("id", "must match filename stem")
    if profile.get("implementation_status") not in IMPLEMENTATION_STATUSES:
        add("implementation_status", f"must be one of {sorted(IMPLEMENTATION_STATUSES)}")
    if profile.get("validation_state") not in VALIDATION_STATES:
        add("validation_state", f"must be one of {sorted(VALIDATION_STATES)}")
    for field in ("version", "description", "primary_goal"):
        if not profile.get(field):
            add(field, "required")

    validation_gates = profile.get("validation_gates") or {}
    if not isinstance(validation_gates, Mapping):
        add("validation_gates", "must be a mapping")
        validation_gates = {}
    required_evidence = validation_gates.get("required_evidence") or []
    if not isinstance(required_evidence, list) or not required_evidence:
        add("validation_gates.required_evidence", "must be a non-empty list")
    minimum_state = validation_gates.get("minimum_validation_state_for_research_claim")
    if minimum_state not in VALIDATION_STATES:
        add("validation_gates.minimum_validation_state_for_research_claim", f"must be one of {sorted(VALIDATION_STATES)}")

    reports = profile.get("required_reports") or []
    if not isinstance(reports, list) or not reports:
        add("required_reports", "must be a non-empty list")

    research_basis = profile.get("research_basis") or []
    if not isinstance(research_basis, list) or not research_basis:
        add("research_basis", "must be a non-empty list")
    else:
        for research_id in research_basis:
            if str(research_id) not in known_research_ids:
                add("research_basis", f"unknown research id {research_id!r}")

    comparison_profile = profile.get("comparison_profile") or {}
    if comparison_profile:
        if not isinstance(comparison_profile, Mapping):
            add("comparison_profile", "must be a mapping")
        else:
            for key in COMPARISON_PROFILE_KEYS:
                try:
                    metric_list(comparison_profile.get(key))
                except TypeError as exc:
                    add(f"comparison_profile.{key}", str(exc))
    return errors


def audit_profiles(root: Path = OBJECTIVES_DIR) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    registry = load_registry()
    known_research_ids = set((registry.get("entries") or {}).keys())
    profiles = load_objective_profiles(root)
    errors: list[dict[str, str]] = []
    for profile in profiles.values():
        errors.extend(audit_profile(profile, known_research_ids=known_research_ids))
    return profiles, errors


def render_profile_list(profiles: Mapping[str, Mapping[str, Any]]) -> None:
    table = Table(title="Objective Profiles")
    table.add_column("ID")
    table.add_column("Implementation")
    table.add_column("Validation")
    table.add_column("Primary Metric")
    for objective_id, profile in sorted(profiles.items()):
        primary = profile.get("primary_goal") or {}
        metric = primary.get("metric") if isinstance(primary, Mapping) else ""
        table.add_row(
            objective_id,
            str(profile.get("implementation_status") or ""),
            str(profile.get("validation_state") or ""),
            str(metric or ""),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and audit Model Forge objective profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured objective profiles")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one objective profile")
    show_parser.add_argument("objective_id")
    show_parser.add_argument("--json", action="store_true")

    audit_parser = subparsers.add_parser("audit", help="Validate objective profiles")
    audit_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "list":
        profiles = load_objective_profiles()
        if args.json:
            print(json.dumps({key: {k: v for k, v in value.items() if k != "_path"} for key, value in profiles.items()}, indent=2, sort_keys=True, default=str))
        else:
            render_profile_list(profiles)
        return

    if args.command == "show":
        profiles = load_objective_profiles()
        if args.objective_id not in profiles:
            raise SystemExit(f"unknown objective profile {args.objective_id!r}")
        profile = {k: v for k, v in profiles[args.objective_id].items() if k != "_path"}
        if args.json:
            print(json.dumps(profile, indent=2, sort_keys=True, default=str))
        else:
            console.print(yaml.safe_dump(profile, sort_keys=False))
        return

    profiles, errors = audit_profiles()
    payload = {"profile_count": len(profiles), "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif errors:
        table = Table(title="Objective Audit Failures")
        table.add_column("Profile")
        table.add_column("Field")
        table.add_column("Message")
        for error in errors:
            table.add_row(error["profile"], error["field"], error["message"])
        console.print(table)
    else:
        console.print(f"[green]objective audit OK[/green]: {len(profiles)} profile(s)")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
