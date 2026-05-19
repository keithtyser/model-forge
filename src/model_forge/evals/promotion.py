from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table


REPO_DIR = Path(__file__).resolve().parents[3]
console = Console()


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


def load_comparison(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected comparison object in {path}")
    return data


def score_lookup(comparison: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparison.get("score_rows", []):
        if isinstance(row, dict):
            lookup[(str(row.get("bucket", "")), str(row.get("metric", "")))] = row
    return lookup


def numeric(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def target_value(gate: dict[str, Any], row: dict[str, Any], reference: str) -> float | None:
    target = gate.get("target")
    if target == "reference":
        return numeric(row.get(reference))
    return numeric(target)


def evaluate_gate(gate: dict[str, Any], row: dict[str, Any], candidate: str, reference: str) -> dict[str, Any]:
    value = numeric(row.get(candidate))
    target = target_value(gate, row, reference)
    operator = str(gate.get("operator", ">="))
    tolerance = float(gate.get("tolerance", 0.0) or 0.0)
    if value is None:
        passed = False
        reason = f"candidate {candidate!r} has no value"
    elif target is None:
        passed = False
        reason = "target has no value"
    elif operator == ">=":
        passed = value + tolerance >= target
        reason = f"{value:g} >= {target:g}"
    elif operator == "<=":
        passed = value - tolerance <= target
        reason = f"{value:g} <= {target:g}"
    elif operator == ">":
        passed = value > target + tolerance
        reason = f"{value:g} > {target:g}"
    elif operator == "<":
        passed = value < target - tolerance
        reason = f"{value:g} < {target:g}"
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "name": gate["name"],
        "bucket": gate["bucket"],
        "metric": gate["metric"],
        "operator": operator,
        "candidate_value": value,
        "reference_value": numeric(row.get(reference)),
        "target_value": target,
        "tolerance": tolerance,
        "passed": passed,
        "reason": reason,
    }


def evaluate_profile(config: dict[str, Any], profile_name: str, comparison: dict[str, Any]) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        valid = ", ".join(sorted(profiles))
        raise SystemExit(f"unknown promotion profile {profile_name!r}; valid profiles: {valid}")
    profile = profiles[profile_name]
    candidate = str(profile["candidate"])
    reference = str(profile.get("reference", "base"))
    rows = score_lookup(comparison)
    gate_results = []
    for gate in profile.get("gates", []):
        row = rows.get((str(gate["bucket"]), str(gate["metric"])))
        if row is None:
            gate_results.append({
                "name": gate["name"],
                "bucket": gate["bucket"],
                "metric": gate["metric"],
                "passed": False,
                "reason": "metric row missing from comparison",
            })
        else:
            gate_results.append(evaluate_gate(gate, row, candidate, reference))
    passed = bool(gate_results) and all(result["passed"] for result in gate_results)
    labels = profile.get("decision_labels", {})
    return {
        "family": config["family"],
        "profile": profile_name,
        "objective": profile.get("objective", "general_assistant"),
        "candidate": candidate,
        "reference": reference,
        "decision": labels.get("pass" if passed else "fail", "pass" if passed else "hold"),
        "passed": passed,
        "gates": gate_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def report_markdown(report: dict[str, Any]) -> str:
    gates = []
    for gate in report["gates"]:
        status = "PASS" if gate.get("passed") else "FAIL"
        gates.append(
            "| {status} | {name} | `{bucket}.{metric}` | {value} | {target} | {reason} |".format(
                status=status,
                name=gate.get("name", ""),
                bucket=gate.get("bucket", ""),
                metric=gate.get("metric", ""),
                value=gate.get("candidate_value"),
                target=gate.get("target_value"),
                reason=gate.get("reason", ""),
            )
        )
    gate_rows = "\n".join(gates)
    return f"""# Promotion Report: {report['profile']}

- Family: `{report['family']}`
- Candidate: `{report['candidate']}`
- Reference: `{report['reference']}`
- Objective: `{report['objective']}`
- Decision: `{report['decision']}`
- Passed: `{str(report['passed']).lower()}`

| Status | Gate | Metric | Candidate | Target | Reason |
|---|---|---|---:|---:|---|
{gate_rows}
"""


def write_report(config: dict[str, Any], profile_name: str) -> dict[str, Path]:
    comparison_path = resolve_repo_path(config["comparison_path"])
    comparison = load_comparison(comparison_path)
    report = evaluate_profile(config, profile_name, comparison)
    output_dir = resolve_repo_path(config.get("output_dir", "reports/generated/promotion"))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{profile_name}.json"
    md_path = output_dir / f"{profile_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def print_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    table = Table(title=f"Promotion: {report['profile']}")
    table.add_column("Gate")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")
    for gate in report["gates"]:
        table.add_row(
            gate["name"],
            f"{gate['bucket']}.{gate['metric']}",
            str(gate.get("candidate_value")),
            str(gate.get("target_value")),
            "PASS" if gate.get("passed") else "FAIL",
        )
    console.print(table)
    console.print(f"Decision: {report['decision']} ({'passed' if report['passed'] else 'held'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write eval promotion reports from a model-forge comparison.json")
    parser.add_argument("profile", nargs="?", help="Promotion profile to evaluate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="List configured profiles")
    args = parser.parse_args()
    config = load_yaml(resolve_repo_path(args.config))
    if args.list:
        for profile in sorted(config.get("profiles", {})):
            print(profile)
        return
    profile = args.profile or next(iter(config.get("profiles", {})), "")
    if not profile:
        raise SystemExit("promotion config has no profiles")
    outputs = write_report(config, profile)
    print_report(outputs["json"])
    console.print(f"Wrote {display_path(outputs['json'])}")
    console.print(f"Wrote {display_path(outputs['markdown'])}")


if __name__ == "__main__":
    main()
