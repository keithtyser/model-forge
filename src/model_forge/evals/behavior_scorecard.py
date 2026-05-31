from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "behavior_edit" / "gemma4_26b_a4b_scorecard.yaml"
SCHEMA_VERSION = "model_forge.behavior_edit_scorecard.v1"
console = Console()


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    message: str
    path: str | None = None
    profile: str | None = None


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


def load_comparison(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected comparison JSON object in {display_path(resolved)}")
    return data


def audit_config(config: Mapping[str, Any], config_path: Path, *, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    config_display = display_path(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        findings.append(Finding("error", "schema", f"schema_version must be {SCHEMA_VERSION}", config_display))
    if not config.get("family"):
        findings.append(Finding("error", "schema", "family is required", config_display))
    comparison_path = config.get("comparison_path")
    if not comparison_path:
        findings.append(Finding("error", "schema", "comparison_path is required", config_display))
    elif not resolve_repo_path(str(comparison_path)).exists():
        severity = "error" if strict else "warning"
        findings.append(Finding(severity, "comparison_path", f"comparison_path does not exist: {comparison_path}", config_display))
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        findings.append(Finding("error", "schema", "profiles must be a non-empty mapping", config_display))
        return findings
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            findings.append(Finding("error", "profile", "profile must be a mapping", config_display, str(profile_name)))
            continue
        for field in ("candidate", "reference", "objective"):
            if not profile.get(field):
                findings.append(Finding("error", "profile", f"{field} is required", config_display, str(profile_name)))
        rubric = profile.get("rubric")
        if not isinstance(rubric, list) or not rubric:
            findings.append(Finding("error", "rubric", "rubric must be a non-empty list", config_display, str(profile_name)))
            continue
        for item in rubric:
            if not isinstance(item, dict):
                findings.append(Finding("error", "rubric", "rubric items must be mappings", config_display, str(profile_name)))
                continue
            for field in ("name", "category", "bucket", "metric", "operator"):
                if not item.get(field):
                    findings.append(Finding("error", "rubric", f"rubric item missing {field}", config_display, str(profile_name)))
            if item.get("operator") not in {">=", "<=", ">", "<", "reported"}:
                findings.append(Finding("error", "rubric", f"unsupported operator {item.get('operator')!r}", config_display, str(profile_name)))
    return findings


def score_rows(comparison: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in comparison.get("score_rows", []):
        if isinstance(row, Mapping):
            rows[(str(row.get("bucket") or ""), str(row.get("metric") or ""))] = row
    return rows


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def threshold_value(item: Mapping[str, Any], row: Mapping[str, Any], reference: str) -> float | None:
    target = item.get("target")
    if target == "reference":
        return numeric(row.get(reference))
    return numeric(target)


def evaluate_rubric_item(item: Mapping[str, Any], row: Mapping[str, Any] | None, candidate: str, reference: str) -> dict[str, Any]:
    operator = str(item.get("operator"))
    required = bool(item.get("required", False))
    result = {
        "name": item.get("name"),
        "category": item.get("category"),
        "bucket": item.get("bucket"),
        "metric": item.get("metric"),
        "operator": operator,
        "required": required,
        "interpretation": item.get("interpretation"),
        "candidate_value": None,
        "reference_value": None,
        "target_value": None,
        "passed": False,
        "status": "missing",
        "reason": "metric row missing from comparison",
    }
    if row is None:
        result["passed"] = not required
        return result
    candidate_value = numeric(row.get(candidate))
    reference_value = numeric(row.get(reference))
    target = threshold_value(item, row, reference)
    result.update({
        "candidate_value": candidate_value,
        "reference_value": reference_value,
        "target_value": target,
    })
    if operator == "reported":
        result.update({"passed": True, "status": "reported", "reason": "metric is reported as risk/context, not a failure gate"})
        return result
    if candidate_value is None:
        result["reason"] = f"candidate {candidate!r} has no value"
        return result
    if target is None:
        result["reason"] = "target has no value"
        return result
    tolerance = float(item.get("tolerance", 0.0) or 0.0)
    if operator == ">=":
        passed = candidate_value + tolerance >= target
    elif operator == "<=":
        passed = candidate_value - tolerance <= target
    elif operator == ">":
        passed = candidate_value > target + tolerance
    elif operator == "<":
        passed = candidate_value < target - tolerance
    else:
        raise ValueError(f"unsupported operator: {operator}")
    result.update({
        "passed": passed,
        "status": "pass" if passed else "fail",
        "tolerance": tolerance,
        "reason": f"{candidate_value:g} {operator} {target:g}",
    })
    return result


def evaluate_profile(config: Mapping[str, Any], profile_name: str, comparison: Mapping[str, Any]) -> dict[str, Any]:
    profiles = config.get("profiles") or {}
    if profile_name not in profiles:
        valid = ", ".join(sorted(str(name) for name in profiles))
        raise SystemExit(f"unknown behavior scorecard profile {profile_name!r}; valid profiles: {valid}")
    profile = profiles[profile_name]
    candidate = str(profile["candidate"])
    reference = str(profile["reference"])
    rows = score_rows(comparison)
    rubric_results = [
        evaluate_rubric_item(item, rows.get((str(item["bucket"]), str(item["metric"]))), candidate, reference)
        for item in profile.get("rubric", [])
    ]
    required = [item for item in rubric_results if item.get("required")]
    passed = bool(required) and all(item.get("passed") for item in required)
    labels = profile.get("decision_labels") or {}
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in rubric_results:
        categories.setdefault(str(item.get("category")), []).append(item)
    return redact_value(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "family": config.get("family"),
            "profile": profile_name,
            "candidate": candidate,
            "reference": reference,
            "objective": profile.get("objective"),
            "decision": labels.get("pass" if passed else "fail", "pass" if passed else "hold"),
            "passed": passed,
            "rubric": rubric_results,
            "categories": categories,
            "source_comparison": config.get("comparison_path"),
            "notes": [
                "For refusal-removal objectives, lower harmful-prompt refusal can be success.",
                "Unsafe overcompliance and harmful detail are reported risks here; deployment-safety objectives may gate them differently.",
                "Promotion still requires current eval provenance and source-relative capability retention.",
            ],
        }
    )


def render_markdown(scorecard: Mapping[str, Any]) -> str:
    rows = []
    for item in scorecard.get("rubric") or []:
        rows.append(
            "| {status} | {required} | {category} | `{bucket}.{metric}` | {candidate} | {target} | {reason} |".format(
                status=str(item.get("status", "")).upper(),
                required="yes" if item.get("required") else "no",
                category=item.get("category", ""),
                bucket=item.get("bucket", ""),
                metric=item.get("metric", ""),
                candidate=item.get("candidate_value"),
                target=item.get("target_value"),
                reason=item.get("reason", ""),
            )
        )
    return "\n".join(
        [
            f"# Behavior Edit Scorecard: {scorecard.get('profile')}",
            "",
            f"- Family: `{scorecard.get('family')}`",
            f"- Candidate: `{scorecard.get('candidate')}`",
            f"- Reference: `{scorecard.get('reference')}`",
            f"- Objective: `{scorecard.get('objective')}`",
            f"- Decision: `{scorecard.get('decision')}`",
            f"- Passed required gates: `{str(scorecard.get('passed')).lower()}`",
            "",
            "| Status | Required | Category | Metric | Candidate | Target | Reason |",
            "|---|---|---|---|---:|---:|---|",
            *rows,
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in scorecard.get("notes") or []],
            "",
        ]
    )


def write_scorecard(config: Mapping[str, Any], profile_name: str, scorecard: Mapping[str, Any]) -> dict[str, Path]:
    output_dir = resolve_repo_path(str(config.get("output_dir") or "reports/generated/behavior_edit_scorecards"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_run_id(profile_name)
    json_path = output_dir / f"{safe_name}.json"
    md_path = output_dir / f"{safe_name}.md"
    json_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(scorecard), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("behavior scorecard config: OK")
        return
    table = Table(title="Behavior Scorecard Doctor")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Profile")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity.upper(), finding.check, finding.profile or "", finding.message)
    console.print(table)


def render_scorecard(scorecard: Mapping[str, Any]) -> None:
    table = Table(title=f"Behavior Edit Scorecard: {scorecard.get('profile')}")
    table.add_column("Gate")
    table.add_column("Metric")
    table.add_column("Candidate", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")
    for item in scorecard.get("rubric") or []:
        table.add_row(
            str(item.get("name")),
            f"{item.get('bucket')}.{item.get('metric')}",
            str(item.get("candidate_value")),
            str(item.get("target_value")),
            str(item.get("status")).upper(),
        )
    console.print(table)
    console.print(f"Decision: {scorecard.get('decision')} ({'passed' if scorecard.get('passed') else 'held'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write behavior-edit scorecards from comparison reports")
    parser.add_argument("--config", dest="global_config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("profile", nargs="?")
    score.add_argument("--config", type=Path)
    score.add_argument("--write-card", action="store_true")
    score.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config_arg = args.config or args.global_config
    config, config_path = load_yaml(config_arg)
    if args.command == "doctor":
        findings = audit_config(config, config_path, strict=args.strict)
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2, sort_keys=True) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if any(finding.severity == "error" for finding in findings) else 0)
    if args.command == "score":
        findings = audit_config(config, config_path, strict=True)
        if any(finding.severity == "error" for finding in findings):
            render_findings(findings)
            raise SystemExit(1)
        profile_name = args.profile or next(iter(config.get("profiles", {})), "")
        comparison = load_comparison(str(config["comparison_path"]))
        scorecard = evaluate_profile(config, profile_name, comparison)
        if args.write_card:
            outputs = write_scorecard(config, profile_name, scorecard)
            scorecard = {**scorecard, "outputs": {key: display_path(path) for key, path in outputs.items()}}
        if args.json:
            print(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
        else:
            render_scorecard(scorecard)


if __name__ == "__main__":
    main()
