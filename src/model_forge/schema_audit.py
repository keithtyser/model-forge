from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from model_forge.artifacts.validate import SCHEMA_VERSION as ARTIFACT_CARD_SCHEMA_VERSION
from model_forge.evals.behavior_scorecard import SCHEMA_VERSION as BEHAVIOR_SCORECARD_SCHEMA_VERSION
from model_forge.evals.run_eval import EVAL_PROVENANCE_SCHEMA_VERSION
from model_forge.objectives import SCHEMA_VERSION as OBJECTIVE_SCHEMA_VERSION
from model_forge.objectives import audit_profiles
from model_forge.quantization.cli import CARD_SCHEMA_VERSION as QUANTIZATION_CARD_SCHEMA_VERSION
from model_forge.reports.kernel_card import SCHEMA_VERSION as KERNEL_CARD_SCHEMA_VERSION
from model_forge.runs.manifest import SCHEMA_VERSION as RUN_MANIFEST_SCHEMA_VERSION
from model_forge.runs.manifest import build_canonical_manifest
from model_forge.variants.manifest import SCHEMA_VERSION as VARIANT_NODE_SCHEMA_VERSION
from model_forge.variants.manifest import default_variant_node, validate_variant_node


SCHEMA_AUDIT_VERSION = "model_forge.validation_schema_audit.v1"
console = Console()


@dataclass(frozen=True)
class SchemaCheck:
    name: str
    schema_version: str
    artifact_class: str
    passed: bool
    message: str


def valid_schema_version(value: str) -> bool:
    return value.startswith("model_forge.") and value.endswith(".v1")


def check_schema_constant(name: str, schema_version: str, artifact_class: str) -> SchemaCheck:
    return SchemaCheck(
        name=name,
        schema_version=schema_version,
        artifact_class=artifact_class,
        passed=valid_schema_version(schema_version),
        message=f"{schema_version} uses model_forge.*.v1 naming",
    )


def build_schema_audit() -> dict[str, Any]:
    checks: list[SchemaCheck] = [
        check_schema_constant("run_manifest_schema", RUN_MANIFEST_SCHEMA_VERSION, "manifest"),
        check_schema_constant("objective_profile_schema", OBJECTIVE_SCHEMA_VERSION, "objective"),
        check_schema_constant("variant_node_schema", VARIANT_NODE_SCHEMA_VERSION, "variant_node"),
        check_schema_constant("eval_provenance_card_schema", EVAL_PROVENANCE_SCHEMA_VERSION, "card"),
        check_schema_constant("artifact_execution_card_schema", ARTIFACT_CARD_SCHEMA_VERSION, "card"),
        check_schema_constant("behavior_scorecard_schema", BEHAVIOR_SCORECARD_SCHEMA_VERSION, "card"),
        check_schema_constant("kernel_card_schema", KERNEL_CARD_SCHEMA_VERSION, "card"),
        check_schema_constant("quantization_card_schema", QUANTIZATION_CARD_SCHEMA_VERSION, "card"),
    ]

    _profiles, objective_errors = audit_profiles()
    checks.append(
        SchemaCheck(
            name="objective_profiles_validate",
            schema_version=OBJECTIVE_SCHEMA_VERSION,
            artifact_class="objective",
            passed=not objective_errors,
            message=f"{len(objective_errors)} objective profile audit error(s)",
        )
    )

    manifest = build_canonical_manifest(
        run_type="generic",
        status="planned",
        family="gemma4_26b_a4b",
        variant="base",
        objective_profile="capability_sft",
        command="./forge schema audit",
        config_paths=["configs/objectives/capability_sft.yaml"],
        metadata={"purpose": "schema audit fixture"},
        now=datetime(2026, 5, 30, tzinfo=timezone.utc),
    )
    manifest_required = {"schema_version", "run_id", "run_type", "status", "identity", "source", "git", "command", "configs", "hardware", "system", "outputs"}
    missing_manifest = sorted(field for field in manifest_required if field not in manifest)
    checks.append(
        SchemaCheck(
            name="run_manifest_required_fields",
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            artifact_class="manifest",
            passed=manifest.get("schema_version") == RUN_MANIFEST_SCHEMA_VERSION and not missing_manifest,
            message=f"missing required fields: {missing_manifest or 'none'}",
        )
    )

    node = default_variant_node(
        "gemma4_26b_a4b",
        "base",
        command="./forge schema audit",
        retention_decision="research_report_only",
    )
    node_errors = validate_variant_node(node)
    checks.append(
        SchemaCheck(
            name="variant_node_required_fields",
            schema_version=VARIANT_NODE_SCHEMA_VERSION,
            artifact_class="variant_node",
            passed=not node_errors and bool(node.get("validation")) and bool(node.get("retention")),
            message="; ".join(node_errors) or "variant node validation, evidence, and retention fields present",
        )
    )

    payload = {
        "schema_version": SCHEMA_AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "required_artifact_classes": ["manifest", "card", "objective", "variant_node"],
        "notes": [
            "This audit verifies schema-version registration and required validation fields.",
            "It does not prove a specific heavy run completed; run-specific evidence still belongs in manifests and cards.",
        ],
    }
    return payload


def render_audit(report: dict[str, Any]) -> None:
    table = Table(title="Validation Schema Audit")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Class")
    table.add_column("Schema")
    table.add_column("Message")
    for check in report["checks"]:
        table.add_row(
            "PASS" if check["passed"] else "FAIL",
            check["name"],
            check["artifact_class"],
            check["schema_version"],
            check["message"],
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Model Forge validation schemas across manifests, cards, objectives, and variants")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        report = build_schema_audit()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            render_audit(report)
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
