from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.objectives import IMPLEMENTATION_STATUSES, VALIDATION_STATES
from model_forge.variants.graph import ancestry, variant_graph
from model_forge.variants.manifest import (
    DEFAULT_OUTPUT_ROOT,
    PROMOTION_DECISIONS,
    default_variant_node,
    node_output_path,
    validate_variant_node,
    write_variant_node,
)
from model_forge.variants.tokenizer_audit import build_tokenizer_audit


console = Console()


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("KEY must not be empty")
    value = value.strip()
    if value == "":
        return key, ""
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def key_value_mapping(items: list[str] | None) -> dict[str, Any]:
    data = {}
    for item in items or []:
        key, value = parse_key_value(item)
        data[key] = value
    return data


def render_graph(graph: dict[str, Any]) -> None:
    table = Table(title=f"Variant Graph: {graph['family']}")
    table.add_column("Variant")
    table.add_column("Source")
    table.add_column("Transform")
    table.add_column("Objective")
    for node in graph["nodes"]:
        table.add_row(
            str(node["variant"]),
            str(node.get("source_variant") or ""),
            str(node.get("transform_type") or ""),
            str(node.get("objective") or ""),
        )
    console.print(table)


def render_node(node: dict[str, Any]) -> None:
    validation = node.get("validation") or {}
    table = Table(title=f"Variant Node: {node.get('variant')}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in [
        ("family", node.get("family")),
        ("source_variant", node.get("source_variant")),
        ("implementation_status", validation.get("implementation_status")),
        ("validation_state", validation.get("validation_state")),
        ("promotion_decision", validation.get("promotion_decision")),
        ("checkpoint", (node.get("checkpoint") or {}).get("local_path")),
    ]:
        table.add_row(key, str(value or ""))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and write Model Forge variant graph metadata")
    subparsers = parser.add_subparsers(dest="action", required=True)

    graph_parser = subparsers.add_parser("graph", help="Show the configured variant graph for a family")
    graph_parser.add_argument("family")
    graph_parser.add_argument("--variant", help="Show ancestry for one variant")
    graph_parser.add_argument("--json", action="store_true")

    node_parser = subparsers.add_parser("node", help="Build or write a variant_node.json for one variant")
    node_parser.add_argument("family")
    node_parser.add_argument("variant")
    node_parser.add_argument("--implementation-status", default="scaffolded", choices=sorted(IMPLEMENTATION_STATUSES))
    node_parser.add_argument("--validation-state", default="planned", choices=sorted(VALIDATION_STATES))
    node_parser.add_argument("--promotion-decision", default="inconclusive", choices=sorted(PROMOTION_DECISIONS))
    node_parser.add_argument("--command", dest="run_command")
    node_parser.add_argument("--spark-evidence-path")
    node_parser.add_argument("--node-count", type=int)
    node_parser.add_argument("--hardware-profile")
    node_parser.add_argument("--cluster-topology")
    node_parser.add_argument("--baseline-run-id")
    node_parser.add_argument("--artifact", action="append", default=[])
    node_parser.add_argument("--metric", action="append", default=[], type=parse_key_value)
    node_parser.add_argument("--log", action="append", default=[], type=parse_key_value)
    node_parser.add_argument("--retention-decision", default="undecided")
    node_parser.add_argument("--keep-until")
    node_parser.add_argument("--disk-budget-gb", type=float)
    node_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    node_parser.add_argument("--write", action="store_true")
    node_parser.add_argument("--json", action="store_true")

    audit_parser = subparsers.add_parser("audit-node", help="Validate a variant_node.json file")
    audit_parser.add_argument("path", type=Path)
    audit_parser.add_argument("--json", action="store_true")

    tokenizer_parser = subparsers.add_parser("tokenizer-audit", help="Check tokenizer/chat-template preservation for configured variants")
    tokenizer_parser.add_argument("family")
    tokenizer_parser.add_argument("--variant", help="Audit one variant plus any needed source variant")
    tokenizer_parser.add_argument("--models-dir", help="Override the family models directory")
    tokenizer_parser.add_argument("--load-tokenizer", action="store_true", help="Run a live AutoTokenizer chat-template round trip for present local dirs")
    tokenizer_parser.add_argument("--strict", action="store_true", help="treat missing local dirs as errors")
    tokenizer_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.action == "graph":
        graph = variant_graph(args.family)
        if args.variant:
            graph["ancestry"] = ancestry(graph, args.variant)
        if args.json:
            print(json.dumps(graph, indent=2, sort_keys=True))
        else:
            render_graph(graph)
            if args.variant:
                console.print(" -> ".join(graph["ancestry"]))
        return

    if args.action == "node":
        node = default_variant_node(
            args.family,
            args.variant,
            implementation_status=args.implementation_status,
            validation_state=args.validation_state,
            promotion_decision=args.promotion_decision,
            command=args.run_command,
            spark_evidence_path=args.spark_evidence_path,
            node_count=args.node_count,
            hardware_profile=args.hardware_profile,
            cluster_topology=args.cluster_topology,
            baseline_run_id=args.baseline_run_id,
            artifacts=args.artifact,
            metrics=dict(args.metric or []),
            logs=dict(args.log or []),
            retention_decision=args.retention_decision,
            keep_until=args.keep_until,
            disk_budget_gb=args.disk_budget_gb,
        )
        if args.write:
            path = node_output_path(args.family, args.variant, args.output_root)
            write_variant_node(node, path)
            node["output_path"] = str(path)
        if args.json:
            print(json.dumps(node, indent=2, sort_keys=True, default=str))
        else:
            render_node(node)
        return

    if args.action == "tokenizer-audit":
        audit = build_tokenizer_audit(
            args.family,
            variant=args.variant,
            models_dir_override=args.models_dir,
            load_tokenizer=args.load_tokenizer,
            strict=args.strict,
        )
        if args.json:
            print(json.dumps(audit, indent=2, sort_keys=True))
        else:
            table = Table(title=f"Tokenizer Audit: {audit['family']}")
            table.add_column("Variant")
            table.add_column("Exists")
            table.add_column("Source")
            table.add_column("Template")
            table.add_column("Tokenizer")
            for record in audit["records"]:
                files = record.get("files") or {}
                tokenizer_kind = "json" if "tokenizer.json" in files else "model" if "tokenizer.model" in files else ""
                table.add_row(
                    str(record["variant"]),
                    str(record.get("exists")),
                    str(record.get("source_variant") or ""),
                    str(record.get("chat_template_source") or ""),
                    tokenizer_kind,
                )
            console.print(table)
            for finding in audit["findings"]:
                style = "red" if finding["level"] == "error" else "yellow"
                console.print(f"[{style}]{finding['level']} {finding['variant']} {finding['check']}: {finding['message']}[/{style}]")
        raise SystemExit(0 if audit["passed"] else 1)

    data = yaml.safe_load(args.path.read_text(encoding="utf-8")) if args.path.suffix in {".yaml", ".yml"} else json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate_variant_node(data)
    payload = {"path": str(args.path), "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
    else:
        console.print(f"[green]variant node OK[/green]: {args.path}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
