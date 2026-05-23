from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from model_forge.objectives import IMPLEMENTATION_STATUSES, VALIDATION_STATES
from model_forge.runs.manifest import REPO_DIR, display_path, git_metadata, redact_value, sanitize_run_id


SCHEMA_VERSION = "model_forge.variant_node.v1"
PROMOTION_DECISIONS = {"promoted", "rejected", "inconclusive", "research_report_only"}
DEFAULT_OUTPUT_ROOT = REPO_DIR / "reports" / "generated" / "variants"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def family_config_path(family: str) -> Path:
    return REPO_DIR / "configs" / "model_families" / f"{family}.yaml"


def load_family(family: str) -> dict[str, Any]:
    path = family_config_path(family)
    if not path.exists():
        raise ValueError(f"unknown family {family!r}; expected {display_path(path)}")
    config = load_yaml(path)
    config["_path"] = display_path(path)
    return config


def variant_config(family_config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    variants = family_config.get("variants") or {}
    if variant not in variants:
        raise ValueError(f"unknown variant {variant!r}; valid variants: {', '.join(sorted(variants))}")
    raw = variants[variant]
    if not isinstance(raw, Mapping):
        raise ValueError(f"variant {variant!r} must be a mapping")
    return dict(raw)


def models_root(family_config: Mapping[str, Any]) -> str:
    return str(family_config.get("default_models_dir") or "~/models")


def configured_local_path(family_config: Mapping[str, Any], variant: Mapping[str, Any]) -> str | None:
    raw = variant.get("merged_local_dir") or variant.get("local_dir")
    if not raw:
        return None
    raw_path = Path(str(raw))
    if raw_path.is_absolute() or str(raw).startswith("~"):
        return str(raw)
    return str(Path(models_root(family_config)) / raw_path)


def infer_transform_type(variant_name: str, variant: Mapping[str, Any]) -> str:
    if variant.get("transform_type"):
        return str(variant["transform_type"])
    quantization = str(variant.get("quantization") or "")
    if quantization:
        return "quantize"
    if variant.get("adapter") or variant.get("lora_rank") or str(variant.get("serve_strategy") or "") == "merged":
        return "fine_tune"
    lowered = variant_name.lower()
    if "abli" in lowered or "abliterat" in lowered:
        return "behavior_edit"
    if variant.get("base_variant"):
        return "derive"
    return "source"


def infer_objective(transform_type: str, variant_name: str) -> str | None:
    if transform_type == "fine_tune":
        return "capability_sft"
    if transform_type == "behavior_edit":
        return "zero_refusal_capability_retention"
    if transform_type == "quantize":
        return "quantized_quality_retention"
    if "serv" in variant_name:
        return "dgx_spark_latency_throughput"
    return None


def transform_from_variant(variant_name: str, variant: Mapping[str, Any]) -> dict[str, Any]:
    transform_type = infer_transform_type(variant_name, variant)
    transform = {
        "type": transform_type,
        "objective": infer_objective(transform_type, variant_name),
    }
    for key in ("config", "backend", "recipe", "quantization", "serve_strategy", "lora_rank"):
        if variant.get(key) is not None:
            transform[key] = variant[key]
    return {key: value for key, value in transform.items() if value is not None}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: str | Path) -> dict[str, Any]:
    raw = Path(str(path)).expanduser()
    entry: dict[str, Any] = {"path": display_path(raw)}
    try:
        stat = raw.stat()
    except OSError:
        entry["exists"] = False
        return entry
    entry["exists"] = True
    entry["size_bytes"] = stat.st_size
    if raw.is_file():
        entry["sha256"] = file_sha256(raw)
        entry["kind"] = "file"
    elif raw.is_dir():
        entry["kind"] = "directory"
        entry["file_count"] = sum(1 for child in raw.rglob("*") if child.is_file())
    else:
        entry["kind"] = "other"
    return entry


def default_variant_node(
    family: str,
    variant_name: str,
    *,
    implementation_status: str = "scaffolded",
    validation_state: str = "planned",
    promotion_decision: str = "inconclusive",
    command: str | None = None,
    spark_evidence_path: str | None = None,
    node_count: int | None = None,
    hardware_profile: str | None = None,
    cluster_topology: str | None = None,
    baseline_run_id: str | None = None,
    artifacts: list[str | Path] | None = None,
    logs: Mapping[str, str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    retention_decision: str = "undecided",
    keep_until: str | None = None,
    disk_budget_gb: int | float | None = None,
) -> dict[str, Any]:
    if implementation_status not in IMPLEMENTATION_STATUSES:
        raise ValueError(f"unsupported implementation_status {implementation_status!r}")
    if validation_state not in VALIDATION_STATES:
        raise ValueError(f"unsupported validation_state {validation_state!r}")
    if promotion_decision not in PROMOTION_DECISIONS:
        raise ValueError(f"unsupported promotion_decision {promotion_decision!r}")

    family_config = load_family(family)
    variant = variant_config(family_config, variant_name)
    source_variant = variant.get("base_variant")
    transform = transform_from_variant(variant_name, variant)
    local_path = configured_local_path(family_config, variant)
    artifact_entries = [artifact_entry(path) for path in artifacts or []]
    if local_path:
        artifact_entries.insert(
            0,
            {
                "path": local_path,
                "kind": "configured_checkpoint_path",
                "exists": Path(local_path).expanduser().exists(),
            },
        )

    validation = {
        "implementation_status": implementation_status,
        "validation_state": validation_state,
        "spark_evidence_path": spark_evidence_path,
        "node_count": node_count,
        "hardware_profile": hardware_profile,
        "cluster_topology": cluster_topology,
        "command": command,
        "baseline_run_id": baseline_run_id,
        "metrics": dict(metrics or {}),
        "logs": dict(logs or {}),
        "known_failure_modes": [],
        "promotion_decision": promotion_decision,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "variant_id": sanitize_run_id(f"{family}_{variant_name}"),
        "family": family,
        "variant": variant_name,
        "source_variant": source_variant,
        "model": {
            "repo_id": variant.get("repo_id"),
            "served_model_name": variant.get("served_model_name"),
            "downloadable": variant.get("downloadable", True),
        },
        "transforms": [] if transform.get("type") == "source" else [transform],
        "checkpoint": {
            "local_path": local_path,
            "source_revision": variant.get("revision"),
            "format": variant.get("format") or "hf_safetensors",
            "merged_adapters": bool(variant.get("merged_local_dir")),
            "artifacts": artifact_entries,
        },
        "validation": validation,
        "retention": {
            "keep_until": keep_until,
            "disk_budget_gb": disk_budget_gb,
            "publish_or_delete_decision": retention_decision,
        },
        "publication_class": variant.get("publication_class") or "research_report_only",
        "git": git_metadata(),
    }


def validate_variant_node(node: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if node.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for field in ("variant_id", "family", "variant", "validation", "retention"):
        if not node.get(field):
            errors.append(f"{field} is required")
    validation = node.get("validation") or {}
    if not isinstance(validation, Mapping):
        errors.append("validation must be a mapping")
        validation = {}
    if validation.get("implementation_status") not in IMPLEMENTATION_STATUSES:
        errors.append("validation.implementation_status is missing or unsupported")
    if validation.get("validation_state") not in VALIDATION_STATES:
        errors.append("validation.validation_state is missing or unsupported")
    if validation.get("promotion_decision") not in PROMOTION_DECISIONS:
        errors.append("validation.promotion_decision is missing or unsupported")
    retention = node.get("retention") or {}
    if not isinstance(retention, Mapping):
        errors.append("retention must be a mapping")
    elif "publish_or_delete_decision" not in retention:
        errors.append("retention.publish_or_delete_decision is required")
    return errors


def node_output_path(family: str, variant: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / sanitize_run_id(family) / sanitize_run_id(variant) / "variant_node.json"


def write_variant_node(node: Mapping[str, Any], path: Path) -> None:
    errors = validate_variant_node(node)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_value(node), indent=2, sort_keys=True) + "\n", encoding="utf-8")
