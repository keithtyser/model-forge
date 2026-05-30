from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model_forge.runs.manifest import display_path
from model_forge.variants.manifest import load_family, variant_config
from model_forge.variants.tokenizer_audit import variant_local_path


REQUIRED_ARCH_FIELDS = (
    "family",
    "tokenizer_family",
    "context_length",
    "target_discovery",
)
REQUIRED_TARGET_FIELDS = (
    "inspect_before_training_or_ablation",
    "common_attention_patterns",
    "common_mlp_patterns",
    "edit_exclusion_patterns",
    "router_or_expert_policy",
)
ACCEPTED_ROUTER_POLICIES = {
    "not_applicable_dense",
    "inspect_model_config",
    "exclude_router_and_experts_by_default",
    "explicit_recipe_required",
}
MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "router_aux_loss_coef",
)


@dataclass(frozen=True)
class ArchitectureFinding:
    level: str
    check: str
    message: str
    variant: str | None = None


def read_model_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def model_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
        "sliding_window",
        "vocab_size",
        *MOE_KEYS,
    )
    return {key: config[key] for key in keys if key in config}


def config_indicates_moe(config: Mapping[str, Any]) -> bool:
    text = json.dumps(model_config_summary(config), sort_keys=True).lower()
    if "moe" in text or "mixture" in text or "expert" in text:
        return True
    for key in MOE_KEYS:
        value = config.get(key)
        if isinstance(value, (int, float)) and value:
            return True
    return False


def audit_architecture_metadata(family_config: Mapping[str, Any]) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    architecture = family_config.get("architecture") or {}
    if not isinstance(architecture, Mapping):
        return [ArchitectureFinding("error", "architecture", "architecture must be a mapping")]
    for field in REQUIRED_ARCH_FIELDS:
        if not architecture.get(field):
            findings.append(ArchitectureFinding("error", "architecture", f"architecture.{field} is required"))
    target = architecture.get("target_discovery") or {}
    if not isinstance(target, Mapping):
        findings.append(ArchitectureFinding("error", "target_discovery", "architecture.target_discovery must be a mapping"))
        return findings
    for field in REQUIRED_TARGET_FIELDS:
        if target.get(field) in (None, "", []):
            findings.append(ArchitectureFinding("error", "target_discovery", f"target_discovery.{field} is required"))
    policy = target.get("router_or_expert_policy")
    if policy and str(policy) not in ACCEPTED_ROUTER_POLICIES:
        findings.append(
            ArchitectureFinding(
                "error",
                "router_or_expert_policy",
                f"unsupported router_or_expert_policy {policy!r}",
            )
        )
    exclusions = [str(item).lower() for item in target.get("edit_exclusion_patterns", []) if str(item).strip()]
    for required in ("embed", "lm_head"):
        if not any(required in item for item in exclusions):
            findings.append(ArchitectureFinding("error", "edit_exclusion_patterns", f"missing exclusion pattern for {required}"))
    if policy != "not_applicable_dense" and not any("router" in item or "expert" in item for item in exclusions):
        findings.append(
            ArchitectureFinding(
                "error",
                "edit_exclusion_patterns",
                "missing router or expert exclusion pattern for MoE-capable families",
            )
        )
    return findings


def build_architecture_audit(
    family: str,
    *,
    variant: str | None = None,
    models_dir_override: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    family_config = load_family(family)
    findings = audit_architecture_metadata(family_config)
    variants = family_config.get("variants") or {}
    selected = [variant] if variant else ["base"]
    records: list[dict[str, Any]] = []
    target = (family_config.get("architecture") or {}).get("target_discovery") or {}
    policy = str(target.get("router_or_expert_policy") or "")

    for variant_name in selected:
        variant_data = variant_config(family_config, str(variant_name))
        path = variant_local_path(family_config, variant_data, models_dir_override)
        record: dict[str, Any] = {
            "variant": variant_name,
            "path": display_path(path) if path else None,
            "exists": bool(path and path.exists()),
            "source_variant": variant_data.get("base_variant"),
        }
        if not record["exists"]:
            level = "error" if strict else "warning"
            findings.append(ArchitectureFinding(level, "local_dir", "configured local_dir is not present", str(variant_name)))
            records.append(record)
            continue
        config = read_model_config(path)
        record["config_json_exists"] = bool(config)
        record["model_config"] = model_config_summary(config)
        record["moe_detected"] = config_indicates_moe(config)
        if not config:
            level = "error" if strict else "warning"
            findings.append(ArchitectureFinding(level, "model_config", "config.json is missing or unreadable", str(variant_name)))
        if record["moe_detected"] and policy in {"", "not_applicable_dense"}:
            findings.append(
                ArchitectureFinding(
                    "error",
                    "moe_router_policy",
                    "model config indicates MoE/expert behavior but router_or_expert_policy does not require inspection or exclusion",
                    str(variant_name),
                )
            )
        if record["moe_detected"]:
            exclusions = [str(item).lower() for item in target.get("edit_exclusion_patterns", []) if str(item).strip()]
            if not any("router" in item or "expert" in item for item in exclusions):
                findings.append(
                    ArchitectureFinding(
                        "error",
                        "moe_exclusions",
                        "MoE model needs router/expert exclusion patterns before behavior edits or LoRA target reuse",
                        str(variant_name),
                    )
                )
        records.append(record)

    errors = [finding for finding in findings if finding.level == "error"]
    return {
        "schema_version": "model_forge.architecture_audit.v1",
        "family": family,
        "metadata": family_config.get("architecture") or {},
        "variant_count": len(selected),
        "records": records,
        "findings": [asdict(finding) for finding in findings],
        "passed": not errors,
    }
