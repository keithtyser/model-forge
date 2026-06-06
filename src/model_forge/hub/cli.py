from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


SCHEMA_VERSION = "model_forge.hub_publish_plan.v1"
DATASET_PLAN_SCHEMA_VERSION = "model_forge.hub_dataset_publish_plan.v1"
DEFAULT_HUB_CONFIG = REPO_DIR / "configs" / "hub.yaml"
RELEASE_CLASS_DIR = REPO_DIR / "configs" / "release_classes"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "reports" / "generated" / "hub"
REPO_URL = "https://github.com/keithtyser/model-forge"
VALIDATION_RANK = {
    "planned": 0,
    "smoke_validated": 1,
    "spark_single_node_validated": 2,
    "spark_cluster_validated": 3,
    "generalizable": 4,
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(HF_TOKEN|HUGGINGFACE_HUB_TOKEN|API_KEY|SECRET|PASSWORD)\s*="),
)
ABSOLUTE_PRIVATE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(home|Users)/[A-Za-z0-9_.-]+/")

console = Console(stderr=True)


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class ReleaseClassFinding:
    severity: str
    check: str
    message: str
    path: str | None = None
    release_class: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def publish_path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser()
    try:
        return str(resolved.resolve().relative_to(REPO_DIR))
    except (OSError, ValueError):
        return f"<external>/{resolved.name}"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def load_family(family: str) -> dict[str, Any]:
    path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not path.exists():
        raise ValueError(f"unknown model family {family!r}; expected {display_path(path)}")
    return load_yaml(path)


def models_root(family_config: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Path:
    env = env or os.environ
    env_name = str(family_config.get("models_dir_env") or "MODEL_FORGE_MODELS_DIR")
    return Path(str(env.get(env_name) or family_config.get("default_models_dir") or "~/models")).expanduser()


def resolve_variant(family: str, variant: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    if variant not in variants:
        raise ValueError(f"unknown variant {variant!r} for {family!r}; valid: {', '.join(sorted(variants))}")
    raw = dict(variants[variant])
    root = models_root(family_config, env)

    def variant_path(key: str) -> Path | None:
        local_dir = str(raw.get(key) or "")
        if not local_dir:
            return None
        candidate = Path(local_dir).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    adapter_path = variant_path("local_dir")
    merged_path = variant_path("merged_local_dir")
    local_path = merged_path or adapter_path
    return {
        "family": family,
        "family_display_name": family_config.get("display_name") or family,
        "variant": variant,
        "repo_id": raw.get("repo_id"),
        "served_model_name": raw.get("served_model_name") or raw.get("repo_id"),
        "base_variant": raw.get("base_variant"),
        "adapter": bool(raw.get("adapter", False)),
        "quantization": raw.get("quantization"),
        "downloadable": raw.get("downloadable", True),
        "promotion": dict(raw.get("promotion") or {}),
        "local_path": local_path,
        "adapter_path": adapter_path,
        "merged_path": merged_path,
        "raw": raw,
    }


def promotion_blocks_release(variant_info: Mapping[str, Any], release_class: Mapping[str, Any]) -> tuple[bool, str]:
    publishes_artifact = bool(release_class.get("publish_weights", False)) or bool(release_class.get("publish_adapter", False))
    if not publishes_artifact:
        return False, "report-only release does not publish model artifacts"
    promotion = variant_info.get("promotion") or {}
    if not isinstance(promotion, Mapping):
        return False, "no promotion metadata"
    decision = str(promotion.get("decision") or "").strip().lower()
    blocked_actions = {str(item).strip().lower() for item in promotion.get("blocked_actions") or []}
    blocked = decision == "rejected" or "hf_upload" in blocked_actions or "publishing" in blocked_actions or "all" in blocked_actions
    if not blocked:
        return False, "variant is not blocked by promotion metadata"
    reason = str(promotion.get("reason") or f"promotion decision is {decision or 'blocked'}")
    evidence = promotion.get("evidence")
    if isinstance(evidence, list) and evidence:
        reason = f"{reason}; evidence={', '.join(str(item) for item in evidence[:3])}"
    return True, reason


def load_hub_config(path: Path = DEFAULT_HUB_CONFIG) -> dict[str, Any]:
    return load_yaml(path) if path.exists() else {}


def load_release_class(name: str) -> dict[str, Any]:
    path = RELEASE_CLASS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown release class {name!r}; expected {display_path(path)}")
    data = load_yaml(path)
    data.setdefault("id", name)
    return data


def audit_release_classes(root: Path = RELEASE_CLASS_DIR) -> list[ReleaseClassFinding]:
    findings: list[ReleaseClassFinding] = []
    allowed_requirements = {
        "dataset_card_complete",
        "eval_results_present",
        "model_card_complete",
        "no_private_tokens_or_paths",
        "promotion_gates_passed_or_research_report_only",
        "quantization_card_present",
        "risk_report_present_or_not_applicable",
        "serving_card_present",
        "source_license_checked",
        "unsafe_examples_redacted",
    }
    allowed_raw_outputs = {False, "private_only", "redacted_only"}
    if not root.exists():
        return [ReleaseClassFinding("error", "release_class_dir", "release class directory is missing", display_path(root))]
    for path in sorted(root.glob("*.yaml")):
        try:
            data = load_yaml(path)
        except Exception as exc:
            findings.append(ReleaseClassFinding("error", "load", str(exc), display_path(path)))
            continue
        release_id = str(data.get("id") or "")
        expected_id = path.stem
        if release_id != expected_id:
            findings.append(ReleaseClassFinding("error", "id", f"id must match filename stem {expected_id!r}", display_path(path), release_id))
        if data.get("hf_visibility") not in {"public", "private"}:
            findings.append(ReleaseClassFinding("error", "visibility", "hf_visibility must be public or private", display_path(path), release_id))
        if str(data.get("minimum_validation_state") or "") not in VALIDATION_RANK:
            findings.append(ReleaseClassFinding("error", "validation_state", "minimum_validation_state is unknown", display_path(path), release_id))
        for field in ("publish_weights", "publish_adapter", "publish_reports", "allow_public_checkpoint"):
            if not isinstance(data.get(field), bool):
                findings.append(ReleaseClassFinding("error", "schema", f"{field} must be boolean", display_path(path), release_id))
        if data.get("publish_raw_outputs") not in allowed_raw_outputs:
            findings.append(ReleaseClassFinding("error", "raw_outputs", "publish_raw_outputs has unsupported value", display_path(path), release_id))
        requirements = data.get("requires")
        if not isinstance(requirements, list) or not requirements:
            findings.append(ReleaseClassFinding("error", "requires", "requires must be a non-empty list", display_path(path), release_id))
        else:
            for requirement in requirements:
                if requirement not in allowed_requirements:
                    findings.append(ReleaseClassFinding("error", "requires", f"unknown requirement {requirement!r}", display_path(path), release_id))
        public_full_checkpoint = data.get("hf_visibility") == "public" and data.get("publish_weights") and not data.get("publish_adapter")
        if public_full_checkpoint and not data.get("allow_public_checkpoint"):
            findings.append(ReleaseClassFinding("warning", "public_checkpoint", "public full checkpoint release is blocked by allow_public_checkpoint=false", display_path(path), release_id))
        if data.get("hf_visibility") == "public" and data.get("publish_raw_outputs") not in {False, "redacted_only"}:
            findings.append(ReleaseClassFinding("error", "raw_outputs", "public releases cannot publish private-only raw outputs", display_path(path), release_id))
    return findings


def token_source(env: Mapping[str, str] | None = None) -> tuple[str, bool]:
    env = env or os.environ
    if env.get("HF_TOKEN"):
        return "HF_TOKEN", True
    if env.get("HUGGINGFACE_HUB_TOKEN"):
        return "HUGGINGFACE_HUB_TOKEN", True
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
    except Exception:
        token = None
    return ("hf-cli-cache", True) if token else ("none", False)


def token_value(env: Mapping[str, str] | None = None) -> str | None:
    env = env or os.environ
    if env.get("HF_TOKEN"):
        return str(env["HF_TOKEN"])
    if env.get("HUGGINGFACE_HUB_TOKEN"):
        return str(env["HUGGINGFACE_HUB_TOKEN"])
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def hf_status(*, offline: bool = False, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source, authenticated = token_source(env)
    user = None
    error = None
    if authenticated and not offline:
        try:
            from huggingface_hub import HfApi

            token = token_value(env)
            user = (HfApi(token=token).whoami(token=token) or {}).get("name")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    config = load_hub_config()
    return {
        "authenticated": authenticated,
        "user": user,
        "token_source": source,
        "default_owner": config.get("default_owner"),
        "default_visibility": config.get("default_visibility", "private"),
        "cache_dir": config.get("cache_dir", "${HF_HOME:-~/.cache/huggingface}"),
        "dry_run_default": bool(config.get("default_dry_run", True)),
        "offline": offline,
        "error": error,
    }


def file_should_scan(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        return False
    if path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".png", ".jpg", ".jpeg", ".webp", ".pdf"}:
        return False
    return True


def scan_text_file(path: Path) -> list[str]:
    if not file_should_scan(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [f"failed to read {publish_path_label(path)}"]
    findings = []
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        findings.append(f"secret-like literal in {publish_path_label(path)}")
    if ABSOLUTE_PRIVATE_PATH_RE.search(text):
        findings.append(f"private absolute path in {publish_path_label(path)}")
    return findings


def scan_paths(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if path.is_dir():
            for child in path.rglob("*"):
                findings.extend(scan_text_file(child))
        else:
            findings.extend(scan_text_file(path))
    return findings


def resolve_supporting_path(path: str | Path | None, *, kind: str) -> tuple[Path | None, dict[str, str] | None]:
    if not path:
        return None, None
    resolved = resolve_path(path)
    if kind == "eval_results" and resolved.is_dir() and (resolved / "scores.csv").is_file():
        used = resolved / "scores.csv"
        return used, {
            "kind": kind,
            "input": publish_path_label(resolved) or str(resolved),
            "used": publish_path_label(used) or str(used),
            "reason": "serving-eval directories can contain private run manifests; scores.csv is the sanitized public evidence file",
        }
    return resolved, None


def list_model_files(path: Path | None, *, include_weights: bool, include_adapter: bool, limit: int = 200) -> tuple[list[str], list[str]]:
    if not path or not path.exists():
        return [], []
    include_suffixes = {".json", ".yaml", ".yml", ".txt", ".md", ".jinja"}
    if include_weights:
        include_suffixes.update({".safetensors", ".bin", ".pt", ".gguf"})
    if include_adapter:
        include_suffixes.update({".safetensors", ".bin"})
    included: list[str] = []
    excluded: list[str] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = str(child.relative_to(path))
        if any(part.startswith(".") for part in child.relative_to(path).parts):
            excluded.append(rel)
        elif child.suffix.lower() in include_suffixes:
            included.append(rel)
        else:
            excluded.append(rel)
        if len(included) + len(excluded) >= limit:
            excluded.append(f"... truncated after {limit} files")
            break
    return included, excluded


def slug_for(family: str, variant: str) -> str:
    return sanitize_run_id(f"model-forge-{family.replace('_', '-')}-{variant.replace('_', '-')}")


def default_repo_id(family: str, variant: str, config: Mapping[str, Any]) -> str:
    owner = str(config.get("default_owner") or "model-forge")
    return f"{owner}/{slug_for(family, variant)}"


def generate_model_card(
    *,
    family: str,
    variant: str,
    variant_info: Mapping[str, Any],
    release_class: Mapping[str, Any],
    repo_id: str,
    validation_state: str,
) -> str:
    base_model = variant_info.get("repo_id") or variant_info.get("served_model_name") or "unknown"
    quantization = variant_info.get("quantization")
    adapter = bool(variant_info.get("adapter"))
    tags = ["model-forge", "post-training"]
    if quantization:
        tags.append(str(quantization))
    if adapter:
        tags.append("adapter")
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    return f"""---
language:
  - en
library_name: transformers
pipeline_tag: text-generation
base_model: {base_model}
tags:
{tags_yaml}
---

# {repo_id}

This repository is a Model Forge release artifact for `{family}` / `{variant}`.

## Source Model

- Source/base model: `{base_model}`
- Served model name: `{variant_info.get('served_model_name') or 'n/a'}`
- Base variant: `{variant_info.get('base_variant') or 'n/a'}`

## What Changed

- Release class: `{release_class.get('id')}`
- Adapter release: `{adapter}`
- Quantization: `{quantization or 'none recorded'}`
- Validation state at planning time: `{validation_state}`

## Evidence

This card is generated from a dry-run Model Forge Hub plan. Attach eval,
serving, quantization, artifact execution, and promotion reports before making
public quality claims.

## Reproducibility

- GitHub repo: {REPO_URL}
- Model family config: `configs/model_families/{family}.yaml`
- Recommended command: `./forge hf plan-model {family} {variant} --release-class {release_class.get('id')}`

## Limitations

This card may describe a planned release. A non-dry-run upload must pass the
release-class gates and write `hub_publish.json` provenance.
"""


def gate_status(name: str, passed: bool, message: str) -> Gate:
    return Gate(name=name, status="pass" if passed else "fail", message=message)


def optional_gate(name: str, passed: bool, message: str) -> Gate:
    return Gate(name=name, status="pass" if passed else "warn", message=message)


def validation_state_passes(actual: str, minimum: str) -> bool:
    return VALIDATION_RANK.get(actual, -1) >= VALIDATION_RANK.get(minimum, 0)


def build_release_gates(
    *,
    release_class: Mapping[str, Any],
    variant_info: Mapping[str, Any],
    included_paths: list[Path],
    model_card: str,
    validation_state: str,
    args: argparse.Namespace,
) -> list[Gate]:
    required = set(str(item) for item in release_class.get("requires") or [])
    gates: list[Gate] = []
    minimum_state = str(release_class.get("minimum_validation_state") or "planned")
    gates.append(
        gate_status(
            "validation_state",
            validation_state_passes(validation_state, minimum_state),
            f"{validation_state} >= {minimum_state}",
        )
    )
    if "model_card_complete" in required:
        required_sections = ("## Source Model", "## Evidence", "## Reproducibility", "## Limitations")
        gates.append(
            gate_status(
                "model_card_complete",
                all(section in model_card for section in required_sections),
                "generated README contains required model-card sections",
            )
        )
    if "source_license_checked" in required:
        gates.append(
            gate_status(
                "source_license_checked",
                bool(args.source_license_checked),
                "pass --source-license-checked after confirming upstream license and data provenance",
            )
        )
    if "eval_results_present" in required:
        gates.append(gate_status("eval_results_present", bool(args.eval_results), "eval results path supplied"))
    if "quantization_card_present" in required:
        gates.append(
            gate_status(
                "quantization_card_present",
                bool(args.quantization_card),
                "quantization card path supplied",
            )
        )
    if "serving_card_present" in required:
        gates.append(gate_status("serving_card_present", bool(args.serving_card), "serving card path supplied"))
    if "promotion_gates_passed_or_research_report_only" in required:
        gates.append(
            optional_gate(
                "promotion_gates_passed_or_research_report_only",
                bool(args.promotion_report) or not bool(release_class.get("publish_weights", False)),
                "promotion report supplied or release is report-only",
            )
        )
    if "risk_report_present_or_not_applicable" in required:
        gates.append(
            optional_gate(
                "risk_report_present_or_not_applicable",
                bool(args.risk_report) or not args.behavior_edited,
                "risk report supplied or not behavior-edited",
            )
        )
    if release_class.get("hf_visibility") == "public" and args.behavior_edited:
        gates.append(
            gate_status(
                "behavior_edit_risk_report",
                bool(args.risk_report),
                "public behavior-edited releases require a risk or behavior-edit scorecard report",
            )
        )
    if "unsafe_examples_redacted" in required:
        gates.append(gate_status("unsafe_examples_redacted", not bool(args.include_raw_outputs), "raw outputs are excluded from this plan"))
    if "no_private_tokens_or_paths" in required:
        findings = scan_paths(included_paths)
        gates.append(
            gate_status(
                "no_private_tokens_or_paths",
                not findings,
                "; ".join(findings[:5]) or "no secret-like literals or private absolute paths found",
            )
        )

    promotion_blocked, promotion_message = promotion_blocks_release(variant_info, release_class)
    gates.append(
        gate_status(
            "variant_promotion_not_blocked",
            not promotion_blocked,
            promotion_message,
        )
    )

    public_checkpoint = (
        release_class.get("hf_visibility") == "public"
        and bool(release_class.get("publish_weights", False))
        and not bool(variant_info.get("adapter"))
    )
    if public_checkpoint:
        gates.append(
            gate_status(
                "public_checkpoint_release_allowed",
                bool(release_class.get("allow_public_checkpoint"))
                and validation_state_passes(validation_state, "spark_single_node_validated"),
                "public full-checkpoint releases require allow_public_checkpoint=true and Spark validation",
            )
        )
    return gates


def build_model_plan(args: argparse.Namespace) -> dict[str, Any]:
    hub_config = load_hub_config()
    variant_info = resolve_variant(args.family, args.variant)
    release_class = load_release_class(args.release_class)
    repo_id = args.repo_id or default_repo_id(args.family, args.variant, hub_config)
    run_id = args.run_id or sanitize_run_id(f"{args.family}_{args.variant}_{args.release_class}_hf_plan")
    output_dir = resolve_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / run_id
    include_weights = bool(release_class.get("publish_weights", False))
    include_adapter = bool(release_class.get("publish_adapter", False))
    include_model_artifact = include_weights or include_adapter
    if args.artifact_path:
        model_path = resolve_path(args.artifact_path)
    elif include_adapter and variant_info.get("adapter_path"):
        model_path = variant_info.get("adapter_path")
    elif include_weights and variant_info.get("merged_path"):
        model_path = variant_info.get("merged_path")
    else:
        model_path = variant_info.get("local_path")
    files_included, files_excluded = list_model_files(
        model_path if include_model_artifact else None,
        include_weights=include_weights,
        include_adapter=include_adapter,
    )

    extra_paths: list[Path] = []
    supporting_path_rewrites: list[dict[str, str]] = []
    for kind, raw_path in [
        ("eval_results", args.eval_results),
        ("serving_card", args.serving_card),
        ("quantization_card", args.quantization_card),
        ("promotion_report", args.promotion_report),
        ("risk_report", args.risk_report),
        ("manifest", args.manifest),
    ]:
        resolved_path, rewrite = resolve_supporting_path(raw_path, kind=kind)
        if resolved_path:
            extra_paths.append(resolved_path)
        if rewrite:
            supporting_path_rewrites.append(rewrite)
    included_scan_paths = ([model_path] if include_model_artifact and model_path and model_path.exists() else []) + extra_paths
    model_card = generate_model_card(
        family=args.family,
        variant=args.variant,
        variant_info=variant_info,
        release_class=release_class,
        repo_id=repo_id,
        validation_state=args.validation_state,
    )
    gates = build_release_gates(
        release_class=release_class,
        variant_info=variant_info,
        included_paths=included_scan_paths,
        model_card=model_card,
        validation_state=args.validation_state,
        args=args,
    )
    blocking_failures = [gate for gate in gates if gate.status == "fail"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "dry_run": True,
        "run_id": run_id,
        "repo_id": repo_id,
        "repo_type": "model",
        "visibility": release_class.get("hf_visibility", "private"),
        "release_class": release_class.get("id"),
        "family": args.family,
        "variant": args.variant,
        "source_model": variant_info.get("repo_id"),
        "local_artifact_path": publish_path_label(model_path),
        "local_artifact_exists": bool(model_path and model_path.exists()),
        "validation_state": args.validation_state,
        "files_included": files_included,
        "files_excluded": files_excluded,
        "supporting_paths": [publish_path_label(path) for path in extra_paths],
        "supporting_path_rewrites": supporting_path_rewrites,
        "release_gates": [gate.__dict__ for gate in gates],
        "blocked": bool(blocking_failures),
        "blocked_until": [f"{gate.name}: {gate.message}" for gate in blocking_failures],
        "model_card_path": "README.md",
        "hub_publish_path": "hub_publish.json",
        "github_repo": REPO_URL,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")
    (output_dir / "hub_publish.json").write_text(
        json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "hub_model_plan.json").write_text(
        json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def count_jsonl_rows(path: Path) -> int | None:
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return None
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def dataset_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file())


def read_small_text(path: Path) -> str:
    if not path.exists() or not path.is_file() or path.stat().st_size > 2_000_000:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def dataset_card_text(files: list[Path]) -> str:
    for name in ("README.md", "dataset_card.md"):
        for path in files:
            if path.name == name:
                return read_small_text(path)
    return ""


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def dataset_publish_gates_for_path(
    *,
    dataset_path: Path,
    files: list[Path],
    visibility: str,
    include_raw_outputs: bool,
) -> tuple[list[Gate], dict[str, Any]]:
    names = {path.name for path in files}
    suffixes = {path.suffix.lower() for path in files}
    card = dataset_card_text(files)
    manifest_text = "\n".join(read_small_text(path) for path in files if path.name in {"manifest.yaml", "manifest.json", "dataset_info.json"})
    redaction_report = next((load_optional_json(path) for path in files if path.name == "redaction_report.json"), None)
    redaction_report = redaction_report or None
    data_candidates = [
        path
        for path in files
        if path.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}
        and path.name not in {"redaction_report.json", "quality_report.json", "generation_report.json", "review_report.json", "hf_publish_plan.json"}
    ]
    row_counts = {
        publish_path_label(path): count
        for path in data_candidates
        for count in [count_jsonl_rows(path)]
        if count is not None
    }
    has_license = "license:" in card.lower() or "license" in manifest_text.lower()
    has_provenance = "provenance" in card.lower() or "source" in manifest_text.lower() or "source" in card.lower()
    card_sections = ("## Purpose", "## Counts", "## Provenance")
    private_findings = []
    token_findings = []
    for path in files:
        findings = scan_text_file(path)
        private_findings.extend(finding for finding in findings if "private absolute path" in finding)
        token_findings.extend(finding for finding in findings if "secret-like literal" in finding)
    public = visibility == "public"
    redacted_or_private = (
        not public
        or not include_raw_outputs
        or bool(redaction_report and not redaction_report.get("raw_message_content_published"))
        or any("redacted" in path.name for path in data_candidates)
    )
    gates = [
        gate_status("license_present", has_license, "dataset card or manifest records a license"),
        gate_status("source_provenance_present", has_provenance, "dataset card or manifest records provenance/source information"),
        gate_status("pii_scan_passed", not private_findings, "; ".join(private_findings[:5]) or "no private absolute paths found"),
        gate_status("unsafe_examples_redacted_or_private", redacted_or_private, "public plans require redaction or raw-output exclusion"),
        gate_status("dataset_card_complete", all(section in card for section in card_sections), "dataset card contains required sections"),
        gate_status("schema_present", bool(data_candidates) and bool({".jsonl", ".json", ".csv", ".parquet"} & suffixes), "dataset file with known schema-like extension is present"),
        gate_status("split_sizes_present", bool(row_counts) or ".parquet" in suffixes or ".csv" in suffixes, f"row_counts={row_counts or '{}'}"),
        gate_status("no_absolute_paths", not private_findings, "; ".join(private_findings[:5]) or "no private absolute paths found"),
        gate_status("no_tokens_or_credentials", not token_findings, "; ".join(token_findings[:5]) or "no token or credential literals found"),
    ]
    metadata = {
        "dataset_path": publish_path_label(dataset_path),
        "files": [publish_path_label(path) for path in files],
        "row_counts": row_counts,
        "redaction_report": redaction_report,
        "data_files": [publish_path_label(path) for path in data_candidates],
    }
    return gates, metadata


def build_dataset_plan(args: argparse.Namespace) -> dict[str, Any]:
    hub_config = load_hub_config()
    dataset_path = resolve_path(args.dataset_path)
    files = dataset_files(dataset_path) if dataset_path.exists() else []
    repo_id = args.repo_id or f"{hub_config.get('default_owner') or 'model-forge'}/{sanitize_run_id(dataset_path.stem or dataset_path.name)}"
    run_id = args.run_id or sanitize_run_id(f"{dataset_path.name}_dataset_hf_plan")
    output_dir = resolve_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / run_id
    gates, metadata = dataset_publish_gates_for_path(
        dataset_path=dataset_path,
        files=files,
        visibility=args.visibility,
        include_raw_outputs=args.include_raw_outputs,
    )
    gates.insert(0, gate_status("dataset_path_exists", dataset_path.exists(), f"path={publish_path_label(dataset_path)}"))
    blocking_failures = [gate for gate in gates if gate.status == "fail"]
    plan = {
        "schema_version": DATASET_PLAN_SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "dry_run": True,
        "run_id": run_id,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "visibility": args.visibility,
        "release_class": args.release_class,
        "dataset_path": publish_path_label(dataset_path),
        "split": args.split,
        "card_template": args.card_template,
        "files_included": metadata["files"],
        "data_files": metadata["data_files"],
        "row_counts": metadata["row_counts"],
        "redaction_report": metadata["redaction_report"],
        "release_gates": [gate.__dict__ for gate in gates],
        "blocked": bool(blocking_failures),
        "blocked_until": [f"{gate.name}: {gate.message}" for gate in blocking_failures],
        "github_repo": REPO_URL,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hub_dataset_plan.json").write_text(
        json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def print_status(data: Mapping[str, Any]) -> None:
    table = Table(title="Hugging Face Status")
    table.add_column("Field")
    table.add_column("Value")
    for key in (
        "authenticated",
        "user",
        "token_source",
        "default_owner",
        "default_visibility",
        "cache_dir",
        "dry_run_default",
        "offline",
        "error",
    ):
        table.add_row(key, str(data.get(key)))
    console.print(table)


def print_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title="HF Model Publish Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key in ("repo_id", "repo_type", "visibility", "release_class", "family", "variant", "validation_state", "blocked"):
        table.add_row(key, str(plan.get(key)))
    console.print(table)
    gate_table = Table(title="Release Gates")
    gate_table.add_column("Gate")
    gate_table.add_column("Status")
    gate_table.add_column("Message")
    for gate in plan.get("release_gates") or []:
        gate_table.add_row(str(gate["name"]), str(gate["status"]), str(gate["message"]))
    console.print(gate_table)


def print_dataset_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title="HF Dataset Publish Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key in ("repo_id", "repo_type", "visibility", "release_class", "dataset_path", "split", "blocked"):
        table.add_row(key, str(plan.get(key)))
    console.print(table)
    gate_table = Table(title="Dataset Release Gates")
    gate_table.add_column("Gate")
    gate_table.add_column("Status")
    gate_table.add_column("Message")
    for gate in plan.get("release_gates") or []:
        gate_table.add_row(str(gate["name"]), str(gate["status"]), str(gate["message"]))
    console.print(gate_table)


def print_release_class_audit(findings: list[ReleaseClassFinding]) -> None:
    if not findings:
        console.print("release class audit: OK")
        return
    table = Table(title="Release Class Audit")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Class")
    table.add_column("Message")
    for finding in findings:
        table.add_row(finding.severity.upper(), finding.check, finding.release_class or "", finding.message)
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Hugging Face Hub publication for Model Forge artifacts")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show safe HF auth/config status")
    status.add_argument("--offline", action="store_true")
    status.add_argument("--json", action="store_true")

    whoami = sub.add_parser("whoami", help="Show authenticated HF username")
    whoami.add_argument("--offline", action="store_true")
    whoami.add_argument("--json", action="store_true")

    login = sub.add_parser("login", help="Validate or optionally store an HF token")
    login.add_argument("--write-token", action="store_true", help="Store token with huggingface_hub.login")
    login.add_argument("--json", action="store_true")

    def add_model_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("family")
        command.add_argument("variant")
        command.add_argument("--repo-id")
        command.add_argument("--release-class", default="report_only")
        command.add_argument("--artifact-path")
        command.add_argument("--validation-state", default="planned", choices=sorted(VALIDATION_RANK))
        command.add_argument("--eval-results")
        command.add_argument("--serving-card")
        command.add_argument("--quantization-card")
        command.add_argument("--promotion-report")
        command.add_argument("--risk-report")
        command.add_argument("--manifest")
        command.add_argument("--source-license-checked", action="store_true")
        command.add_argument("--behavior-edited", action="store_true")
        command.add_argument("--include-raw-outputs", action="store_true")
        command.add_argument("--output-dir")
        command.add_argument("--run-id")
        command.add_argument("--json", action="store_true")

    plan_model = sub.add_parser("plan-model", help="Write a dry-run model publish plan and model card")
    add_model_args(plan_model)
    publish_model = sub.add_parser("publish-model", help="Dry-run model publish plan; real upload is intentionally blocked")
    add_model_args(publish_model)
    publish_model.add_argument("--dry-run", action="store_true", default=True)
    publish_model.add_argument("--execute", action="store_true", help="Reserved for future guarded upload support")
    publish_dataset = sub.add_parser("publish-dataset", help="Dry-run dataset publish plan")
    publish_dataset.add_argument("dataset_path")
    publish_dataset.add_argument("--repo-id")
    publish_dataset.add_argument("--release-class", default="public_dataset")
    publish_dataset.add_argument("--split")
    publish_dataset.add_argument("--card-template", default="dataset")
    publish_dataset.add_argument("--visibility", choices=["public", "private"], default="public")
    publish_dataset.add_argument("--include-raw-outputs", action="store_true")
    publish_dataset.add_argument("--dry-run", action="store_true", default=True)
    publish_dataset.add_argument("--execute", action="store_true", help="Reserved for future guarded dataset upload support")
    publish_dataset.add_argument("--output-dir")
    publish_dataset.add_argument("--run-id")
    publish_dataset.add_argument("--json", action="store_true")
    release_classes = sub.add_parser("release-classes", help="List or audit release-class configs")
    release_classes.add_argument("--audit", action="store_true")
    release_classes.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    if args.command in {"status", "whoami"}:
        data = hf_status(offline=args.offline)
        if args.command == "whoami" and not data.get("user") and not args.offline:
            return 1
        if args.json:
            print(json.dumps(redact_value(data), indent=2, sort_keys=True))
        else:
            print_status(data)
        return 0
    if args.command == "login":
        token = token_value()
        result = {"token_present": bool(token), "token_source": token_source()[0], "stored": False}
        if not token:
            result["error"] = "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN first"
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                console.print("[red]Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN first[/red]")
            return 1
        if args.write_token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            result["stored"] = True
        if args.json:
            print(json.dumps(redact_value(result), indent=2, sort_keys=True))
        else:
            console.print("HF token present; token value was not printed.")
        return 0
    if args.command in {"plan-model", "publish-model"}:
        if args.command == "publish-model" and args.execute:
            console.print(
                "[red]Non-dry-run model upload is not implemented; "
                "use scripts/publish_hf_artifact.py only after reviewing the plan.[/red]"
            )
            return 2
        plan = build_model_plan(args)
        if args.json:
            print(json.dumps(redact_value(plan), indent=2, sort_keys=True))
        else:
            print_plan(plan)
        return 1 if plan.get("blocked") and args.command == "publish-model" else 0
    if args.command == "publish-dataset":
        if args.execute:
            console.print("[red]Non-dry-run dataset upload is not implemented; inspect the dry-run plan first.[/red]")
            return 2
        plan = build_dataset_plan(args)
        if args.json:
            print(json.dumps(redact_value(plan), indent=2, sort_keys=True))
        else:
            print_dataset_plan(plan)
        return 1 if plan.get("blocked") else 0
    if args.command == "release-classes":
        findings = audit_release_classes()
        if args.json:
            print(json.dumps([finding.__dict__ for finding in findings], indent=2, sort_keys=True))
        elif args.audit:
            print_release_class_audit(findings)
        else:
            for path in sorted(RELEASE_CLASS_DIR.glob("*.yaml")):
                print(path.stem)
        return 1 if any(finding.severity == "error" for finding in findings) else 0
    parser.print_help()
    return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
