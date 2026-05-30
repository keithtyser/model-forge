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
        "local_path": local_path,
        "adapter_path": adapter_path,
        "merged_path": merged_path,
        "raw": raw,
    }


def load_hub_config(path: Path = DEFAULT_HUB_CONFIG) -> dict[str, Any]:
    return load_yaml(path) if path.exists() else {}


def load_release_class(name: str) -> dict[str, Any]:
    path = RELEASE_CLASS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown release class {name!r}; expected {display_path(path)}")
    data = load_yaml(path)
    data.setdefault("id", name)
    return data


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

    extra_paths = [
        resolve_path(path)
        for path in [
            args.eval_results,
            args.serving_card,
            args.quantization_card,
            args.promotion_report,
            args.risk_report,
            args.manifest,
        ]
        if path
    ]
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
    parser.print_help()
    return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
