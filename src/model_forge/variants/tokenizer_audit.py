from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model_forge.runs.manifest import display_path
from model_forge.variants.manifest import load_family, transform_from_variant, variant_config


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
PRESERVED_FIELDS = (
    "tokenizer_json_sha256",
    "tokenizer_model_sha256",
    "special_tokens_map_sha256",
    "chat_template_sha256",
    "special_tokens",
)
ROUND_TRIP_MESSAGES = [
    {"role": "system", "content": "You are validating tokenizer behavior."},
    {"role": "user", "content": "Explain one deployment risk and give two checks."},
]


@dataclass(frozen=True)
class TokenizerFinding:
    level: str
    variant: str
    check: str
    message: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def models_dir(family_config: Mapping[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    env_name = str(family_config.get("models_dir_env") or "")
    if env_name and os.environ.get(env_name):
        return Path(str(os.environ[env_name])).expanduser()
    return Path(str(family_config.get("default_models_dir") or "~/models")).expanduser()


def variant_local_path(family_config: Mapping[str, Any], variant: Mapping[str, Any], models_dir_override: str | None = None) -> Path | None:
    raw = variant.get("merged_local_dir") or variant.get("local_dir")
    if not raw:
        return None
    raw_path = Path(str(raw)).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return models_dir(family_config, models_dir_override) / raw_path


def tokenizer_policy(variant: Mapping[str, Any]) -> dict[str, Any]:
    raw = variant.get("tokenizer_policy") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def should_preserve_tokenizer(variant_name: str, variant: Mapping[str, Any]) -> bool:
    policy = tokenizer_policy(variant)
    if policy.get("preserve") is not None:
        return bool(policy["preserve"])
    transform = transform_from_variant(variant_name, variant)
    return str(transform.get("type")) in {"fine_tune", "behavior_edit", "quantize", "derive"}


def tokenizer_record(path: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": display_path(path) if path else None,
        "exists": bool(path and path.exists()),
        "files": {},
    }
    if not path or not path.exists() or not path.is_dir():
        return record

    for filename in TOKENIZER_FILES:
        file_path = path / filename
        if file_path.exists() and file_path.is_file():
            record["files"][filename] = {
                "path": display_path(file_path),
                "sha256": file_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }

    tokenizer_config = read_json(path / "tokenizer_config.json")
    special_tokens_map = read_json(path / "special_tokens_map.json")
    if tokenizer_config:
        record["tokenizer_config_sha256"] = file_sha256(path / "tokenizer_config.json")
        special_tokens = {
            key: tokenizer_config.get(key)
            for key in ("bos_token", "eos_token", "unk_token", "pad_token")
            if tokenizer_config.get(key) is not None
        }
        if special_tokens:
            record["special_tokens"] = special_tokens
            record["special_tokens_sha256"] = value_sha256(special_tokens)
        if tokenizer_config.get("chat_template"):
            record["chat_template_source"] = "tokenizer_config.json"
            record["chat_template_sha256"] = value_sha256(tokenizer_config["chat_template"])
    if special_tokens_map:
        record["special_tokens_map_sha256"] = file_sha256(path / "special_tokens_map.json")
    if (path / "chat_template.jinja").exists():
        record["chat_template_source"] = "chat_template.jinja"
        record["chat_template_sha256"] = file_sha256(path / "chat_template.jinja")
    if (path / "tokenizer.json").exists():
        record["tokenizer_json_sha256"] = file_sha256(path / "tokenizer.json")
    if (path / "tokenizer.model").exists():
        record["tokenizer_model_sha256"] = file_sha256(path / "tokenizer.model")
    return record


def live_round_trip(path: Path, trust_remote_code: bool = False) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        return {"status": "skipped", "reason": f"transformers unavailable: {exc}"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
        rendered = tokenizer.apply_chat_template(ROUND_TRIP_MESSAGES, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, add_special_tokens=False)
        decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 - report audit failure without hiding context
        return {"status": "failed", "reason": str(exc)}
    probes = ("deployment risk", "two checks")
    return {
        "status": "passed" if all(probe in decoded for probe in probes) else "failed",
        "rendered_chars": len(str(rendered)),
        "token_count": len(encoded["input_ids"]),
        "decoded_contains_probes": {probe: probe in decoded for probe in probes},
    }


def compare_records(variant_name: str, child: Mapping[str, Any], parent_name: str, parent: Mapping[str, Any]) -> list[TokenizerFinding]:
    findings: list[TokenizerFinding] = []
    if not parent.get("exists") or not child.get("exists"):
        return findings
    for field in PRESERVED_FIELDS:
        parent_value = parent.get(field)
        child_value = child.get(field)
        if parent_value is None and child_value is None:
            continue
        if parent_value != child_value:
            findings.append(
                TokenizerFinding(
                    "error",
                    variant_name,
                    "tokenizer_preservation",
                    f"{field} differs from source variant {parent_name}",
                )
            )
    return findings


def build_tokenizer_audit(
    family: str,
    *,
    variant: str | None = None,
    models_dir_override: str | None = None,
    load_tokenizer: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    selected = [variant] if variant else sorted(str(name) for name in variants)
    records: dict[str, dict[str, Any]] = {}
    findings: list[TokenizerFinding] = []

    for variant_name in selected:
        variant_data = variant_config(family_config, variant_name)
        path = variant_local_path(family_config, variant_data, models_dir_override)
        record = tokenizer_record(path)
        record["variant"] = variant_name
        record["source_variant"] = variant_data.get("base_variant")
        record["preserve_tokenizer"] = should_preserve_tokenizer(variant_name, variant_data)
        record["documented_change"] = bool(tokenizer_policy(variant_data).get("documented_change"))
        if load_tokenizer and path and path.exists():
            record["round_trip"] = live_round_trip(path, trust_remote_code=bool(variant_data.get("trust_remote_code", False)))
            if record["round_trip"].get("status") == "failed":
                findings.append(
                    TokenizerFinding("error", variant_name, "round_trip", str(record["round_trip"].get("reason") or "round-trip failed"))
                )
            elif record["round_trip"].get("status") == "skipped":
                level = "error" if strict else "warning"
                findings.append(
                    TokenizerFinding(level, variant_name, "round_trip", str(record["round_trip"].get("reason") or "round-trip skipped"))
                )
        records[variant_name] = record

        if not record["exists"]:
            level = "error" if strict else "warning"
            findings.append(TokenizerFinding(level, variant_name, "local_dir", "configured local_dir is not present"))
            continue
        if not any(name in record["files"] for name in ("tokenizer.json", "tokenizer.model")):
            findings.append(TokenizerFinding("error", variant_name, "tokenizer_files", "missing tokenizer.json or tokenizer.model"))
        if "tokenizer_config.json" not in record["files"]:
            findings.append(TokenizerFinding("error", variant_name, "tokenizer_config", "missing tokenizer_config.json"))

    for variant_name in selected:
        variant_data = variant_config(family_config, variant_name)
        parent_name = variant_data.get("base_variant")
        if not parent_name or not should_preserve_tokenizer(variant_name, variant_data):
            continue
        if tokenizer_policy(variant_data).get("documented_change"):
            continue
        if str(parent_name) not in records:
            parent_data = variant_config(family_config, str(parent_name))
            records[str(parent_name)] = tokenizer_record(variant_local_path(family_config, parent_data, models_dir_override))
            if not records[str(parent_name)]["exists"]:
                level = "error" if strict else "warning"
                findings.append(
                    TokenizerFinding(
                        level,
                        variant_name,
                        "source_local_dir",
                        f"source variant {parent_name} local_dir is not present",
                    )
                )
        findings.extend(compare_records(variant_name, records[variant_name], str(parent_name), records[str(parent_name)]))

    errors = [finding for finding in findings if finding.level == "error"]
    return {
        "schema_version": "model_forge.tokenizer_audit.v1",
        "family": family,
        "models_dir": display_path(models_dir(family_config, models_dir_override)),
        "metadata_only": not load_tokenizer,
        "strict": strict,
        "variant_count": len(selected),
        "records": [records[name] for name in sorted(records)],
        "findings": [asdict(finding) for finding in findings],
        "passed": not errors,
    }
