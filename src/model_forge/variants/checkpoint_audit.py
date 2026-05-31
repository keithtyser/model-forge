from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_forge.runs.manifest import display_path
from model_forge.variants.manifest import load_family, variant_config
from model_forge.variants.tokenizer_audit import models_dir, variant_local_path


REQUIRED_METADATA_FILES = ("config.json", "generation_config.json")
TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer.model", "merges.txt", "vocab.json", "tokenizer_config.json", "chat_template.jinja")
ADAPTER_METADATA_FILES = ("adapter_config.json",)


@dataclass(frozen=True)
class CheckpointFinding:
    level: str
    variant: str
    check: str
    message: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def safetensor_record(path: Path) -> dict[str, Any]:
    shards = sorted(path.glob("*.safetensors"))
    index_path = path / "model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
    metadata = index.get("metadata") if isinstance(index.get("metadata"), dict) else {}
    expected_total_bytes = int(float(metadata.get("total_size") or 0))
    referenced = sorted({str(name) for name in weight_map.values()})
    present = sorted(shard.name for shard in shards)
    return {
        "index_path": display_path(index_path),
        "index_exists": index_path.exists(),
        "expected_total_bytes": expected_total_bytes,
        "shard_count": len(shards),
        "total_shard_bytes": sum(shard.stat().st_size for shard in shards),
        "present_shards": present,
        "referenced_shards": referenced,
        "missing_referenced_shards": sorted(set(referenced) - set(present)),
        "extra_shards": sorted(set(present) - set(referenced)) if referenced else [],
    }


def local_path_for_raw(family_config: dict[str, Any], raw: Any, models_dir_override: str | None) -> Path | None:
    if not raw:
        return None
    raw_path = Path(str(raw)).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return models_dir(family_config, models_dir_override) / raw_path


def audit_full_checkpoint(
    *,
    variant_name: str,
    path: Path,
    record: dict[str, Any],
    findings: list[CheckpointFinding],
    strict: bool,
    check_prefix: str = "",
) -> None:
    files = {item.name: item.stat().st_size for item in path.iterdir() if item.is_file()}
    record["files"] = files
    record["metadata_files_present"] = sorted(name for name in REQUIRED_METADATA_FILES if name in files)
    record["tokenizer_files_present"] = sorted(name for name in TOKENIZER_MARKERS if name in files)
    record["safetensors"] = safetensor_record(path)
    download_cache = path / ".cache" / "huggingface" / "download"
    if download_cache.exists():
        incomplete_files = sorted(download_cache.glob("*.incomplete"))
        record["incomplete_download_files"] = sorted(item.name for item in incomplete_files)
        record["incomplete_download_bytes"] = sum(item.stat().st_size for item in incomplete_files)
        if incomplete_files:
            latest = max(item.stat().st_mtime for item in incomplete_files)
            largest = max(incomplete_files, key=lambda item: item.stat().st_size)
            record["incomplete_download_latest_mtime"] = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()
            record["largest_incomplete_download"] = {
                "name": largest.name,
                "bytes": largest.stat().st_size,
            }
        record["lock_files"] = sorted(item.name for item in download_cache.glob("*.lock"))
    expected_total = int(record["safetensors"].get("expected_total_bytes") or 0)
    observed_total = int(record["safetensors"].get("total_shard_bytes") or 0) + int(record.get("incomplete_download_bytes") or 0)
    if expected_total > 0:
        record["download_progress_fraction"] = min(1.0, observed_total / expected_total)

    prefix = f"{check_prefix}_" if check_prefix else ""
    for required in REQUIRED_METADATA_FILES:
        if required not in files:
            findings.append(CheckpointFinding("error", variant_name, f"{prefix}metadata", f"missing {required}"))
    if not record["tokenizer_files_present"]:
        findings.append(CheckpointFinding("error", variant_name, f"{prefix}tokenizer", "no tokenizer/chat-template marker files found"))

    safetensors = record["safetensors"]
    if not safetensors["present_shards"]:
        findings.append(CheckpointFinding("error", variant_name, f"{prefix}weights", "no .safetensors weight shards found"))
    elif safetensors["index_exists"]:
        if safetensors["missing_referenced_shards"]:
            missing = ", ".join(safetensors["missing_referenced_shards"][:5])
            findings.append(CheckpointFinding("error", variant_name, f"{prefix}weights", f"index references missing shards: {missing}"))
    elif len(safetensors["present_shards"]) > 1:
        findings.append(CheckpointFinding("error", variant_name, f"{prefix}weights", "multiple safetensor shards found without model.safetensors.index.json"))

    if record.get("incomplete_download_files"):
        level = "error" if strict else "warning"
        findings.append(CheckpointFinding(level, variant_name, f"{prefix}download", "Hugging Face incomplete download files are present"))


def build_checkpoint_audit(
    family: str,
    *,
    variant: str | None = None,
    models_dir_override: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    selected = [variant] if variant else sorted(str(name) for name in variants)
    records: list[dict[str, Any]] = []
    findings: list[CheckpointFinding] = []

    for variant_name in selected:
        variant_data = variant_config(family_config, variant_name)
        adapter = bool(variant_data.get("adapter"))
        path = (
            local_path_for_raw(family_config, variant_data.get("local_dir"), models_dir_override)
            if adapter
            else variant_local_path(family_config, variant_data, models_dir_override)
        )
        merged_path = local_path_for_raw(family_config, variant_data.get("merged_local_dir"), models_dir_override)
        record: dict[str, Any] = {
            "variant": variant_name,
            "path": display_path(path) if path else None,
            "merged_path": display_path(merged_path) if merged_path else None,
            "exists": bool(path and path.exists() and path.is_dir()),
            "source_variant": variant_data.get("base_variant"),
            "adapter": adapter,
            "quantization": variant_data.get("quantization"),
        }
        if not path or not path.exists() or not path.is_dir():
            level = "error" if strict else "warning"
            findings.append(CheckpointFinding(level, variant_name, "local_dir", "configured local_dir is not present"))
            records.append(record)
            continue

        if adapter:
            files = {item.name: item.stat().st_size for item in path.iterdir() if item.is_file()}
            record["files"] = files
            record["adapter_metadata_files_present"] = sorted(name for name in ADAPTER_METADATA_FILES if name in files)
            record["tokenizer_files_present"] = sorted(name for name in TOKENIZER_MARKERS if name in files)
            if "adapter_model.safetensors" not in files:
                findings.append(CheckpointFinding("error", variant_name, "adapter", "adapter variant is missing adapter_model.safetensors"))
            for required in ADAPTER_METADATA_FILES:
                if required not in files:
                    findings.append(CheckpointFinding("error", variant_name, "adapter", f"missing {required}"))
            if not record["tokenizer_files_present"]:
                findings.append(CheckpointFinding("error", variant_name, "tokenizer", "no tokenizer/chat-template marker files found"))
            if merged_path:
                merged_record: dict[str, Any] = {
                    "path": display_path(merged_path),
                    "exists": merged_path.exists() and merged_path.is_dir(),
                }
                record["merged_checkpoint"] = merged_record
                if not merged_path.exists() or not merged_path.is_dir():
                    findings.append(CheckpointFinding("error", variant_name, "merged_local_dir", "configured merged_local_dir is not present"))
                else:
                    audit_full_checkpoint(
                        variant_name=variant_name,
                        path=merged_path,
                        record=merged_record,
                        findings=findings,
                        strict=strict,
                        check_prefix="merged",
                    )
        else:
            audit_full_checkpoint(
                variant_name=variant_name,
                path=path,
                record=record,
                findings=findings,
                strict=strict,
            )
        records.append(record)

    errors = [finding for finding in findings if finding.level == "error"]
    return {
        "schema_version": "model_forge.checkpoint_audit.v1",
        "family": family,
        "variant_count": len(selected),
        "strict": strict,
        "records": records,
        "findings": [asdict(finding) for finding in findings],
        "passed": not errors,
    }
