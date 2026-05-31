from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forge.runs.manifest import display_path
from model_forge.variants.manifest import load_family, variant_config
from model_forge.variants.tokenizer_audit import variant_local_path


REQUIRED_METADATA_FILES = ("config.json", "generation_config.json")
TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer.model", "merges.txt", "vocab.json", "tokenizer_config.json", "chat_template.jinja")


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
    referenced = sorted({str(name) for name in weight_map.values()})
    present = sorted(shard.name for shard in shards)
    return {
        "index_path": display_path(index_path),
        "index_exists": index_path.exists(),
        "shard_count": len(shards),
        "total_shard_bytes": sum(shard.stat().st_size for shard in shards),
        "present_shards": present,
        "referenced_shards": referenced,
        "missing_referenced_shards": sorted(set(referenced) - set(present)),
        "extra_shards": sorted(set(present) - set(referenced)) if referenced else [],
    }


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
        path = variant_local_path(family_config, variant_data, models_dir_override)
        record: dict[str, Any] = {
            "variant": variant_name,
            "path": display_path(path) if path else None,
            "exists": bool(path and path.exists() and path.is_dir()),
            "source_variant": variant_data.get("base_variant"),
            "adapter": bool(variant_data.get("adapter")),
            "quantization": variant_data.get("quantization"),
        }
        if not path or not path.exists() or not path.is_dir():
            level = "error" if strict else "warning"
            findings.append(CheckpointFinding(level, variant_name, "local_dir", "configured local_dir is not present"))
            records.append(record)
            continue

        files = {item.name: item.stat().st_size for item in path.iterdir() if item.is_file()}
        record["files"] = files
        record["metadata_files_present"] = sorted(name for name in REQUIRED_METADATA_FILES if name in files)
        record["tokenizer_files_present"] = sorted(name for name in TOKENIZER_MARKERS if name in files)
        record["safetensors"] = safetensor_record(path)
        download_cache = path / ".cache" / "huggingface" / "download"
        if download_cache.exists():
            record["incomplete_download_files"] = sorted(item.name for item in download_cache.glob("*.incomplete"))
            record["lock_files"] = sorted(item.name for item in download_cache.glob("*.lock"))

        for required in REQUIRED_METADATA_FILES:
            if required not in files:
                findings.append(CheckpointFinding("error", variant_name, "metadata", f"missing {required}"))
        if not record["tokenizer_files_present"]:
            findings.append(CheckpointFinding("error", variant_name, "tokenizer", "no tokenizer/chat-template marker files found"))

        safetensors = record["safetensors"]
        if variant_data.get("adapter"):
            if "adapter_model.safetensors" not in files:
                findings.append(CheckpointFinding("error", variant_name, "adapter", "adapter variant is missing adapter_model.safetensors"))
        elif not safetensors["present_shards"]:
            findings.append(CheckpointFinding("error", variant_name, "weights", "no .safetensors weight shards found"))
        elif safetensors["index_exists"]:
            if safetensors["missing_referenced_shards"]:
                missing = ", ".join(safetensors["missing_referenced_shards"][:5])
                findings.append(CheckpointFinding("error", variant_name, "weights", f"index references missing shards: {missing}"))
        elif len(safetensors["present_shards"]) > 1:
            findings.append(CheckpointFinding("error", variant_name, "weights", "multiple safetensor shards found without model.safetensors.index.json"))

        if record.get("incomplete_download_files"):
            level = "error" if strict else "warning"
            findings.append(CheckpointFinding(level, variant_name, "download", "Hugging Face incomplete download files are present"))
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
