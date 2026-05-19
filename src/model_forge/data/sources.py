from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REPO_DIR = Path(__file__).resolve().parents[3]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_source_registry(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"version": "0.0.0", "sources": {}}
    registry_path = resolve_repo_path(path)
    registry = load_yaml(registry_path)
    sources = registry.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError(f"source registry must contain a sources mapping: {registry_path}")
    normalized: dict[str, Any] = {}
    for source_id, raw in sources.items():
        if not isinstance(raw, dict):
            raise ValueError(f"source registry entry must be a mapping: {source_id}")
        entry = copy.deepcopy(raw)
        entry.setdefault("id", str(source_id))
        entry.setdefault("quality_tier", "candidate")
        entry.setdefault("roles", [])
        entry.setdefault("license", "unknown")
        normalized[str(source_id)] = entry
    return {
        "version": str(registry.get("version", "0.0.0")),
        "path": display_path(registry_path),
        "sources": normalized,
    }


def merge_source_override(registry_entry: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(registry_entry)
    for key, value in override.items():
        if key == "id":
            continue
        merged[key] = copy.deepcopy(value)
    merged.setdefault("name", merged.get("id", ""))
    if "role" not in merged and merged.get("roles"):
        merged["role"] = list(merged["roles"])[0]
    return merged


def resolve_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    registry = load_source_registry(config.get("source_registry"))
    registry_sources = registry["sources"]
    resolved: list[dict[str, Any]] = []
    for item in config.get("sources", []):
        if not isinstance(item, dict):
            raise ValueError("source entries must be mappings")
        source_id = item.get("id")
        if source_id:
            if str(source_id) not in registry_sources:
                raise ValueError(f"source id {source_id!r} not found in registry {registry.get('path', '<none>')}")
            resolved.append(merge_source_override(registry_sources[str(source_id)], item))
        else:
            entry = copy.deepcopy(item)
            entry.setdefault("id", entry.get("name", "inline_source"))
            entry.setdefault("quality_tier", "candidate")
            resolved.append(entry)
    for source_id in config.get("source_ids", []):
        if str(source_id) not in registry_sources:
            raise ValueError(f"source id {source_id!r} not found in registry {registry.get('path', '<none>')}")
        resolved.append(merge_source_override(registry_sources[str(source_id)], {}))
    return resolved


def registry_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    if not config.get("source_registry"):
        return None
    registry = load_source_registry(config.get("source_registry"))
    selected = resolve_sources(config)
    return {
        "path": registry.get("path"),
        "version": registry.get("version"),
        "selected_source_ids": [str(source.get("id", "")) for source in selected],
        "selected_sources": selected,
    }
