"""Model family and variant registry.

One deep home for "given a family name (and optionally a variant), tell me where
its config lives, where its weights resolve on disk, and what the variant
declares". Before this module, every CLI re-derived the same logic: load the
family YAML, look up the variant, resolve its relative ``local_dir`` against the
family ``models_dir``, and apply the same precedence rules -- each returning a
slightly different shape. The resolution invariant now lives here; callers ask
instead of re-deriving.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from model_forge.runs.manifest import REPO_DIR, display_path

__all__ = [
    "REPO_DIR",
    "display_path",
    "MODEL_FAMILIES_DIR",
    "load_yaml",
    "resolve_repo_path",
    "family_config_path",
    "load_family",
    "models_dir",
    "Variant",
    "resolve_variant",
]

MODEL_FAMILIES_DIR = REPO_DIR / "configs" / "model_families"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and require it to be a mapping."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(Path(path))}")
    return data


def resolve_repo_path(value: str | Path, base: Path | None = None) -> Path:
    """Resolve ``value`` against the repo root (or ``base``) unless absolute."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base or REPO_DIR) / path


def family_config_path(name: str) -> Path:
    return MODEL_FAMILIES_DIR / f"{name}.yaml"


def load_family(name: str) -> dict[str, Any]:
    """Load ``configs/model_families/<name>.yaml`` or raise a clear error."""
    path = family_config_path(name)
    if not path.exists():
        raise ValueError(f"unknown family {name!r}; expected {display_path(path)}")
    return load_yaml(path)


def models_dir(family_config: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Path:
    """Root directory that a family's relative local paths resolve against."""
    env = env if env is not None else os.environ
    env_name = str(family_config.get("models_dir_env") or "MODEL_FORGE_MODELS_DIR")
    raw = env.get(env_name) or family_config.get("default_models_dir") or "~/models"
    return Path(str(raw)).expanduser()


@dataclass(frozen=True)
class Variant:
    """A resolved family variant: declared metadata plus on-disk locations.

    ``local_path`` is the merged checkpoint when present, otherwise the adapter
    directory. ``served_model_name`` falls back to ``repo_id``; callers that need
    other fallbacks (e.g. local-dir-as-model-id) read ``raw``.
    """

    family: str
    variant: str
    family_display_name: str
    repo_id: str | None
    served_model_name: str | None
    base_variant: str | None
    adapter: bool
    quantization: Any
    downloadable: bool
    promotion: dict[str, Any]
    hub_slug: str | None
    local_path: Path | None
    adapter_path: Path | None
    merged_path: Path | None
    models_root: Path
    raw: dict[str, Any]


def resolve_variant(family: str, variant: str, env: Mapping[str, str] | None = None) -> Variant:
    """Resolve ``family``/``variant`` to a :class:`Variant`.

    Raises ``ValueError`` if the family or variant is unknown.
    """
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    if variant not in variants:
        raise ValueError(
            f"unknown variant {variant!r} for {family!r}; valid: {', '.join(sorted(variants))}"
        )
    raw = dict(variants[variant])
    root = models_dir(family_config, env)

    def variant_path(key: str) -> Path | None:
        local_dir = str(raw.get(key) or "")
        if not local_dir:
            return None
        candidate = Path(local_dir).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    adapter_path = variant_path("local_dir")
    merged_path = variant_path("merged_local_dir")
    return Variant(
        family=family,
        variant=variant,
        family_display_name=family_config.get("display_name") or family,
        repo_id=raw.get("repo_id"),
        served_model_name=raw.get("served_model_name") or raw.get("repo_id"),
        base_variant=raw.get("base_variant"),
        adapter=bool(raw.get("adapter", False)),
        quantization=raw.get("quantization"),
        downloadable=raw.get("downloadable", True),
        promotion=dict(raw.get("promotion") or {}),
        hub_slug=raw.get("hub_slug") or raw.get("publish_slug"),
        local_path=merged_path or adapter_path,
        adapter_path=adapter_path,
        merged_path=merged_path,
        models_root=root,
        raw=raw,
    )
