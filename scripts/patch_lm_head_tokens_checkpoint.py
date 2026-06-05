#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


PRESERVED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def load_structured_file(path: Path) -> dict[str, Any]:
    data = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(data)
    else:
        loaded = yaml.safe_load(data)
    if not isinstance(loaded, dict):
        raise SystemExit(f"expected mapping in config: {path}")
    return loaded


def load_index(checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid safetensors index weight_map: {index_path}")
    return data


def find_lm_head_tensor(weight_map: dict[str, str], configured: str | None) -> str:
    if configured:
        if configured not in weight_map:
            raise SystemExit(f"configured lm_head tensor is not in checkpoint index: {configured}")
        return configured
    candidates = [name for name in weight_map if name.endswith("lm_head.weight")]
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one lm_head.weight tensor, found {len(candidates)}")
    return candidates[0]


def copy_preserved_files(source: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in PRESERVED_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output dir already exists; pass --overwrite: {output_dir}")
        if not output_dir.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def check_resources(output_dir: Path, required_bytes: int, min_ram_fraction: float, min_disk_fraction: float) -> dict[str, Any]:
    mem = psutil.virtual_memory()
    ram_fraction = mem.available / mem.total
    if ram_fraction < min_ram_fraction:
        raise SystemExit(f"available RAM fraction {ram_fraction:.3f} is below guard {min_ram_fraction:.3f}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir.parent)
    projected_free = usage.free - required_bytes
    projected_fraction = projected_free / usage.total
    if projected_fraction < min_disk_fraction:
        raise SystemExit(
            "free disk fraction would breach guard after lm-head token patch: "
            f"{projected_fraction:.3f} < {min_disk_fraction:.3f} "
            f"(need {required_bytes / (1024 ** 3):.1f} GiB)"
        )
    return {
        "available_ram_fraction": round(ram_fraction, 4),
        "free_disk_fraction_before": round(usage.free / usage.total, 4),
        "projected_free_disk_fraction_after": round(projected_fraction, 4),
        "required_bytes": required_bytes,
    }


def tokenizer_needed(patches: list[dict[str, Any]]) -> bool:
    for patch in patches:
        if patch.get("token_id") is None and patch.get("token") is not None:
            return True
        if patch.get("replacement_token_id") is None and patch.get("replacement") is not None:
            return True
    return False


def load_tokenizer(tokenizer_dir: Path, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=trust_remote_code)


def single_token_id(tokenizer: Any, text: str, *, field: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise SystemExit(f"{field} must resolve to exactly one token: {text!r} -> {token_ids}")
    return int(token_ids[0])


def decode_token(tokenizer: Any | None, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    return str(tokenizer.decode([token_id]))


def normalize_patch_specs(
    raw_patches: list[Any],
    *,
    tokenizer_dir: Path,
    trust_remote_code: bool,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_patches, start=1):
        if not isinstance(raw, dict):
            raise SystemExit(f"patch {index} must be a mapping")
        patches.append(dict(raw))
    if not patches:
        raise SystemExit("at least one token patch is required")

    tokenizer = load_tokenizer(tokenizer_dir, trust_remote_code) if tokenizer_needed(patches) else None
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, patch in enumerate(patches, start=1):
        token_id = patch.get("token_id")
        if token_id is None:
            if tokenizer is None or patch.get("token") is None:
                raise SystemExit(f"patch {index} needs token_id or token")
            token_id = single_token_id(tokenizer, str(patch["token"]), field=f"patch {index} token")
        token_id = int(token_id)
        if token_id in seen:
            raise SystemExit(f"duplicate target token_id in patch config: {token_id}")
        seen.add(token_id)

        replacement_token_id = patch.get("replacement_token_id")
        if replacement_token_id is None and patch.get("replacement") is not None:
            if tokenizer is None:
                raise SystemExit(f"patch {index} replacement text requires tokenizer")
            replacement_token_id = single_token_id(tokenizer, str(patch["replacement"]), field=f"patch {index} replacement")
        if replacement_token_id is not None:
            replacement_token_id = int(replacement_token_id)

        alpha = float(patch.get("alpha", 1.0 if replacement_token_id is not None else 0.0))
        scale = float(patch.get("scale", 1.0))
        if replacement_token_id is None and scale == 1.0:
            raise SystemExit(f"patch {index} must set replacement/replacement_token_id or scale")
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit(f"patch {index} alpha must be in [0, 1]")
        if scale < 0.0:
            raise SystemExit(f"patch {index} scale must be non-negative")
        normalized.append({
            "name": str(patch.get("name") or f"patch_{index}"),
            "token_id": token_id,
            "token": patch.get("token"),
            "decoded_token": decode_token(tokenizer, token_id),
            "replacement_token_id": replacement_token_id,
            "replacement": patch.get("replacement"),
            "decoded_replacement": decode_token(tokenizer, replacement_token_id) if replacement_token_id is not None else None,
            "alpha": alpha,
            "scale": scale,
            "preserve_norm": bool(patch.get("preserve_norm", replacement_token_id is not None and scale == 1.0)),
        })
    return normalized


def patch_row(lm_head: torch.Tensor, patch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    token_id = int(patch["token_id"])
    if token_id < 0 or token_id >= lm_head.shape[0]:
        raise SystemExit(f"target token_id out of range for lm_head rows: {token_id}")
    old = lm_head[token_id].float()
    new = old.clone()
    replacement_token_id = patch.get("replacement_token_id")
    if replacement_token_id is not None:
        replacement_token_id = int(replacement_token_id)
        if replacement_token_id < 0 or replacement_token_id >= lm_head.shape[0]:
            raise SystemExit(f"replacement token_id out of range for lm_head rows: {replacement_token_id}")
        replacement = lm_head[replacement_token_id].float()
        new = old.add(replacement.sub(old), alpha=float(patch["alpha"]))
    if float(patch["scale"]) != 1.0:
        new = new.mul(float(patch["scale"]))
    old_norm = float(torch.linalg.vector_norm(old).item())
    new_norm_before = float(torch.linalg.vector_norm(new).item())
    if patch.get("preserve_norm") and old_norm > 0.0 and new_norm_before > 0.0:
        new = new.mul(old_norm / new_norm_before)
    final_norm = float(torch.linalg.vector_norm(new).item())
    return new.to(dtype=lm_head.dtype), {
        **patch,
        "old_norm": old_norm,
        "new_norm_before_preserve": new_norm_before,
        "new_norm": final_norm,
    }


def link_or_copy(src: Path, dst: Path, *, copy_unchanged: bool) -> str:
    if copy_unchanged:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def patch_checkpoint(
    *,
    source_dir: Path,
    output_dir: Path,
    patches: list[dict[str, Any]],
    lm_head_tensor: str | None,
    overwrite: bool,
    copy_unchanged: bool,
    min_ram_fraction: float,
    min_disk_fraction: float,
    dry_run: bool,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == source_dir:
        raise SystemExit("output dir must differ from source checkpoint dir")
    index = load_index(source_dir)
    weight_map = {str(k): str(v) for k, v in index["weight_map"].items()}
    lm_head_name = find_lm_head_tensor(weight_map, lm_head_tensor)
    lm_head_shard = weight_map[lm_head_name]
    shard_path = source_dir / lm_head_shard
    if not shard_path.exists():
        raise SystemExit(f"lm_head shard is missing: {shard_path}")
    required_bytes = shard_path.stat().st_size
    if copy_unchanged:
        required_bytes = sum((source_dir / name).stat().st_size for name in set(weight_map.values()))
    resources = check_resources(output_dir, required_bytes, min_ram_fraction, min_disk_fraction)
    manifest: dict[str, Any] = {
        "schema_version": "model_forge.lm_head_token_patch.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "lm_head_tensor": lm_head_name,
        "lm_head_shard": lm_head_shard,
        "copy_unchanged": copy_unchanged,
        "resource_preflight": resources,
        "patches": patches,
        "dry_run": dry_run,
    }
    if dry_run:
        manifest["shard_count"] = len(set(weight_map.values()))
        return manifest

    prepare_output(output_dir, overwrite)
    copied_files = copy_preserved_files(source_dir, output_dir)
    shutil.copy2(source_dir / "model.safetensors.index.json", output_dir / "model.safetensors.index.json")

    shard_actions: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        src = source_dir / shard_name
        dst = output_dir / shard_name
        if shard_name != lm_head_shard:
            shard_actions[shard_name] = link_or_copy(src, dst, copy_unchanged=copy_unchanged)

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    lm_head = tensors[lm_head_name]
    if not torch.is_floating_point(lm_head):
        raise SystemExit(f"lm_head tensor must be floating point: {lm_head.dtype}")
    applied: list[dict[str, Any]] = []
    edited = lm_head.clone()
    for patch in patches:
        edited[int(patch["token_id"])], record = patch_row(edited, patch)
        applied.append(record)
    tensors[lm_head_name] = edited
    tmp_path = output_dir / f".{lm_head_shard}.tmp"
    save_file(tensors, tmp_path, metadata=metadata)
    tmp_path.replace(output_dir / lm_head_shard)
    shard_actions[lm_head_shard] = "rewrite"

    manifest["copied_files"] = copied_files
    manifest["patches"] = applied
    manifest["shard_actions"] = shard_actions
    (output_dir / "model_forge_lm_head_token_patch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    check_resources(output_dir, 0, min_ram_fraction, min_disk_fraction)
    return manifest


def config_values(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model") or {}
    patch = config.get("patch") or config.get("lm_head_token_patch") or {}
    if not isinstance(model, dict) or not isinstance(patch, dict):
        raise SystemExit("config needs model and patch mappings")
    return {
        "source_dir": model.get("local_dir") or model.get("source_dir"),
        "output_dir": model.get("output_dir"),
        "trust_remote_code": bool(model.get("trust_remote_code", True)),
        "tokenizer_dir": patch.get("tokenizer_dir") or model.get("local_dir") or model.get("source_dir"),
        "lm_head_tensor": patch.get("lm_head_tensor"),
        "patches": patch.get("patches") or [],
        "copy_unchanged": bool(patch.get("copy_unchanged", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch selected lm_head token rows in a sharded safetensors checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config with model and patch sections")
    parser.add_argument("--source", type=Path, default=None, help="Source checkpoint directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output checkpoint directory")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Tokenizer directory for text token patches")
    parser.add_argument("--lm-head-tensor", default=None, help="Tensor name for lm_head.weight if auto-detection is ambiguous")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate output directory if present")
    parser.add_argument("--copy-unchanged", action="store_true", help="Copy unchanged shards instead of hardlinking them")
    parser.add_argument("--dry-run", action="store_true", help="Validate resources and token specs without writing checkpoint")
    parser.add_argument("--min-available-ram-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")))
    parser.add_argument("--min-free-disk-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded: dict[str, Any] = {}
    if args.config:
        loaded = config_values(load_structured_file(args.config.expanduser().resolve()))
    source_dir = args.source or (Path(str(loaded["source_dir"])) if loaded.get("source_dir") else None)
    output_dir = args.output_dir or (Path(str(loaded["output_dir"])) if loaded.get("output_dir") else None)
    tokenizer_dir = args.tokenizer_dir or (Path(str(loaded["tokenizer_dir"])) if loaded.get("tokenizer_dir") else None)
    if source_dir is None:
        raise SystemExit("--source or config model.local_dir is required")
    if output_dir is None:
        raise SystemExit("--output-dir or config model.output_dir is required")
    if tokenizer_dir is None:
        tokenizer_dir = source_dir
    trust_remote_code = bool(loaded.get("trust_remote_code", True))
    raw_patches = loaded.get("patches") or []
    patches = normalize_patch_specs(
        raw_patches,
        tokenizer_dir=tokenizer_dir.expanduser().resolve(),
        trust_remote_code=trust_remote_code,
    )
    manifest = patch_checkpoint(
        source_dir=source_dir,
        output_dir=output_dir,
        patches=patches,
        lm_head_tensor=args.lm_head_tensor or loaded.get("lm_head_tensor"),
        overwrite=args.overwrite,
        copy_unchanged=bool(args.copy_unchanged or loaded.get("copy_unchanged", False)),
        min_ram_fraction=args.min_available_ram_fraction,
        min_disk_fraction=args.min_free_disk_fraction,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
