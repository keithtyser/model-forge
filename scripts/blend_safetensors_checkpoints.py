#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
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


def load_index(checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid safetensors index weight_map: {index_path}")
    return data


def grouped_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        grouped.setdefault(str(shard_name), []).append(str(tensor_name))
    return {name: sorted(tensors) for name, tensors in sorted(grouped.items())}


def checkpoint_total_size(index: dict[str, Any], checkpoint: Path) -> int:
    metadata = index.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("total_size") is not None:
        return int(metadata["total_size"])
    return sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))


def check_resources(output_dir: Path, required_bytes: int, min_ram_fraction: float, min_disk_fraction: float) -> dict[str, Any]:
    mem = psutil.virtual_memory()
    ram_fraction = mem.available / mem.total
    if ram_fraction < min_ram_fraction:
        raise SystemExit(f"available RAM fraction {ram_fraction:.3f} is below guard {min_ram_fraction:.3f}")
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_parent)
    projected_free = usage.free - required_bytes
    projected_fraction = projected_free / usage.total
    if projected_fraction < min_disk_fraction:
        raise SystemExit(
            "free disk fraction would breach guard after blend export: "
            f"{projected_fraction:.3f} < {min_disk_fraction:.3f} "
            f"(need {required_bytes / (1024 ** 3):.1f} GiB)"
        )
    return {
        "available_ram_fraction": round(ram_fraction, 4),
        "free_disk_fraction_before": round(usage.free / usage.total, 4),
        "projected_free_disk_fraction_after": round(projected_fraction, 4),
        "required_bytes": required_bytes,
    }


def compile_regex(raw: str | None) -> re.Pattern[str] | None:
    if not raw:
        return None
    return re.compile(raw)


def should_blend(name: str, include: re.Pattern[str] | None, exclude: re.Pattern[str] | None) -> bool:
    if include is not None and not include.search(name):
        return False
    if exclude is not None and exclude.search(name):
        return False
    return True


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output dir already exists; pass --overwrite: {output_dir}")
        if not output_dir.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_preserved_files(source: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in PRESERVED_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


def blend_tensor(base: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    if base.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(base.shape)} != {tuple(target.shape)}")
    if base.dtype != target.dtype:
        raise ValueError(f"dtype mismatch {base.dtype} != {target.dtype}")
    if not torch.is_floating_point(base):
        return base
    blended = base.float().add_(target.float().sub(base.float()), alpha=alpha)
    return blended.to(dtype=base.dtype)


def write_blended_shard(
    base_dir: Path,
    target_dir: Path,
    output_dir: Path,
    shard_name: str,
    tensor_names: list[str],
    alpha: float,
    include: re.Pattern[str] | None,
    exclude: re.Pattern[str] | None,
) -> dict[str, int]:
    stats = {"blended": 0, "copied": 0}
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(base_dir / shard_name, framework="pt", device="cpu") as base_handle, safe_open(
        target_dir / shard_name,
        framework="pt",
        device="cpu",
    ) as target_handle:
        metadata = base_handle.metadata()
        for name in tensor_names:
            base_tensor = base_handle.get_tensor(name)
            target_tensor = target_handle.get_tensor(name)
            if should_blend(name, include, exclude) and torch.is_floating_point(base_tensor):
                try:
                    tensors[name] = blend_tensor(base_tensor, target_tensor, alpha)
                except ValueError as exc:
                    raise SystemExit(f"cannot blend {name}: {exc}") from exc
                stats["blended"] += 1
            else:
                tensors[name] = base_tensor
                stats["copied"] += 1
    tmp_path = output_dir / f".{shard_name}.tmp"
    save_file(tensors, tmp_path, metadata=metadata)
    tmp_path.replace(output_dir / shard_name)
    return stats


def blend_checkpoints(
    base_dir: Path,
    target_dir: Path,
    output_dir: Path,
    alpha: float,
    include_regex: str | None,
    exclude_regex: str | None,
    overwrite: bool,
    min_ram_fraction: float,
    min_disk_fraction: float,
    dry_run: bool,
) -> dict[str, Any]:
    base_dir = base_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir in {base_dir, target_dir}:
        raise SystemExit("output dir must differ from base and target checkpoint dirs")
    base_index = load_index(base_dir)
    target_index = load_index(target_dir)
    base_map = {str(k): str(v) for k, v in base_index["weight_map"].items()}
    target_map = {str(k): str(v) for k, v in target_index["weight_map"].items()}
    if base_map != target_map:
        raise SystemExit("base and target checkpoints must have identical tensor-to-shard maps")
    grouped = grouped_by_shard(base_map)
    required_bytes = checkpoint_total_size(base_index, base_dir)
    resources = check_resources(output_dir, required_bytes, min_ram_fraction, min_disk_fraction)
    include = compile_regex(include_regex)
    exclude = compile_regex(exclude_regex)
    manifest: dict[str, Any] = {
        "schema_version": "model_forge.checkpoint_blend.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir),
        "target_dir": str(target_dir),
        "output_dir": str(output_dir),
        "alpha": alpha,
        "formula": "output = base + alpha * (target - base)",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "resource_preflight": resources,
        "shards": [],
        "dry_run": dry_run,
    }
    if dry_run:
        manifest["shard_count"] = len(grouped)
        manifest["tensor_count"] = len(base_map)
        return manifest
    prepare_output(output_dir, overwrite)
    copied_files = copy_preserved_files(base_dir, output_dir)
    shutil.copy2(base_dir / "model.safetensors.index.json", output_dir / "model.safetensors.index.json")
    totals = {"blended": 0, "copied": 0}
    for shard_index, (shard_name, tensor_names) in enumerate(grouped.items(), start=1):
        print(f"[model-forge] blending shard {shard_index}/{len(grouped)}: {shard_name}", flush=True)
        stats = write_blended_shard(base_dir, target_dir, output_dir, shard_name, tensor_names, alpha, include, exclude)
        totals["blended"] += stats["blended"]
        totals["copied"] += stats["copied"]
        manifest["shards"].append({"name": shard_name, "tensor_count": len(tensor_names), **stats})
        check_resources(output_dir, 0, min_ram_fraction, min_disk_fraction)
    manifest["copied_files"] = copied_files
    manifest["tensor_stats"] = totals
    (output_dir / "model_forge_checkpoint_blend.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-blend two matching safetensors checkpoints")
    parser.add_argument("--base", type=Path, required=True, help="Base/source checkpoint directory")
    parser.add_argument("--target", type=Path, required=True, help="Target checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output checkpoint directory")
    parser.add_argument("--alpha", type=float, required=True, help="Blend factor: output = base + alpha * (target - base)")
    parser.add_argument("--include-regex", default=None, help="Only blend tensor names matching this regex")
    parser.add_argument("--exclude-regex", default=None, help="Copy tensor names matching this regex from base")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate output directory if present")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/resources without writing checkpoint")
    parser.add_argument("--min-available-ram-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")))
    parser.add_argument("--min-free-disk-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = blend_checkpoints(
        base_dir=args.base,
        target_dir=args.target,
        output_dir=args.output_dir,
        alpha=args.alpha,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        overwrite=args.overwrite,
        min_ram_fraction=args.min_available_ram_fraction,
        min_disk_fraction=args.min_free_disk_fraction,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
