#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
    "special_tokens_map.json",
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


def checkpoint_total_size(index: dict[str, Any], checkpoint: Path) -> int:
    metadata = index.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("total_size") is not None:
        return int(metadata["total_size"])
    return sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))


def grouped_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        grouped.setdefault(str(shard_name), []).append(str(tensor_name))
    return {name: sorted(tensors) for name, tensors in sorted(grouped.items())}


def compile_regex(raw: str | None) -> re.Pattern[str] | None:
    if not raw:
        return None
    return re.compile(raw)


def selected(name: str, include: re.Pattern[str] | None, exclude: re.Pattern[str] | None) -> bool:
    if include is not None and not include.search(name):
        return False
    if exclude is not None and exclude.search(name):
        return False
    return True


def available_ram_fraction() -> float:
    mem = psutil.virtual_memory()
    return mem.available / mem.total


def check_resources(path: Path, required_bytes: int, min_ram_fraction: float, min_disk_fraction: float) -> dict[str, Any]:
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram_fraction:
        raise SystemExit(f"available RAM fraction {ram_fraction:.3f} is below guard {min_ram_fraction:.3f}")
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    projected_free = usage.free - required_bytes
    projected_fraction = projected_free / usage.total
    if projected_fraction < min_disk_fraction:
        raise SystemExit(
            "free disk fraction would breach guard during source tether: "
            f"{projected_fraction:.3f} < {min_disk_fraction:.3f} "
            f"(need {required_bytes / (1024 ** 3):.1f} GiB transient)"
        )
    return {
        "available_ram_fraction": round(ram_fraction, 4),
        "free_disk_fraction_before": round(usage.free / usage.total, 4),
        "projected_free_disk_fraction_after": round(projected_fraction, 4),
        "required_bytes": required_bytes,
    }


def prepare_output(candidate_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir == candidate_dir:
        return
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output dir already exists; pass --overwrite: {output_dir}")
        if not output_dir.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_preserved_files(source_dir: Path, candidate_dir: Path, output_dir: Path, preserve_from: str) -> list[str]:
    copied: list[str] = []
    primary = source_dir if preserve_from == "source" else candidate_dir
    fallback = candidate_dir if preserve_from == "source" else source_dir
    for name in PRESERVED_FILES:
        src = primary / name
        if not src.exists():
            src = fallback / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


def tensor_stats(source: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if source.shape != candidate.shape:
        raise ValueError(f"shape mismatch {tuple(source.shape)} != {tuple(candidate.shape)}")
    if source.dtype != candidate.dtype:
        raise ValueError(f"dtype mismatch {source.dtype} != {candidate.dtype}")
    if not torch.is_floating_point(source):
        return {"floating": False}
    delta = candidate.float().sub(source.float())
    source_norm = float(source.float().norm().item())
    delta_norm = float(delta.norm().item())
    return {
        "floating": True,
        "mean_abs_delta": float(delta.abs().mean().item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "delta_l2": delta_norm,
        "source_l2": source_norm,
        "relative_l2": delta_norm / source_norm if source_norm > 0 else None,
    }


def collect_drift_stats(
    source_dir: Path,
    candidate_dir: Path,
    grouped: dict[str, list[str]],
    include: re.Pattern[str] | None,
    exclude: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for shard_name, tensor_names in grouped.items():
        with safe_open(source_dir / shard_name, framework="pt", device="cpu") as source_handle, safe_open(
            candidate_dir / shard_name,
            framework="pt",
            device="cpu",
        ) as candidate_handle:
            for name in tensor_names:
                source_tensor = source_handle.get_tensor(name)
                candidate_tensor = candidate_handle.get_tensor(name)
                entry = {
                    "name": name,
                    "shard": shard_name,
                    "selected": selected(name, include, exclude),
                    "dtype": str(candidate_tensor.dtype),
                    "shape": list(candidate_tensor.shape),
                }
                try:
                    entry.update(tensor_stats(source_tensor, candidate_tensor))
                except ValueError as exc:
                    raise SystemExit(f"cannot compare {name}: {exc}") from exc
                stats.append(entry)
    return stats


def choose_resets(stats: list[dict[str, Any]], restore_top_k: int, drift_metric: str) -> set[str]:
    if restore_top_k <= 0:
        return set()
    candidates = [
        item for item in stats
        if item.get("selected") and item.get("floating") and item.get(drift_metric) is not None
    ]
    candidates.sort(key=lambda item: float(item[drift_metric]), reverse=True)
    return {str(item["name"]) for item in candidates[:restore_top_k]}


def tether_tensor(source: torch.Tensor, candidate: torch.Tensor, alpha: float, reset: bool) -> torch.Tensor:
    if source.shape != candidate.shape:
        raise ValueError(f"shape mismatch {tuple(source.shape)} != {tuple(candidate.shape)}")
    if source.dtype != candidate.dtype:
        raise ValueError(f"dtype mismatch {source.dtype} != {candidate.dtype}")
    if reset or not torch.is_floating_point(source):
        return source.clone()
    tethered = source.float().add_(candidate.float().sub(source.float()), alpha=alpha)
    return tethered.to(dtype=source.dtype)


def write_tethered_shard(
    source_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    shard_name: str,
    tensor_names: list[str],
    alpha: float,
    reset_names: set[str],
    include: re.Pattern[str] | None,
    exclude: re.Pattern[str] | None,
) -> dict[str, int]:
    stats = {"tethered": 0, "reset_to_source": 0, "copied_candidate": 0}
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source_dir / shard_name, framework="pt", device="cpu") as source_handle, safe_open(
        candidate_dir / shard_name,
        framework="pt",
        device="cpu",
    ) as candidate_handle:
        metadata = candidate_handle.metadata()
        for name in tensor_names:
            source_tensor = source_handle.get_tensor(name)
            candidate_tensor = candidate_handle.get_tensor(name)
            if selected(name, include, exclude) and torch.is_floating_point(candidate_tensor):
                try:
                    tensors[name] = tether_tensor(source_tensor, candidate_tensor, alpha, name in reset_names)
                except ValueError as exc:
                    raise SystemExit(f"cannot source-tether {name}: {exc}") from exc
                if name in reset_names:
                    stats["reset_to_source"] += 1
                else:
                    stats["tethered"] += 1
            else:
                tensors[name] = candidate_tensor
                stats["copied_candidate"] += 1
    tmp_path = output_dir / f".{shard_name}.tmp"
    save_file(tensors, tmp_path, metadata=metadata)
    tmp_path.replace(output_dir / shard_name)
    del tensors
    gc.collect()
    return stats


def source_tether_checkpoint(
    source_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    alpha: float,
    restore_top_k: int,
    drift_metric: str,
    include_regex: str | None,
    exclude_regex: str | None,
    preserve_from: str,
    overwrite: bool,
    in_place: bool,
    min_ram_fraction: float,
    min_disk_fraction: float,
    dry_run: bool,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    output_dir = candidate_dir if in_place else output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"source dir does not exist: {source_dir}")
    if not candidate_dir.is_dir():
        raise SystemExit(f"candidate dir does not exist: {candidate_dir}")
    if output_dir in {source_dir}:
        raise SystemExit("output dir must differ from source checkpoint dir")
    if not 0 <= alpha <= 1:
        raise SystemExit("--alpha must be between 0 and 1")
    if drift_metric not in {"mean_abs_delta", "max_abs_delta", "delta_l2", "relative_l2"}:
        raise SystemExit(f"unsupported drift metric: {drift_metric}")

    source_index = load_index(source_dir)
    candidate_index = load_index(candidate_dir)
    source_map = {str(k): str(v) for k, v in source_index["weight_map"].items()}
    candidate_map = {str(k): str(v) for k, v in candidate_index["weight_map"].items()}
    if source_map != candidate_map:
        raise SystemExit("source and candidate checkpoints must have identical tensor-to-shard maps")

    grouped = grouped_by_shard(candidate_map)
    include = compile_regex(include_regex)
    exclude = compile_regex(exclude_regex)
    required_bytes = max(
        path.stat().st_size for path in candidate_dir.glob("*.safetensors")
    ) if in_place else checkpoint_total_size(candidate_index, candidate_dir)
    resources = check_resources(output_dir.parent if not in_place else output_dir, required_bytes, min_ram_fraction, min_disk_fraction)
    drift_stats = collect_drift_stats(source_dir, candidate_dir, grouped, include, exclude)
    reset_names = choose_resets(drift_stats, restore_top_k, drift_metric)

    selected_count = sum(1 for item in drift_stats if item.get("selected") and item.get("floating"))
    manifest: dict[str, Any] = {
        "schema_version": "model_forge.source_tether.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "alpha": alpha,
        "formula": "output = source + alpha * (candidate - source)",
        "restore_top_k": restore_top_k,
        "drift_metric": drift_metric,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "preserve_from": preserve_from,
        "in_place": in_place,
        "resource_preflight": resources,
        "tensor_count": len(candidate_map),
        "selected_floating_tensor_count": selected_count,
        "reset_tensor_count": len(reset_names),
        "reset_tensors": [
            item for item in sorted(drift_stats, key=lambda entry: float(entry.get(drift_metric) or 0), reverse=True)
            if item["name"] in reset_names
        ],
        "dry_run": dry_run,
        "shards": [],
    }
    if dry_run:
        return manifest

    prepare_output(candidate_dir, output_dir, overwrite)
    if output_dir != candidate_dir:
        shutil.copy2(candidate_dir / "model.safetensors.index.json", output_dir / "model.safetensors.index.json")
    copied_files = copy_preserved_files(source_dir, candidate_dir, output_dir, preserve_from)
    totals = {"tethered": 0, "reset_to_source": 0, "copied_candidate": 0}
    for shard_index, (shard_name, tensor_names) in enumerate(grouped.items(), start=1):
        print(f"[model-forge] source-tethering shard {shard_index}/{len(grouped)}: {shard_name}", flush=True)
        check_resources(output_dir, (candidate_dir / shard_name).stat().st_size, min_ram_fraction, min_disk_fraction)
        shard_stats = write_tethered_shard(
            source_dir=source_dir,
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            shard_name=shard_name,
            tensor_names=tensor_names,
            alpha=alpha,
            reset_names=reset_names,
            include=include,
            exclude=exclude,
        )
        for key, value in shard_stats.items():
            totals[key] += value
        manifest["shards"].append({"name": shard_name, "tensor_count": len(tensor_names), **shard_stats})
    manifest["copied_files"] = copied_files
    manifest["tensor_stats"] = totals
    (output_dir / "model_forge_source_tether.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-tether a safetensors checkpoint toward a preservation source")
    parser.add_argument("--source", type=Path, required=True, help="Preservation/source checkpoint directory")
    parser.add_argument("--candidate", type=Path, required=True, help="Behavior-edited candidate checkpoint directory")
    parser.add_argument("--output-dir", type=Path, default=Path("source-tethered"), help="Output checkpoint directory unless --in-place is used")
    parser.add_argument("--alpha", type=float, default=0.895, help="Tether factor: output = source + alpha * (candidate - source)")
    parser.add_argument("--restore-top-k", type=int, default=0, help="Restore this many highest-drift selected tensors exactly to source")
    parser.add_argument("--drift-metric", default="mean_abs_delta", choices=["mean_abs_delta", "max_abs_delta", "delta_l2", "relative_l2"])
    parser.add_argument("--include-regex", default=None, help="Only tether tensor names matching this regex")
    parser.add_argument("--exclude-regex", default=None, help="Copy candidate tensor names matching this regex unchanged")
    parser.add_argument("--preserve-from", default="source", choices=["source", "candidate"], help="Where to copy config/tokenizer metadata from")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate output directory if present")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the candidate checkpoint in place using per-shard temp files")
    parser.add_argument("--dry-run", action="store_true", help="Validate and rank drift without writing checkpoint")
    parser.add_argument("--min-available-ram-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")))
    parser.add_argument("--min-free-disk-fraction", type=float, default=float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = source_tether_checkpoint(
        source_dir=args.source,
        candidate_dir=args.candidate,
        output_dir=args.output_dir,
        alpha=args.alpha,
        restore_top_k=args.restore_top_k,
        drift_metric=args.drift_metric,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        preserve_from=args.preserve_from,
        overwrite=args.overwrite,
        in_place=args.in_place,
        min_ram_fraction=args.min_available_ram_fraction,
        min_disk_fraction=args.min_free_disk_fraction,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
