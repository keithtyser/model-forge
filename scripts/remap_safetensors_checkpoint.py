#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import psutil
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_PRESERVE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "processor_config.json",
)


def parse_prefix_map(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("prefix maps must be FROM=TO")
    source, target = raw.split("=", 1)
    if not source:
        raise argparse.ArgumentTypeError("source prefix must not be empty")
    return source, target


def directory_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def available_ram_fraction() -> float:
    mem = psutil.virtual_memory()
    return mem.available / mem.total


def guard_resources(output_dir: Path, min_ram_fraction: float, min_disk_fraction: float, transient_bytes: int) -> None:
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram_fraction:
        raise SystemExit(f"available RAM fraction {ram_fraction:.3f} is below guard {min_ram_fraction:.3f}")
    usage = shutil.disk_usage(output_dir)
    projected_free = usage.free - transient_bytes
    projected_fraction = projected_free / usage.total
    if projected_fraction < min_disk_fraction:
        raise SystemExit(
            "free disk fraction would breach guard during shard rewrite: "
            f"{projected_fraction:.3f} < {min_disk_fraction:.3f}"
        )


def remap_name(name: str, prefix_maps: list[tuple[str, str]]) -> str:
    for source, target in prefix_maps:
        if name.startswith(source):
            return target + name[len(source):]
    return name


def load_index(path: Path) -> dict:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing index: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def remapped_weight_map(weight_map: dict[str, str], prefix_maps: list[tuple[str, str]]) -> dict[str, str]:
    remapped: dict[str, str] = {}
    for name, shard in weight_map.items():
        target = remap_name(str(name), prefix_maps)
        if target in remapped:
            raise SystemExit(f"prefix maps create duplicate tensor key: {target}")
        remapped[target] = str(shard)
    return remapped


def verify_against_reference(reference_dir: Path, candidate_weight_map: dict[str, str]) -> None:
    reference_map = load_index(reference_dir).get("weight_map") or {}
    reference_keys = set(reference_map)
    candidate_keys = set(candidate_weight_map)
    if reference_keys != candidate_keys:
        only_reference = sorted(reference_keys - candidate_keys)[:20]
        only_candidate = sorted(candidate_keys - reference_keys)[:20]
        raise SystemExit(
            "remapped tensor keys do not match reference checkpoint keys\n"
            f"only in reference ({len(reference_keys - candidate_keys)}): {only_reference}\n"
            f"only in candidate ({len(candidate_keys - reference_keys)}): {only_candidate}"
        )


def copy_preserved_files(reference_dir: Path, output_dir: Path, names: tuple[str, ...]) -> list[str]:
    copied = []
    for name in names:
        source = reference_dir / name
        if not source.exists():
            continue
        shutil.copy2(source, output_dir / name)
        copied.append(name)
    return copied


def rewrite_shard(
    checkpoint_dir: Path,
    shard: str,
    names: list[str],
    target_names: dict[str, str],
    min_ram_fraction: float,
    min_disk_fraction: float,
) -> None:
    shard_path = checkpoint_dir / shard
    if not shard_path.exists():
        raise SystemExit(f"missing shard: {shard_path}")
    guard_resources(checkpoint_dir, min_ram_fraction, min_disk_fraction, shard_path.stat().st_size)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    tensors = {}
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for name in names:
            tensors[target_names[name]] = handle.get_tensor(name)
    save_file(tensors, tmp_path, metadata=metadata)
    del tensors
    gc.collect()

    with safe_open(tmp_path, framework="pt", device="cpu") as handle:
        observed = set(handle.keys())
    expected = {target_names[name] for name in names}
    if observed != expected:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"rewritten shard key mismatch for {shard}")
    tmp_path.replace(shard_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap safetensors checkpoint tensor key prefixes.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--map-prefix", required=True, action="append", type=parse_prefix_map)
    parser.add_argument("--preserve-file", action="append", default=[])
    parser.add_argument("--skip-preserve-defaults", action="store_true")
    parser.add_argument("--min-available-ram-fraction", type=float, default=float(os.getenv("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")))
    parser.add_argument("--min-free-disk-fraction", type=float, default=float(os.getenv("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10")))
    parser.add_argument("--verify-reference-keys", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint dir does not exist: {checkpoint_dir}")
    if not reference_dir.is_dir():
        raise SystemExit(f"reference dir does not exist: {reference_dir}")

    index = load_index(checkpoint_dir)
    weight_map = index.get("weight_map") or {}
    target_weight_map = remapped_weight_map(weight_map, args.map_prefix)
    if args.verify_reference_keys:
        verify_against_reference(reference_dir, target_weight_map)

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(str(shard), []).append(str(name))

    total_bytes = directory_size_bytes(checkpoint_dir)
    print(f"[model-forge] remapping {len(weight_map)} tensors across {len(by_shard)} shards in {checkpoint_dir}")
    print(f"[model-forge] checkpoint size before rewrite: {total_bytes} bytes")
    for shard in sorted(by_shard):
        print(f"[model-forge] rewriting {shard} ({len(by_shard[shard])} tensors)", flush=True)
        rewrite_shard(
            checkpoint_dir,
            shard,
            by_shard[shard],
            {name: remap_name(name, args.map_prefix) for name in by_shard[shard]},
            args.min_available_ram_fraction,
            args.min_free_disk_fraction,
        )

    index["weight_map"] = target_weight_map
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preserved = [] if args.skip_preserve_defaults else list(DEFAULT_PRESERVE_FILES)
    preserved.extend(args.preserve_file)
    copied = copy_preserved_files(reference_dir, checkpoint_dir, tuple(dict.fromkeys(preserved)))
    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "reference_dir": str(reference_dir),
        "prefix_maps": [{"from": source, "to": target} for source, target in args.map_prefix],
        "tensor_count": len(weight_map),
        "shard_count": len(by_shard),
        "preserved_files": copied,
        "verified_reference_keys": bool(args.verify_reference_keys),
    }
    (checkpoint_dir / "model_forge_key_remap.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[model-forge] remap complete")


if __name__ == "__main__":
    main()
