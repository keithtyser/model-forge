#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def scale_adapter(source: Path, output: Path, scale: float, overwrite: bool = False) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source adapter does not exist: {source}")
    adapter_model = source / "adapter_model.safetensors"
    if not adapter_model.is_file():
        raise SystemExit(f"source adapter is missing adapter_model.safetensors: {adapter_model}")
    if scale <= 0:
        raise SystemExit("--scale must be greater than zero")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.name == "adapter_model.safetensors":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        elif item.is_file():
            shutil.copy2(item, destination)

    state = load_file(adapter_model)
    scaled_state: dict[str, torch.Tensor] = {}
    scaled_tensors = 0
    copied_tensors = 0
    for name, tensor in state.items():
        if name.endswith("lora_B.weight"):
            scaled_state[name] = tensor * scale
            scaled_tensors += 1
        else:
            scaled_state[name] = tensor
            copied_tensors += 1
    save_file(scaled_state, output / "adapter_model.safetensors")

    manifest = {
        "schema_version": "model_forge.scaled_lora_adapter.v1",
        "source_adapter": str(source),
        "output_adapter": str(output),
        "scale": scale,
        "scale_method": "multiply_lora_B_weight",
        "scaled_tensors": scaled_tensors,
        "copied_tensors": copied_tensors,
        "created_at_unix": time.time(),
    }
    (output / "model_forge_scaled_adapter.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a scaled copy of a PEFT LoRA adapter without writing a full model checkpoint")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = scale_adapter(args.source, args.output, args.scale, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
