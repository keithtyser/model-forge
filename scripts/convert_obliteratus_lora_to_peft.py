#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


ADAPTER_FILE = "abliteration_lora_adapters.pt"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "generation_config.json",
)


KEY_RE = re.compile(r"^layer\.(?P<layer>\d+)\.(?P<section>attn|ffn)\.(?P<weight>[^.]+)\.lora_(?P<side>[AB])$")


def load_torch(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"expected dict payload in {path}")
    return payload


def parse_obliteratus_adapters(payload: dict[str, torch.Tensor]) -> dict[tuple[int, str, str], dict[str, torch.Tensor]]:
    grouped: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in payload.items():
        match = KEY_RE.match(str(key))
        if match is None:
            raise SystemExit(f"unsupported OBLITERATUS LoRA key: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"adapter value is not a tensor for key: {key}")
        item = (
            int(match.group("layer")),
            match.group("section"),
            match.group("weight"),
        )
        grouped.setdefault(item, {})[match.group("side")] = tensor.detach().cpu()
    for item, sides in grouped.items():
        if set(sides) != {"A", "B"}:
            raise SystemExit(f"incomplete LoRA pair for {item}: found {sorted(sides)}")
        a = sides["A"]
        b = sides["B"]
        if a.ndim != 2 or b.ndim != 2:
            raise SystemExit(f"LoRA tensors must be rank-2 for {item}: A={tuple(a.shape)} B={tuple(b.shape)}")
        if a.shape[0] != b.shape[1]:
            raise SystemExit(f"LoRA rank mismatch for {item}: A={tuple(a.shape)} B={tuple(b.shape)}")
    return grouped


def peft_module_name(section: str, attn_module_name: str, ffn_module_name: str) -> str:
    if section == "attn":
        return attn_module_name
    if section == "ffn":
        return ffn_module_name
    raise SystemExit(f"unknown section {section!r}")


def convert_adapters(
    input_dir: Path,
    output_dir: Path,
    *,
    base_model_name_or_path: str,
    key_template: str,
    attn_module_name: str,
    ffn_module_name: str,
    lora_alpha: int | None = None,
    copy_metadata: bool = True,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    adapter_path = input_dir / ADAPTER_FILE
    if not adapter_path.is_file():
        raise SystemExit(f"missing OBLITERATUS adapter payload: {adapter_path}")

    grouped = parse_obliteratus_adapters(load_torch(adapter_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    ranks: set[int] = set()
    records: list[dict[str, Any]] = []
    for (layer, section, weight), sides in sorted(grouped.items()):
        module_name = peft_module_name(section, attn_module_name, ffn_module_name)
        module_path = key_template.format(layer=layer, module=module_name, weight=weight)
        a = sides["A"].contiguous()
        b = sides["B"].contiguous()
        rank = int(a.shape[0])
        ranks.add(rank)
        target_modules.add(weight)
        tensors[f"{module_path}.lora_A.weight"] = a
        tensors[f"{module_path}.lora_B.weight"] = b
        records.append({
            "layer": layer,
            "section": section,
            "module": module_name,
            "weight": weight,
            "rank": rank,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        })

    if not tensors:
        raise SystemExit("no LoRA tensors found to convert")
    if len(ranks) != 1:
        raise SystemExit(f"mixed LoRA ranks are not supported for PEFT export yet: {sorted(ranks)}")
    rank = next(iter(ranks))
    alpha = rank if lora_alpha is None else int(lora_alpha)

    save_file(tensors, output_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    copied: list[str] = []
    if copy_metadata:
        for name in TOKENIZER_FILES:
            source = input_dir / name
            destination = output_dir / name
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
                copied.append(name)

    manifest = {
        "schema_version": "model_forge.obliteratus_lora_peft_adapter.v1",
        "source_adapter": str(adapter_path),
        "output_dir": str(output_dir),
        "base_model_name_or_path": base_model_name_or_path,
        "key_template": key_template,
        "attn_module_name": attn_module_name,
        "ffn_module_name": ffn_module_name,
        "adapter_count": len(records),
        "rank": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(target_modules),
        "copied_metadata_files": copied,
        "records": records,
        "scaling_note": "OBLITERATUS stores the full delta in B@A; lora_alpha defaults to rank so PEFT scaling is 1.0.",
    }
    (output_dir / "model_forge_obliteratus_lora_peft_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = (
        "# OBLITERATUS LoRA PEFT Adapter\n\n"
        "Converted by Model Forge from OBLITERATUS reversible ablation LoRA tensors.\n\n"
        f"- Base model: `{base_model_name_or_path}`\n"
        f"- Adapter count: `{len(records)}`\n"
        f"- Rank: `{rank}`\n"
        f"- LoRA alpha: `{alpha}`\n"
        f"- Target modules: `{', '.join(sorted(target_modules))}`\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OBLITERATUS reversible LoRA tensors to a PEFT adapter")
    parser.add_argument("--input-dir", required=True, help="Directory containing abliteration_lora_adapters.pt")
    parser.add_argument("--output-dir", required=True, help="PEFT adapter output directory")
    parser.add_argument("--base-model", required=True, help="Base model path or repo id for adapter_config.json")
    parser.add_argument(
        "--key-template",
        default="base_model.model.model.layers.{layer}.{module}.{weight}",
        help="PEFT module key template before .lora_[AB].weight",
    )
    parser.add_argument("--attn-module-name", default="self_attn")
    parser.add_argument("--ffn-module-name", default="mlp")
    parser.add_argument("--lora-alpha", type=int, default=None, help="Defaults to adapter rank for exact B@A scaling")
    parser.add_argument("--no-copy-metadata", action="store_true", help="Do not copy tokenizer/generation metadata from input dir")
    args = parser.parse_args()

    manifest = convert_adapters(
        Path(args.input_dir),
        Path(args.output_dir),
        base_model_name_or_path=args.base_model,
        key_template=args.key_template,
        attn_module_name=args.attn_module_name,
        ffn_module_name=args.ffn_module_name,
        lora_alpha=args.lora_alpha,
        copy_metadata=not args.no_copy_metadata,
    )
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "output_dir": manifest["output_dir"],
        "adapter_count": manifest["adapter_count"],
        "rank": manifest["rank"],
    }, indent=2))


if __name__ == "__main__":
    main()
