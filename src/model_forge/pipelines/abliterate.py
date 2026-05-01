from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_forge.hardware import detect_hardware_profile, recommended_training_env

console = Console()


REPO_DIR = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return data


def resolve_repo_path(raw: str | Path, base: Path | None = None) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base or REPO_DIR) / path


def resolve_model_source(raw: str | Path | None) -> str:
    if raw is None:
        raise SystemExit("model source is required")
    value = str(raw)
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith("~"):
        return str(path)
    return value


def load_prompts(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text()) or {}
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", [])
    else:
        prompts = []
    if not all(isinstance(item, str) and item.strip() for item in prompts):
        raise SystemExit(f"prompt file must contain non-empty string prompts: {path}")
    return [item.strip() for item in prompts]


def cuda_free_gb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    if not values:
        return None
    return min(values) / 1024


def build_plan(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    model = config.get("model", {})
    data = config.get("data", {})
    activation = config.get("activation_collection", {})
    safety = config.get("safety", {})
    hardware = detect_hardware_profile()
    training_env = recommended_training_env()

    harmful_path = resolve_repo_path(data.get("harmful_prompts", ""), config_path.parent)
    benign_path = resolve_repo_path(data.get("benign_prompts", ""), config_path.parent)
    harmful_count = len(load_prompts(harmful_path))
    benign_count = len(load_prompts(benign_path))
    max_pairs = int(activation.get("max_pairs", min(harmful_count, benign_count)))
    usable_pairs = min(harmful_count, benign_count, max_pairs)

    return {
        "name": config.get("name", config_path.stem),
        "method": config.get("method", "contrastive_refusal_direction"),
        "model": {
            "source": model.get("source"),
            "local_dir": model.get("local_dir"),
            "output_dir": model.get("output_dir"),
            "dtype": model.get("dtype", "bfloat16"),
            "device_map": model.get("device_map", "auto"),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
        },
        "data": {
            "harmful_prompts": str(harmful_path),
            "benign_prompts": str(benign_path),
            "harmful_count": harmful_count,
            "benign_count": benign_count,
            "usable_pairs": usable_pairs,
        },
        "activation_collection": {
            "batch_size": int(activation.get("batch_size", 1)),
            "max_seq_len": int(activation.get("max_seq_len", 1024)),
            "max_pairs": max_pairs,
            "preprocessing_parallelism": activation.get("preprocessing_parallelism", "auto"),
            "effective_parallelism": int(training_env.get("MODEL_FORGE_PARALLELISM", "1")),
            "high_parallelism_c": int(activation.get("high_parallelism_c", training_env.get("MODEL_FORGE_HIGH_PARALLELISM", "1"))),
            "layer_skip_first": int(activation.get("layer_skip_first", 0)),
            "layer_skip_last": int(activation.get("layer_skip_last", 0)),
        },
        "edit": config.get("edit", {}),
        "safety": {
            "min_free_cuda_gb": float(safety.get("min_free_cuda_gb", 8)),
            "require_execute_flag": safety.get("require_execute_flag", True),
            "free_cuda_gb": cuda_free_gb(),
        },
        "hardware": {
            "profile": hardware.name,
            "label": hardware.label,
            "gpus": [{"name": gpu.name, "memory_total_mb": gpu.memory_total_mb} for gpu in hardware.gpus],
            "vllm_env": dict(hardware.vllm_env),
            "training_env": dict(training_env),
            "notes": list(hardware.notes),
        },
    }


def print_plan(plan: dict[str, Any]) -> None:
    console.print(Panel.fit(
        "\n".join([
            f"[bold]Name[/bold]: {plan['name']}",
            f"[bold]Method[/bold]: {plan['method']}",
            f"[bold]Model[/bold]: {plan['model']['source']}",
            f"[bold]Output[/bold]: {plan['model']['output_dir']}",
            f"[bold]Hardware[/bold]: {plan['hardware']['label']}",
        ]),
        title="[bold cyan]Abliteration Plan[/bold cyan]",
        border_style="cyan",
    ))
    table = Table(title="Prompt And Memory Guard")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("harmful prompts", str(plan["data"]["harmful_count"]))
    table.add_row("benign controls", str(plan["data"]["benign_count"]))
    table.add_row("usable contrast pairs", str(plan["data"]["usable_pairs"]))
    table.add_row("batch size", str(plan["activation_collection"]["batch_size"]))
    table.add_row("max sequence length", str(plan["activation_collection"]["max_seq_len"]))
    table.add_row("effective preprocessing c", str(plan["activation_collection"]["effective_parallelism"]))
    table.add_row("high-throughput c", str(plan["activation_collection"]["high_parallelism_c"]))
    free = plan["safety"]["free_cuda_gb"]
    table.add_row("free CUDA GB", "unknown" if free is None else f"{free:.1f}")
    table.add_row("required free CUDA GB", str(plan["safety"]["min_free_cuda_gb"]))
    console.print(table)
    if plan["hardware"]["notes"]:
        console.print("[bold]Profile notes[/bold]")
        for note in plan["hardware"]["notes"]:
            console.print(f"- {note}")


def guard_execute(plan: dict[str, Any], execute: bool) -> None:
    if not execute:
        raise SystemExit("dry run only; pass --execute to load a model")
    if plan["hardware"]["profile"] == "cpu" and os.environ.get("MODEL_FORGE_ALLOW_CPU_ABLATION") != "1":
        raise SystemExit(
            "no CUDA GPU detected; refusing to load a large model on CPU. "
            "Set MODEL_FORGE_ALLOW_CPU_ABLATION=1 only for small test models."
        )
    free = plan["safety"]["free_cuda_gb"]
    required = plan["safety"]["min_free_cuda_gb"]
    if free is not None and free < required and os.environ.get("MODEL_FORGE_SKIP_MEMORY_GUARD") != "1":
        raise SystemExit(
            f"free CUDA memory is {free:.1f} GB, below configured guard {required:.1f} GB; "
            "stop other jobs, lower the plan size, or set MODEL_FORGE_SKIP_MEMORY_GUARD=1"
        )


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def collect_directions(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plan = build_plan(config, config_path)
    model_cfg = plan["model"]
    source = resolve_model_source(model_cfg["local_dir"] or model_cfg["source"])

    harmful = load_prompts(Path(plan["data"]["harmful_prompts"]))[: plan["data"]["usable_pairs"]]
    benign = load_prompts(Path(plan["data"]["benign_prompts"]))[: plan["data"]["usable_pairs"]]
    activation = plan["activation_collection"]

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=model_cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = str(model_cfg["device_map"])
    load_kwargs = {
        "torch_dtype": _torch_dtype(torch, model_cfg["dtype"]),
        "trust_remote_code": model_cfg["trust_remote_code"],
    }
    if device_map == "auto":
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    if device_map in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise SystemExit("device_map=cuda requested but CUDA is not available")
        model.to(torch.device("cuda:0"))
    model.eval()
    first_device = next(model.parameters()).device

    def prompt_vectors(prompts: list[str]) -> dict[int, list[Any]]:
        vectors: dict[int, list[Any]] = {}
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=activation["max_seq_len"],
                padding=False,
            )
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            last_index = int(inputs["attention_mask"][0].sum().item()) - 1
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            for layer_index, states in enumerate(hidden_states):
                vectors.setdefault(layer_index, []).append(states[0, last_index, :].detach().float().cpu())
        return vectors

    harmful_vectors = prompt_vectors(harmful)
    benign_vectors = prompt_vectors(benign)
    layer_count = min(len(harmful_vectors), len(benign_vectors))
    first = activation["layer_skip_first"]
    last_exclusive = layer_count - activation["layer_skip_last"]
    directions = {}
    for layer_index in range(first, max(first, last_exclusive)):
        harmful_mean = torch.stack(harmful_vectors[layer_index]).mean(dim=0)
        benign_mean = torch.stack(benign_vectors[layer_index]).mean(dim=0)
        direction = harmful_mean - benign_mean
        norm = torch.linalg.vector_norm(direction)
        if norm > 0:
            direction = direction / norm
        directions[layer_index] = direction

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(directions, output_dir / "refusal_directions.pt")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "direction_layers": sorted(directions),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    console.print(f"[bold green]Wrote[/bold green] {output_dir / 'refusal_directions.pt'}")


def language_layer_index(name: str) -> int | None:
    match = re.search(r"model[.]language_model[.]layers[.](\d+)[.]", name)
    if not match:
        return None
    return int(match.group(1))


def is_projection_target(name: str, edit: dict[str, Any]) -> bool:
    layer = language_layer_index(name)
    if layer is None:
        return False
    if layer < int(edit.get("layer_start", 0)):
        return False
    if layer > int(edit.get("layer_end", 10**9)):
        return False
    suffixes = tuple(edit.get("target_weight_suffixes") or ())
    return bool(suffixes) and name.endswith(suffixes)


def copy_non_weight_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        target = output_dir / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def export_projection(config: dict[str, Any], config_path: Path, directions_path: Path, overwrite: bool) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan = build_plan(config, config_path)
    model_cfg = plan["model"]
    edit = plan["edit"]
    source_dir = Path(resolve_model_source(model_cfg["local_dir"] or model_cfg["source"]))
    output_dir = resolve_repo_path(model_cfg["output_dir"])
    strength = float(edit.get("strength", 1.0))

    if edit.get("mode") != "projection":
        raise SystemExit(f"unsupported export edit mode: {edit.get('mode')!r}")
    if not source_dir.exists():
        raise SystemExit(f"missing base model directory: {source_dir}")
    if not directions_path.exists():
        raise SystemExit(f"missing refusal directions: {directions_path}")
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    index_path = source_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    raw_directions = torch.load(directions_path, map_location="cpu")
    directions = {int(layer): vector.float() for layer, vector in raw_directions.items()}

    copy_non_weight_files(source_dir, output_dir)
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    changed: list[dict[str, Any]] = []
    for shard in shards:
        names = [name for name, filename in weight_map.items() if filename == shard]
        shard_tensors: dict[str, Any] = {}
        console.print(f"[cyan]Writing[/cyan] {shard} ({len(names)} tensors)")
        with safe_open(source_dir / shard, framework="pt", device="cpu") as base_file:
            for name in names:
                base_tensor = base_file.get_tensor(name)
                layer = language_layer_index(name)
                if not is_projection_target(name, edit) or layer not in directions:
                    shard_tensors[name] = base_tensor
                    continue
                direction = directions[layer]
                if base_tensor.ndim != 2 or base_tensor.shape[0] != direction.numel():
                    raise SystemExit(
                        f"cannot project {name}: tensor shape {tuple(base_tensor.shape)} "
                        f"does not match direction length {direction.numel()}"
                    )
                direction = direction / direction.norm().clamp_min(1e-6)
                base_float = base_tensor.float()
                output_tensor = (base_float - strength * torch.outer(direction, direction @ base_float)).to(base_tensor.dtype)
                delta = output_tensor.float() - base_float
                changed.append({
                    "name": name,
                    "shard": shard,
                    "shape": list(base_tensor.shape),
                    "dtype": str(base_tensor.dtype),
                    "mean_abs_delta": float(delta.abs().mean()),
                    "max_abs_delta": float(delta.abs().max()),
                })
                shard_tensors[name] = output_tensor
        save_file(shard_tensors, output_dir / shard)
        del shard_tensors

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "projection",
        "base_model": str(source_dir),
        "directions": str(directions_path),
        "output_dir": str(output_dir),
        "strength": strength,
        "layer_start": edit.get("layer_start"),
        "layer_end": edit.get("layer_end"),
        "target_weight_suffixes": edit.get("target_weight_suffixes"),
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
    }
    (output_dir / "model_forge_abliteration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    console.print(Panel.fit(
        "\n".join([
            f"[bold]Output[/bold]: {output_dir}",
            f"[bold]Changed tensors[/bold]: {len(changed)}",
            f"[bold]Strength[/bold]: {strength}",
        ]),
        title="[bold green]Abliterated Base Checkpoint Exported[/bold green]",
        border_style="green",
    ))


def command_plan(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    plan = build_plan(load_yaml(config_path), config_path)
    print_plan(plan)


def command_collect(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    plan = build_plan(config, config_path)
    print_plan(plan)
    if not args.execute:
        console.print("[yellow]Dry run only; pass --execute to load the model and collect directions.[/yellow]")
        return
    guard_execute(plan, args.execute)
    output_dir = resolve_repo_path(args.output_dir or config.get("artifacts_dir", "artifacts/abliteration/refusal_directions"))
    collect_directions(config, config_path, output_dir)


def command_export(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    plan = build_plan(config, config_path)
    print_plan(plan)
    if not args.execute:
        console.print("[yellow]Dry run only; export is blocked until collected directions are reviewed.[/yellow]")
        return
    directions_path = resolve_repo_path(args.directions or Path(config.get("artifacts_dir", "artifacts/abliteration/refusal_directions")) / "refusal_directions.pt")
    export_projection(config, config_path, directions_path=directions_path, overwrite=args.overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and run memory-guarded model abliteration steps")
    parser.add_argument("--config", default="configs/abliteration/gemma4_26b_a4b_local_abli.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Inspect config, prompts, hardware, and memory guard without loading a model")
    plan.set_defaults(func=command_plan)

    collect = sub.add_parser("collect", help="Collect contrastive refusal directions")
    collect.add_argument("--execute", action="store_true", help="Actually load the model and collect activations")
    collect.add_argument("--output-dir", default=None)
    collect.set_defaults(func=command_collect)

    export = sub.add_parser("export", help="Reserved for reviewed weight-edit export")
    export.add_argument("--execute", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--directions", default=None)
    export.set_defaults(func=command_export)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
