from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
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
    if path.suffix.lower() == ".txt":
        prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not prompts:
            raise SystemExit(f"prompt file must contain at least one non-empty line: {path}")
        return prompts
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
            "token_position": activation.get("token_position", "final_prompt_token"),
            "direction_extraction": activation.get("direction_extraction", "mean_difference"),
            "direction_components": int(activation.get("direction_components", 1)),
            "direction_source_layer": activation.get("direction_source_layer"),
            "replicate_source_direction": bool(activation.get("replicate_source_direction", False)),
            "use_chat_template": bool(activation.get("use_chat_template", False)),
            "winsorize_quantile": activation.get("winsorize_quantile"),
            "harmful_suffix": activation.get("harmful_suffix"),
            "benign_suffix": activation.get("benign_suffix"),
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
    table.add_row("token position", str(plan["activation_collection"]["token_position"]))
    table.add_row("direction extraction", str(plan["activation_collection"]["direction_extraction"]))
    table.add_row("direction components", str(plan["activation_collection"]["direction_components"]))
    table.add_row("direction source layer", str(plan["activation_collection"]["direction_source_layer"]))
    table.add_row("chat template", str(plan["activation_collection"]["use_chat_template"]))
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

    def prompt_vectors(prompts: list[str], suffix: str | None) -> dict[int, list[Any]]:
        vectors: dict[int, list[Any]] = {}
        for prompt in prompts:
            full_prompt = prompt if not suffix else prompt.rstrip() + suffix
            if activation.get("use_chat_template"):
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": full_prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=activation["max_seq_len"],
                )
            else:
                inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=activation["max_seq_len"],
                    padding=False,
                )
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            last_index = int(inputs["attention_mask"][0].sum().item()) - 1
            first_pool_index = last_index
            if suffix and activation["token_position"] == "suffix_mean":
                if activation.get("use_chat_template"):
                    prompt_inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=activation["max_seq_len"],
                    )
                else:
                    prompt_inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=activation["max_seq_len"],
                        padding=False,
                    )
                prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())
                first_pool_index = min(max(prompt_len, 0), last_index)
            with torch.no_grad():
                if activation["token_position"] == "generation_last_token":
                    outputs = model.generate(
                        **inputs,
                        use_cache=False,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        do_sample=False,
                    )
                    hidden_states = outputs.hidden_states[0][1:]
                else:
                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                    hidden_states = outputs.hidden_states[1:]
            for layer_index, states in enumerate(hidden_states):
                if activation["token_position"] == "generation_last_token":
                    vector = states[0, -1, :]
                elif activation["token_position"] == "suffix_mean" and suffix:
                    vector = states[0, first_pool_index : last_index + 1, :].mean(dim=0)
                else:
                    vector = states[0, last_index, :]
                vectors.setdefault(layer_index, []).append(vector.detach().float().cpu())
        return vectors

    harmful_vectors = prompt_vectors(harmful, activation.get("harmful_suffix"))
    benign_vectors = prompt_vectors(benign, activation.get("benign_suffix"))
    layer_count = min(len(harmful_vectors), len(benign_vectors))
    first = activation["layer_skip_first"]
    last_exclusive = layer_count - activation["layer_skip_last"]
    directions = {}
    harmful_means = {}
    benign_means = {}
    def maybe_winsorize(values: Any) -> Any:
        quantile = activation.get("winsorize_quantile")
        if quantile is None:
            return values
        q = float(quantile)
        if q <= 0 or q >= 0.5:
            raise SystemExit("winsorize_quantile must be between 0 and 0.5")
        low = torch.quantile(values, q, dim=0)
        high = torch.quantile(values, 1 - q, dim=0)
        return values.clamp(low, high)

    def normalize_direction_basis(direction: Any) -> Any:
        if direction.ndim == 1:
            return direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
        if direction.ndim != 2:
            raise SystemExit(f"direction tensor must be 1D or 2D, got shape {tuple(direction.shape)}")
        norms = torch.linalg.vector_norm(direction, dim=1)
        direction = direction[norms > 1e-6]
        if direction.numel() == 0:
            return direction
        q, _ = torch.linalg.qr(direction.float().T, mode="reduced")
        return q.T.contiguous()

    def extract_direction(harmful_stack: Any, benign_stack: Any) -> Any:
        method = str(activation.get("direction_extraction", "mean_difference")).lower()
        components = max(1, int(activation.get("direction_components", 1)))
        mean_direction = harmful_stack.mean(dim=0) - benign_stack.mean(dim=0)
        if method == "mean_difference":
            return mean_direction
        contrast = harmful_stack - benign_stack
        if method == "paired_svd":
            centered = contrast - contrast.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
            direction = vh[:components]
        elif method == "whitened_paired_svd":
            pooled = torch.cat([harmful_stack, benign_stack], dim=0)
            scale = pooled.float().std(dim=0).clamp_min(1e-6)
            whitened = contrast.float() / scale
            whitened = whitened - whitened.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(whitened, full_matrices=False)
            direction = vh[:components] / scale
        elif method == "mean_plus_paired_svd":
            centered = contrast - contrast.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
            direction = torch.vstack([mean_direction.float(), vh[: max(0, components - 1)]])
        else:
            raise SystemExit(f"unsupported direction_extraction: {method!r}")
        if direction.ndim == 1:
            if torch.dot(direction.float(), mean_direction.float()) < 0:
                direction = -direction
        else:
            for idx in range(direction.shape[0]):
                if torch.dot(direction[idx].float(), mean_direction.float()) < 0:
                    direction[idx] = -direction[idx]
        return direction

    target_layers = list(range(first, max(first, last_exclusive)))
    source_layer = activation.get("direction_source_layer")
    if source_layer is not None:
        if isinstance(source_layer, str) and source_layer.endswith("%"):
            source_layer_index = int(layer_count * (float(source_layer.rstrip("%")) / 100.0))
        elif isinstance(source_layer, float) and 0 < source_layer < 1:
            source_layer_index = int(layer_count * source_layer)
        else:
            source_layer_index = int(source_layer)
        if source_layer_index < 0 or source_layer_index >= layer_count:
            raise SystemExit(f"direction_source_layer {source_layer!r} resolved outside available layers 0..{layer_count - 1}")
        if source_layer_index not in target_layers:
            target_layers.append(source_layer_index)
    else:
        source_layer_index = None

    extracted_directions = {}
    for layer_index in sorted(target_layers):
        harmful_stack = maybe_winsorize(torch.stack(harmful_vectors[layer_index]))
        benign_stack = maybe_winsorize(torch.stack(benign_vectors[layer_index]))
        harmful_mean = harmful_stack.mean(dim=0)
        benign_mean = benign_stack.mean(dim=0)
        direction = extract_direction(harmful_stack, benign_stack)
        direction = normalize_direction_basis(direction)
        extracted_directions[layer_index] = direction
        directions[layer_index] = direction
        harmful_means[layer_index] = harmful_mean
        benign_means[layer_index] = benign_mean

    if source_layer_index is not None and activation.get("replicate_source_direction"):
        source_direction = extracted_directions[source_layer_index]
        directions = {layer_index: source_direction.clone() for layer_index in range(first, max(first, last_exclusive))}

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(directions, output_dir / "refusal_directions.pt")
    torch.save(
        {
            "refusal_directions": directions,
            "harmful_means": harmful_means,
            "benign_means": benign_means,
            "direction_method": activation["token_position"],
            "direction_extraction": activation["direction_extraction"],
            "direction_components": int(activation.get("direction_components", 1)),
            "direction_source_layer": source_layer_index,
            "replicate_source_direction": bool(activation.get("replicate_source_direction", False)),
            "use_chat_template": bool(activation.get("use_chat_template", False)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "direction_artifact.pt",
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "direction_layers": sorted(directions),
        "artifact_format": "direction_artifact_v1",
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


def configured_target_layers(edit: dict[str, Any]) -> list[int]:
    start = int(edit.get("layer_start", 0))
    end = int(edit.get("layer_end", start - 1))
    if end < start:
        return []
    return list(range(start, end + 1))


def missing_direction_layers(edit: dict[str, Any], directions: dict[int, Any]) -> list[int]:
    return [layer for layer in configured_target_layers(edit) if layer not in directions]


def _is_layer_tensor_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(key, int) for key in value)


def load_direction_artifact(path: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(path, map_location="cpu")
    if _is_layer_tensor_dict(raw):
        return {
            "refusal_directions": {int(layer): vector.float() for layer, vector in raw.items()},
            "harmful_means": {},
            "benign_means": {},
            "source_path": str(path),
            "format": "legacy_refusal_directions",
        }
    directions = raw.get("refusal_directions", raw.get("directions"))
    if not _is_layer_tensor_dict(directions):
        raise SystemExit(f"direction artifact does not contain layer-indexed directions: {path}")
    return {
        "refusal_directions": {int(layer): vector.float() for layer, vector in directions.items()},
        "harmful_means": {int(layer): vector.float() for layer, vector in raw.get("harmful_means", {}).items()},
        "benign_means": {int(layer): vector.float() for layer, vector in raw.get("benign_means", {}).items()},
        "source_path": str(path),
        "format": raw.get("format", "direction_artifact_v1"),
    }


def tensor_strength(name: str, layer: int, edit: dict[str, Any], default: float) -> float:
    strength = default
    for suffix, value in (edit.get("module_strengths") or {}).items():
        if name.endswith(suffix):
            strength *= float(value)
    layer_strengths = edit.get("layer_strengths") or {}
    if str(layer) in layer_strengths:
        strength *= float(layer_strengths[str(layer)])
    elif layer in layer_strengths:
        strength *= float(layer_strengths[layer])
    return strength


def direction_width(direction: Any) -> int:
    return int(direction.shape[-1])


def normalize_intervention_direction(direction: Any) -> Any:
    import torch

    direction = direction.float()
    if direction.ndim == 1:
        return direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
    if direction.ndim != 2:
        raise SystemExit(f"direction tensor must be 1D or 2D, got shape {tuple(direction.shape)}")
    norms = torch.linalg.vector_norm(direction, dim=1)
    direction = direction[norms > 1e-6]
    if direction.numel() == 0:
        raise SystemExit("direction subspace is empty after dropping zero-norm components")
    q, _ = torch.linalg.qr(direction.T, mode="reduced")
    return q.T.contiguous()


def intervention_direction(layer: int, artifact: dict[str, Any], edit: dict[str, Any]) -> Any:
    import torch

    direction = artifact["refusal_directions"][layer].float()
    mode = str(edit.get("direction_transform", "raw")).lower()
    if mode in {"biprojection", "orthogonalized", "projected"}:
        benign = artifact.get("benign_means", {}).get(layer)
        if benign is None:
            raise SystemExit(
                f"direction_transform={mode} requires benign_means for layer {layer}; "
                "re-run collection to create direction_artifact.pt"
            )
        benign = benign.float()
        benign = benign / torch.linalg.vector_norm(benign).clamp_min(1e-6)
        if direction.ndim == 1:
            direction = direction - benign * torch.dot(direction, benign)
        else:
            direction = direction - torch.outer(direction @ benign, benign)
    elif mode != "raw":
        raise SystemExit(f"unsupported direction_transform: {mode!r}")
    return normalize_intervention_direction(direction)


def apply_projection(base_tensor: Any, direction: Any, strength: float, norm_preserve: bool = False) -> Any:
    import torch

    direction = normalize_intervention_direction(direction)
    base_float = base_tensor.float()
    if direction.ndim == 1:
        if base_tensor.ndim == 1:
            output_float = base_float - strength * direction * torch.dot(direction, base_float)
        else:
            output_float = base_float - strength * torch.outer(direction, direction @ base_float)
    else:
        if base_tensor.ndim == 1:
            output_float = base_float - strength * (direction.T @ (direction @ base_float))
        else:
            output_float = base_float - strength * (direction.T @ (direction @ base_float))
    if base_tensor.ndim == 2 and norm_preserve:
        base_norms = base_float.norm(dim=1, keepdim=True)
        output_norms = output_float.norm(dim=1, keepdim=True).clamp_min(1e-6)
        output_float = output_float * (base_norms / output_norms)
    return output_float


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


def sota_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "preferred_backend": "obliteratus",
        "output_dir": config.get("model", {}).get("output_dir"),
        "work_dir": str(Path(config.get("artifacts_dir", "artifacts/abliteration/sota")) / "sota"),
        "license_notice": "External SOTA backends may be AGPL-licensed; review license terms before redistribution or service use.",
        "backends": {
            "obliteratus": {
                "install": "pip install 'git+https://github.com/elder-plinius/OBLITERATUS.git'",
                "method": "advanced",
                "max_seq_length": 512,
                "telemetry": False,
            },
            "heretic": {
                "install": "pip install -U 'heretic-llm[research]'",
                "quantization": "none",
                "device_map": "auto",
                "n_trials": 200,
                "n_startup_trials": 60,
                "orthogonalize_direction": True,
                "row_normalization": "full",
                "winsorization_quantile": 1.0,
            },
        },
    }
    user = config.get("sota", {})
    merged = json.loads(json.dumps(defaults))
    for key, value in user.items():
        if key == "backends":
            for backend, backend_cfg in value.items():
                merged.setdefault("backends", {}).setdefault(backend, {}).update(backend_cfg or {})
        else:
            merged[key] = value
    return merged


def build_sota_plan(config: dict[str, Any], config_path: Path, backend: str | None = None) -> dict[str, Any]:
    plan = build_plan(config, config_path)
    sota = sota_config(config)
    selected = backend or sota.get("preferred_backend", "obliteratus")
    backends = sota.get("backends", {})
    if selected not in backends:
        raise SystemExit(f"unknown SOTA backend {selected!r}; valid backends: {', '.join(sorted(backends))}")
    model_cfg = plan["model"]
    source = resolve_model_source(model_cfg["local_dir"] or model_cfg["source"])
    output_dir = resolve_repo_path(sota.get("output_dir") or model_cfg["output_dir"])
    work_dir = resolve_repo_path(sota.get("work_dir", Path(config.get("artifacts_dir", "artifacts/abliteration/sota")) / "sota"))
    return {
        "name": plan["name"],
        "backend": selected,
        "source_model": source,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "backend_config": backends[selected],
        "install": backends[selected].get("install"),
        "license_notice": sota.get("license_notice"),
        "all_backends": backends,
    }


def print_sota_plan(plan: dict[str, Any]) -> None:
    lines = [
        f"[bold]Backend[/bold]: {plan['backend']}",
        f"[bold]Source[/bold]: {plan['source_model']}",
        f"[bold]Output[/bold]: {plan['output_dir']}",
        f"[bold]Work dir[/bold]: {plan['work_dir']}",
        f"[bold]Install[/bold]: {plan['install']}",
    ]
    console.print(Panel.fit("\n".join(lines), title="[bold cyan]SOTA Abliteration Backend[/bold cyan]", border_style="cyan"))
    console.print(f"[yellow]{plan['license_notice']}[/yellow]")


def write_obliteratus_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_obliteratus.py"
    script = f'''from __future__ import annotations

import json
import os
from pathlib import Path

from obliteratus.abliterate import AbliterationPipeline

model_name = {plan["source_model"]!r}
output_dir = {plan["output_dir"]!r}
method = {backend.get("method", "advanced")!r}
max_seq_length = {int(backend.get("max_seq_length", 512))}

if {not bool(backend.get("telemetry", False))!r}:
    os.environ.setdefault("OBLITERATUS_TELEMETRY", "0")

pipeline = AbliterationPipeline(
    model_name=model_name,
    method=method,
    output_dir=output_dir,
    max_seq_length=max_seq_length,
)
result = pipeline.run()
Path(output_dir).mkdir(parents=True, exist_ok=True)
summary_path = Path(output_dir) / "model_forge_sota_obliteratus.json"
summary_path.write_text(json.dumps({{
    "backend": "obliteratus",
    "method": method,
    "model_name": model_name,
    "output_dir": output_dir,
    "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else repr(result),
}}, indent=2) + "\\n")
print(f"Wrote {{summary_path}}")
'''
    runner.write_text(script)
    return runner


def write_heretic_config(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "config.toml"
    lines = [
        f'model = "{plan["source_model"]}"',
        'dtypes = ["auto", "float16", "bfloat16", "float32"]',
        f'quantization = "{backend.get("quantization", "none")}"',
        f'device_map = "{backend.get("device_map", "auto")}"',
        f'batch_size = {int(backend.get("batch_size", 0))}',
        f'max_batch_size = {int(backend.get("max_batch_size", 128))}',
        f'max_response_length = {int(backend.get("max_response_length", 100))}',
        f'kl_divergence_scale = {float(backend.get("kl_divergence_scale", 1.0))}',
        f'kl_divergence_target = {float(backend.get("kl_divergence_target", 0.01))}',
        f'n_trials = {int(backend.get("n_trials", 200))}',
        f'n_startup_trials = {int(backend.get("n_startup_trials", 60))}',
        f'orthogonalize_direction = {str(bool(backend.get("orthogonalize_direction", True))).lower()}',
        f'row_normalization = "{backend.get("row_normalization", "full")}"',
        f'full_normalization_lora_rank = {int(backend.get("full_normalization_lora_rank", 3))}',
        f'winsorization_quantile = {float(backend.get("winsorization_quantile", 1.0))}',
        f'study_checkpoint_dir = "{work_dir / "heretic_checkpoints"}"',
    ]
    if backend.get("max_memory"):
        items = ", ".join(f'"{key}" = "{value}"' for key, value in backend["max_memory"].items())
        lines.append(f"max_memory = {{ {items} }}")
    for section in ["good_prompts", "bad_prompts", "good_evaluation_prompts", "bad_evaluation_prompts"]:
        prompt_cfg = backend.get(section)
        if not prompt_cfg:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key in ["dataset", "split", "column", "prefix", "suffix", "system_prompt"]:
            if key in prompt_cfg:
                value = prompt_cfg[key]
                if value is None:
                    lines.append(f"{key} = null")
                else:
                    lines.append(f"{key} = {json.dumps(str(value))}")
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def write_heretic_runner(plan: dict[str, Any]) -> Path:
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_heretic_auto.py"
    script = f'''from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import questionary
from questionary import Choice

import heretic.main as heretic_main
import heretic.model as heretic_model
from heretic.config import RowNormalization
from peft import LoraConfig, PeftModel, get_peft_model

work_dir = Path({str(work_dir)!r})
output_dir = Path({plan["output_dir"]!r})
state = {{"selected_trial": None, "save_requested": False, "saved": False}}


def _choice_title(choice):
    return choice.title if isinstance(choice, Choice) else str(choice)


def _choice_value(choice):
    return choice.value if isinstance(choice, Choice) else choice


def prompt_select(message, choices):
    if message == "How would you like to proceed?":
        return "continue"
    if message == "Which trial do you want to use?":
        if state["saved"]:
            return ""
        for choice in choices:
            value = _choice_value(choice)
            if value != "continue" and value != "":
                state["selected_trial"] = _choice_title(choice)
                return value
        return ""
    if message == "What do you want to do with the decensored model?":
        if not state["save_requested"]:
            state["save_requested"] = True
            return "Save the model to a local folder"
        state["saved"] = True
        return "Return to the trial selection menu"
    if message == "How do you want to proceed?":
        return "merge"
    return _choice_value(choices[0]) if choices else None


def prompt_path(message):
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def prompt_text(message, default="", qmark="?", unsafe=False):
    return default


def prompt_password(message):
    return ""


def apply_lora_exact_language_modules(self):
    target_ids = set()
    for layer_index in range(len(self.get_layers())):
        for modules in self.get_layer_modules(layer_index).values():
            for module in modules:
                target_ids.add(id(module))

    target_modules = [
        name for name, module in self.model.named_modules()
        if id(module) in target_ids
    ]
    if not target_modules:
        raise RuntimeError("No exact language-layer Heretic targets found")

    if self.settings.row_normalization != RowNormalization.FULL:
        lora_rank = 1
    else:
        lora_rank = self.settings.full_normalization_lora_rank

    self.peft_config = LoraConfig(
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    self.model = get_peft_model(self.model, self.peft_config)
    print("* LoRA adapters initialized with exact language-layer targets")


def main():
    os.chdir(work_dir)
    sys.argv = ["heretic"]
    heretic_main.prompt_select = prompt_select
    heretic_main.prompt_path = prompt_path
    heretic_main.prompt_text = prompt_text
    heretic_main.prompt_password = prompt_password
    heretic_model.Model._apply_lora = apply_lora_exact_language_modules
    questionary.select = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected interactive prompt"))
    heretic_main.run()
    if not state["saved"]:
        raise SystemExit("Heretic exited before a model was saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "model_forge_sota_heretic.json"
    summary_path.write_text(json.dumps({{
        "backend": "heretic",
        "source_model": {plan["source_model"]!r},
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "selected_trial": state["selected_trial"],
        "saved": state["saved"],
        "config": str(work_dir / "config.toml"),
    }}, indent=2) + "\\n")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_sota_artifacts(config: dict[str, Any], config_path: Path, backend: str | None = None) -> dict[str, Any]:
    selected_plan = build_sota_plan(config, config_path, backend)
    work_dir = Path(selected_plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in selected_plan["all_backends"]:
        plan = build_sota_plan(config, config_path, name)
        if name == "obliteratus":
            paths["obliteratus_runner"] = str(write_obliteratus_runner(plan))
        elif name == "heretic":
            paths["heretic_config"] = str(write_heretic_config(plan))
            paths["heretic_runner"] = str(write_heretic_runner(plan))
    readme = work_dir / "README.md"
    readme.write_text(
        "\n".join([
            "# SOTA Abliteration Backend",
            "",
            f"Source model: `{selected_plan['source_model']}`",
            f"Output dir: `{selected_plan['output_dir']}`",
            "",
            "Preferred backend: `" + selected_plan["backend"] + "`",
            "",
            "Install commands:",
            "",
            "```bash",
            *(backend_cfg.get("install", "") for backend_cfg in selected_plan["all_backends"].values()),
            "```",
            "",
            "Run OBLITERATUS:",
            "",
            "```bash",
            f"{sys.executable} {paths.get('obliteratus_runner', '<runner>')}",
            "```",
            "",
            "Run Heretic:",
            "",
            "```bash",
            f"cd {work_dir} && {sys.executable} {paths.get('heretic_runner', '<runner>')}",
            "```",
            "",
            "Heretic reads `config.toml` from the working directory. The generated runner patches Heretic's prompts so batch runs save the selected Pareto trial to the configured output directory.",
            "",
            selected_plan["license_notice"],
            "",
        ])
    )
    return {"plan": selected_plan, "paths": paths, "readme": str(readme)}


def command_sota_plan(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    plan = build_sota_plan(load_yaml(config_path), config_path, args.backend)
    print_sota_plan(plan)


def command_sota_prepare(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    result = write_sota_artifacts(load_yaml(config_path), config_path, args.backend)
    print_sota_plan(result["plan"])
    console.print("[bold green]Wrote SOTA backend artifacts[/bold green]")
    for label, path in result["paths"].items():
        console.print(f"- {label}: {path}")
    console.print(f"- README: {result['readme']}")


def command_sota_run(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    result = write_sota_artifacts(config, config_path, args.backend)
    plan = result["plan"]
    print_sota_plan(plan)
    if not args.execute:
        console.print("[yellow]Dry run only; pass --execute to run the external backend.[/yellow]")
        return
    if plan["backend"] == "obliteratus":
        runner = result["paths"].get("obliteratus_runner")
        if runner is None:
            raise SystemExit("missing generated OBLITERATUS runner")
        subprocess.run([sys.executable, runner], cwd=REPO_DIR, check=True)
    elif plan["backend"] == "heretic":
        runner = result["paths"].get("heretic_runner")
        if runner is None:
            raise SystemExit("missing generated Heretic runner")
        subprocess.run([sys.executable, runner], cwd=Path(plan["work_dir"]), check=True)
    else:
        raise SystemExit(f"unsupported SOTA backend: {plan['backend']}")


def export_projection(
    config: dict[str, Any],
    config_path: Path,
    directions_path: Path,
    overwrite: bool,
    strength_override: float | None = None,
    output_dir_override: str | None = None,
) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan = build_plan(config, config_path)
    model_cfg = plan["model"]
    edit = plan["edit"]
    source_dir = Path(resolve_model_source(model_cfg["local_dir"] or model_cfg["source"]))
    output_dir = resolve_repo_path(output_dir_override or model_cfg["output_dir"])
    strength = float(strength_override if strength_override is not None else edit.get("strength", 1.0))

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
    artifact = load_direction_artifact(directions_path)
    directions = artifact["refusal_directions"]
    missing_layers = missing_direction_layers(edit, directions)
    if missing_layers and edit.get("require_all_target_directions", True):
        raise SystemExit(
            "refusing export because target layers lack directions: "
            f"{missing_layers}. Re-run collection with matching layer_skip settings, "
            "or set edit.require_all_target_directions=false for exploratory exports."
        )

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
                direction = intervention_direction(layer, artifact, edit)
                if base_tensor.ndim not in {1, 2} or base_tensor.shape[0] != direction_width(direction):
                    raise SystemExit(
                        f"cannot project {name}: tensor shape {tuple(base_tensor.shape)} "
                        f"does not match direction width {direction_width(direction)}"
                    )
                local_strength = tensor_strength(name, layer, edit, strength)
                base_float = base_tensor.float()
                output_float = apply_projection(
                    base_tensor,
                    direction,
                    local_strength,
                    norm_preserve=bool(edit.get("norm_preserve", False)),
                )
                output_tensor = output_float.to(base_tensor.dtype)
                delta = output_tensor.float() - base_float
                changed.append({
                    "name": name,
                    "shard": shard,
                    "shape": list(base_tensor.shape),
                    "dtype": str(base_tensor.dtype),
                    "strength": local_strength,
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
        "direction_artifact_format": artifact["format"],
        "direction_transform": edit.get("direction_transform", "raw"),
        "norm_preserve": bool(edit.get("norm_preserve", False)),
        "module_strengths": edit.get("module_strengths", {}),
        "layer_start": edit.get("layer_start"),
        "layer_end": edit.get("layer_end"),
        "target_weight_suffixes": edit.get("target_weight_suffixes"),
        "required_target_layers": configured_target_layers(edit),
        "missing_direction_layers": missing_layers,
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


def _projection_delta(base_tensor: Any, direction: Any, strength: float, norm_preserve: bool = False) -> Any:
    base_float = base_tensor.float()
    output_tensor = apply_projection(base_tensor, direction, strength, norm_preserve=norm_preserve)
    return output_tensor - base_float


def analyze_reference(
    config: dict[str, Any],
    config_path: Path,
    reference_model: str,
    directions_path: Path | None,
    output_path: Path | None,
    strength_override: float | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    plan = build_plan(config, config_path)
    model_cfg = plan["model"]
    edit = plan["edit"]
    source_dir = Path(resolve_model_source(model_cfg["local_dir"] or model_cfg["source"]))
    reference_dir = Path(resolve_model_source(reference_model))
    strength = float(strength_override if strength_override is not None else edit.get("strength", 1.0))
    if not source_dir.exists():
        raise SystemExit(f"missing base model directory: {source_dir}")
    if not reference_dir.exists():
        raise SystemExit(f"missing reference model directory: {reference_dir}")

    index = json.loads((source_dir / "model.safetensors.index.json").read_text())
    reference_index = json.loads((reference_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    reference_map = reference_index["weight_map"]
    artifact: dict[str, Any] = {"refusal_directions": {}, "harmful_means": {}, "benign_means": {}, "format": "none"}
    directions: dict[int, Any] = {}
    if directions_path and directions_path.exists():
        artifact = load_direction_artifact(directions_path)
        directions = artifact["refusal_directions"]

    changed: list[dict[str, Any]] = []
    by_suffix: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    cosines: list[float] = []
    for name, shard in weight_map.items():
        reference_shard = reference_map.get(name)
        if reference_shard is None:
            continue
        with safe_open(source_dir / shard, framework="pt", device="cpu") as base_file:
            base_tensor = base_file.get_tensor(name)
        with safe_open(reference_dir / reference_shard, framework="pt", device="cpu") as reference_file:
            reference_tensor = reference_file.get_tensor(name)
        reference_delta = reference_tensor.float() - base_tensor.float()
        mean_abs_delta = float(reference_delta.abs().mean())
        if mean_abs_delta <= 0:
            continue
        layer = language_layer_index(name)
        suffix = next((item for item in edit.get("target_weight_suffixes", []) if name.endswith(item)), "<outside-targets>")
        entry: dict[str, Any] = {
            "name": name,
            "layer": layer,
            "suffix": suffix,
            "shape": list(base_tensor.shape),
            "reference_mean_abs_delta": mean_abs_delta,
            "reference_max_abs_delta": float(reference_delta.abs().max()),
            "is_configured_target": is_projection_target(name, edit),
        }
        if layer is not None:
            by_layer[str(layer)] = by_layer.get(str(layer), 0) + 1
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        if is_projection_target(name, edit) and layer in directions and base_tensor.ndim in {1, 2}:
            direction = intervention_direction(layer, artifact, edit)
            local_strength = tensor_strength(name, layer, edit, strength)
            projected_delta = _projection_delta(
                base_tensor,
                direction,
                local_strength,
                norm_preserve=bool(edit.get("norm_preserve", False)),
            )
            reference_flat = reference_delta.flatten()
            projected_flat = projected_delta.flatten()
            cosine = torch.nn.functional.cosine_similarity(reference_flat, projected_flat, dim=0).item()
            entry["projection_reference_cosine"] = float(cosine)
            entry["projection_mean_abs_delta"] = float(projected_delta.abs().mean())
            entry["projection_reference_mean_abs_ratio"] = float(projected_delta.abs().mean() / reference_delta.abs().mean().clamp_min(1e-12))
            entry["projection_strength"] = local_strength
            cosines.append(float(cosine))
        changed.append(entry)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(source_dir),
        "reference_model": str(reference_dir),
        "directions": str(directions_path) if directions_path else None,
        "direction_artifact_format": artifact["format"],
        "strength": strength,
        "direction_transform": edit.get("direction_transform", "raw"),
        "norm_preserve": bool(edit.get("norm_preserve", False)),
        "changed_tensor_count": len(changed),
        "changed_by_suffix": dict(sorted(by_suffix.items())),
        "changed_by_layer": dict(sorted(by_layer.items(), key=lambda item: int(item[0]))),
        "configured_target_layers": configured_target_layers(edit),
        "missing_direction_layers": missing_direction_layers(edit, directions) if directions else [],
        "projection_reference_cosine_mean": sum(cosines) / len(cosines) if cosines else None,
        "changed_tensors": changed,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    if not quiet:
        console.print(Panel.fit(
            "\n".join([
                f"[bold]Reference[/bold]: {reference_dir}",
                f"[bold]Changed tensors[/bold]: {len(changed)}",
                f"[bold]Changed suffixes[/bold]: {summary['changed_by_suffix']}",
                f"[bold]Projection cosine mean[/bold]: {summary['projection_reference_cosine_mean']}",
            ]),
            title="[bold cyan]Reference Ablation Diagnostics[/bold cyan]",
            border_style="cyan",
        ))
    return summary


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
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts/abliteration/refusal_directions"))
    default_directions = artifacts_dir / "direction_artifact.pt"
    if not resolve_repo_path(default_directions).exists():
        default_directions = artifacts_dir / "refusal_directions.pt"
    directions_path = resolve_repo_path(args.directions or default_directions)
    export_projection(
        config,
        config_path,
        directions_path=directions_path,
        overwrite=args.overwrite,
        strength_override=args.strength,
        output_dir_override=args.output_dir,
    )


def command_analyze_reference(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    diagnostics = config.get("diagnostics", {})
    reference_model = args.reference_model or diagnostics.get("reference_model")
    if not reference_model:
        raise SystemExit("reference model is required; pass --reference-model or set diagnostics.reference_model")
    directions_path = None
    if args.directions or config.get("artifacts_dir"):
        artifacts_dir = Path(config.get("artifacts_dir"))
        default_directions = artifacts_dir / "direction_artifact.pt"
        if not resolve_repo_path(default_directions).exists():
            default_directions = artifacts_dir / "refusal_directions.pt"
        directions_path = resolve_repo_path(args.directions or default_directions)
    output_path = resolve_repo_path(args.output) if args.output else None
    analyze_reference(
        config,
        config_path,
        reference_model=reference_model,
        directions_path=directions_path,
        output_path=output_path,
        strength_override=args.strength,
    )


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("expected at least one numeric value")
    return values


def command_sweep_reference(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    diagnostics = config.get("diagnostics", {})
    reference_model = args.reference_model or diagnostics.get("reference_model")
    if not reference_model:
        raise SystemExit("reference model is required; pass --reference-model or set diagnostics.reference_model")
    directions_path = resolve_repo_path(args.directions or Path(config.get("artifacts_dir", "artifacts/abliteration/refusal_directions")) / "direction_artifact.pt")
    if not directions_path.exists():
        legacy = resolve_repo_path(Path(config.get("artifacts_dir", "artifacts/abliteration/refusal_directions")) / "refusal_directions.pt")
        directions_path = legacy
    strengths = parse_float_list(args.strengths)
    transforms = [item.strip() for item in args.transforms.split(",") if item.strip()]
    norm_options = [False, True] if args.include_norm_preserve else [bool(config.get("edit", {}).get("norm_preserve", False))]

    rows: list[dict[str, Any]] = []
    for transform in transforms:
        for norm_preserve in norm_options:
            for strength in strengths:
                candidate = json.loads(json.dumps(config))
                candidate.setdefault("edit", {})["direction_transform"] = transform
                candidate["edit"]["norm_preserve"] = norm_preserve
                try:
                    summary = analyze_reference(
                        candidate,
                        config_path,
                        reference_model=reference_model,
                        directions_path=directions_path,
                        output_path=None,
                        strength_override=strength,
                        quiet=True,
                    )
                    ratios = [
                        item["projection_reference_mean_abs_ratio"]
                        for item in summary["changed_tensors"]
                        if "projection_reference_mean_abs_ratio" in item
                    ]
                    rows.append({
                        "transform": transform,
                        "norm_preserve": norm_preserve,
                        "strength": strength,
                        "cosine_mean": summary["projection_reference_cosine_mean"],
                        "mean_abs_ratio": sum(ratios) / len(ratios) if ratios else None,
                        "missing_layers": summary["missing_direction_layers"],
                        "error": None,
                    })
                except SystemExit as exc:
                    rows.append({
                        "transform": transform,
                        "norm_preserve": norm_preserve,
                        "strength": strength,
                        "cosine_mean": None,
                        "mean_abs_ratio": None,
                        "missing_layers": [],
                        "error": str(exc),
                    })

    rows.sort(key=lambda item: (item["cosine_mean"] is None, -(item["cosine_mean"] or -999), abs((item["mean_abs_ratio"] or 0) - 1)))
    output_path = resolve_repo_path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2) + "\n")

    table = Table(title="Reference Alignment Sweep")
    table.add_column("transform")
    table.add_column("norm")
    table.add_column("strength")
    table.add_column("cosine")
    table.add_column("abs ratio")
    table.add_column("missing")
    table.add_column("error")
    for row in rows[: args.top_k]:
        table.add_row(
            str(row["transform"]),
            str(row["norm_preserve"]),
            f"{row['strength']:.3g}",
            "n/a" if row["cosine_mean"] is None else f"{row['cosine_mean']:.4f}",
            "n/a" if row["mean_abs_ratio"] is None else f"{row['mean_abs_ratio']:.4f}",
            ",".join(str(item) for item in row["missing_layers"]),
            "" if row["error"] is None else row["error"][:80],
        )
    console.print(table)


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
    export.add_argument("--strength", type=float, default=None, help="Override edit.strength for this export")
    export.add_argument("--output-dir", default=None, help="Override model.output_dir for this export")
    export.set_defaults(func=command_export)

    analyze = sub.add_parser("analyze-reference", help="Compare base-to-reference deltas against configured targets")
    analyze.add_argument("--reference-model", default=None)
    analyze.add_argument("--directions", default=None)
    analyze.add_argument("--strength", type=float, default=None)
    analyze.add_argument("--output", default=None)
    analyze.set_defaults(func=command_analyze_reference)

    sweep = sub.add_parser("sweep-reference", help="Rank training-free edit settings against a reference checkpoint")
    sweep.add_argument("--reference-model", default=None)
    sweep.add_argument("--directions", default=None)
    sweep.add_argument("--strengths", default="0.5,1.0,1.5,2.0,3.0,4.0")
    sweep.add_argument("--transforms", default="raw,biprojection")
    sweep.add_argument("--include-norm-preserve", action="store_true")
    sweep.add_argument("--top-k", type=int, default=12)
    sweep.add_argument("--output", default=None)
    sweep.set_defaults(func=command_sweep_reference)

    sota_plan = sub.add_parser("sota-plan", help="Inspect SOTA external backend plan")
    sota_plan.add_argument("--backend", choices=["obliteratus", "heretic"], default=None)
    sota_plan.set_defaults(func=command_sota_plan)

    sota_prepare = sub.add_parser("sota-prepare", help="Write backend-specific SOTA runner/config files")
    sota_prepare.add_argument("--backend", choices=["obliteratus", "heretic"], default=None)
    sota_prepare.set_defaults(func=command_sota_prepare)

    sota_run = sub.add_parser("sota-run", help="Run a prepared external SOTA backend")
    sota_run.add_argument("--backend", choices=["obliteratus", "heretic"], default=None)
    sota_run.add_argument("--execute", action="store_true")
    sota_run.set_defaults(func=command_sota_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
