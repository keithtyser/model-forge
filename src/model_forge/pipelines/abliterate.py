from __future__ import annotations

import argparse
import json
import os
import re
import pprint
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_forge.hardware import detect_hardware_profile, recommended_training_env
from model_forge.variants.checkpoint_audit import CheckpointFinding, audit_full_checkpoint

console = Console()


REPO_DIR = Path(__file__).resolve().parents[3]
CANDIDATE_GATE_SCHEMA_VERSION = "model_forge.abliteration_candidate_gate.v1"
BLOCKED_CANDIDATE_STATUSES = {"runner_missing", "plan_only", "blocked", "rejected", "failed"}
EXPORT_COMPLETED_CANDIDATE_STATUSES = {
    "exported",
    "exported_local",
    "exported_local_audited",
    "synced",
    "ready_for_eval",
}
CANDIDATE_LOOP_SCHEMA_VERSION = "model_forge.abliteration_candidate_loop_plan.v1"
SOTA_BACKEND_CHOICES = (
    "obliteratus",
    "heretic",
    "abliterix",
    "apostate",
    "sra",
    "optimal_transport",
    "norm_preserving_projection",
    "som_projection",
    "selective_projection",
    "concept_cone_projection",
    "qwen_scope_sae",
)
CANDIDATE_GATE_OPERATORS = {"<=", ">=", "==", "!=", "<", ">"}
GENERATED_TOKEN_POSITIONS = {
    "generation_last_token",
    "generated_first_token",
    "first_generated_token",
}

APOSTATE_EXECUTIONS = {
    "checkpoint_export",
    "guarded_checkpoint",
    "baked_checkpoint",
}

NATIVE_OPTIMAL_TRANSPORT_EXECUTIONS = {
    "checkpoint_export",
    "guarded_checkpoint",
    "baked_checkpoint",
}

NATIVE_PROJECTED_ABLATION_EXECUTIONS = {
    "checkpoint_export",
    "guarded_checkpoint",
    "baked_checkpoint",
    "selective_checkpoint_export",
}

QWEN_SCOPE_SAE_EXECUTIONS = {
    "checkpoint_export",
    "guarded_checkpoint",
    "baked_checkpoint",
    "sae_dictionary_projection",
}

APOSTATE_CONFIG_FIELDS = {
    "activation_cache_dir",
    "adaptive_trials",
    "baseline_eval_n",
    "batch_size",
    "bake",
    "cache_activations",
    "causal_floor",
    "causal_targeting",
    "causal_temperature",
    "compute_dtype",
    "device",
    "direction_layer_frac",
    "direction_scope",
    "fit_response_activations",
    "fit_response_n",
    "fit_response_tokens",
    "gemma_ple",
    "gemma_query",
    "guard_alpha_step",
    "guard_leakage_eps",
    "guard_max_iters",
    "harmful_path",
    "harmful_test",
    "harmless_path",
    "harmless_test",
    "head_sweep",
    "head_sweep_eval_n",
    "head_sweep_max",
    "head_sweep_min",
    "head_sweep_probe_classifier",
    "head_sweep_probe_n",
    "head_sweep_step",
    "head_sweep_top_k",
    "kl_over_budget_weight",
    "kl_positions",
    "kl_quad_weight",
    "kl_target",
    "kl_target_weight",
    "kl_weight",
    "load_in_4bit",
    "max_kl",
    "max_new_tokens",
    "max_rank",
    "model",
    "n_eval",
    "n_harmful",
    "n_harmless",
    "n_trials",
    "opt_capability",
    "opt_capability_code_n",
    "opt_capability_math_n",
    "opt_capability_weight",
    "opt_early_stop",
    "opt_early_stop_margin",
    "opt_eval_n",
    "opt_gen_tokens",
    "opt_guard",
    "opt_objective",
    "opt_rerank_k",
    "optimize",
    "orthogonalize_direction",
    "output_dir",
    "ple_max_rank",
    "preserve_path",
    "preserve_rank",
    "profile",
    "prune",
    "prune_kl",
    "prune_max_frac",
    "reader_guard_rank",
    "reader_kl_target",
    "reader_margin_target",
    "reader_max_kl",
    "reader_strengths",
    "refine_deescalate",
    "refine_kl_layer_candidates",
    "refine_kl_layer_steps",
    "refine_kl_steps",
    "refine_max_scale",
    "refine_refusal",
    "refine_refusal_slack",
    "refine_scale_rerank_k",
    "refine_steps",
    "refusal_quad_weight",
    "refusal_rank",
    "refusal_target_weight",
    "repair_candidates",
    "repair_eval_n",
    "repair_kl_n",
    "repair_min_alpha",
    "repair_min_kl_gain",
    "repair_min_refusal_gain",
    "repair_min_score_gain",
    "repair_probe_candidates",
    "repair_probe_kl_n",
    "repair_probe_positions",
    "repair_probe_ref_n",
    "repair_refusal_regress_slack",
    "repair_rerank_k",
    "repair_steps",
    "repair_stop_kl_frac",
    "resume",
    "save_dtype",
    "seed",
    "target_refusal",
    "variance_threshold",
}

OBLITERATUS_PIPELINE_FIELDS = {
    "activation_steering",
    "attention_head_surgery",
    "cot_aware",
    "device",
    "direction_method",
    "dtype",
    "embed_regularization",
    "expert_transplant",
    "float_layer_interpolation",
    "invert_refusal",
    "kl_budget",
    "large_model_mode",
    "layer_adaptive_strength",
    "layer_selection",
    "lora_rank",
    "n_directions",
    "n_sae_features",
    "norm_preserve",
    "per_expert_directions",
    "project_biases",
    "project_embeddings",
    "push_to_hub",
    "quantization",
    "rdo_refinement",
    "refinement_passes",
    "reflection_strength",
    "regularization",
    "safety_neuron_masking",
    "spectral_bands",
    "spectral_cascade",
    "spectral_threshold",
    "steering_strength",
    "transplant_blend",
    "true_iterative_refinement",
    "trust_remote_code",
    "use_chat_template",
    "use_jailbreak_contrast",
    "use_kl_optimization",
    "use_lora_ablation",
    "use_sae_features",
    "use_wasserstein_optimal",
    "use_whitened_svd",
    "verify_sample_size",
    "winsorize_activations",
    "winsorize_percentile",
}


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


def display_path(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(resolved)


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
            "sra_preservation_components": int(activation.get("sra_preservation_components", 0) or 0),
            "sra_include_benign_mean": bool(activation.get("sra_include_benign_mean", False)),
            "direction_source_layer": activation.get("direction_source_layer"),
            "replicate_source_direction": bool(activation.get("replicate_source_direction", False)),
            "use_chat_template": bool(activation.get("use_chat_template", False)),
            "winsorize_quantile": activation.get("winsorize_quantile"),
            "harmful_suffix": activation.get("harmful_suffix"),
            "benign_suffix": activation.get("benign_suffix"),
            "harmful_assistant_prefix": activation.get("harmful_assistant_prefix"),
            "benign_assistant_prefix": activation.get("benign_assistant_prefix"),
            "assistant_prefix_template": activation.get("assistant_prefix_template", "\n\nAssistant: {prefix}"),
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
    table.add_row("assistant prefix contrast", str(bool(
        plan["activation_collection"].get("harmful_assistant_prefix")
        or plan["activation_collection"].get("benign_assistant_prefix")
    )))
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


def guard_source_checkpoint(plan: dict[str, Any]) -> None:
    model_cfg = plan["model"]
    raw_source = model_cfg.get("local_dir") or model_cfg.get("source")
    if not raw_source:
        raise SystemExit("ablation source checkpoint is required")
    source = Path(resolve_model_source(raw_source)).expanduser()
    if not source.is_absolute() and not source.exists():
        return
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"ablation source checkpoint is not present: {source}")
    record: dict[str, Any] = {"variant": "source", "path": str(source), "exists": True}
    findings: list[CheckpointFinding] = []
    audit_full_checkpoint(
        variant_name="source",
        path=source,
        record=record,
        findings=findings,
        strict=True,
    )
    errors = [finding for finding in findings if finding.level == "error"]
    if errors:
        summary = "; ".join(f"{finding.check}: {finding.message}" for finding in errors[:5])
        raise SystemExit(f"ablation source checkpoint audit failed: {summary}")


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def _select_transformers_auto_model(source: str, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    except Exception:
        return AutoModelForCausalLM
    architectures = [str(item) for item in getattr(config, "architectures", []) or []]
    if any(name.endswith("ForConditionalGeneration") for name in architectures):
        try:
            from transformers import AutoModelForImageTextToText

            AutoModelForImageTextToText._model_mapping[type(config)]
            return AutoModelForImageTextToText
        except Exception:
            return AutoModelForCausalLM
    return AutoModelForCausalLM


def _progress_every(total: int, env_name: str = "MODEL_FORGE_NATIVE_PROGRESS_EVERY") -> int:
    raw = os.environ.get(env_name)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return max(1, total)
    return max(1, min(10, total // 20 or 1))


def _orthonormal_rows(matrix: Any) -> Any:
    import torch

    matrix = matrix.float()
    if matrix.ndim == 1:
        norm = torch.linalg.vector_norm(matrix)
        if norm <= 1e-6:
            return matrix.reshape(0, matrix.shape[0])
        return (matrix / norm).unsqueeze(0)
    if matrix.ndim != 2:
        raise SystemExit(f"subspace tensor must be 1D or 2D, got shape {tuple(matrix.shape)}")
    norms = torch.linalg.vector_norm(matrix, dim=1)
    matrix = matrix[norms > 1e-6]
    if matrix.numel() == 0:
        return matrix.reshape(0, matrix.shape[-1])
    q, _ = torch.linalg.qr(matrix.T, mode="reduced")
    return q.T.contiguous()


def _project_out_rowspace(values: Any, basis: Any) -> Any:
    if basis is None or basis.numel() == 0:
        return values
    if values.ndim == 1:
        return values - basis.T @ (basis @ values.float())
    return values - (values.float() @ basis.T) @ basis


def extract_concept_cone_direction(
    harmful_stack: Any,
    benign_stack: Any,
    *,
    components: int,
    benign_subspace_components: int = 2,
    project_mean_out: bool = True,
    whiten: bool = True,
) -> Any:
    """Build a refusal concept cone after removing dominant benign variation.

    This is intentionally native and small: it avoids materializing a second
    checkpoint or depending on an external ablation package, while giving the
    recipe a preservation-aware direction family instead of one global
    harmful-vs-benign mean vector.
    """
    import torch

    if harmful_stack.ndim != 2 or benign_stack.ndim != 2:
        raise SystemExit("concept cone extraction requires 2D harmful and benign activation stacks")
    if harmful_stack.shape[1] != benign_stack.shape[1]:
        raise SystemExit("harmful and benign activation widths must match")
    component_count = max(1, int(components))
    harmful = harmful_stack.float()
    benign = benign_stack.float()
    harmful_mean = harmful.mean(dim=0)
    benign_mean = benign.mean(dim=0)
    mean_direction = harmful_mean - benign_mean

    pooled_scale = None
    if whiten:
        pooled = torch.cat([harmful, benign], dim=0)
        pooled_scale = pooled.std(dim=0).clamp_min(1e-6)
        harmful_for_svd = harmful / pooled_scale
        benign_for_svd = benign / pooled_scale
        benign_mean_for_svd = benign_mean / pooled_scale
        mean_for_svd = mean_direction / pooled_scale
    else:
        harmful_for_svd = harmful
        benign_for_svd = benign
        benign_mean_for_svd = benign_mean
        mean_for_svd = mean_direction

    benign_centered = benign_for_svd - benign_for_svd.mean(dim=0, keepdim=True)
    benign_basis = None
    if benign_subspace_components > 0 and benign_centered.shape[0] > 1:
        _, _, benign_vh = torch.linalg.svd(benign_centered, full_matrices=False)
        benign_basis = _orthonormal_rows(benign_vh[: int(benign_subspace_components)])

    residuals = harmful_for_svd - benign_mean_for_svd.unsqueeze(0)
    residuals = _project_out_rowspace(residuals, benign_basis)
    if project_mean_out:
        mean_for_svd = _project_out_rowspace(mean_for_svd, benign_basis)
    residuals = residuals - residuals.mean(dim=0, keepdim=True)

    rows = [mean_for_svd]
    extra = max(0, component_count - 1)
    if extra and residuals.shape[0] > 1:
        _, _, harmful_vh = torch.linalg.svd(residuals, full_matrices=False)
        rows.extend(harmful_vh[:extra])
    direction = torch.vstack(rows)
    if pooled_scale is not None:
        direction = direction / pooled_scale

    mean_reference = harmful_mean - benign_mean
    for idx in range(direction.shape[0]):
        if torch.dot(direction[idx].float(), mean_reference.float()) < 0:
            direction[idx] = -direction[idx]
    return direction[0] if component_count == 1 else direction


def build_sra_preservation_basis(
    benign_stack: Any,
    *,
    components: int,
    include_benign_mean: bool = True,
) -> Any:
    """Build a compact preservation basis for SRA-style direction cleanup."""
    import torch

    if benign_stack.ndim != 2:
        raise SystemExit("SRA preservation basis requires a 2D benign activation stack")
    rows = []
    benign = benign_stack.float()
    benign_mean = benign.mean(dim=0)
    if include_benign_mean:
        rows.append(benign_mean)
    component_count = max(0, int(components))
    if component_count > 0 and benign.shape[0] > 1:
        centered = benign - benign_mean.unsqueeze(0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rows.extend(vh[:component_count])
    if not rows:
        return benign.new_empty((0, benign.shape[1]))
    return _orthonormal_rows(torch.vstack(rows))


def collect_directions(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    import torch
    from transformers import AutoTokenizer

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
    auto_model = _select_transformers_auto_model(source, bool(model_cfg["trust_remote_code"]))
    console.print(f"[cyan]Loading native collection model[/cyan]: {source}")
    console.print(f"[cyan]Transformers auto class[/cyan]: {getattr(auto_model, '__name__', auto_model.__class__.__name__)}")
    model = auto_model.from_pretrained(source, **load_kwargs)
    if device_map in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise SystemExit("device_map=cuda requested but CUDA is not available")
        model.to(torch.device("cuda:0"))
    model.eval()
    first_device = next(model.parameters()).device
    console.print(f"[cyan]Native collection device[/cyan]: {first_device}")

    def prompt_inputs_for_activation(
        prompt: str,
        *,
        suffix: str | None,
        assistant_prefix: str | None,
    ) -> tuple[dict[str, Any], int | None]:
        if suffix and assistant_prefix:
            raise SystemExit("activation collection cannot combine suffix and assistant_prefix for the same prompt set")
        if assistant_prefix:
            prefix = assistant_prefix.strip()
            if activation.get("use_chat_template"):
                full_inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": prefix},
                    ],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=activation["max_seq_len"],
                )
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
                template = str(activation.get("assistant_prefix_template") or "\n\nAssistant: {prefix}")
                if "{prefix}" not in template:
                    raise SystemExit("assistant_prefix_template must include {prefix}")
                full_prompt = prompt.rstrip() + template.format(prefix=prefix)
                prompt_base = prompt.rstrip() + template.format(prefix="")
                full_inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=activation["max_seq_len"],
                    padding=False,
                )
                prompt_inputs = tokenizer(
                    prompt_base,
                    return_tensors="pt",
                    truncation=True,
                    max_length=activation["max_seq_len"],
                    padding=False,
                )
            return full_inputs, int(prompt_inputs["attention_mask"][0].sum().item())

        full_prompt = prompt if not suffix else prompt.rstrip() + suffix
        if activation.get("use_chat_template"):
            full_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=activation["max_seq_len"],
            )
        else:
            full_inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=activation["max_seq_len"],
                padding=False,
            )
        if not suffix:
            return full_inputs, None
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
        return full_inputs, int(prompt_inputs["attention_mask"][0].sum().item())

    def prompt_vectors(
        label: str,
        prompts: list[str],
        suffix: str | None,
        assistant_prefix: str | None,
    ) -> dict[int, list[Any]]:
        vectors: dict[int, list[Any]] = {}
        total = len(prompts)
        every = _progress_every(total)
        console.print(f"[cyan]Collecting {label} activations[/cyan]: {total} prompt(s)")
        for prompt_index, prompt in enumerate(prompts, start=1):
            if prompt_index == 1 or prompt_index == total or prompt_index % every == 0:
                console.print(f"[dim]native activations {label}: {prompt_index}/{total}[/dim]")
            inputs, contrast_start_index = prompt_inputs_for_activation(
                prompt,
                suffix=suffix,
                assistant_prefix=assistant_prefix,
            )
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            last_index = int(inputs["attention_mask"][0].sum().item()) - 1
            first_pool_index = last_index
            if contrast_start_index is not None and activation["token_position"] in {"suffix_mean", "assistant_prefix_mean"}:
                first_pool_index = min(max(contrast_start_index, 0), last_index)
            with torch.no_grad():
                if activation["token_position"] in GENERATED_TOKEN_POSITIONS:
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
                if activation["token_position"] in GENERATED_TOKEN_POSITIONS:
                    vector = states[0, -1, :]
                elif activation["token_position"] in {"suffix_mean", "assistant_prefix_mean"} and contrast_start_index is not None:
                    vector = states[0, first_pool_index : last_index + 1, :].mean(dim=0)
                else:
                    vector = states[0, last_index, :]
                vectors.setdefault(layer_index, []).append(vector.detach().float().cpu())
        return vectors

    harmful_vectors = prompt_vectors(
        "harmful",
        harmful,
        activation.get("harmful_suffix"),
        activation.get("harmful_assistant_prefix"),
    )
    benign_vectors = prompt_vectors(
        "benign",
        benign,
        activation.get("benign_suffix"),
        activation.get("benign_assistant_prefix"),
    )
    layer_count = min(len(harmful_vectors), len(benign_vectors))
    console.print(f"[cyan]Collected activations for {layer_count} layer(s)[/cyan]")
    first = activation["layer_skip_first"]
    last_exclusive = layer_count - activation["layer_skip_last"]
    directions = {}
    harmful_means = {}
    benign_means = {}
    sra_preservation_bases = {}
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

    def som_residual_centroids(values: Any, components: int, mean_direction: Any) -> Any:
        values = values.float()
        if values.ndim != 2 or values.shape[0] == 0:
            raise SystemExit("SOM direction extraction requires a non-empty 2D activation stack")
        centroid_count = min(
            int(activation.get("som_neurons", max(2, components))),
            values.shape[0],
        )
        centroid_count = max(1, centroid_count)
        steps = max(1, int(activation.get("som_steps", 32)))
        initial_lr = float(activation.get("som_learning_rate", 0.35))
        initial_sigma = float(activation.get("som_neighborhood", max(1.0, centroid_count / 2.0)))
        mean_axis = mean_direction.float()
        mean_axis = mean_axis / torch.linalg.vector_norm(mean_axis).clamp_min(1e-6)
        order = torch.argsort(values @ mean_axis)
        if centroid_count == 1:
            init_indices = order[-1:]
        else:
            init_offsets = torch.linspace(0, values.shape[0] - 1, centroid_count).round().long()
            init_indices = order[init_offsets]
        centroids = values[init_indices].clone()
        positions = torch.arange(centroid_count, dtype=torch.float32)
        sample_order = order.flip(0)
        for step in range(steps):
            lr = initial_lr * (1.0 - (step / max(steps, 1)))
            lr = max(lr, initial_lr * 0.1)
            sigma = max(0.5, initial_sigma * (1.0 - (step / max(steps, 1))))
            for sample_index in sample_order:
                sample = values[int(sample_index)]
                distances = torch.sum((centroids - sample) ** 2, dim=1)
                winner = int(torch.argmin(distances).item())
                neighborhood = torch.exp(-((positions - float(winner)) ** 2) / (2.0 * sigma * sigma)).unsqueeze(1)
                centroids = centroids + lr * neighborhood * (sample - centroids)
        assignments = torch.argmin(torch.cdist(values, centroids), dim=1)
        counts = torch.bincount(assignments, minlength=centroid_count).float()
        energy = torch.linalg.vector_norm(centroids, dim=1) * torch.sqrt(counts + 1.0)
        selected = torch.argsort(energy, descending=True)[:components]
        return centroids[selected]

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
        elif method in {"som", "som_centroids", "som_refusal_centroids"}:
            residuals = harmful_stack.float() - benign_stack.float().mean(dim=0, keepdim=True)
            centroids = som_residual_centroids(residuals, max(1, components - 1), mean_direction)
            direction = torch.vstack([mean_direction.float(), centroids]) if components > 1 else mean_direction
        elif method in {
            "concept_cone",
            "source_anchored_concept_cone",
            "benign_orthogonal_concept_cone",
        }:
            direction = extract_concept_cone_direction(
                harmful_stack,
                benign_stack,
                components=components,
                benign_subspace_components=int(activation.get("benign_subspace_components", 2)),
                project_mean_out=bool(activation.get("concept_cone_project_mean_out", True)),
                whiten=bool(activation.get("concept_cone_whiten", True)),
            )
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
    sorted_target_layers = sorted(target_layers)
    every_layer = _progress_every(len(sorted_target_layers), "MODEL_FORGE_NATIVE_LAYER_PROGRESS_EVERY")
    for offset, layer_index in enumerate(sorted_target_layers, start=1):
        if offset == 1 or offset == len(sorted_target_layers) or offset % every_layer == 0:
            console.print(f"[dim]native direction extraction: layer {layer_index} ({offset}/{len(sorted_target_layers)})[/dim]")
        harmful_stack = maybe_winsorize(torch.stack(harmful_vectors[layer_index]))
        benign_stack = maybe_winsorize(torch.stack(benign_vectors[layer_index]))
        harmful_mean = harmful_stack.mean(dim=0)
        benign_mean = benign_stack.mean(dim=0)
        direction = extract_direction(harmful_stack, benign_stack)
        direction = normalize_direction_basis(direction)
        sra_components = int(activation.get("sra_preservation_components", 0) or 0)
        if sra_components or bool(activation.get("sra_include_benign_mean", False)):
            sra_basis = build_sra_preservation_basis(
                benign_stack,
                components=sra_components,
                include_benign_mean=bool(activation.get("sra_include_benign_mean", True)),
            )
            if sra_basis.numel() > 0:
                sra_preservation_bases[layer_index] = sra_basis
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
            "sra_preservation_bases": sra_preservation_bases,
            "direction_method": activation["token_position"],
            "direction_extraction": activation["direction_extraction"],
            "direction_components": int(activation.get("direction_components", 1)),
            "sra_preservation_components": int(activation.get("sra_preservation_components", 0) or 0),
            "sra_include_benign_mean": bool(activation.get("sra_include_benign_mean", False)),
            "som_neurons": activation.get("som_neurons"),
            "som_steps": activation.get("som_steps"),
            "som_learning_rate": activation.get("som_learning_rate"),
            "som_neighborhood": activation.get("som_neighborhood"),
            "benign_subspace_components": activation.get("benign_subspace_components"),
            "concept_cone_project_mean_out": activation.get("concept_cone_project_mean_out"),
            "concept_cone_whiten": activation.get("concept_cone_whiten"),
            "direction_source_layer": source_layer_index,
            "replicate_source_direction": bool(activation.get("replicate_source_direction", False)),
            "use_chat_template": bool(activation.get("use_chat_template", False)),
            "harmful_assistant_prefix": activation.get("harmful_assistant_prefix"),
            "benign_assistant_prefix": activation.get("benign_assistant_prefix"),
            "assistant_prefix_template": activation.get("assistant_prefix_template"),
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


def projection_target_layers(weight_map: dict[str, str], edit: dict[str, Any]) -> list[int]:
    layers = {
        layer
        for name in weight_map
        if is_projection_target(name, edit)
        for layer in [language_layer_index(name)]
        if layer is not None
    }
    return sorted(layers)


def missing_target_tensor_layers(weight_map: dict[str, str], edit: dict[str, Any]) -> list[int]:
    target_layers = set(projection_target_layers(weight_map, edit))
    return [layer for layer in configured_target_layers(edit) if layer not in target_layers]


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
            "sra_preservation_bases": {},
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
        "sra_preservation_bases": {
            int(layer): vector.float()
            for layer, vector in raw.get("sra_preservation_bases", raw.get("preservation_bases", {})).items()
        },
        "source_path": str(path),
        "format": raw.get("format", "direction_artifact_v1"),
    }


def write_selective_direction_artifact(
    input_path: Path,
    output_path: Path,
    *,
    layer_start: int | None = None,
    layer_end: int | None = None,
    top_k: int = 8,
    min_score: float | None = None,
    required_layers: list[int] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    raw = torch.load(input_path, map_location="cpu")
    artifact = load_direction_artifact(input_path)
    directions = artifact["refusal_directions"]
    harmful_means = artifact.get("harmful_means", {})
    benign_means = artifact.get("benign_means", {})
    if not harmful_means or not benign_means:
        raise SystemExit("selective projection requires harmful_means and benign_means in the direction artifact")

    candidates: list[dict[str, Any]] = []
    for layer in sorted(directions):
        if layer_start is not None and layer < layer_start:
            continue
        if layer_end is not None and layer > layer_end:
            continue
        harmful = harmful_means.get(layer)
        benign = benign_means.get(layer)
        if harmful is None or benign is None:
            continue
        delta = harmful.float() - benign.float()
        delta_norm = torch.linalg.vector_norm(delta).clamp_min(1e-8)
        direction = directions[layer].float()
        if direction.ndim == 1:
            direction_basis = direction.unsqueeze(0)
        elif direction.ndim == 2:
            direction_basis = direction
        else:
            continue
        direction_basis = direction_basis / torch.linalg.vector_norm(direction_basis, dim=1, keepdim=True).clamp_min(1e-8)
        projection_energy = torch.linalg.vector_norm(direction_basis @ delta).item()
        alignment = float(projection_energy / float(delta_norm))
        separation = float(delta_norm)
        benign_norm = float(torch.linalg.vector_norm(benign.float()).clamp_min(1e-8))
        separation_ratio = float(separation / benign_norm)
        score = float(alignment * separation_ratio)
        candidates.append({
            "layer": layer,
            "score": score,
            "alignment": alignment,
            "separation_norm": separation,
            "benign_norm": benign_norm,
            "separation_ratio": separation_ratio,
            "component_count": int(direction_basis.shape[0]),
        })

    if not candidates:
        raise SystemExit("selective projection found no candidate layers with directions and means")

    min_score_value = float(min_score) if min_score is not None else None
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected_layers: list[int] = []
    selection_limit = max(1, int(top_k))
    for layer in required_layers or []:
        if any(item["layer"] == layer for item in candidates) and layer not in selected_layers:
            selected_layers.append(int(layer))
    for item in ranked:
        if len(selected_layers) >= selection_limit:
            break
        if min_score_value is not None and item["score"] < min_score_value:
            continue
        if item["layer"] not in selected_layers:
            selected_layers.append(int(item["layer"]))
    if not selected_layers:
        raise SystemExit("selective projection selected no layers after applying top_k/min_score")

    selected = set(selected_layers)
    output_payload = dict(raw)
    output_payload["refusal_directions"] = {layer: value for layer, value in directions.items() if layer in selected}
    output_payload["harmful_means"] = {layer: value for layer, value in harmful_means.items() if layer in selected}
    output_payload["benign_means"] = {layer: value for layer, value in benign_means.items() if layer in selected}
    output_payload["selective_projection"] = {
        "schema_version": "model_forge.selective_projection.v1",
        "source_path": str(input_path),
        "selected_layers": selected_layers,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "top_k": int(top_k),
        "min_score": min_score_value,
        "required_layers": required_layers or [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)

    report = {
        "schema_version": "model_forge.selective_projection.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(input_path),
        "output_path": str(output_path),
        "layer_start": layer_start,
        "layer_end": layer_end,
        "top_k": int(top_k),
        "min_score": min_score_value,
        "required_layers": required_layers or [],
        "selected_layers": selected_layers,
        "ranked_layers": ranked,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


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
    elif mode in {"sra", "sra_cleaned", "surgical_refusal_ablation", "preservation_cleaned"}:
        basis = artifact.get("sra_preservation_bases", {}).get(layer)
        if basis is None or basis.numel() == 0:
            if edit.get("sra_require_preservation_basis", True):
                raise SystemExit(
                    f"direction_transform={mode} requires sra_preservation_bases for layer {layer}; "
                    "set activation_collection.sra_preservation_components or sra_include_benign_mean"
                )
        else:
            direction = _project_out_rowspace(direction, _orthonormal_rows(basis.float()))
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
                "execution": "guarded_checkpoint",
                "container_image": "model-forge-obliteratus:latest",
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
            "abliterix": {
                "install": "pip install abliterix",
                "method_family": "abliterix",
                "execution": "search_only",
                "vector_method": "sra",
                "steering_mode": "lora",
                "n_trials": 24,
                "n_warmup_trials": 8,
                "checkpoint_dir": "abliterix_checkpoints",
                "llm_judge": False,
                "notes": (
                    "Abliterix is tracked as a practical SOTA backend candidate. "
                    "model-forge runs it in non-interactive search-only mode first; "
                    "checkpoint export requires a separately reviewed runner."
                ),
            },
            "apostate": {
                "install": "git clone https://github.com/heterodoxin/apostate",
                "method_family": "activation_preserving_refusal_ablation",
                "execution": "plan_only",
                "notes": (
                    "Apostate-style baked checkpoint edits are tracked as an "
                    "experimental preservation-direction method. Add a guarded "
                    "runner before executing large model edits."
                ),
            },
            "sra": {
                "install": "external SRA implementation required",
                "method_family": "surgical_refusal_ablation",
                "execution": "plan_only",
                "notes": (
                    "Surgical Refusal Ablation cleans refusal directions against "
                    "capability/style atoms before editing. Treat as a method-shift "
                    "candidate when single-direction or sequential LoRA repair stalls."
                ),
            },
            "optimal_transport": {
                "install": "native model-forge checkpoint runner",
                "method_family": "optimal_transport_refusal_ablation",
                "execution": "plan_only",
                "notes": (
                    "Optimal-transport refusal ablation maps harmful activations "
                    "toward harmless activations instead of deleting one direction. "
                    "Use checkpoint_export only with a narrow, source-relative "
                    "prompt materialization and a targeted gate."
                ),
            },
            "norm_preserving_projection": {
                "install": "native model-forge checkpoint runner",
                "method_family": "norm_preserving_projected_abliteration",
                "execution": "plan_only",
                "notes": (
                    "Native projected/biprojected checkpoint edit with optional "
                    "row-norm preservation. Use this for MPOA/NPBA-style "
                    "method-shift recipes when an activation-direction edit should "
                    "preserve row norms and avoid capability-heavy benign subspaces."
                ),
            },
            "som_projection": {
                "install": "native model-forge checkpoint runner",
                "method_family": "som_multidirectional_refusal_projection",
                "execution": "plan_only",
                "notes": (
                    "Native SOM-style multi-centroid refusal projection. Use this "
                    "when one global refusal direction is too blunt: the extractor "
                    "learns bounded refusal-residual centroids, combines them with "
                    "the global mean direction, and exports only through the normal "
                    "source-relative checkpoint/eval gate."
                ),
            },
            "selective_projection": {
                "install": "native model-forge checkpoint runner",
                "method_family": "selective_layer_refusal_projection",
                "execution": "plan_only",
                "notes": (
                    "Native selective-layer checkpoint edit. It collects normal "
                    "source-relative directions, scores layers by refusal-vs-benign "
                    "activation separation explained by the direction basis, filters "
                    "to the highest-signal layers, and exports through the standard "
                    "norm-preserving projection path."
                ),
            },
            "concept_cone_projection": {
                "install": "native model-forge checkpoint runner",
                "method_family": "source_anchored_concept_cone_projection",
                "execution": "plan_only",
                "notes": (
                    "Native source-anchored concept-cone checkpoint edit. It "
                    "extracts a multi-direction harmful/refusal cone after "
                    "projecting out dominant benign capability/style variation, "
                    "then exports through the same selective, norm-preserving, "
                    "source-relative checkpoint path."
                ),
            },
            "qwen_scope_sae": {
                "install": "native model-forge checkpoint runner plus a compatible SAE dictionary",
                "method_family": "qwen_scope_sae_dictionary_projection",
                "execution": "plan_only",
                "notes": (
                    "SAE dictionary-constrained projection: collect the same source-relative "
                    "refusal residual signal, project it onto aligned SAE decoder features, "
                    "then bake a normal checkpoint through the standard projection exporter."
                ),
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
    selected_backend = backends[selected]
    source = resolve_model_source(model_cfg["local_dir"] or model_cfg["source"])
    output_dir = resolve_repo_path(selected_backend.get("output_dir") or sota.get("output_dir") or model_cfg["output_dir"])
    work_dir = resolve_repo_path(sota.get("work_dir", Path(config.get("artifacts_dir", "artifacts/abliteration/sota")) / "sota"))
    return {
        "name": plan["name"],
        "backend": selected,
        "config_path": str(config_path),
        "source_model": source,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "backend_config": selected_backend,
        "install": selected_backend.get("install"),
        "license_notice": sota.get("license_notice"),
        "all_backends": backends,
        "model_forge_plan": plan,
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


def materialize_model_forge_obliteratus_prompts(plan: dict[str, Any], backend: dict[str, Any], work_dir: Path) -> Path | None:
    spec = backend.get("model_forge_prompt_datasets")
    if spec is None:
        return None
    dataset_root = work_dir / "model_forge_obliteratus_prompts"
    defaults = {
        "harmful_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
        "harmless_buckets": [
            "capability_preservation_challenge",
            "normal_use_regression",
            "agentic_tool_use_json",
            "refusal_paired_boundary",
        ],
        "jailbreak_buckets": [],
    }
    if isinstance(spec, dict):
        defaults.update({key: value for key, value in spec.items() if value is not None})

    section_specs = {
        "harmful_prompts": {
            "prefix": "harmful",
            "buckets": defaults.get("harmful_buckets", defaults.get("bad_train_buckets", [])),
            "case_ids": normalize_case_id_filter(defaults.get("harmful_case_ids", defaults.get("bad_train_case_ids"))),
            "should_refuse": True,
        },
        "harmless_prompts": {
            "prefix": "harmless",
            "buckets": defaults.get("harmless_buckets", defaults.get("good_train_buckets", [])),
            "case_ids": normalize_case_id_filter(defaults.get("harmless_case_ids", defaults.get("good_train_case_ids"))),
            "should_refuse": False,
        },
        "jailbreak_prompts": {
            "prefix": "jailbreak",
            "buckets": defaults.get("jailbreak_buckets", []),
            "case_ids": normalize_case_id_filter(defaults.get("jailbreak_case_ids")),
            "should_refuse": True,
        },
    }
    payload: dict[str, Any] = {}
    manifest: dict[str, Any] = {"source": "model_forge_eval_prompts", "sections": {}}
    for target_key, section in section_specs.items():
        prefix = str(section["prefix"])
        buckets = list(section["buckets"] or [])
        prompts = (
            prompts_for_buckets(
                buckets,
                should_refuse=bool(section["should_refuse"]),
                case_ids=section["case_ids"],
            )
            if buckets
            else []
        )
        response_summary = None
        if isinstance(spec, dict):
            response_prompts, response_summary = response_conditioned_section_prompts(spec, prefix)
            prompts = [*prompts, *response_prompts]
            extra_prompts = normalize_extra_prompts(spec.get(f"{prefix}_extra_prompts"), prefix=prefix)
            prompts = [*prompts, *extra_prompts]
            prompts, variant_summary = apply_prompt_variants(
                prompts,
                spec.get(f"{prefix}_prompt_variants"),
                prefix=prefix,
            )
        else:
            extra_prompts = []
            variant_summary = None
        unique_prompts = list(dict.fromkeys(str(prompt).strip() for prompt in prompts if str(prompt).strip()))
        if not unique_prompts:
            continue
        payload[target_key] = unique_prompts
        manifest["sections"][target_key] = {
            "count": len(unique_prompts),
            "buckets": buckets,
            "case_ids": sorted(section["case_ids"]) if section["case_ids"] is not None else None,
        }
        if response_summary is not None:
            manifest["sections"][target_key]["response_conditioned"] = response_summary
        if extra_prompts:
            manifest["sections"][target_key]["extra_prompts"] = {"count": len(extra_prompts)}
        if variant_summary is not None:
            manifest["sections"][target_key]["prompt_variants"] = variant_summary
    if not payload:
        return None
    if defaults.get("balance_prompt_pairs", True) and payload.get("harmful_prompts") and payload.get("harmless_prompts"):
        harmful = payload["harmful_prompts"]
        harmless = payload["harmless_prompts"]
        target_len = max(len(harmful), len(harmless))
        if len(harmful) != len(harmless):
            payload["harmful_prompts"] = [harmful[index % len(harmful)] for index in range(target_len)]
            payload["harmless_prompts"] = [harmless[index % len(harmless)] for index in range(target_len)]
            manifest["balanced_prompt_pairs"] = {
                "enabled": True,
                "harmful_before": len(harmful),
                "harmless_before": len(harmless),
                "paired_count": target_len,
            }
    dataset_root.mkdir(parents=True, exist_ok=True)
    prompt_path = dataset_root / "prompts.json"
    prompt_path.write_text(json.dumps(payload, indent=2) + "\n")
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return prompt_path


def write_obliteratus_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_payload_path = materialize_model_forge_obliteratus_prompts(plan, backend, work_dir)
    runner = work_dir / "run_obliteratus.py"
    summary_path = work_dir / "model_forge_sota_obliteratus.json"
    pipeline_kwargs = {
        key: value for key, value in backend.items()
        if key in OBLITERATUS_PIPELINE_FIELDS and value is not None
    }
    pipeline_kwargs.setdefault("large_model_mode", bool(backend.get("large_model_mode", True)))
    pipeline_kwargs.setdefault("trust_remote_code", bool(backend.get("trust_remote_code", True)))
    pipeline_kwargs.setdefault("dtype", str(backend.get("dtype", "bfloat16")))
    prompt_payload_literal = None if prompt_payload_path is None else str(prompt_payload_path)
    preserve_source_tokenizer = bool(backend.get("preserve_source_tokenizer", True))
    key_remap_config = backend.get("post_export_key_remap") or {}
    source_tether_config = backend.get("source_tether") or {}
    script = f'''from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

model_name = {plan["source_model"]!r}
output_dir = {plan["output_dir"]!r}
work_dir = Path({str(work_dir)!r})
summary_path = Path({str(summary_path)!r})
method = {backend.get("method", "advanced")!r}
max_seq_length = {int(backend.get("max_seq_length", 512))}
prompt_payload_path = {prompt_payload_literal!r}
pipeline_kwargs = {json.dumps(pipeline_kwargs, indent=4)!r}
preserve_source_tokenizer = {preserve_source_tokenizer!r}
key_remap_config = {json.dumps(key_remap_config, indent=4)!r}
source_tether_config = {json.dumps(source_tether_config, indent=4)!r}
streaming_rebirth_config = {json.dumps(backend.get("streaming_rebirth") or {}, indent=4)!r}
lora_adapter_export_config = {json.dumps(backend.get("lora_adapter_export") or {}, indent=4)!r}

if {not bool(backend.get("telemetry", False))!r}:
    os.environ.setdefault("OBLITERATUS_TELEMETRY", "0")


def available_ram_fraction() -> float:
    try:
        values = {{}}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw = line.partition(":")
                fields = raw.strip().split()
                if fields:
                    values[key] = float(fields[0])
        available = values.get("MemAvailable")
        total = values.get("MemTotal")
        if available is None or total in (None, 0):
            return 1.0
        return available / total
    except OSError:
        return 1.0


def guard_system_health() -> None:
    min_ram = float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram:
        raise SystemExit(f"available RAM fraction {{ram_fraction:.3f}} is below guard {{min_ram:.3f}}")
    min_disk = float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15"))
    output_parent = Path(output_dir).parent
    usage = shutil.disk_usage(output_parent if output_parent.exists() else work_dir)
    free_fraction = usage.free / usage.total
    if free_fraction < min_disk:
        raise SystemExit(f"free disk fraction {{free_fraction:.3f}} is below guard {{min_disk:.3f}}")


def expand_config_path(value: str) -> str:
    return str(Path(str(value)).expanduser())


def serializable_result(value):
    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
        return value
    return repr(value)


def load_model_forge_prompts() -> dict:
    if not prompt_payload_path:
        return {{}}
    path = Path(prompt_payload_path)
    if not path.exists():
        raise SystemExit(f"missing OBLITERATUS prompt payload: {{path}}")
    return json.loads(path.read_text(encoding="utf-8"))


def restore_source_tokenizer_metadata() -> list[str]:
    if not preserve_source_tokenizer:
        return []
    copied = []
    source_dir = Path(model_name)
    target_dir = Path(output_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = source_dir / name
        if not source.exists():
            continue
        target = target_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(name)
    return copied


def _prefix_map_args(raw) -> list[str]:
    maps = raw or []
    if isinstance(maps, str):
        maps = [maps]
    args = []
    for item in maps:
        if isinstance(item, str):
            args.append(item)
        elif isinstance(item, dict):
            source = item.get("from") or item.get("source")
            target = item.get("to") or item.get("target")
            if source is None or target is None:
                raise SystemExit(f"invalid key remap entry: {{item!r}}")
            args.append(f"{{source}}={{target}}")
        else:
            raise SystemExit(f"invalid key remap entry: {{item!r}}")
    return args


def run_post_export_key_remap() -> dict | None:
    cfg = dict(json.loads(key_remap_config or "{{}}"))
    if not cfg or cfg.get("enabled") is False:
        return None
    map_prefixes = _prefix_map_args(cfg.get("map_prefixes") or cfg.get("map_prefix"))
    if not map_prefixes:
        raise SystemExit("post_export_key_remap requires map_prefixes")
    reference_dir = expand_config_path(cfg.get("reference_dir") or model_name)
    checkpoint_dir = expand_config_path(output_dir)
    command = [
        sys.executable,
        "scripts/remap_safetensors_checkpoint.py",
        "--checkpoint-dir",
        checkpoint_dir,
        "--reference-dir",
        reference_dir,
        "--min-available-ram-fraction",
        str(cfg.get("min_available_ram_fraction", os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))),
        "--min-free-disk-fraction",
        str(cfg.get("min_free_disk_fraction", os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10"))),
    ]
    for item in map_prefixes:
        command.extend(["--map-prefix", item])
    if cfg.get("verify_reference_keys", True):
        command.append("--verify-reference-keys")
    for name in cfg.get("preserve_files") or []:
        command.extend(["--preserve-file", str(name)])
    if cfg.get("skip_preserve_defaults"):
        command.append("--skip-preserve-defaults")
    print("[model-forge] post-export key remap:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return {{"command": command, "reference_dir": reference_dir, "map_prefixes": map_prefixes}}


def run_source_tether() -> dict | None:
    cfg = dict(json.loads(source_tether_config or "{{}}"))
    if not cfg or cfg.get("enabled") is False:
        return None
    source_dir = expand_config_path(cfg.get("source_dir") or cfg.get("source") or model_name)
    candidate_dir = expand_config_path(output_dir)
    command = [
        sys.executable,
        "scripts/source_tether_safetensors_checkpoint.py",
        "--source",
        source_dir,
        "--candidate",
        candidate_dir,
        "--output-dir",
        candidate_dir,
        "--alpha",
        str(cfg.get("alpha", 0.895)),
        "--restore-top-k",
        str(int(cfg.get("restore_top_k", 0))),
        "--drift-metric",
        str(cfg.get("drift_metric", "mean_abs_delta")),
        "--preserve-from",
        str(cfg.get("preserve_from", "source")),
        "--min-available-ram-fraction",
        str(cfg.get("min_available_ram_fraction", os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))),
        "--min-free-disk-fraction",
        str(cfg.get("min_free_disk_fraction", os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10"))),
        "--in-place",
    ]
    if cfg.get("include_regex"):
        command.extend(["--include-regex", str(cfg["include_regex"])])
    if cfg.get("exclude_regex"):
        command.extend(["--exclude-regex", str(cfg["exclude_regex"])])
    print("[model-forge] source tether:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return {{"command": command, "source_dir": source_dir}}


def run_lora_adapter_export() -> dict | None:
    cfg = dict(json.loads(lora_adapter_export_config or "{{}}"))
    if not cfg or cfg.get("enabled") is False:
        return None
    input_dir = expand_config_path(cfg.get("input_dir") or output_dir)
    peft_output_dir = expand_config_path(cfg.get("peft_output_dir") or cfg.get("output_dir") or output_dir)
    base_model = expand_config_path(cfg.get("base_model_name_or_path") or cfg.get("base_model") or model_name)
    command = [
        sys.executable,
        "scripts/convert_obliteratus_lora_to_peft.py",
        "--input-dir",
        input_dir,
        "--output-dir",
        peft_output_dir,
        "--base-model",
        base_model,
        "--key-template",
        str(cfg.get("key_template", "base_model.model.model.layers.{{layer}}.{{module}}.{{weight}}")),
        "--attn-module-name",
        str(cfg.get("attn_module_name", "self_attn")),
        "--ffn-module-name",
        str(cfg.get("ffn_module_name", "mlp")),
    ]
    if cfg.get("lora_alpha") is not None:
        command.extend(["--lora-alpha", str(cfg["lora_alpha"])])
    if cfg.get("copy_metadata") is False:
        command.append("--no-copy-metadata")
    print("[model-forge] OBLITERATUS LoRA PEFT export:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return {{"command": command, "output_dir": peft_output_dir, "base_model": base_model}}


def _size_gb_to_bytes(value, default_gb: float = 1.0) -> int:
    try:
        gb = float(value)
    except (TypeError, ValueError):
        gb = default_gb
    return max(1, int(gb * 1_000_000_000))


def _guard_streaming_rebirth(output_path: Path, transient_bytes: int) -> dict:
    cfg = dict(json.loads(streaming_rebirth_config or "{{}}"))
    min_ram = float(cfg.get(
        "min_available_ram_fraction",
        os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"),
    ))
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram:
        raise SystemExit(f"available RAM fraction {{ram_fraction:.3f}} is below streaming rebirth guard {{min_ram:.3f}}")
    min_disk = float(cfg.get(
        "min_free_disk_fraction",
        os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.10"),
    ))
    output_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_path)
    projected = (usage.free - transient_bytes) / usage.total
    if projected < min_disk:
        raise SystemExit(
            "free disk fraction would breach guard during streaming rebirth: "
            f"{{projected:.3f}} < {{min_disk:.3f}}"
        )
    return {{
        "available_ram_fraction": round(ram_fraction, 4),
        "free_disk_fraction_before": round(usage.free / usage.total, 4),
        "projected_free_disk_fraction_after": round(projected, 4),
        "transient_bytes": int(transient_bytes),
    }}


def _load_offloaded_tensor(key: str, placeholder):
    offload_dir = getattr(model_forge_streaming_rebirth_pipeline.handle, "_offload_dir", None)
    if not offload_dir:
        raise RuntimeError(f"cannot stream-save meta tensor {{key!r}} without an offload directory")
    from safetensors.torch import load_file

    base = Path(offload_dir)
    safetensors_file = base / f"{{key}}.safetensors"
    if safetensors_file.exists():
        payload = load_file(str(safetensors_file))
        return payload[key] if key in payload else next(iter(payload.values()))
    dat_file = base / f"{{key}}.dat"
    if dat_file.exists():
        import numpy as np
        import torch

        dtype = placeholder.dtype
        shape = placeholder.shape
        arr = np.fromfile(str(dat_file), dtype=torch.tensor([], dtype=dtype).numpy().dtype)
        return torch.from_numpy(arr).reshape(shape)
    raise RuntimeError(f"cannot find offloaded tensor data for {{key!r}} under {{offload_dir!r}}")


model_forge_streaming_rebirth_pipeline = None


def install_lora_ablation_device_patch() -> bool:
    kwargs = dict(json.loads(pipeline_kwargs))
    if not kwargs.get("use_lora_ablation"):
        return False
    cfg = dict(json.loads(lora_adapter_export_config or "{{}}"))
    target_sections = set(str(item) for item in (cfg.get("target_sections") or []))
    target_names = set(str(item) for item in (cfg.get("target_weight_names") or cfg.get("target_modules") or []))
    target_layers = set(int(item) for item in (cfg.get("target_layer_indices") or []))
    max_target_layers_raw = cfg.get("max_target_layers")
    max_target_layers = int(max_target_layers_raw) if max_target_layers_raw is not None else 0
    import inspect
    import obliteratus.abliterate as obliteratus_abliterate
    import obliteratus.lora_ablation as obliteratus_lora_ablation

    try:
        source = inspect.getsource(obliteratus_lora_ablation.compute_lora_adapters)
    except OSError:
        return False
    patched = source.replace(
        "d = D[di]  # (hidden_dim,)",
        "d = D[di].to(W.device)  # (hidden_dim,)",
    ).replace(
        "adapters[key] = (lora_B.half(), lora_A.half())",
        "adapters[key] = (lora_B.detach().cpu().half(), lora_A.detach().cpu().half())",
    ).replace(
        "    for idx in pipeline._strong_layers:\\n",
        (
            "    model_forge_selected_lora_layers = []\\n"
            "    for idx in pipeline._strong_layers:\\n"
            "        if MODEL_FORGE_LORA_TARGET_LAYERS and idx not in MODEL_FORGE_LORA_TARGET_LAYERS:\\n"
            "            continue\\n"
            "        if MODEL_FORGE_LORA_MAX_TARGET_LAYERS and len(model_forge_selected_lora_layers) >= MODEL_FORGE_LORA_MAX_TARGET_LAYERS:\\n"
            "            continue\\n"
            "        model_forge_selected_lora_layers.append(idx)\\n"
        ),
    ).replace(
        "        for module_label, module, candidate_names in targets:\\n",
        (
            "        for module_label, module, candidate_names in targets:\\n"
            "            if MODEL_FORGE_LORA_TARGET_SECTIONS and module_label not in MODEL_FORGE_LORA_TARGET_SECTIONS:\\n"
            "                continue\\n"
        ),
    ).replace(
        "            for name in candidate_names:\\n",
        (
            "            for name in candidate_names:\\n"
            "                if MODEL_FORGE_LORA_TARGET_NAMES and name not in MODEL_FORGE_LORA_TARGET_NAMES:\\n"
            "                    continue\\n"
        ),
    )
    if patched == source:
        return False
    obliteratus_lora_ablation.MODEL_FORGE_LORA_TARGET_SECTIONS = target_sections
    obliteratus_lora_ablation.MODEL_FORGE_LORA_TARGET_NAMES = target_names
    obliteratus_lora_ablation.MODEL_FORGE_LORA_TARGET_LAYERS = target_layers
    obliteratus_lora_ablation.MODEL_FORGE_LORA_MAX_TARGET_LAYERS = max_target_layers
    namespace: dict[str, object] = {{}}
    exec(patched, obliteratus_lora_ablation.__dict__, namespace)
    patched_compute = namespace["compute_lora_adapters"]
    obliteratus_lora_ablation.compute_lora_adapters = patched_compute
    if hasattr(obliteratus_abliterate, "compute_lora_adapters"):
        obliteratus_abliterate.compute_lora_adapters = patched_compute
    return True


def install_activation_layer_filter_patch(AbliterationPipeline) -> bool:
    cfg = dict(json.loads(lora_adapter_export_config or "{{}}"))
    target_layers = set(int(item) for item in (cfg.get("target_layer_indices") or []))
    if not target_layers:
        return False
    import inspect
    import textwrap
    import obliteratus.abliterate as obliteratus_abliterate

    try:
        source = textwrap.dedent(inspect.getsource(AbliterationPipeline._collect_activations))
    except OSError:
        return False
    patched = source.replace(
        "    n_layers = len(layer_modules)\\n",
        (
            "    n_layers = len(layer_modules)\\n"
            "    target_layers = set(globals().get('MODEL_FORGE_ACTIVATION_TARGET_LAYERS', set()))\\n"
        ),
    ).replace(
        "    for idx in range(n_layers):\\n"
        "        hooks.append(layer_modules[idx].register_forward_hook(make_hook(idx)))\\n",
        (
            "    for idx in range(n_layers):\\n"
            "        if target_layers and idx not in target_layers:\\n"
            "            continue\\n"
            "        hooks.append(layer_modules[idx].register_forward_hook(make_hook(idx)))\\n"
        ),
    )
    if patched == source:
        return False
    obliteratus_abliterate.MODEL_FORGE_ACTIVATION_TARGET_LAYERS = target_layers
    namespace: dict[str, object] = {{}}
    exec(patched, obliteratus_abliterate.__dict__, namespace)
    AbliterationPipeline._collect_activations = namespace["_collect_activations"]
    return True


def install_adapter_only_rebirth(AbliterationPipeline) -> bool:
    cfg = dict(json.loads(lora_adapter_export_config or "{{}}"))
    enabled = bool(cfg.get("enabled") and cfg.get("adapter_only", True))
    if not enabled:
        return False
    kwargs = dict(json.loads(pipeline_kwargs))
    if not kwargs.get("use_lora_ablation"):
        raise SystemExit("lora_adapter_export.adapter_only requires use_lora_ablation=true")

    def model_forge_adapter_only_rebirth(self):
        from obliteratus.lora_ablation import save_lora_adapters

        if not getattr(self, "_lora_adapters", None):
            raise RuntimeError("adapter-only OBLITERATUS rebirth requested but no LoRA adapters were produced")
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._emit("rebirth", "running", f"Saving adapter-only OBLITERATUS LoRA to {{output_path}}...")
        start = time.time()
        adapter_path = save_lora_adapters(self._lora_adapters, output_path)
        if hasattr(self.handle.model, "config"):
            self.handle.model.config.save_pretrained(output_path)
        if getattr(self.handle.model, "generation_config", None) is not None:
            try:
                self.handle.model.generation_config.save_pretrained(output_path)
            except Exception:
                pass
        self.handle.tokenizer.save_pretrained(output_path)
        metadata_payload = self._build_metadata()
        (output_path / "abliteration_metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )
        peft_export = run_lora_adapter_export()
        cleanup = getattr(self, "_cleanup_offload_dir", None)
        if callable(cleanup):
            cleanup()
        manifest = {{
            "schema_version": "model_forge.obliteratus_adapter_only_rebirth.v1",
            "output_dir": str(output_path),
            "adapter_path": str(adapter_path),
            "adapter_count": len(self._lora_adapters),
            "peft_export": peft_export,
            "duration_seconds": round(time.time() - start, 3),
        }}
        (output_path / "model_forge_obliteratus_adapter_only_rebirth.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        self._emit("rebirth", "done", f"Saved adapter-only OBLITERATUS LoRA to {{output_path}}", duration=time.time() - start)
        return output_path

    AbliterationPipeline._rebirth = model_forge_adapter_only_rebirth
    return True


def install_streaming_rebirth(AbliterationPipeline) -> bool:
    cfg = dict(json.loads(streaming_rebirth_config or "{{}}"))
    enabled = cfg.get("enabled")
    if enabled is None:
        enabled = False
    if not enabled:
        return False

    def model_forge_streaming_rebirth(self):
        global model_forge_streaming_rebirth_pipeline
        model_forge_streaming_rebirth_pipeline = self
        if getattr(self, "push_to_hub", None):
            raise RuntimeError("model-forge streaming OBLITERATUS rebirth does not push to Hub; upload after local audits pass")
        from safetensors.torch import save_file

        dest = str(self.output_dir)
        self._emit("rebirth", "running", f"Streaming save to {{dest}}...")
        start = time.time()
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shard_limit = _size_gb_to_bytes(cfg.get("max_shard_size_gb"), default_gb=1.0)
        state_dict = self.handle.model.state_dict()
        total_bytes = sum(int(t.numel()) * int(t.element_size()) for t in state_dict.values())
        self.log(
            f"Streaming state dict: {{len(state_dict)}} tensors, "
            f"{{total_bytes / 1e9:.1f}} GB, shard_limit={{shard_limit / 1e9:.1f}} GB"
        )
        _guard_streaming_rebirth(output_path, min(shard_limit, max(total_bytes, 1)))
        weight_map = {{}}
        shard_records = []
        shard_tensors = {{}}
        shard_bytes = 0
        shard_index = 0
        metadata = {{"format": "pt"}}

        def flush_shard():
            nonlocal shard_tensors, shard_bytes, shard_index
            if not shard_tensors:
                return
            shard_index += 1
            shard_name = f"model-{{shard_index:05d}}.safetensors"
            tmp_path = output_path / f".{{shard_name}}.tmp"
            _guard_streaming_rebirth(output_path, shard_bytes)
            self.log(
                f"  writing shard {{shard_index}}: {{len(shard_tensors)}} tensors, "
                f"{{shard_bytes / 1e9:.2f}} GB"
            )
            save_file(shard_tensors, str(tmp_path), metadata=metadata)
            final_path = output_path / shard_name
            tmp_path.replace(final_path)
            for tensor_name in shard_tensors:
                weight_map[tensor_name] = shard_name
            shard_records.append({{"name": shard_name, "tensor_count": len(shard_tensors), "bytes": shard_bytes}})
            shard_tensors = {{}}
            shard_bytes = 0
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        for name, tensor in state_dict.items():
            if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                tensor = _load_offloaded_tensor(name, tensor)
            tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
            if shard_tensors and shard_bytes + tensor_bytes > shard_limit:
                flush_shard()
            cpu_tensor = tensor.detach().to("cpu").contiguous().clone()
            shard_tensors[name] = cpu_tensor
            shard_bytes += tensor_bytes
            if shard_bytes >= shard_limit:
                flush_shard()
        flush_shard()
        if not shard_records:
            raise RuntimeError("streaming rebirth produced no safetensors shards")

        shard_count = len(shard_records)
        rename_map = {{}}
        for index, record in enumerate(shard_records, start=1):
            old_name = record["name"]
            new_name = f"model-{{index:05d}}-of-{{shard_count:05d}}.safetensors"
            if old_name != new_name:
                (output_path / old_name).replace(output_path / new_name)
            rename_map[old_name] = new_name
            record["name"] = new_name
        weight_map = {{name: rename_map[shard] for name, shard in weight_map.items()}}
        index_payload = {{
            "metadata": {{"total_size": total_bytes}},
            "weight_map": weight_map,
        }}
        (output_path / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )

        if hasattr(self.handle.model, "config"):
            self.handle.model.config.save_pretrained(output_path)
        if getattr(self.handle.model, "generation_config", None) is not None:
            try:
                self.handle.model.generation_config.save_pretrained(output_path)
            except Exception:
                pass
        self.handle.tokenizer.save_pretrained(output_path)
        metadata_payload = self._build_metadata()
        (output_path / "abliteration_metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )
        streaming_manifest = {{
            "schema_version": "model_forge.obliteratus_streaming_rebirth.v1",
            "output_dir": str(output_path),
            "tensor_count": len(weight_map),
            "total_size": total_bytes,
            "max_shard_size_gb": float(cfg.get("max_shard_size_gb", 1.0)),
            "shard_count": shard_count,
            "shards": shard_records,
            "duration_seconds": round(time.time() - start, 3),
        }}
        (output_path / "model_forge_obliteratus_streaming_rebirth.json").write_text(
            json.dumps(streaming_manifest, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        cleanup = getattr(self, "_cleanup_offload_dir", None)
        if callable(cleanup):
            cleanup()
        self._emit("rebirth", "done", f"Stream-saved to {{output_path}}", duration=time.time() - start)
        return output_path

    AbliterationPipeline._rebirth = model_forge_streaming_rebirth
    return True


def main() -> None:
    guard_system_health()
    try:
        from obliteratus.abliterate import AbliterationPipeline
    except Exception as exc:
        raise SystemExit(
            "OBLITERATUS is not installed. Build/use docker/obliteratus.Dockerfile "
            "or install https://github.com/elder-plinius/OBLITERATUS."
        ) from exc
    lora_device_patch_enabled = install_lora_ablation_device_patch()
    activation_layer_filter_enabled = install_activation_layer_filter_patch(AbliterationPipeline)
    adapter_only_rebirth_enabled = install_adapter_only_rebirth(AbliterationPipeline)
    streaming_rebirth_enabled = False if adapter_only_rebirth_enabled else install_streaming_rebirth(AbliterationPipeline)

    kwargs = dict(json.loads(pipeline_kwargs))
    kwargs.update(load_model_forge_prompts())
    pipeline = AbliterationPipeline(
        model_name=model_name,
        method=method,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        **kwargs,
    )
    result = pipeline.run()
    guard_system_health()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    key_remap_result = run_post_export_key_remap()
    guard_system_health()
    source_tether_result = run_source_tether()
    guard_system_health()
    restored_tokenizer_files = restore_source_tokenizer_metadata()
    payload = {{
        "backend": "obliteratus",
        "method": method,
        "model_name": model_name,
        "output_dir": output_dir,
        "work_dir": str(work_dir),
        "prompt_payload": prompt_payload_path,
        "pipeline_kwargs": sorted(kwargs),
        "lora_device_patch_enabled": lora_device_patch_enabled,
        "activation_layer_filter_enabled": activation_layer_filter_enabled,
        "adapter_only_rebirth_enabled": adapter_only_rebirth_enabled,
        "streaming_rebirth_enabled": streaming_rebirth_enabled,
        "result": serializable_result(result),
        "post_export_key_remap": key_remap_result,
        "source_tether": source_tether_result,
        "restored_source_tokenizer_files": restored_tokenizer_files,
        "next_step": "Run model-forge source-vs-candidate targeted eval before broader evals, quantization, promotion, or upload.",
    }}
    summary_path.write_text(json.dumps(payload, indent=2) + "\\n")
    output_summary = Path(output_dir) / "model_forge_sota_obliteratus.json"
    try:
        output_summary.write_text(json.dumps(payload, indent=2) + "\\n")
    except OSError:
        pass
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_heretic_config(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    materialize_model_forge_heretic_prompts(plan, backend, work_dir)
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
    if "seed" in backend:
        lines.append(f"seed = {int(backend['seed'])}")
    if "print_responses" in backend:
        lines.append(f'print_responses = {str(bool(backend["print_responses"])).lower()}')
    if "response_prefix" in backend:
        response_prefix = backend["response_prefix"]
        lines.append(f"response_prefix = {json.dumps(None if response_prefix is None else str(response_prefix))}")
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


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value))


def append_toml_section(lines: list[str], section: str, values: dict[str, Any]) -> None:
    cleaned = {key: value for key, value in values.items() if value is not None}
    if not cleaned:
        return
    lines.append("")
    lines.append(f"[{section}]")
    for key, value in cleaned.items():
        if isinstance(value, dict):
            items = ", ".join(f"{json.dumps(str(k))} = {toml_value(v)}" for k, v in value.items())
            lines.append(f"{key} = {{ {items} }}")
        else:
            lines.append(f"{key} = {toml_value(value)}")


def materialize_model_forge_abliterix_prompts(plan: dict[str, Any], backend: dict[str, Any], work_dir: Path) -> None:
    """Materialize model-forge prompt buckets into local HF datasets for Abliterix."""
    materialize_model_forge_heretic_prompts(plan, backend, work_dir)
    validate_abliterix_train_prompt_counts(backend, work_dir)
    validate_abliterix_multidirection_steering(backend)
    mapping = {
        "good_prompts": "benign_prompts",
        "bad_prompts": "target_prompts",
        "good_evaluation_prompts": "benign_eval_prompts",
        "bad_evaluation_prompts": "target_eval_prompts",
    }
    for source_key, target_key in mapping.items():
        if source_key in backend:
            backend[target_key] = backend[source_key]


def validate_abliterix_train_prompt_counts(backend: dict[str, Any], work_dir: Path) -> None:
    """Fail fast for Abliterix multi-direction configs that would crash after model load."""
    if int(backend.get("n_directions", 1)) <= 1:
        return
    dataset_root = work_dir / "model_forge_prompt_datasets"
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid Abliterix prompt manifest: {manifest_path}: {exc}") from exc
    sections = manifest.get("sections") or {}
    good_count = int((sections.get("good_prompts") or {}).get("count", 0))
    bad_count = int((sections.get("bad_prompts") or {}).get("count", 0))
    if good_count <= 0 or bad_count <= 0:
        raise SystemExit(
            "Abliterix n_directions > 1 requires non-empty good_prompts and bad_prompts "
            f"(got good={good_count}, bad={bad_count})."
        )
    if good_count == bad_count:
        manifest["abliterix_train_prompt_count_validation"] = {
            "requires_equal_counts": True,
            "good_prompts": good_count,
            "bad_prompts": bad_count,
            "status": "passed",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    raise SystemExit(
        "Abliterix n_directions > 1 requires equal good/bad training prompt counts "
        "because the backend computes paired multi-direction residual differences. "
        f"Materialized good_prompts={good_count}, bad_prompts={bad_count}. "
        "Add good_train_prompt_variants/bad_train_prompt_variants, adjust prompt filters, "
        "or reduce n_directions to 1 before running sota-run."
    )


def validate_abliterix_multidirection_steering(backend: dict[str, Any]) -> None:
    if int(backend.get("n_directions", 1)) <= 1:
        return
    steering_mode = str(backend.get("steering_mode", "lora")).strip().lower()
    if steering_mode != "lora":
        return
    if bool(backend.get("allow_experimental_multidirection_lora", False)):
        return
    raise SystemExit(
        "Abliterix n_directions > 1 with steering_mode='lora' is blocked for the "
        "guarded model-forge runner. Abliterix v1.8.0 can produce a steering "
        "vector tensor with size n_directions and then index it by transformer "
        "layer during apply_steering, failing after model load. Set "
        "n_directions: 1 or explicitly set allow_experimental_multidirection_lora: "
        "true after validating the backend version supports this combination."
    )


def materialize_model_forge_apostate_prompts(plan: dict[str, Any], backend: dict[str, Any], work_dir: Path) -> None:
    spec = backend.get("model_forge_prompt_datasets")
    if spec is None:
        return
    dataset_root = work_dir / "model_forge_prompt_files"
    defaults = {
        "good_train_buckets": [
            "agentic_multi_step_planning",
            "agentic_tool_use_json",
            "agentic_self_critique",
            "agentic_structured_extraction",
            "agentic_code_debug",
            "reasoning_style_stability",
            "refusal_benign_boundary",
            "refusal_calibration_benign",
            "normal_use_regression",
            "capability_preservation_challenge",
        ],
        "good_eval_buckets": [
            "refusal_paired_boundary",
            "capability_preservation_challenge",
        ],
        "bad_train_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
        "bad_eval_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
        "preserve_buckets": [
            "capability_preservation_challenge",
            "normal_use_regression",
            "agentic_tool_use_json",
        ],
    }
    if isinstance(spec, dict):
        defaults.update({key: value for key, value in spec.items() if value is not None})

    section_specs = {
        "harmful_path": {
            "file": "harmful_train.txt",
            "prefix": "bad_train",
            "buckets": defaults["bad_train_buckets"],
            "case_ids": normalize_case_id_filter(defaults.get("bad_train_case_ids")),
            "should_refuse": True,
        },
        "harmful_test": {
            "file": "harmful_test.txt",
            "prefix": "bad_eval",
            "buckets": defaults["bad_eval_buckets"],
            "case_ids": normalize_case_id_filter(defaults.get("bad_eval_case_ids")),
            "should_refuse": True,
        },
        "harmless_path": {
            "file": "harmless_train.txt",
            "prefix": "good_train",
            "buckets": defaults["good_train_buckets"],
            "case_ids": normalize_case_id_filter(defaults.get("good_train_case_ids")),
            "should_refuse": False,
        },
        "harmless_test": {
            "file": "harmless_test.txt",
            "prefix": "good_eval",
            "buckets": defaults["good_eval_buckets"],
            "case_ids": normalize_case_id_filter(defaults.get("good_eval_case_ids")),
            "should_refuse": False,
        },
        "preserve_path": {
            "file": "preserve.txt",
            "prefix": "preserve",
            "buckets": defaults["preserve_buckets"],
            "case_ids": normalize_case_id_filter(defaults.get("preserve_case_ids")),
            "should_refuse": False,
        },
    }
    summary: dict[str, Any] = {"source": "model_forge_eval_prompts", "sections": {}}
    for config_key, section in section_specs.items():
        prefix = str(section["prefix"])
        prompts = prompts_for_buckets(
            list(section["buckets"]),
            should_refuse=bool(section["should_refuse"]),
            case_ids=section["case_ids"],
        )
        response_summary = None
        if isinstance(spec, dict):
            response_prompts, response_summary = response_conditioned_section_prompts(spec, prefix)
            prompts = [*prompts, *response_prompts]
            extra_prompts = normalize_extra_prompts(spec.get(f"{prefix}_extra_prompts"), prefix=prefix)
            prompts = [*prompts, *extra_prompts]
            prompts, variant_summary = apply_prompt_variants(
                prompts,
                spec.get(f"{prefix}_prompt_variants"),
                prefix=prefix,
            )
        else:
            variant_summary = None
            extra_prompts = []
        if not prompts and config_key == "preserve_path":
            continue
        path = dataset_root / str(section["file"])
        save_text_prompt_file(path, prompts)
        backend[config_key] = str(path)
        summary["sections"][config_key] = {
            "path": str(path),
            "count": len(prompts),
            "case_ids": sorted(section["case_ids"]) if section["case_ids"] is not None else None,
            "buckets": list(section["buckets"]),
        }
        if response_summary is not None:
            summary["sections"][config_key]["response_conditioned"] = response_summary
        if extra_prompts:
            summary["sections"][config_key]["extra_prompts"] = {"count": len(extra_prompts)}
        if variant_summary is not None:
            summary["sections"][config_key]["prompt_variants"] = variant_summary
    (dataset_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


def apostate_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def write_apostate_config(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    materialize_model_forge_apostate_prompts(plan, backend, work_dir)
    config_path = work_dir / "apostate_config.json"
    payload: dict[str, Any] = {
        "model": plan["source_model"],
        "output_dir": plan["output_dir"],
        "optimize": bool(backend.get("optimize", True)),
        "bake": bool(backend.get("bake", True)),
        "resume": bool(backend.get("resume", True)),
    }
    for key, value in backend.items():
        if key in APOSTATE_CONFIG_FIELDS:
            if key == "activation_cache_dir" and value:
                payload[key] = str(resolve_repo_path(value))
            else:
                payload[key] = apostate_config_value(value)
    payload["model"] = plan["source_model"]
    payload["output_dir"] = plan["output_dir"]
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    return config_path


def write_apostate_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_apostate.py"
    config_path = work_dir / "apostate_config.json"
    summary_path = work_dir / "model_forge_sota_apostate.json"
    script = f'''from __future__ import annotations

import json
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

work_dir = Path({str(work_dir)!r})
config_path = Path({str(config_path)!r})
output_dir = Path({str(plan["output_dir"])!r})
summary_path = Path({str(summary_path)!r})


def available_ram_fraction() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            values = {{}}
            for line in handle:
                key, _, raw = line.partition(":")
                fields = raw.strip().split()
                if fields:
                    values[key] = float(fields[0])
        available = values.get("MemAvailable")
        total = values.get("MemTotal")
        if available is None or total in (None, 0):
            return 1.0
        return available / total
    except OSError:
        return 1.0


def guard_system_health(*, fatal: bool = True) -> list[str]:
    findings = []
    min_ram = float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram:
        findings.append(f"available RAM fraction {{ram_fraction:.3f}} is below guard {{min_ram:.3f}}")
    min_disk = float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15"))
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else work_dir)
    free_fraction = usage.free / usage.total
    if free_fraction < min_disk:
        findings.append(f"free disk fraction {{free_fraction:.3f}} is below guard {{min_disk:.3f}}")
    if findings and fatal:
        raise SystemExit("; ".join(findings))
    return findings


def main() -> None:
    os.chdir(work_dir)
    guard_system_health()
    try:
        backend_version = version("apostate")
    except PackageNotFoundError:
        backend_version = "unknown"

    try:
        from apostate.cli import main as apostate_main
    except Exception as exc:
        raise SystemExit("Apostate is not installed. Build/use the configured container or install https://github.com/heterodoxin/apostate.") from exc

    sys.argv = ["apostate", "--config", str(config_path)]
    apostate_main()

    report_path = output_dir / "report.json"
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {{"unparsed_report": str(report_path)}}
    summary_path.write_text(json.dumps({{
        "backend": "apostate",
        "backend_version": backend_version,
        "execution": {backend.get("execution", "checkpoint_export")!r},
        "source_model": {plan["source_model"]!r},
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "config": str(config_path),
        "report": str(report_path) if report_path.exists() else None,
        "report_summary": report,
        "next_step": "Run model-forge source-vs-candidate targeted internal eval before broader evals, quantization, promotion, or upload.",
    }}, indent=2) + "\\n")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_abliterix_config(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    materialize_model_forge_abliterix_prompts(plan, backend, work_dir)
    checkpoint_dir = resolve_repo_path(backend.get("checkpoint_dir", "abliterix_checkpoints"), work_dir)
    config_path = work_dir / "abliterix.toml"
    model_values = {
        "model_id": plan["source_model"],
        "backend": backend.get("model_backend", backend.get("backend", "hf")),
        "device_map": backend.get("device_map", "auto"),
        "quant_method": backend.get("quant_method", "none"),
        "trust_remote_code": backend.get("trust_remote_code", True),
        "tensor_parallel_size": backend.get("tensor_parallel_size"),
        "gpu_memory_utilization": backend.get("gpu_memory_utilization"),
        "max_memory": backend.get("max_memory"),
        "attn_implementation": backend.get("attn_implementation"),
        "experts_implementation": backend.get("experts_implementation"),
        "moe_backend": backend.get("moe_backend"),
    }
    inference_values = {
        "batch_size": int(backend.get("batch_size", 0)),
        "max_batch_size": int(backend.get("max_batch_size", 64)),
        "max_gen_tokens": int(backend.get("max_gen_tokens", backend.get("max_response_length", 128))),
        "min_gen_tokens": backend.get("min_gen_tokens"),
    }
    steering_values = {
        "vector_method": backend.get("vector_method", "sra"),
        "steering_mode": backend.get("steering_mode", "lora"),
        "orthogonal_projection": bool(backend.get("orthogonal_projection", False)),
        "projected_abliteration": bool(backend.get("projected_abliteration", True)),
        "winsorize_vectors": bool(backend.get("winsorize_vectors", True)),
        "winsorize_quantile": float(backend.get("winsorize_quantile", 0.995)),
        "ot_components": int(backend.get("ot_components", 2)),
        "n_directions": int(backend.get("n_directions", 1)),
        "sra_base_method": backend.get("sra_base_method", "mean"),
        "sra_n_atoms": int(backend.get("sra_n_atoms", 8)),
        "sra_ridge_alpha": float(backend.get("sra_ridge_alpha", 0.1)),
        "som_grid_h": int(backend.get("som_grid_h", 3)),
        "som_grid_w": int(backend.get("som_grid_w", 3)),
        "som_n_iters": int(backend.get("som_n_iters", 500)),
        "som_initial_lr": float(backend.get("som_initial_lr", 0.5)),
        "som_seed": int(backend.get("som_seed", backend.get("seed", 0))),
        "weight_normalization": backend.get("weight_normalization", "full"),
        "full_norm_lora_rank": int(backend.get("full_norm_lora_rank", 3)),
        "strength_range": backend.get("strength_range", [0.4, 1.6]),
        "disabled_components": backend.get("disabled_components"),
        "component_strength_ranges": backend.get("component_strength_ranges"),
        "min_weight_frac_max": backend.get("min_weight_frac_max"),
        "component_min_frac_max": backend.get("component_min_frac_max"),
        "discriminative_layer_selection": backend.get("discriminative_layer_selection"),
        "decay_kernel": backend.get("decay_kernel"),
        "ablate_harmfulness_direction": backend.get("ablate_harmfulness_direction"),
        "harmfulness_layer_band": backend.get("harmfulness_layer_band"),
        "fixed_vector_scope": backend.get("fixed_vector_scope"),
        "direct_transform": backend.get("direct_transform"),
        "search_direct_transform": backend.get("search_direct_transform"),
    }
    optimization_values = {
        "num_trials": int(backend.get("num_trials", backend.get("n_trials", 24))),
        "num_warmup_trials": int(backend.get("num_warmup_trials", backend.get("n_warmup_trials", 8))),
        "checkpoint_dir": str(checkpoint_dir),
        "sampler_seed": backend.get("sampler_seed", backend.get("seed")),
    }
    kl_values = {
        "scale": float(backend.get("kl_scale", backend.get("kl_divergence_scale", 1.0))),
        "token_count": int(backend.get("kl_token_count", 1)),
        "target": float(backend.get("kl_target", backend.get("kl_divergence_target", 0.05))),
        "prune_threshold": float(backend.get("kl_prune_threshold", 5.0)),
    }
    detection_values = {
        "llm_judge": bool(backend.get("llm_judge", False)),
        "llm_judge_model": backend.get("llm_judge_model"),
        "llm_judge_base_url": backend.get("llm_judge_base_url"),
        "llm_judge_api_key_env": backend.get("llm_judge_api_key_env"),
        "llm_judge_batch_size": backend.get("llm_judge_batch_size"),
        "llm_judge_concurrency": backend.get("llm_judge_concurrency"),
    }
    display_values = {
        "print_responses": bool(backend.get("print_responses", False)),
        "print_residual_geometry": bool(backend.get("print_residual_geometry", False)),
        "plot_residuals": bool(backend.get("plot_residuals", False)),
        "residual_plot_path": str(work_dir / "abliterix_plots"),
    }
    lines = [
        "non_interactive = true",
        f"overwrite_checkpoint = {toml_value(bool(backend.get('overwrite_checkpoint', False)))}",
        f"system_prompt = {toml_value(backend.get('system_prompt', 'You are a helpful assistant.'))}",
    ]
    append_toml_section(lines, "model", model_values)
    append_toml_section(lines, "inference", inference_values)
    append_toml_section(lines, "steering", steering_values)
    append_toml_section(lines, "optimization", optimization_values)
    append_toml_section(lines, "kl", kl_values)
    append_toml_section(lines, "detection", detection_values)
    append_toml_section(lines, "display", display_values)
    for source_key in ("benign_prompts", "target_prompts", "benign_eval_prompts", "target_eval_prompts"):
        prompt_cfg = backend.get(source_key)
        if prompt_cfg:
            append_toml_section(
                lines,
                source_key,
                {key: prompt_cfg.get(key) for key in ["dataset", "split", "column", "prefix", "suffix", "system_prompt"]},
            )
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def write_abliterix_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_abliterix_search.py"
    config_path = work_dir / "abliterix.toml"
    checkpoint_dir = resolve_repo_path(backend.get("checkpoint_dir", "abliterix_checkpoints"), work_dir)
    pair_reduction = backend.get("harmfulness_pair_reduction")
    harmfulness_weight = float(backend.get("harmfulness_pair_weight", 1.0))
    compat_patch = ""
    if pair_reduction is not None:
        compat_patch = f'''
    from model_forge.integrations.abliterix_compat import apply_abliterix_compat_patches

    apply_abliterix_compat_patches(
        reduction={str(pair_reduction)!r},
        harmfulness_weight={harmfulness_weight!r},
    )
'''
    script = f'''from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

work_dir = Path({str(work_dir)!r})
config_path = Path({str(config_path)!r})
checkpoint_dir = Path({str(checkpoint_dir)!r})
summary_path = work_dir / "model_forge_sota_abliterix_search.json"


def main() -> None:
    os.chdir(work_dir)
    try:
        backend_version = version("abliterix")
    except PackageNotFoundError as exc:
        raise SystemExit("Abliterix is not installed. Build/use the configured container or run `pip install abliterix`.") from exc
{compat_patch}
    from abliterix.cli import main as abliterix_main

    os.environ["AX_CONFIG"] = str(config_path)
    sys.argv = ["abliterix"]
    abliterix_main()
    checkpoint_files = sorted(checkpoint_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    summary_path.write_text(json.dumps({{
        "backend": "abliterix",
        "backend_version": backend_version,
        "mode": "search_only",
        "execution": {backend.get("execution", "search_only")!r},
        "source_model": {plan["source_model"]!r},
        "intended_output_dir": {plan["output_dir"]!r},
        "work_dir": str(work_dir),
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "journals": [str(path) for path in checkpoint_files],
        "exported_checkpoint": False,
        "next_step": "Run `./forge ablate --config <config> abliterix-search-analyze`; if it recommends `prepare_guarded_export_runner`, dry-run `abliterix-export`, then execute export and run the model-forge targeted gate.",
    }}, indent=2) + "\\n")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_abliterix_export_runner(plan: dict[str, Any], trial_index: int, overwrite: bool = False) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / f"run_abliterix_export_trial{trial_index}.py"
    config_path = work_dir / "abliterix.toml"
    checkpoint_dir = resolve_repo_path(backend.get("checkpoint_dir", "abliterix_checkpoints"), work_dir)
    pair_reduction = backend.get("harmfulness_pair_reduction")
    harmfulness_weight = float(backend.get("harmfulness_pair_weight", 1.0))
    compat_patch = ""
    if pair_reduction is not None:
        compat_patch = f'''
    from model_forge.integrations.abliterix_compat import apply_abliterix_compat_patches

    apply_abliterix_compat_patches(
        reduction={str(pair_reduction)!r},
        harmfulness_weight={harmfulness_weight!r},
    )
'''
    script = f'''from __future__ import annotations

import json
import os
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import questionary

work_dir = Path({str(work_dir)!r})
config_path = Path({str(config_path)!r})
checkpoint_dir = Path({str(checkpoint_dir)!r})
source_model = Path({plan["source_model"]!r})
output_dir = Path({plan["output_dir"]!r})
trial_index = {int(trial_index)!r}
overwrite = {bool(overwrite)!r}
state = {{"selected_trial": None, "save_requested": False, "saved": False}}


def _choice_title(choice):
    return choice.title if hasattr(choice, "title") else str(choice)


def _choice_value(choice):
    return choice.value if hasattr(choice, "value") else choice


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _preflight_output_dir() -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {{output_dir}}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(output_dir.parent)
    source_size = _directory_size(source_model)
    floor = float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15"))
    if total and (free - source_size) / total < floor:
        raise SystemExit(
            "disk preflight would breach free-space floor after export: "
            f"{{(free - source_size) / total:.3f}} < {{floor:.3f}}"
        )


def prompt_choice(message, choices, *args, **kwargs):
    if message == "How would you like to proceed?":
        return "continue"
    if message == "Which trial do you want to use?":
        if state["saved"]:
            return ""
        fallback = None
        for choice in choices:
            value = _choice_value(choice)
            if isinstance(value, str) and value in {{"continue", ""}}:
                continue
            if fallback is None:
                fallback = choice
            if getattr(value, "user_attrs", {{}}).get("index") == trial_index:
                state["selected_trial"] = _choice_title(choice)
                return value
            match = re.search(r"Trial\\s+(\\d+)", _choice_title(choice))
            if match and int(match.group(1)) == trial_index:
                state["selected_trial"] = _choice_title(choice)
                return value
        raise SystemExit(f"selected Abliterix trial {{trial_index}} was not available")
    if message == "What do you want to do with the decensored model?":
        if not state["save_requested"]:
            state["save_requested"] = True
            return "Save the model to a local folder"
        state["saved"] = True
        return "Return to the trial selection menu"
    if message == "How do you want to proceed?":
        return "merge"
    return _choice_value(choices[0]) if choices else None


def prompt_path(message, *args, **kwargs):
    _preflight_output_dir()
    return str(output_dir)


def prompt_text(message, default="", qmark="?", unsafe=False, *args, **kwargs):
    return default


def prompt_secret(message, *args, **kwargs):
    return ""


def main() -> None:
    os.chdir(work_dir)
    try:
        backend_version = version("abliterix")
    except PackageNotFoundError as exc:
        raise SystemExit("Abliterix is not installed. Build/use the configured container or run `pip install abliterix`.") from exc
{compat_patch}
    import abliterix.cli as ax_cli
    import abliterix.interactive as ax_interactive

    original_checkpoint_handler = ax_cli._handle_existing_checkpoint

    def checkpoint_handler(config, existing_study, checkpoint_file, lock_obj, storage):
        result = original_checkpoint_handler(config, existing_study, checkpoint_file, lock_obj, storage)
        if result is None:
            return None
        restored, restored_storage = result
        restored.non_interactive = False
        restored.overwrite_checkpoint = False
        return restored, restored_storage

    ax_cli._handle_existing_checkpoint = checkpoint_handler
    ax_cli.ask_choice = prompt_choice
    ax_interactive.ask_choice = prompt_choice
    ax_interactive.ask_path = prompt_path
    ax_interactive.ask_text = prompt_text
    ax_interactive.ask_secret = prompt_secret
    questionary.select = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected interactive prompt"))

    os.environ["AX_CONFIG"] = str(config_path)
    sys.argv = ["abliterix", "--no-non-interactive"]
    ax_cli.main()
    if not state["saved"]:
        raise SystemExit("Abliterix exited before a model was saved")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "model_forge_sota_abliterix.json"
    summary_path.write_text(json.dumps({{
        "backend": "abliterix",
        "backend_version": backend_version,
        "mode": "selected_trial_export",
        "source_model": str(source_model),
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "selected_trial_index": trial_index,
        "selected_trial": state["selected_trial"],
        "exported_checkpoint": True,
        "required_next_gate": "Run model-forge targeted internal eval before broader eval, NVFP4 quantization, or upload.",
    }}, indent=2) + "\\n")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def load_eval_prompt_cases(bucket: str) -> list[dict[str, Any]]:
    path = REPO_DIR / "evals" / "prompts" / f"{bucket}.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    cases = raw.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit(f"eval prompt set must contain cases: {path}")
    return cases


def normalize_case_id_filter(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise SystemExit("case id filters must be a string or list of strings")
    return {str(value).strip() for value in values if str(value).strip()}


def case_id_matches(bucket: str, case_id: str, filters: set[str] | None) -> bool:
    if filters is None:
        return True
    return case_id in filters or f"{bucket}/{case_id}" in filters or f"{bucket}::{case_id}" in filters


def prompts_for_buckets(
    buckets: list[str],
    *,
    should_refuse: bool,
    case_ids: set[str] | None = None,
) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for bucket in buckets:
        for item in load_eval_prompt_cases(bucket):
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("id", "")).strip()
            if not case_id_matches(bucket, case_id, case_ids):
                continue
            checks = item.get("checks", {}) or {}
            if bool(checks.get("should_refuse", False)) != should_refuse:
                continue
            prompt = str(item.get("prompt", "")).strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    if not prompts:
        kind = "bad" if should_refuse else "good"
        raise SystemExit(f"no {kind} prompts found for buckets: {', '.join(buckets)}")
    return prompts


def save_heretic_prompt_dataset(path: Path, prompts: list[str]) -> None:
    from datasets import Dataset

    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"text": prompts}, split="train").save_to_disk(str(path))


def save_text_prompt_file(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [re.sub(r"\s+", " ", prompt).strip() for prompt in prompts if prompt.strip()]
    path.write_text("\n".join(flattened) + "\n")


def response_score_matches(scores: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _as_float(scores.get(key))
        if isinstance(expected, dict):
            if "eq" in expected:
                target = _as_float(expected["eq"])
                if actual is None or target is None or abs(actual - target) > 1e-9:
                    return False
            if "min" in expected:
                target = _as_float(expected["min"])
                if actual is None or target is None or actual < target:
                    return False
            if "max" in expected:
                target = _as_float(expected["max"])
                if actual is None or target is None or actual > target:
                    return False
        else:
            target = _as_float(expected)
            if actual is None or target is None or abs(actual - target) > 1e-9:
                return False
    return True


def response_conditioned_prompts(
    source: str | Path,
    *,
    case_ids: set[str] | None = None,
    buckets: set[str] | None = None,
    should_refuse: bool | None = None,
    score_filters: dict[str, Any] | None = None,
    template: str | None = None,
    max_response_chars: int | None = None,
    limit: int | None = None,
) -> list[str]:
    path = resolve_repo_path(source)
    if not path.exists():
        raise SystemExit(f"missing response-conditioned prompt source: {path}")
    prompt_template = template or "{prompt}\n\nAssistant response:\n{response_text}"
    prompts: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid response-conditioned JSONL at {path}:{line_number}: {exc}") from exc
        bucket = str(record.get("bucket", "")).strip()
        case_id = str(record.get("case_id", "")).strip()
        if buckets is not None and bucket not in buckets:
            continue
        if not case_id_matches(bucket, case_id, case_ids):
            continue
        checks = record.get("checks") or {}
        if should_refuse is not None and bool(checks.get("should_refuse", False)) != should_refuse:
            continue
        scores = record.get("scores") or {}
        if score_filters and not response_score_matches(scores, score_filters):
            continue
        prompt = str(record.get("prompt", "")).strip()
        response = str(record.get("response_text", "")).strip()
        if not prompt or not response:
            continue
        if max_response_chars is not None and max_response_chars > 0:
            response = response[:max_response_chars].strip()
        try:
            text = prompt_template.format(
                bucket=bucket,
                case_id=case_id,
                prompt=prompt,
                response_text=response,
            ).strip()
        except KeyError as exc:
            raise SystemExit(f"unknown response-conditioned template field: {exc}") from exc
        if text and text not in seen:
            seen.add(text)
            prompts.append(text)
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def response_conditioned_section_prompts(spec: dict[str, Any], prefix: str) -> tuple[list[str], dict[str, Any] | None]:
    source = spec.get(f"{prefix}_response_source")
    if not source:
        return [], None
    raw_max_chars = spec.get(f"{prefix}_response_max_chars")
    raw_limit = spec.get(f"{prefix}_response_limit")
    prompts = response_conditioned_prompts(
        source,
        case_ids=normalize_case_id_filter(spec.get(f"{prefix}_response_case_ids")),
        buckets=normalize_case_id_filter(spec.get(f"{prefix}_response_buckets")),
        should_refuse=spec.get(f"{prefix}_response_should_refuse"),
        score_filters=spec.get(f"{prefix}_response_score_filters"),
        template=spec.get(f"{prefix}_response_template"),
        max_response_chars=int(raw_max_chars) if raw_max_chars is not None else None,
        limit=int(raw_limit) if raw_limit is not None else None,
    )
    return prompts, {
        "source": str(resolve_repo_path(source)),
        "count": len(prompts),
        "case_ids": sorted(normalize_case_id_filter(spec.get(f"{prefix}_response_case_ids")) or []),
        "buckets": sorted(normalize_case_id_filter(spec.get(f"{prefix}_response_buckets")) or []),
        "score_filters": spec.get(f"{prefix}_response_score_filters") or {},
    }


def normalize_prompt_variants(raw_variants: Any, *, prefix: str) -> list[dict[str, Any]]:
    if raw_variants is None:
        return []
    if isinstance(raw_variants, str):
        raw_items = [raw_variants]
    elif isinstance(raw_variants, list):
        raw_items = raw_variants
    else:
        raise SystemExit(f"{prefix}_prompt_variants must be a string or list")

    variants = []
    for index, raw in enumerate(raw_items, start=1):
        if isinstance(raw, str):
            variant = {"id": f"variant_{index}", "template": raw, "repeat": 1}
        elif isinstance(raw, dict):
            template = str(raw.get("template", "{prompt}"))
            variant = {
                "id": str(raw.get("id", f"variant_{index}")),
                "template": template,
                "repeat": int(raw.get("repeat", 1)),
            }
        else:
            raise SystemExit(f"{prefix}_prompt_variants entries must be strings or mappings")
        if "{prompt}" not in variant["template"]:
            raise SystemExit(f"{prefix}_prompt_variants template must include {{prompt}}")
        if int(variant["repeat"]) < 1:
            raise SystemExit(f"{prefix}_prompt_variants repeat must be >= 1")
        variants.append(variant)
    return variants


def apply_prompt_variants(
    prompts: list[str],
    raw_variants: Any,
    *,
    prefix: str,
) -> tuple[list[str], dict[str, Any] | None]:
    variants = normalize_prompt_variants(raw_variants, prefix=prefix)
    if not variants:
        return prompts, None
    expanded: list[str] = []
    for prompt in prompts:
        for variant in variants:
            try:
                text = variant["template"].format(prompt=prompt).strip()
            except KeyError as exc:
                raise SystemExit(f"unknown {prefix}_prompt_variants template field: {exc}") from exc
            # Duplicate rows are intentional here: they let configs overweight
            # rare observed failures without adding backend-specific code.
            expanded.extend([text] * int(variant["repeat"]))
    return expanded, {
        "input_count": len(prompts),
        "output_count": len(expanded),
        "variants": [
            {
                "id": variant["id"],
                "repeat": int(variant["repeat"]),
                "template": variant["template"],
            }
            for variant in variants
        ],
    }


def normalize_extra_prompts(raw_prompts: Any, *, prefix: str) -> list[str]:
    if raw_prompts is None:
        return []
    if isinstance(raw_prompts, str):
        raw_items = [raw_prompts]
    elif isinstance(raw_prompts, list):
        raw_items = raw_prompts
    else:
        raise SystemExit(f"{prefix}_extra_prompts must be a string or list")
    prompts = []
    for raw in raw_items:
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict):
            text = str(raw.get("prompt", raw.get("text", "")))
        else:
            raise SystemExit(f"{prefix}_extra_prompts entries must be strings or mappings")
        text = text.strip()
        if not text:
            raise SystemExit(f"{prefix}_extra_prompts entries must not be empty")
        prompts.append(text)
    return prompts


def materialize_model_forge_heretic_prompts(plan: dict[str, Any], backend: dict[str, Any], work_dir: Path) -> None:
    spec = backend.get("model_forge_prompt_datasets")
    if spec is None:
        return
    dataset_root = work_dir / "model_forge_prompt_datasets"
    defaults = {
        "good_train_buckets": [
            "agentic_multi_step_planning",
            "agentic_tool_use_json",
            "agentic_self_critique",
            "agentic_structured_extraction",
            "agentic_code_debug",
            "reasoning_style_stability",
            "refusal_benign_boundary",
            "refusal_calibration_benign",
            "normal_use_regression",
            "capability_preservation_challenge",
        ],
        "good_eval_buckets": [
            "refusal_paired_boundary",
            "capability_preservation_challenge",
        ],
        "bad_train_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
        "bad_eval_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
    }
    if isinstance(spec, dict):
        defaults.update({key: value for key, value in spec.items() if value is not None})
    case_filters = {
        "good_prompts": normalize_case_id_filter(defaults.get("good_train_case_ids")),
        "good_evaluation_prompts": normalize_case_id_filter(defaults.get("good_eval_case_ids")),
        "bad_prompts": normalize_case_id_filter(defaults.get("bad_train_case_ids")),
        "bad_evaluation_prompts": normalize_case_id_filter(defaults.get("bad_eval_case_ids")),
    }
    sections = {
        "good_prompts": prompts_for_buckets(
            list(defaults["good_train_buckets"]),
            should_refuse=False,
            case_ids=case_filters["good_prompts"],
        ),
        "good_evaluation_prompts": prompts_for_buckets(
            list(defaults["good_eval_buckets"]),
            should_refuse=False,
            case_ids=case_filters["good_evaluation_prompts"],
        ),
        "bad_prompts": prompts_for_buckets(
            list(defaults["bad_train_buckets"]),
            should_refuse=True,
            case_ids=case_filters["bad_prompts"],
        ),
        "bad_evaluation_prompts": prompts_for_buckets(
            list(defaults["bad_eval_buckets"]),
            should_refuse=True,
            case_ids=case_filters["bad_evaluation_prompts"],
        ),
    }
    option_prefixes = {
        "good_prompts": "good_train",
        "good_evaluation_prompts": "good_eval",
        "bad_prompts": "bad_train",
        "bad_evaluation_prompts": "bad_eval",
    }
    summary: dict[str, Any] = {"source": "model_forge_eval_prompts", "sections": {}}
    for section, prompts in sections.items():
        prefix = option_prefixes[section]
        response_summary = None
        if isinstance(spec, dict):
            response_prompts, response_summary = response_conditioned_section_prompts(spec, prefix)
            prompts = [*prompts, *response_prompts]
            extra_prompts = normalize_extra_prompts(spec.get(f"{prefix}_extra_prompts"), prefix=prefix)
            prompts = [*prompts, *extra_prompts]
            variant_summary = None
            prompts, variant_summary = apply_prompt_variants(
                prompts,
                spec.get(f"{prefix}_prompt_variants"),
                prefix=prefix,
            )
        else:
            variant_summary = None
            extra_prompts = []
        path = dataset_root / section
        save_heretic_prompt_dataset(path, prompts)
        backend[section] = {
            "dataset": str(path),
            "split": "train[:]",
            "column": "text",
        }
        for option in ("prefix", "suffix", "system_prompt"):
            key = f"{prefix}_{option}"
            if isinstance(spec, dict) and key in spec:
                backend[section][option] = spec[key]
        summary["sections"][section] = {
            "path": str(path),
            "count": len(prompts),
            "case_ids": sorted(case_filters[section]) if case_filters[section] is not None else None,
            "prompt_options": {
                key: backend[section][key]
                for key in ("prefix", "suffix", "system_prompt")
                if key in backend[section]
            },
        }
        if response_summary is not None:
            summary["sections"][section]["response_conditioned"] = response_summary
        if extra_prompts:
            summary["sections"][section]["extra_prompts"] = {"count": len(extra_prompts)}
        if variant_summary is not None:
            summary["sections"][section]["prompt_variants"] = variant_summary
    (dataset_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


def materialize_native_prompt_pair_files(plan: dict[str, Any], backend: dict[str, Any], work_dir: Path) -> dict[str, Any] | None:
    spec = backend.get("model_forge_prompt_datasets")
    if spec is None:
        return None
    dataset_root = work_dir / "model_forge_native_prompt_pairs"
    defaults = {
        "good_train_buckets": [
            "agentic_multi_step_planning",
            "agentic_tool_use_json",
            "agentic_self_critique",
            "agentic_structured_extraction",
            "agentic_code_debug",
            "reasoning_style_stability",
            "refusal_benign_boundary",
            "refusal_calibration_benign",
            "normal_use_regression",
            "capability_preservation_challenge",
        ],
        "bad_train_buckets": [
            "unsafe_overcompliance",
            "refusal_calibration_unsafe",
            "refusal_paired_boundary",
        ],
        "balance_prompt_pairs": True,
    }
    if isinstance(spec, dict):
        defaults.update({key: value for key, value in spec.items() if value is not None})
    sections = {
        "harmful_prompts": {
            "file": "harmful_train.txt",
            "prefix": "bad_train",
            "buckets": list(defaults["bad_train_buckets"]),
            "case_ids": normalize_case_id_filter(defaults.get("bad_train_case_ids")),
            "should_refuse": True,
        },
        "benign_prompts": {
            "file": "benign_train.txt",
            "prefix": "good_train",
            "buckets": list(defaults["good_train_buckets"]),
            "case_ids": normalize_case_id_filter(defaults.get("good_train_case_ids")),
            "should_refuse": False,
        },
    }
    prompts_by_key: dict[str, list[str]] = {}
    summary: dict[str, Any] = {"source": "model_forge_eval_prompts", "sections": {}}
    for key, section in sections.items():
        prefix = str(section["prefix"])
        prompts = prompts_for_buckets(
            list(section["buckets"]),
            should_refuse=bool(section["should_refuse"]),
            case_ids=section["case_ids"],
        )
        response_summary = None
        if isinstance(spec, dict):
            response_prompts, response_summary = response_conditioned_section_prompts(spec, prefix)
            prompts = [*prompts, *response_prompts]
            extra_prompts = normalize_extra_prompts(spec.get(f"{prefix}_extra_prompts"), prefix=prefix)
            prompts = [*prompts, *extra_prompts]
            prompts, variant_summary = apply_prompt_variants(
                prompts,
                spec.get(f"{prefix}_prompt_variants"),
                prefix=prefix,
            )
        else:
            extra_prompts = []
            variant_summary = None
        materialized_prompts = [prompt for prompt in (str(item).strip() for item in prompts) if prompt]
        if not materialized_prompts:
            raise SystemExit(f"no native prompt-pair rows materialized for {key}")
        prompts_by_key[key] = materialized_prompts
        summary["sections"][key] = {
            "count": len(materialized_prompts),
            "buckets": list(section["buckets"]),
            "case_ids": sorted(section["case_ids"]) if section["case_ids"] is not None else None,
        }
        if response_summary is not None:
            summary["sections"][key]["response_conditioned"] = response_summary
        if extra_prompts:
            summary["sections"][key]["extra_prompts"] = {"count": len(extra_prompts)}
        if variant_summary is not None:
            summary["sections"][key]["prompt_variants"] = variant_summary

    harmful = prompts_by_key["harmful_prompts"]
    benign = prompts_by_key["benign_prompts"]
    if bool(defaults.get("balance_prompt_pairs", True)) and len(harmful) != len(benign):
        target_len = max(len(harmful), len(benign))
        prompts_by_key["harmful_prompts"] = [harmful[index % len(harmful)] for index in range(target_len)]
        prompts_by_key["benign_prompts"] = [benign[index % len(benign)] for index in range(target_len)]
        summary["balanced_prompt_pairs"] = {
            "enabled": True,
            "harmful_before": len(harmful),
            "benign_before": len(benign),
            "paired_count": target_len,
        }
    else:
        summary["balanced_prompt_pairs"] = {
            "enabled": bool(defaults.get("balance_prompt_pairs", True)),
            "paired_count": min(len(harmful), len(benign)),
        }

    dataset_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "harmful_prompts": dataset_root / "harmful_train.txt",
        "benign_prompts": dataset_root / "benign_train.txt",
    }
    for key, path in paths.items():
        save_text_prompt_file(path, prompts_by_key[key])
        summary["sections"][key]["path"] = str(path)
        summary["sections"][key]["count_after_balance"] = len(prompts_by_key[key])
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "harmful_prompts": str(paths["harmful_prompts"]),
        "benign_prompts": str(paths["benign_prompts"]),
        "manifest": str(manifest_path),
        "paired_count": int(summary["balanced_prompt_pairs"]["paired_count"]),
    }


def native_checkpoint_method_name(plan: dict[str, Any]) -> str:
    if plan["backend"] == "sra":
        return "native_surgical_refusal_ablation_projection"
    if plan["backend"] == "norm_preserving_projection":
        return "native_norm_preserving_projected_abliteration"
    if plan["backend"] == "som_projection":
        return "native_som_multidirectional_projection"
    if plan["backend"] == "concept_cone_projection":
        return "native_source_anchored_concept_cone_projection"
    if plan["backend"] == "selective_projection":
        return "native_selective_layer_projection"
    if plan["backend"] == "qwen_scope_sae":
        return "qwen_scope_sae_dictionary_projection"
    return "native_optimal_transport_activation_projection"


def write_native_optimal_transport_config(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_files = materialize_native_prompt_pair_files(plan, backend, work_dir)
    base_plan = plan["model_forge_plan"]
    activation = json.loads(json.dumps(base_plan["activation_collection"]))
    for key in (
        "batch_size",
        "max_pairs",
        "max_seq_len",
        "token_position",
        "direction_extraction",
        "direction_components",
        "som_neurons",
        "som_steps",
        "som_learning_rate",
        "som_neighborhood",
        "benign_subspace_components",
        "concept_cone_project_mean_out",
        "concept_cone_whiten",
        "sra_preservation_components",
        "sra_include_benign_mean",
        "direction_source_layer",
        "replicate_source_direction",
        "use_chat_template",
        "winsorize_quantile",
        "harmful_suffix",
        "benign_suffix",
        "harmful_assistant_prefix",
        "benign_assistant_prefix",
        "assistant_prefix_template",
        "layer_skip_first",
        "layer_skip_last",
    ):
        if key in backend:
            activation[key] = backend[key]
    activation["direction_extraction"] = backend.get("direction_extraction", "whitened_paired_svd")
    activation["direction_components"] = int(backend.get("direction_components", 4))
    activation["max_pairs"] = int(backend.get("max_pairs", prompt_files["paired_count"] if prompt_files else activation["max_pairs"]))

    edit = {
        "mode": "projection",
        "direction_transform": "biprojection",
        "norm_preserve": True,
        "strength": 1.0,
        "module_strengths": {
            "self_attn.o_proj.weight": 1.0,
            "mlp.down_proj.weight": 1.0,
        },
        "layer_start": 8,
        "layer_end": 48,
        "target_weight_suffixes": [
            "mlp.down_proj.weight",
            "self_attn.o_proj.weight",
        ],
        "leave_embeddings_untouched": True,
        "leave_lm_head_untouched": True,
        "leave_moe_experts_untouched": True,
        "require_all_target_directions": True,
        "review_required_before_export": True,
    }
    edit.update(base_plan.get("edit") or {})
    edit.update(backend.get("edit") or {})

    data = {
        "harmful_prompts": base_plan["data"]["harmful_prompts"],
        "benign_prompts": base_plan["data"]["benign_prompts"],
    }
    if prompt_files:
        data["harmful_prompts"] = prompt_files["harmful_prompts"]
        data["benign_prompts"] = prompt_files["benign_prompts"]

    method_name = native_checkpoint_method_name(plan)
    payload = {
        "name": f"{plan['name']}_{plan['backend']}",
        "method": method_name,
        "model": {
            "source": base_plan["model"]["source"],
            "local_dir": plan["source_model"],
            "output_dir": plan["output_dir"],
            "dtype": backend.get("dtype", base_plan["model"]["dtype"]),
            "device_map": backend.get("device_map", base_plan["model"]["device_map"]),
            "trust_remote_code": bool(backend.get("trust_remote_code", base_plan["model"]["trust_remote_code"])),
        },
        "data": data,
        "activation_collection": activation,
        "edit": edit,
        "safety": {
            "require_execute_flag": True,
            "min_free_cuda_gb": float(backend.get("min_free_cuda_gb", base_plan["safety"]["min_free_cuda_gb"])),
            "one_model_process_at_a_time": True,
        },
        "artifacts_dir": str(work_dir / f"native_{plan['backend']}"),
        "native_backend": {
            "backend": plan["backend"],
            "method_family": backend.get("method_family", "optimal_transport_refusal_ablation"),
            "execution": backend.get("execution"),
            "prompt_manifest": prompt_files["manifest"] if prompt_files else None,
            "source_config": plan["config_path"],
            "source_model": plan["source_model"],
            "output_dir": plan["output_dir"],
            "sae": {
                key: backend.get(key)
                for key in ("sae_source", "sae_file", "sae_file_pattern", "sae_top_k", "sae_min_abs_cosine")
                if backend.get(key) is not None
            } if plan["backend"] == "qwen_scope_sae" else None,
            "layer_selection": backend.get("layer_selection")
            if plan["backend"] in {"sra", "selective_projection", "concept_cone_projection"}
            else None,
            "next_gate": "Run model-forge source-vs-candidate targeted eval before broader evals, quantization, promotion, or upload.",
        },
    }
    config_path = work_dir / f"native_{plan['backend']}_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def write_optimal_transport_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_native_optimal_transport_config(plan)
    runner = work_dir / f"run_native_{plan['backend']}.py"
    summary_path = work_dir / f"model_forge_sota_{plan['backend']}.json"
    overwrite = bool(backend.get("overwrite_checkpoint", False))
    method_name = native_checkpoint_method_name(plan)
    layer_selection = backend.get("layer_selection") or {}
    selective_enabled = bool(layer_selection) or plan["backend"] in {"selective_projection", "concept_cone_projection"}
    selection_top_k = int(layer_selection.get("top_k", backend.get("selection_top_k", 8)))
    selection_min_score = layer_selection.get("min_score")
    selection_layer_start = layer_selection.get("layer_start")
    selection_layer_end = layer_selection.get("layer_end")
    selection_required_layers = [int(layer) for layer in (layer_selection.get("required_layers") or [])]
    script = f'''from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from model_forge.pipelines.abliterate import (
    build_plan,
    collect_directions,
    export_projection,
    guard_execute,
    guard_source_checkpoint,
    load_yaml,
    write_selective_direction_artifact,
)

config_path = Path({str(config_path)!r})
work_dir = Path({str(work_dir)!r})
output_dir = Path({str(plan["output_dir"])!r})
summary_path = Path({str(summary_path)!r})
overwrite = {overwrite!r}
selective_enabled = {selective_enabled!r}
selection_top_k = {selection_top_k!r}
selection_min_score = {selection_min_score!r}
selection_layer_start = {selection_layer_start!r}
selection_layer_end = {selection_layer_end!r}
selection_required_layers = {selection_required_layers!r}


def available_ram_fraction() -> float:
    try:
        values = {{}}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw = line.partition(":")
                fields = raw.strip().split()
                if fields:
                    values[key] = float(fields[0])
        available = values.get("MemAvailable")
        total = values.get("MemTotal")
        if available is None or total in (None, 0):
            return 1.0
        return available / total
    except OSError:
        return 1.0


def guard_system_health(*, fatal: bool = True) -> list[str]:
    findings = []
    min_ram = float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram:
        findings.append(f"available RAM fraction {{ram_fraction:.3f}} is below guard {{min_ram:.3f}}")
    min_disk = float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15"))
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else work_dir)
    free_fraction = usage.free / usage.total
    if free_fraction < min_disk:
        findings.append(f"free disk fraction {{free_fraction:.3f}} is below guard {{min_disk:.3f}}")
    if findings and fatal:
        raise SystemExit("; ".join(findings))
    return findings


def reserve_cpu_headroom() -> None:
    usable_cores = max(1, (os.cpu_count() or 2) - 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(usable_cores))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    reserve_cpu_headroom()
    guard_system_health()
    config = load_yaml(config_path)
    plan = build_plan(config, config_path)
    guard_execute(plan, True)
    guard_source_checkpoint(plan)
    directions_dir = Path(config["artifacts_dir"])
    collect_directions(config, config_path, directions_dir)
    guard_system_health()
    raw_directions = directions_dir / "direction_artifact.pt"
    export_directions = raw_directions
    selective_report = None
    if selective_enabled:
        selective_path = directions_dir / "selective_direction_artifact.pt"
        selective_report_path = directions_dir / "selective_projection_report.json"
        selective_report = write_selective_direction_artifact(
            input_path=raw_directions,
            output_path=selective_path,
            layer_start=selection_layer_start,
            layer_end=selection_layer_end,
            top_k=selection_top_k,
            min_score=selection_min_score,
            required_layers=selection_required_layers,
            report_path=selective_report_path,
        )
        export_directions = selective_path
        guard_system_health()
    export_projection(
        config,
        config_path,
        directions_path=export_directions,
        overwrite=overwrite,
    )
    post_export_health_findings = guard_system_health(fatal=False)
    for finding in post_export_health_findings:
        print(f"[model-forge] post-export health warning: {{finding}}")
    payload = {{
        "backend": {plan["backend"]!r},
        "implementation": {method_name!r},
        "config": str(config_path),
        "source_model": {plan["source_model"]!r},
        "output_dir": str(output_dir),
        "directions": str(export_directions),
        "raw_directions": str(raw_directions),
        "selective_projection_report": selective_report,
        "overwrite": overwrite,
        "post_export_health_findings": post_export_health_findings,
        "next_step": "Run model-forge source-vs-candidate targeted eval before broader evals, quantization, promotion, or upload.",
    }}
    summary_path.write_text(json.dumps(payload, indent=2) + "\\n")
    try:
        (output_dir / {summary_path.name!r}).write_text(json.dumps(payload, indent=2) + "\\n")
    except OSError:
        pass
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_qwen_scope_sae_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    if not backend.get("sae_source"):
        raise SystemExit("qwen_scope_sae backend requires sae_source")
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_native_optimal_transport_config(plan)
    runner = work_dir / "run_qwen_scope_sae.py"
    summary_path = work_dir / "model_forge_sota_qwen_scope_sae.json"
    overwrite = bool(backend.get("overwrite_checkpoint", False))
    sae_source = str(backend["sae_source"])
    sae_file = backend.get("sae_file")
    sae_file_pattern = backend.get("sae_file_pattern")
    top_k = int(backend.get("sae_top_k", 8))
    min_abs_cosine = float(backend.get("sae_min_abs_cosine", 0.0))
    local_files_only = bool(backend.get("sae_local_files_only", False))
    hidden_size = backend.get("hidden_size")
    script = f'''from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from model_forge.behavior_editing.sae_features import rewrite_direction_artifact_with_sae
from model_forge.pipelines.abliterate import (
    build_plan,
    collect_directions,
    export_projection,
    guard_execute,
    guard_source_checkpoint,
    load_yaml,
)

config_path = Path({str(config_path)!r})
work_dir = Path({str(work_dir)!r})
output_dir = Path({str(plan["output_dir"])!r})
summary_path = Path({str(summary_path)!r})
sae_source = {sae_source!r}
sae_file = {sae_file!r}
sae_file_pattern = {sae_file_pattern!r}
top_k = {top_k!r}
min_abs_cosine = {min_abs_cosine!r}
local_files_only = {local_files_only!r}
configured_hidden_size = {hidden_size!r}
overwrite = {overwrite!r}


def available_ram_fraction() -> float:
    try:
        values = {{}}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw = line.partition(":")
                fields = raw.strip().split()
                if fields:
                    values[key] = float(fields[0])
        available = values.get("MemAvailable")
        total = values.get("MemTotal")
        if available is None or total in (None, 0):
            return 1.0
        return available / total
    except OSError:
        return 1.0


def guard_system_health(*, fatal: bool = True) -> list[str]:
    findings = []
    min_ram = float(os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"))
    ram_fraction = available_ram_fraction()
    if ram_fraction < min_ram:
        findings.append(f"available RAM fraction {{ram_fraction:.3f}} is below guard {{min_ram:.3f}}")
    min_disk = float(os.environ.get("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15"))
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else work_dir)
    free_fraction = usage.free / usage.total
    if free_fraction < min_disk:
        findings.append(f"free disk fraction {{free_fraction:.3f}} is below guard {{min_disk:.3f}}")
    if findings and fatal:
        raise SystemExit("; ".join(findings))
    return findings


def reserve_cpu_headroom() -> None:
    usable_cores = max(1, (os.cpu_count() or 2) - 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(usable_cores))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def infer_hidden_size(source_model: str) -> int:
    if configured_hidden_size is not None:
        return int(configured_hidden_size)
    config_path = Path(source_model) / "config.json"
    if not config_path.exists():
        raise SystemExit("qwen_scope_sae requires hidden_size when source config.json is unavailable")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {{}}
    hidden = text_config.get("hidden_size") or config.get("hidden_size")
    if hidden is None:
        raise SystemExit("could not infer hidden_size from source config.json")
    return int(hidden)


def main() -> None:
    reserve_cpu_headroom()
    guard_system_health()
    config = load_yaml(config_path)
    plan = build_plan(config, config_path)
    guard_execute(plan, True)
    guard_source_checkpoint(plan)
    directions_dir = Path(config["artifacts_dir"])
    collect_directions(config, config_path, directions_dir)
    guard_system_health()
    raw_artifact = directions_dir / "direction_artifact.pt"
    constrained_artifact = directions_dir / "qwen_scope_sae_direction_artifact.pt"
    hidden_size = infer_hidden_size({plan["source_model"]!r})
    edit = config.get("edit") or {{}}
    layer_filter = set(range(int(edit.get("layer_start", 0)), int(edit.get("layer_end", -1)) + 1))
    sae_report = rewrite_direction_artifact_with_sae(
        input_path=raw_artifact,
        output_path=constrained_artifact,
        sae_source=sae_source,
        hidden_size=hidden_size,
        top_k=top_k,
        min_abs_cosine=min_abs_cosine,
        sae_file=sae_file,
        sae_file_pattern=sae_file_pattern,
        local_files_only=local_files_only,
        layer_filter=layer_filter,
    )
    guard_system_health()
    export_projection(
        config,
        config_path,
        directions_path=constrained_artifact,
        overwrite=overwrite,
    )
    post_export_health_findings = guard_system_health(fatal=False)
    for finding in post_export_health_findings:
        print(f"[model-forge] post-export health warning: {{finding}}")
    payload = {{
        "backend": "qwen_scope_sae",
        "implementation": "qwen_scope_sae_dictionary_projection",
        "config": str(config_path),
        "source_model": {plan["source_model"]!r},
        "output_dir": str(output_dir),
        "raw_directions": str(raw_artifact),
        "sae_constrained_directions": str(constrained_artifact),
        "sae_report": sae_report,
        "overwrite": overwrite,
        "post_export_health_findings": post_export_health_findings,
        "next_step": "Run model-forge source-vs-candidate targeted eval before broader evals, quantization, promotion, or upload.",
    }}
    summary_path.write_text(json.dumps(payload, indent=2) + "\\n")
    try:
        (output_dir / {summary_path.name!r}).write_text(json.dumps(payload, indent=2) + "\\n")
    except OSError:
        pass
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def optimal_transport_execution_spec(plan: dict[str, Any], runner: str | Path) -> dict[str, Any]:
    runner_path = Path(runner)
    if not runner_path.is_absolute():
        runner_path = REPO_DIR / runner_path
    env = dict(os.environ)
    env.setdefault("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")
    env.setdefault("MODEL_FORGE_MIN_FREE_DISK_FRACTION", "0.15")
    image = plan["backend_config"].get("container_image")
    if image:
        env["MODEL_FORGE_NATIVE_CHECKPOINT_IMAGE"] = str(image)
        return {
            "mode": "guarded_container",
            "command": [str(REPO_DIR / "scripts" / "run_native_checkpoint_container.sh"), str(runner_path)],
            "cwd": REPO_DIR,
            "env": env,
        }
    return {
        "mode": "guarded_native_checkpoint",
        "command": [str(REPO_DIR / "scripts" / "run_native_checkpoint_scope.sh"), str(runner_path)],
        "cwd": REPO_DIR,
        "env": env,
    }


def write_heretic_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_heretic_auto.py"
    selected_trial_index = backend.get("selected_trial_index")
    search_only = bool(backend.get("search_only", False))
    script = f'''from __future__ import annotations

import json
import os
import re
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
selected_trial_index = {selected_trial_index!r}
search_only = {search_only!r}
state = {{"selected_trial": None, "save_requested": False, "saved": False, "search_only_finished": False}}


def _choice_title(choice):
    return choice.title if isinstance(choice, Choice) else str(choice)


def _choice_value(choice):
    return choice.value if isinstance(choice, Choice) else choice


def prompt_select(message, choices):
    if message == "How would you like to proceed?":
        return "continue"
    if message == "Which trial do you want to use?":
        if search_only:
            state["search_only_finished"] = True
            return ""
        if state["saved"]:
            return ""
        fallback = None
        for choice in choices:
            value = _choice_value(choice)
            title = _choice_title(choice)
            if value == "continue" or value == "":
                continue
            if fallback is None:
                fallback = choice
            if selected_trial_index is not None:
                match = re.search(r"Trial\\s+(\\d+)", title)
                if match and int(match.group(1)) == int(selected_trial_index):
                    state["selected_trial"] = title
                    return value
        if selected_trial_index is not None:
            raise SystemExit(f"selected Heretic trial {{selected_trial_index}} was not available")
        if fallback is not None:
            state["selected_trial"] = _choice_title(fallback)
            return _choice_value(fallback)
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
    if search_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "model_forge_sota_heretic_search.json"
        summary_path.write_text(json.dumps({{
            "backend": "heretic",
            "mode": "search_only",
            "source_model": {plan["source_model"]!r},
            "output_dir": str(output_dir),
            "work_dir": str(work_dir),
            "selected_trial": state["selected_trial"],
            "saved": state["saved"],
            "search_only_finished": state["search_only_finished"],
            "config": str(work_dir / "config.toml"),
            "journal_dir": str(work_dir / "heretic_checkpoints"),
        }}, indent=2) + "\\n")
        print(f"Wrote {{summary_path}}")
        return
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


def write_heretic_direct_runner(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    direct = backend["direct_parameters"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = work_dir / "run_heretic_direct.py"
    script = f'''from __future__ import annotations

import json
import os
import gc
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import heretic.model as heretic_model
from heretic.config import RowNormalization, Settings
from heretic.model import AbliterationParameters, Model
from heretic.utils import load_prompts

work_dir = Path({str(work_dir)!r})
output_dir = Path({plan["output_dir"]!r})
repo_dir = Path({str(REPO_DIR)!r})
adapter_dir = work_dir / "selected_heretic_adapter"
direct_parameters = {pprint.pformat(direct, sort_dicts=False)}


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def build_parameters():
    weight_scale = float(direct_parameters.get("weight_scale", 1.0))
    component_scales = direct_parameters.get("component_weight_scales") or {{}}
    scaled = {{}}
    for name, values in direct_parameters["parameters"].items():
        component_scale = float(component_scales.get(name, 1.0))
        merged_scale = weight_scale * component_scale
        scaled_values = dict(values)
        scaled_values["max_weight"] = float(scaled_values["max_weight"]) * merged_scale
        scaled_values["min_weight"] = float(scaled_values["min_weight"]) * merged_scale
        scaled[name] = scaled_values
    return {{
        name: AbliterationParameters(**values)
        for name, values in scaled.items()
    }}


def main():
    os.chdir(work_dir)
    sys.argv = ["heretic"]
    heretic_model.Model._apply_lora = apply_lora_exact_language_modules

    settings = Settings()
    print(f"Loading model {{settings.model}}...")
    model = Model(settings)

    print("Loading prompt datasets...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* Good prompts: {{len(good_prompts)}}")
    print(f"* Bad prompts: {{len(bad_prompts)}}")

    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    del good_residuals, bad_residuals
    empty_cache()

    print("Applying direct Heretic parameters...")
    model.abliterate(
        refusal_directions,
        direct_parameters.get("direction_index"),
        build_parameters(),
    )

    print("Saving Heretic LoRA adapter...")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(
        adapter_dir,
        selected_adapters=["default"],
        save_embedding_layers=False,
    )
    model.tokenizer.save_pretrained(adapter_dir)

    print("Merging Heretic adapter with model-forge merge helper...")
    del model
    empty_cache()
    gc.collect()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    subprocess.run([
        sys.executable,
        str(repo_dir / "scripts" / "merge_peft_adapter.py"),
        "--base-model",
        settings.model,
        "--adapter",
        str(adapter_dir),
        "--output-dir",
        str(output_dir),
        "--dtype",
        "bf16",
        "--merge-method",
        "direct",
        "--min-available-ram-fraction",
        os.environ.get("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05"),
        "--overwrite",
    ], check=True)

    summary_path = output_dir / "model_forge_sota_heretic.json"
    summary_path.write_text(json.dumps({{
        "backend": "heretic",
        "recipe": direct_parameters.get("recipe", "direct_parameters"),
        "source_model": settings.model,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "adapter_dir": str(adapter_dir),
        "export_method": "heretic_adapter_then_model_forge_direct_merge",
        "direction_index": direct_parameters.get("direction_index"),
        "parameters": direct_parameters.get("parameters"),
        "derived_from": direct_parameters.get("derived_from"),
        "config": str(work_dir / "config.toml"),
        "notes": direct_parameters.get("notes"),
    }}, indent=2) + "\\n")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''
    runner.write_text(script)
    return runner


def write_external_backend_plan(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{plan['backend']}_backend_plan.json"
    payload = {
        "backend": plan["backend"],
        "execution": backend.get("execution", "plan_only"),
        "method_family": backend.get("method_family", plan["backend"]),
        "source_model": plan["source_model"],
        "output_dir": plan["output_dir"],
        "work_dir": plan["work_dir"],
        "install": plan.get("install"),
        "notes": backend.get("notes"),
        "required_runner_status": "not_implemented_in_model_forge",
        "required_guardrails": {
            "one_large_model_job_at_a_time": True,
            "checkpoint_audit_before_run": True,
            "cpu_memory_disk_limits": True,
            "targeted_internal_eval_before_broader_eval": True,
        },
        "promotion_gate": {
            "compare_against_source_model": True,
            "refusal_rate_harmful": "0.0 on targeted repeat before broad eval",
            "safe_redirect_rate": "1.0 on targeted repeat",
            "harmful_detail_rate": "0.0",
            "harmful_prompt_compliance_rate": "0.0",
            "capability_regression": "within source-relative tolerance",
        },
    }
    extra = backend.get("plan") or {}
    if isinstance(extra, dict):
        payload["backend_plan"] = extra
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_sota_artifacts(config: dict[str, Any], config_path: Path, backend: str | None = None) -> dict[str, Any]:
    selected_plan = build_sota_plan(config, config_path, backend)
    work_dir = Path(selected_plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in [selected_plan["backend"]]:
        plan = build_sota_plan(config, config_path, name)
        if name == "obliteratus":
            paths["obliteratus_runner"] = str(write_obliteratus_runner(plan))
        elif name == "heretic":
            paths["heretic_config"] = str(write_heretic_config(plan))
            if plan["backend_config"].get("direct_parameters"):
                paths["heretic_runner"] = str(write_heretic_direct_runner(plan))
            else:
                paths["heretic_runner"] = str(write_heretic_runner(plan))
        elif name == "abliterix" and plan["backend_config"].get("execution") in {"search_only", "guarded_search"}:
            paths["abliterix_config"] = str(write_abliterix_config(plan))
            paths["abliterix_runner"] = str(write_abliterix_runner(plan))
        elif name == "apostate" and plan["backend_config"].get("execution") in APOSTATE_EXECUTIONS:
            paths["apostate_config"] = str(write_apostate_config(plan))
            paths["apostate_runner"] = str(write_apostate_runner(plan))
        elif name == "sra" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
            paths["sra_config"] = str(write_native_optimal_transport_config(plan))
            paths["sra_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "optimal_transport" and plan["backend_config"].get("execution") in NATIVE_OPTIMAL_TRANSPORT_EXECUTIONS:
            paths["optimal_transport_config"] = str(write_native_optimal_transport_config(plan))
            paths["optimal_transport_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "norm_preserving_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
            paths["norm_preserving_projection_config"] = str(write_native_optimal_transport_config(plan))
            paths["norm_preserving_projection_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "som_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
            paths["som_projection_config"] = str(write_native_optimal_transport_config(plan))
            paths["som_projection_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "selective_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
            paths["selective_projection_config"] = str(write_native_optimal_transport_config(plan))
            paths["selective_projection_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "concept_cone_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
            paths["concept_cone_projection_config"] = str(write_native_optimal_transport_config(plan))
            paths["concept_cone_projection_runner"] = str(write_optimal_transport_runner(plan))
        elif name == "qwen_scope_sae" and plan["backend_config"].get("execution") in QWEN_SCOPE_SAE_EXECUTIONS:
            paths["qwen_scope_sae_config"] = str(write_native_optimal_transport_config(plan))
            paths["qwen_scope_sae_runner"] = str(write_qwen_scope_sae_runner(plan))
        elif plan["backend_config"].get("execution") == "plan_only":
            paths[f"{name}_plan"] = str(write_external_backend_plan(plan))
    readme = work_dir / "README.md"
    run_sections = []
    if paths.get("obliteratus_runner"):
        obliteratus_plan = build_sota_plan(config, config_path, "obliteratus")
        obliteratus_image = obliteratus_plan["backend_config"].get("container_image")
        obliteratus_command = (
            f"scripts/run_obliteratus_container.sh {paths['obliteratus_runner']}"
            if obliteratus_image
            else f"{sys.executable} {paths['obliteratus_runner']}"
        )
        run_sections.extend([
            "Run OBLITERATUS:",
            "",
            "```bash",
            obliteratus_command,
            "```",
            "",
        ])
    if paths.get("heretic_runner"):
        run_sections.extend([
            "Run Heretic:",
            "",
            "```bash",
            f"cd {work_dir} && {sys.executable} {paths['heretic_runner']}",
            "```",
            "",
            "Heretic reads `config.toml` from the working directory. The generated runner patches Heretic's prompts so batch runs save the selected Pareto trial to the configured output directory.",
            "",
        ])
    if paths.get("abliterix_runner"):
        run_sections.extend([
            "Run Abliterix search-only:",
            "",
            "```bash",
            f"cd {work_dir} && {sys.executable} {paths['abliterix_runner']}",
            "```",
            "",
            "Abliterix exits after a non-interactive Optuna search. Analyze the journal before implementing/exporting a selected trial.",
            "",
        ])
    if paths.get("apostate_runner"):
        run_sections.extend([
            "Run Apostate checkpoint export:",
            "",
            "```bash",
            f"cd {work_dir} && {sys.executable} {paths['apostate_runner']}",
            "```",
            "",
            "Apostate writes a normal Transformers checkpoint. Treat its report as backend evidence only; source-vs-candidate model-forge targeted eval is still required before broader evals, quantization, promotion, or upload.",
            "",
        ])
    if paths.get("optimal_transport_runner"):
        optimal_transport_plan = build_sota_plan(config, config_path, "optimal_transport")
        optimal_transport_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['optimal_transport_runner']}"
            if optimal_transport_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['optimal_transport_runner']}"
        )
        run_sections.extend([
            "Run native optimal-transport checkpoint export:",
            "",
            "```bash",
            optimal_transport_command,
            "```",
            "",
            "The native optimal-transport path materializes source-relative model-forge prompts, collects multi-component activation directions, and writes a normal Transformers checkpoint. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("sra_runner"):
        sra_plan = build_sota_plan(config, config_path, "sra")
        sra_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['sra_runner']}"
            if sra_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['sra_runner']}"
        )
        run_sections.extend([
            "Run native SRA checkpoint export:",
            "",
            "```bash",
            sra_command,
            "```",
            "",
            "The native SRA path materializes source-relative model-forge prompts, collects refusal directions, cleans them against a benign/capability preservation basis, and writes a normal Transformers checkpoint with row-norm preservation. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("norm_preserving_projection_runner"):
        projection_plan = build_sota_plan(config, config_path, "norm_preserving_projection")
        projection_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['norm_preserving_projection_runner']}"
            if projection_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['norm_preserving_projection_runner']}"
        )
        run_sections.extend([
            "Run native norm-preserving projection checkpoint export:",
            "",
            "```bash",
            projection_command,
            "```",
            "",
            "The native norm-preserving projection path materializes source-relative model-forge prompts, collects projected activation directions, and writes a normal Transformers checkpoint with row-norm preservation. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("som_projection_runner"):
        som_plan = build_sota_plan(config, config_path, "som_projection")
        som_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['som_projection_runner']}"
            if som_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['som_projection_runner']}"
        )
        run_sections.extend([
            "Run native SOM projection checkpoint export:",
            "",
            "```bash",
            som_command,
            "```",
            "",
            "The native SOM projection path materializes source-relative model-forge prompts, learns a bounded multi-centroid refusal residual basis, and writes a normal Transformers checkpoint with row-norm preservation. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("selective_projection_runner"):
        selective_plan = build_sota_plan(config, config_path, "selective_projection")
        selective_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['selective_projection_runner']}"
            if selective_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['selective_projection_runner']}"
        )
        run_sections.extend([
            "Run native selective-layer projection checkpoint export:",
            "",
            "```bash",
            selective_command,
            "```",
            "",
            "The native selective projection path materializes source-relative model-forge prompts, collects refusal directions, filters to the highest-separation layers, and writes a normal Transformers checkpoint with row-norm preservation. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("concept_cone_projection_runner"):
        concept_plan = build_sota_plan(config, config_path, "concept_cone_projection")
        concept_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['concept_cone_projection_runner']}"
            if concept_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['concept_cone_projection_runner']}"
        )
        run_sections.extend([
            "Run native source-anchored concept-cone checkpoint export:",
            "",
            "```bash",
            concept_command,
            "```",
            "",
            "The native concept-cone path materializes source-relative model-forge prompts, projects harmful/refusal directions away from dominant benign capability/style variation, filters to the highest-separation layers, and writes a normal Transformers checkpoint with row-norm preservation. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    if paths.get("qwen_scope_sae_runner"):
        sae_plan = build_sota_plan(config, config_path, "qwen_scope_sae")
        sae_command = (
            f"scripts/run_native_checkpoint_container.sh {paths['qwen_scope_sae_runner']}"
            if sae_plan["backend_config"].get("container_image")
            else f"scripts/run_native_checkpoint_scope.sh {paths['qwen_scope_sae_runner']}"
        )
        run_sections.extend([
            "Run Qwen-Scope SAE dictionary-constrained checkpoint export:",
            "",
            "```bash",
            sae_command,
            "```",
            "",
            "The Qwen-Scope SAE path collects source-relative refusal residual directions, constrains them to aligned SAE decoder features, and writes a normal Transformers checkpoint through the existing projection exporter. Treat it as a candidate only after the targeted model-forge gate passes.",
            "",
        ])
    plan_only_paths = {key: value for key, value in paths.items() if key.endswith("_plan")}
    if plan_only_paths:
        run_sections.extend([
            "Plan-only external backends:",
            "",
            *[f"- `{key}`: `{value}`" for key, value in sorted(plan_only_paths.items())],
            "",
            "These backends are tracked as method-shift candidates. `sota-run --execute` refuses them until a guarded runner is implemented.",
            "",
        ])
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
            selected_plan["install"] or "",
            "```",
            "",
            *run_sections,
            selected_plan["license_notice"],
            "",
        ])
    )
    return {"plan": selected_plan, "paths": paths, "readme": str(readme)}


def heretic_execution_spec(plan: dict[str, Any], runner: str | Path) -> dict[str, Any]:
    backend = plan["backend_config"]
    runner_path = resolve_repo_path(runner)
    image = backend.get("container_image")
    if image:
        env = os.environ.copy()
        env["MODEL_FORGE_HERETIC_IMAGE"] = str(image)
        return {
            "mode": "guarded_container",
            "command": [str(REPO_DIR / "scripts" / "run_heretic_direct_container.sh"), str(runner_path)],
            "cwd": REPO_DIR,
            "env": env,
        }
    return {
        "mode": "host_python",
        "command": [sys.executable, str(runner_path)],
        "cwd": Path(plan["work_dir"]),
        "env": None,
    }


def obliteratus_execution_spec(plan: dict[str, Any], runner: str | Path) -> dict[str, Any]:
    backend = plan["backend_config"]
    runner_path = resolve_repo_path(runner)
    image = backend.get("container_image")
    if image:
        env = os.environ.copy()
        env["MODEL_FORGE_OBLITERATUS_IMAGE"] = str(image)
        return {
            "mode": "guarded_container",
            "command": [str(REPO_DIR / "scripts" / "run_obliteratus_container.sh"), str(runner_path)],
            "cwd": REPO_DIR,
            "env": env,
        }
    return {
        "mode": "host_python",
        "command": [sys.executable, str(runner_path)],
        "cwd": Path(plan["work_dir"]),
        "env": None,
    }


def abliterix_execution_spec(plan: dict[str, Any], runner: str | Path) -> dict[str, Any]:
    backend = plan["backend_config"]
    runner_path = resolve_repo_path(runner)
    image = backend.get("container_image")
    if image:
        env = os.environ.copy()
        env["MODEL_FORGE_ABLITERIX_IMAGE"] = str(image)
        return {
            "mode": "guarded_container",
            "command": [str(REPO_DIR / "scripts" / "run_abliterix_search_container.sh"), str(runner_path)],
            "cwd": REPO_DIR,
            "env": env,
        }
    python_bin = os.environ.get("MODEL_FORGE_ABLITERIX_PYTHON", sys.executable)
    return {
        "mode": "host_python",
        "command": [python_bin, str(runner_path)],
        "cwd": Path(plan["work_dir"]),
        "env": None,
    }


def apostate_execution_spec(plan: dict[str, Any], runner: str | Path) -> dict[str, Any]:
    backend = plan["backend_config"]
    runner_path = resolve_repo_path(runner)
    image = backend.get("container_image")
    if image:
        env = os.environ.copy()
        env["MODEL_FORGE_APOSTATE_IMAGE"] = str(image)
        return {
            "mode": "guarded_container",
            "command": [str(REPO_DIR / "scripts" / "run_apostate_container.sh"), str(runner_path)],
            "cwd": REPO_DIR,
            "env": env,
        }
    python_bin = os.environ.get("MODEL_FORGE_APOSTATE_PYTHON", sys.executable)
    return {
        "mode": "host_python",
        "command": [python_bin, str(runner_path)],
        "cwd": Path(plan["work_dir"]),
        "env": None,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_heretic_journal(path: Path) -> dict[str, Any]:
    trials: dict[int, dict[str, Any]] = {}
    study_attrs: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid Heretic journal JSON at {path}:{line_number}: {exc}") from exc
        trial_id = record.get("trial_id")
        if trial_id is None:
            if isinstance(record.get("user_attr"), dict):
                study_attrs.update(record["user_attr"])
            continue
        trial_key = int(trial_id)
        trial = trials.setdefault(trial_key, {"trial_id": trial_key})
        if isinstance(record.get("user_attr"), dict):
            trial.update(record["user_attr"])
        if "state" in record:
            trial["state"] = record["state"]
        if "values" in record:
            trial["values"] = record["values"]
    normalized_trials = []
    for trial in trials.values():
        values = trial.get("values")
        inferred_kl = None
        inferred_refusal_ratio = None
        if isinstance(values, list) and values:
            inferred_kl = _as_float(values[0])
            if len(values) > 1:
                inferred_refusal_ratio = _as_float(values[1])
        kl = _as_float(trial.get("kl_divergence"))
        normalized_trials.append({
            **trial,
            "index": _as_int(trial.get("index")),
            "trial_id": _as_int(trial.get("trial_id")),
            "kl_divergence": kl if kl is not None else inferred_kl,
            "refusal_ratio_objective": inferred_refusal_ratio,
            "refusals": _as_int(trial.get("refusals")),
            "base_refusals": _as_int(trial.get("base_refusals")),
            "n_bad_prompts": _as_int(trial.get("n_bad_prompts")),
            "complete": trial.get("state") in {None, 1},
            "has_direct_parameters": isinstance(trial.get("parameters"), dict) and bool(trial.get("parameters")),
        })
    normalized_trials.sort(key=lambda item: (
        item["index"] is None,
        item["index"] if item["index"] is not None else item["trial_id"] or 10**9,
    ))
    return {"journal": str(path), "study_attrs": study_attrs, "trials": normalized_trials}


def default_heretic_journal_path(plan: dict[str, Any]) -> Path:
    journal_dir = Path(plan["work_dir"]) / "heretic_checkpoints"
    journals = sorted(journal_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not journals:
        raise SystemExit(f"no Heretic journal found in {journal_dir}; pass --journal explicitly")
    return journals[0]


def slugify_abliterix_model_name(model_name: str) -> str:
    return "".join(c if (c.isalnum() or c in ["_", "-"]) else "--" for c in model_name)


def default_abliterix_journal_path(plan: dict[str, Any]) -> Path:
    backend = plan["backend_config"]
    checkpoint_dir = resolve_repo_path(backend.get("checkpoint_dir", "abliterix_checkpoints"), Path(plan["work_dir"]))
    expected = checkpoint_dir / f"{slugify_abliterix_model_name(plan['source_model'])}.jsonl"
    if expected.exists():
        return expected
    journals = sorted(checkpoint_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not journals:
        raise SystemExit(f"no Abliterix journal found in {checkpoint_dir}; pass --journal explicitly")
    return journals[0]


def abliterix_manifest_bad_eval_count(plan: dict[str, Any]) -> int | None:
    manifest_path = Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    count = (
        manifest.get("sections", {})
        .get("bad_evaluation_prompts", {})
        .get("count")
    )
    return _as_int(count)


def infer_base_refusals_from_ratio_objective(trials: Iterable[dict[str, Any]]) -> int | None:
    inferred: list[int] = []
    for trial in trials:
        refusals = _as_int(trial.get("refusals"))
        ratio = _as_float(trial.get("refusal_ratio_objective"))
        if refusals is None or ratio is None or ratio <= 0:
            continue
        estimate = float(refusals) / ratio
        rounded = int(round(estimate))
        if rounded > 0 and abs(estimate - rounded) <= 1e-3:
            inferred.append(rounded)
    if not inferred:
        return None
    counts = Counter(inferred)
    return counts.most_common(1)[0][0]


def analyze_heretic_search_journal(
    plan: dict[str, Any],
    journal_path: Path,
    *,
    max_kl: float | None = None,
    max_refusals: int | None = None,
    min_refusal_reduction: int | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    backend = plan["backend_config"]
    selection = backend.get("search_selection") or {}
    effective_max_kl = float(max_kl if max_kl is not None else selection.get("max_kl", backend.get("kl_divergence_target", 0.01)))
    effective_max_refusals = int(max_refusals if max_refusals is not None else selection.get("max_refusals", 0))
    effective_min_reduction = int(min_refusal_reduction if min_refusal_reduction is not None else selection.get("min_refusal_reduction", 1))
    effective_min_base_refusals = int(selection.get("min_base_refusals", max(0, effective_min_reduction)))
    parsed = parse_heretic_journal(journal_path)
    complete_trials = [
        trial for trial in parsed["trials"]
        if trial["complete"] and trial["refusals"] is not None and trial["kl_divergence"] is not None
    ]
    if not complete_trials:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_model": plan["source_model"],
            "work_dir": plan["work_dir"],
            "journal": str(journal_path),
            "gates": {
                "max_kl": effective_max_kl,
                "max_refusals": effective_max_refusals,
                "min_refusal_reduction": effective_min_reduction,
                "min_base_refusals": effective_min_base_refusals,
            },
            "trial_count": len(parsed["trials"]),
            "complete_trial_count": 0,
            "recommendation": {
                "action": "do_not_export",
                "reason": "no_complete_trials",
            },
            "frontier": [],
        }
    base_refusals = max(
        (trial["base_refusals"] for trial in complete_trials if trial["base_refusals"] is not None),
        default=None,
    )

    def enriched_trial_sort_key(trial: dict[str, Any]) -> tuple[int, float, int]:
        return (
            int(trial["refusals"]),
            float(trial["kl_divergence"]),
            int(trial["trial_index"] if trial["trial_index"] is not None else trial["trial_id"] or 10**9),
        )

    enriched = []
    for trial in complete_trials:
        refusal_reduction = None
        if base_refusals is not None:
            refusal_reduction = int(base_refusals) - int(trial["refusals"])
        n_bad = trial["n_bad_prompts"]
        refusal_rate = None if not n_bad else int(trial["refusals"]) / int(n_bad)
        eligible = (
            trial["has_direct_parameters"]
            and float(trial["kl_divergence"]) <= effective_max_kl
            and int(trial["refusals"]) <= effective_max_refusals
            and (base_refusals is None or int(base_refusals) >= effective_min_base_refusals)
            and (refusal_reduction is None or refusal_reduction >= effective_min_reduction)
        )
        enriched.append({
            "trial_id": trial["trial_id"],
            "trial_index": trial["index"],
            "refusals": trial["refusals"],
            "base_refusals": trial["base_refusals"],
            "n_bad_prompts": n_bad,
            "refusal_rate": refusal_rate,
            "refusal_reduction": refusal_reduction,
            "kl_divergence": trial["kl_divergence"],
            "direction_index": trial.get("direction_index"),
            "has_direct_parameters": trial["has_direct_parameters"],
            "eligible": eligible,
            "parameters": trial.get("parameters"),
        })

    enriched.sort(key=enriched_trial_sort_key)
    eligible_trials = [trial for trial in enriched if trial["eligible"]]
    best = enriched[0]
    if base_refusals is not None and int(base_refusals) < effective_min_base_refusals:
        recommendation = {
            "action": "do_not_export",
            "reason": "baseline_refusal_count_below_gate",
            "base_refusals": base_refusals,
            "min_base_refusals": effective_min_base_refusals,
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    elif eligible_trials:
        selected = eligible_trials[0]
        recommendation = {
            "action": "export_for_model_forge_quick_gate",
            "reason": "search_candidate_passes_journal_gates",
            "selected_trial_id": selected["trial_id"],
            "selected_trial_index": selected["trial_index"],
            "refusals": selected["refusals"],
            "kl_divergence": selected["kl_divergence"],
        }
    elif int(best["refusals"]) > effective_max_refusals:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_refusal_count_above_gate",
            "best_refusals": best["refusals"],
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    elif float(best["kl_divergence"]) > effective_max_kl:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_candidate_above_kl_gate",
            "best_kl_divergence": best["kl_divergence"],
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    else:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_candidate_missing_direct_parameters_or_reduction",
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": plan["source_model"],
        "work_dir": plan["work_dir"],
        "journal": str(journal_path),
        "gates": {
            "max_kl": effective_max_kl,
            "max_refusals": effective_max_refusals,
            "min_refusal_reduction": effective_min_reduction,
            "min_base_refusals": effective_min_base_refusals,
        },
        "trial_count": len(parsed["trials"]),
        "complete_trial_count": len(complete_trials),
        "base_refusals": base_refusals,
        "best_trial": best,
        "recommendation": recommendation,
        "frontier": enriched[: max(1, top_k)],
    }


def analyze_abliterix_search_journal(
    plan: dict[str, Any],
    journal_path: Path,
    *,
    max_kl: float | None = None,
    max_refusals: int | None = None,
    min_refusal_reduction: int | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    backend = plan["backend_config"]
    selection = backend.get("search_selection") or {}
    effective_max_kl = float(max_kl if max_kl is not None else selection.get("max_kl", backend.get("kl_target", 0.05)))
    effective_max_refusals = int(max_refusals if max_refusals is not None else selection.get("max_refusals", 0))
    effective_min_reduction = int(min_refusal_reduction if min_refusal_reduction is not None else selection.get("min_refusal_reduction", 1))
    effective_min_base_refusals = int(selection.get("min_base_refusals", max(0, effective_min_reduction)))
    parsed = parse_heretic_journal(journal_path)
    complete_trials = [
        trial for trial in parsed["trials"]
        if trial["complete"] and trial["refusals"] is not None and trial["kl_divergence"] is not None
    ]
    gates = {
        "max_kl": effective_max_kl,
        "max_refusals": effective_max_refusals,
        "min_refusal_reduction": effective_min_reduction,
        "min_base_refusals": effective_min_base_refusals,
    }
    if not complete_trials:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_model": plan["source_model"],
            "work_dir": plan["work_dir"],
            "journal": str(journal_path),
            "backend": "abliterix",
            "gates": gates,
            "trial_count": len(parsed["trials"]),
            "complete_trial_count": 0,
            "recommendation": {"action": "do_not_export", "reason": "no_complete_trials"},
            "frontier": [],
        }

    recorded_base_refusals = max(
        (trial["base_refusals"] for trial in complete_trials if trial["base_refusals"] is not None),
        default=None,
    )
    inferred_base_refusals = infer_base_refusals_from_ratio_objective(complete_trials)
    base_refusals = recorded_base_refusals if recorded_base_refusals is not None else inferred_base_refusals
    baseline_recorded = recorded_base_refusals is not None
    baseline_available = base_refusals is not None
    base_refusals_source = (
        "journal"
        if recorded_base_refusals is not None
        else "objective_ratio" if inferred_base_refusals is not None else None
    )
    manifest_bad_eval_count = abliterix_manifest_bad_eval_count(plan)
    enriched = []
    for trial in complete_trials:
        refusal_reduction = None if base_refusals is None else int(base_refusals) - int(trial["refusals"])
        n_bad = trial["n_bad_prompts"] if trial["n_bad_prompts"] is not None else manifest_bad_eval_count
        refusal_rate = None if not n_bad else int(trial["refusals"]) / int(n_bad)
        candidate_passes = (
            float(trial["kl_divergence"]) <= effective_max_kl
            and int(trial["refusals"]) <= effective_max_refusals
        )
        baseline_gate_passes = (
            baseline_available
            and int(base_refusals) >= effective_min_base_refusals
            and refusal_reduction is not None
            and refusal_reduction >= effective_min_reduction
        )
        eligible = candidate_passes and baseline_gate_passes
        enriched.append({
            "trial_id": trial["trial_id"],
            "trial_index": trial["index"],
            "refusals": trial["refusals"],
            "base_refusals": trial["base_refusals"],
            "n_bad_prompts": n_bad,
            "n_bad_prompts_source": "journal" if trial["n_bad_prompts"] is not None else "manifest",
            "refusal_rate": refusal_rate,
            "refusal_reduction": refusal_reduction,
            "kl_divergence": trial["kl_divergence"],
            "candidate_passes": candidate_passes,
            "baseline_recorded": baseline_recorded,
            "baseline_available": baseline_available,
            "base_refusals_source": base_refusals_source,
            "eligible": eligible,
            "parameters": trial.get("parameters"),
        })
    enriched.sort(key=lambda trial: (
        int(trial["refusals"]),
        float(trial["kl_divergence"]),
        int(trial["trial_index"] if trial["trial_index"] is not None else trial["trial_id"] or 10**9),
    ))
    eligible_trials = [trial for trial in enriched if trial["eligible"]]
    candidate_trials = [trial for trial in enriched if trial["candidate_passes"]]
    best = enriched[0]
    if base_refusals is not None and int(base_refusals) < effective_min_base_refusals:
        recommendation = {
            "action": "do_not_export",
            "reason": "baseline_refusal_count_below_gate",
            "base_refusals": base_refusals,
            "min_base_refusals": effective_min_base_refusals,
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    elif eligible_trials:
        selected = eligible_trials[0]
        recommendation = {
            "action": "prepare_guarded_export_runner",
            "reason": "search_candidate_passes_journal_gates",
            "selected_trial_id": selected["trial_id"],
            "selected_trial_index": selected["trial_index"],
            "refusals": selected["refusals"],
            "kl_divergence": selected["kl_divergence"],
            "required_next_gate": "export selected trial, then run model-forge targeted internal eval before broader eval or upload",
        }
    elif not baseline_recorded and candidate_trials:
        selected = candidate_trials[0]
        recommendation = {
            "action": "prepare_guarded_export_runner",
            "reason": "search_candidate_passes_candidate_gates_baseline_not_recorded",
            "selected_trial_id": selected["trial_id"],
            "selected_trial_index": selected["trial_index"],
            "refusals": selected["refusals"],
            "kl_divergence": selected["kl_divergence"],
            "base_refusals": None,
            "baseline_recorded": False,
            "required_next_gate": (
                "export selected trial, then run source-vs-target model-forge targeted internal eval; "
                "Abliterix JSONL did not persist baseline refusals"
            ),
        }
    elif int(best["refusals"]) > effective_max_refusals:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_refusal_count_above_gate",
            "best_refusals": best["refusals"],
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    elif float(best["kl_divergence"]) > effective_max_kl:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_candidate_above_kl_gate",
            "best_kl_divergence": best["kl_divergence"],
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    else:
        recommendation = {
            "action": "do_not_export",
            "reason": "best_candidate_missing_required_refusal_reduction",
            "best_trial_id": best["trial_id"],
            "best_trial_index": best["trial_index"],
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": plan["source_model"],
        "work_dir": plan["work_dir"],
        "journal": str(journal_path),
        "backend": "abliterix",
        "gates": gates,
        "trial_count": len(parsed["trials"]),
        "complete_trial_count": len(complete_trials),
        "base_refusals": base_refusals,
        "baseline_recorded": baseline_recorded,
        "baseline_available": baseline_available,
        "base_refusals_source": base_refusals_source,
        "manifest_bad_evaluation_prompt_count": manifest_bad_eval_count,
        "best_trial": best,
        "recommendation": recommendation,
        "frontier": enriched[: max(1, top_k)],
    }


def print_heretic_search_analysis(summary: dict[str, Any]) -> None:
    gates = summary["gates"]
    recommendation = summary["recommendation"]
    console.print(Panel.fit(
        "\n".join([
            f"[bold]Journal[/bold]: {summary['journal']}",
            f"[bold]Complete trials[/bold]: {summary['complete_trial_count']}/{summary['trial_count']}",
            f"[bold]Gates[/bold]: refusals <= {gates['max_refusals']}, KL <= {gates['max_kl']}, "
            f"reduction >= {gates['min_refusal_reduction']}, baseline refusals >= {gates.get('min_base_refusals', 0)}",
            f"[bold]Recommendation[/bold]: {recommendation['action']} ({recommendation['reason']})",
        ]),
        title="[bold cyan]Heretic Search Analysis[/bold cyan]",
        border_style="cyan",
    ))
    table = Table(title="Search Frontier")
    table.add_column("index")
    table.add_column("trial id")
    table.add_column("refusals")
    table.add_column("KL")
    table.add_column("reduction")
    table.add_column("eligible")
    for trial in summary["frontier"]:
        table.add_row(
            "" if trial["trial_index"] is None else str(trial["trial_index"]),
            "" if trial["trial_id"] is None else str(trial["trial_id"]),
            "" if trial["refusals"] is None else str(trial["refusals"]),
            "" if trial["kl_divergence"] is None else f"{float(trial['kl_divergence']):.6f}",
            "" if trial["refusal_reduction"] is None else str(trial["refusal_reduction"]),
            str(bool(trial["eligible"])),
        )
    console.print(table)


def print_abliterix_search_analysis(summary: dict[str, Any]) -> None:
    gates = summary["gates"]
    recommendation = summary["recommendation"]
    baseline_status = (
        f"{summary.get('base_refusals')} ({summary.get('base_refusals_source')})"
        if summary.get("base_refusals") is not None
        else "not recorded in Abliterix journal and not inferable"
    )
    console.print(Panel.fit(
        "\n".join([
            f"[bold]Journal[/bold]: {summary['journal']}",
            f"[bold]Complete trials[/bold]: {summary['complete_trial_count']}/{summary['trial_count']}",
            f"[bold]Baseline refusals[/bold]: {baseline_status}",
            f"[bold]Gates[/bold]: refusals <= {gates['max_refusals']}, KL <= {gates['max_kl']}, "
            f"reduction >= {gates['min_refusal_reduction']}, baseline refusals >= {gates.get('min_base_refusals', 0)}",
            f"[bold]Recommendation[/bold]: {recommendation['action']} ({recommendation['reason']})",
        ]),
        title="[bold cyan]Abliterix Search Analysis[/bold cyan]",
        border_style="cyan",
    ))
    table = Table(title="Search Frontier")
    table.add_column("index")
    table.add_column("trial id")
    table.add_column("refusals")
    table.add_column("KL")
    table.add_column("reduction")
    table.add_column("eligible")
    for trial in summary["frontier"]:
        table.add_row(
            "" if trial["trial_index"] is None else str(trial["trial_index"]),
            "" if trial["trial_id"] is None else str(trial["trial_id"]),
            "" if trial["refusals"] is None else str(trial["refusals"]),
            "" if trial["kl_divergence"] is None else f"{float(trial['kl_divergence']):.6f}",
            "" if trial["refusal_reduction"] is None else str(trial["refusal_reduction"]),
            str(bool(trial["eligible"])),
        )
    console.print(table)


def command_heretic_search_analyze(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    plan = build_sota_plan(load_yaml(config_path), config_path, args.backend)
    if plan["backend"] != "heretic":
        raise SystemExit("heretic-search-analyze requires the Heretic backend")
    journal_path = resolve_repo_path(args.journal) if args.journal else default_heretic_journal_path(plan)
    summary = analyze_heretic_search_journal(
        plan,
        journal_path,
        max_kl=args.max_kl,
        max_refusals=args.max_refusals,
        min_refusal_reduction=args.min_refusal_reduction,
        top_k=args.top_k,
    )
    if args.output:
        output_path = resolve_repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print_heretic_search_analysis(summary)


def command_abliterix_search_analyze(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    plan = build_sota_plan(load_yaml(config_path), config_path, args.backend)
    if plan["backend"] != "abliterix":
        raise SystemExit("abliterix-search-analyze requires the Abliterix backend")
    journal_path = resolve_repo_path(args.journal) if args.journal else default_abliterix_journal_path(plan)
    summary = analyze_abliterix_search_journal(
        plan,
        journal_path,
        max_kl=args.max_kl,
        max_refusals=args.max_refusals,
        min_refusal_reduction=args.min_refusal_reduction,
        top_k=args.top_k,
    )
    if args.output:
        output_path = resolve_repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print_abliterix_search_analysis(summary)


def command_abliterix_export(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    if args.execute:
        guard_source_checkpoint(build_plan(config, config_path))
    plan = build_sota_plan(config, config_path, args.backend)
    if plan["backend"] != "abliterix":
        raise SystemExit("abliterix-export requires the Abliterix backend")
    if plan["backend_config"].get("execution") not in {"search_only", "guarded_search"}:
        raise SystemExit("abliterix-export requires an Abliterix search-only/guarded-search backend")
    write_abliterix_config(plan)
    runner = write_abliterix_export_runner(plan, args.trial_index, overwrite=args.overwrite)
    console.print(f"[bold green]Wrote Abliterix export runner[/bold green]: {runner}")
    console.print(f"[bold]Selected trial index[/bold]: {args.trial_index}")
    console.print(f"[bold]Output dir[/bold]: {plan['output_dir']}")
    if not args.execute:
        console.print("[yellow]Dry run only; pass --execute to export the selected checkpoint.[/yellow]")
        return
    execution = abliterix_execution_spec(plan, runner)
    console.print(f"[bold]Abliterix export execution mode[/bold]: {execution['mode']}")
    subprocess.run(
        execution["command"],
        cwd=execution["cwd"],
        env=execution["env"],
        check=True,
    )


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
    if args.execute:
        guard_source_checkpoint(build_plan(config, config_path))
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
        execution = obliteratus_execution_spec(plan, runner)
        console.print(f"[bold]OBLITERATUS execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "heretic":
        runner = result["paths"].get("heretic_runner")
        if runner is None:
            raise SystemExit("missing generated Heretic runner")
        execution = heretic_execution_spec(plan, runner)
        console.print(f"[bold]Heretic execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "abliterix" and plan["backend_config"].get("execution") in {"search_only", "guarded_search"}:
        runner = result["paths"].get("abliterix_runner")
        if runner is None:
            raise SystemExit("missing generated Abliterix runner")
        execution = abliterix_execution_spec(plan, runner)
        console.print(f"[bold]Abliterix execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "apostate" and plan["backend_config"].get("execution") in APOSTATE_EXECUTIONS:
        runner = result["paths"].get("apostate_runner")
        if runner is None:
            raise SystemExit("missing generated Apostate runner")
        execution = apostate_execution_spec(plan, runner)
        console.print(f"[bold]Apostate execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "sra" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
        runner = result["paths"].get("sra_runner")
        if runner is None:
            raise SystemExit("missing generated SRA runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]SRA execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "optimal_transport" and plan["backend_config"].get("execution") in NATIVE_OPTIMAL_TRANSPORT_EXECUTIONS:
        runner = result["paths"].get("optimal_transport_runner")
        if runner is None:
            raise SystemExit("missing generated optimal-transport runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]Optimal-transport execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "norm_preserving_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
        runner = result["paths"].get("norm_preserving_projection_runner")
        if runner is None:
            raise SystemExit("missing generated norm-preserving projection runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]Norm-preserving projection execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "som_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
        runner = result["paths"].get("som_projection_runner")
        if runner is None:
            raise SystemExit("missing generated SOM projection runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]SOM projection execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "selective_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
        runner = result["paths"].get("selective_projection_runner")
        if runner is None:
            raise SystemExit("missing generated selective projection runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]Selective projection execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "concept_cone_projection" and plan["backend_config"].get("execution") in NATIVE_PROJECTED_ABLATION_EXECUTIONS:
        runner = result["paths"].get("concept_cone_projection_runner")
        if runner is None:
            raise SystemExit("missing generated concept-cone projection runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]Concept-cone projection execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend"] == "qwen_scope_sae" and plan["backend_config"].get("execution") in QWEN_SCOPE_SAE_EXECUTIONS:
        runner = result["paths"].get("qwen_scope_sae_runner")
        if runner is None:
            raise SystemExit("missing generated Qwen-Scope SAE runner")
        execution = optimal_transport_execution_spec(plan, runner)
        console.print(f"[bold]Qwen-Scope SAE execution mode[/bold]: {execution['mode']}")
        subprocess.run(
            execution["command"],
            cwd=execution["cwd"],
            env=execution["env"],
            check=True,
        )
    elif plan["backend_config"].get("execution") == "plan_only":
        plan_path = result["paths"].get(f"{plan['backend']}_plan")
        raise SystemExit(
            f"SOTA backend {plan['backend']!r} is plan-only in model-forge. "
            f"Implement a guarded runner before --execute. Plan: {plan_path}"
        )
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
    weight_map = index["weight_map"]
    artifact = load_direction_artifact(directions_path)
    directions = artifact["refusal_directions"]
    missing_layers = missing_direction_layers(edit, directions)
    if missing_layers and edit.get("require_all_target_directions", True):
        raise SystemExit(
            "refusing export because target layers lack directions: "
            f"{missing_layers}. Re-run collection with matching layer_skip settings, "
            "or set edit.require_all_target_directions=false for exploratory exports."
        )
    missing_tensor_layers = missing_target_tensor_layers(weight_map, edit)
    if missing_tensor_layers and edit.get("require_target_tensor_per_layer", False):
        raise SystemExit(
            "refusing export because target layers have no tensors matching "
            f"target_weight_suffixes: {missing_tensor_layers}. Inspect the model "
            "architecture and include the correct attention/MLP suffixes for this family, "
            "or set edit.require_target_tensor_per_layer=false for exploratory exports."
        )

    copy_non_weight_files(source_dir, output_dir)
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

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
        "available_direction_layers": sorted(directions),
        "missing_direction_layers": missing_layers,
        "target_tensor_layers": projection_target_layers(weight_map, edit),
        "missing_target_tensor_layers": missing_tensor_layers,
        "require_target_tensor_per_layer": bool(edit.get("require_target_tensor_per_layer", False)),
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
    guard_source_checkpoint(plan)
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
    guard_source_checkpoint(plan)
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


def candidate_gate_defaults() -> dict[str, Any]:
    return {
        "trials": 3,
        "temperature": 1,
        "requirements": [],
    }


def candidate_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = candidate_gate_defaults()
    user = config.get("candidate_selection") or {}
    if isinstance(user, dict):
        for key in ("objective", "trials", "temperature"):
            if user.get(key) is not None:
                merged[key] = user[key]
    gate = user.get("gate") if isinstance(user.get("gate"), dict) else user
    if isinstance(gate, dict):
        for key, value in gate.items():
            if key == "requirements" and value:
                merged["requirements"] = list(value)
            elif key != "candidates":
                merged[key] = value
    default_min_count = int(merged.get("trials") or 1)
    normalized = []
    for index, item in enumerate(merged.get("requirements") or []):
        if not isinstance(item, dict):
            raise SystemExit(f"candidate gate requirement {index} must be a mapping")
        requirement = dict(item)
        requirement.setdefault("name", f"{requirement.get('bucket', 'bucket')}.{requirement.get('metric', 'metric')}")
        requirement.setdefault("operator", "==")
        requirement.setdefault("required", True)
        requirement.setdefault("min_count", default_min_count)
        missing = [key for key in ("bucket", "metric", "value") if requirement.get(key) is None]
        if missing:
            raise SystemExit(f"candidate gate requirement {requirement['name']!r} missing: {', '.join(missing)}")
        if requirement["operator"] not in CANDIDATE_GATE_OPERATORS:
            raise SystemExit(
                f"candidate gate requirement {requirement['name']!r} has unsupported operator "
                f"{requirement['operator']!r}; valid: {', '.join(sorted(CANDIDATE_GATE_OPERATORS))}"
            )
        normalized.append(requirement)
    if not normalized:
        raise SystemExit(
            "candidate-gate requires candidate_selection.gate.requirements in the config; "
            "define the target bucket/case/metric requirements for this model family"
        )
    merged["requirements"] = normalized
    return merged


def parse_candidate_gate_arg(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise SystemExit("empty --candidate entry")
    if "," in raw:
        parsed: dict[str, str] = {}
        for item in raw.split(","):
            key, sep, value = item.partition("=")
            if not sep:
                raise SystemExit(f"candidate entries must be key=value pairs, got {item!r}")
            parsed[key.strip()] = value.strip()
        eval_dir = parsed.get("eval") or parsed.get("eval_dir") or parsed.get("path")
        if not eval_dir:
            raise SystemExit("--candidate key-value entries require eval=<dir>")
        return {
            "name": parsed.get("name") or parsed.get("variant") or Path(eval_dir).name,
            "variant": parsed.get("variant"),
            "eval_dir": eval_dir,
            **{key: value for key, value in parsed.items() if key not in {"name", "variant", "eval", "eval_dir", "path"}},
        }
    name, sep, eval_dir = raw.partition("=")
    if sep:
        return {"name": name.strip(), "eval_dir": eval_dir.strip()}
    return {"name": Path(raw).name, "eval_dir": raw}


def candidate_gate_entries(config: dict[str, Any], cli_candidates: list[str] | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if cli_candidates:
        entries.extend(parse_candidate_gate_arg(raw) for raw in cli_candidates)
    configured = (config.get("candidate_selection") or {}).get("candidates") or []
    if not entries and configured:
        for item in configured:
            if isinstance(item, str):
                entries.append(parse_candidate_gate_arg(item))
            elif isinstance(item, dict):
                eval_dir = item.get("eval_dir") or item.get("eval") or item.get("path")
                if not eval_dir:
                    raise SystemExit(f"candidate selection entry missing eval_dir/eval/path: {item}")
                entry = dict(item)
                entry["eval_dir"] = str(eval_dir)
                entry.setdefault("name", entry.get("variant") or Path(str(eval_dir)).name)
                entries.append(entry)
            else:
                raise SystemExit(f"candidate selection entries must be mappings or strings, got {type(item).__name__}")
    if not entries:
        raise SystemExit("candidate-gate needs --candidate entries or candidate_selection.candidates in the config")
    return entries


def load_candidate_response_rows(eval_dir: str | Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(eval_dir)
    responses_path = path / "responses.jsonl" if path.is_dir() else path
    if not responses_path.exists():
        raise SystemExit(f"missing candidate responses file: {responses_path}")
    rows = []
    with responses_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON in {responses_path}:{line_number}: {exc}") from exc
    return rows


def compare_candidate_gate_value(actual: float, operator: str, target: float) -> bool:
    tolerance = 1e-9
    if operator == "==":
        return abs(actual - target) <= tolerance
    if operator == "!=":
        return abs(actual - target) > tolerance
    if operator == "<=":
        return actual <= target + tolerance
    if operator == ">=":
        return actual >= target - tolerance
    if operator == "<":
        return actual < target - tolerance
    if operator == ">":
        return actual > target + tolerance
    raise SystemExit(f"unsupported candidate gate operator: {operator}")


def candidate_gate_deficit(actual: float | None, operator: str, target: float, missing: bool) -> float:
    if missing or actual is None:
        return 100.0
    if operator in {"==", "!="}:
        if compare_candidate_gate_value(actual, operator, target):
            return 0.0
        return abs(actual - target) if operator == "==" else 1.0
    if operator == "<=":
        return max(0.0, actual - target)
    if operator == "<":
        return max(0.0, actual - target + 1e-9)
    if operator == ">=":
        return max(0.0, target - actual)
    if operator == ">":
        return max(0.0, target - actual + 1e-9)
    return 100.0


def summarize_candidate_requirement(rows: list[dict[str, Any]], requirement: dict[str, Any]) -> dict[str, Any]:
    bucket = str(requirement["bucket"])
    case_id = requirement.get("case_id")
    metric = str(requirement["metric"])
    target = float(requirement["value"])
    operator = str(requirement["operator"])
    min_count = int(requirement.get("min_count") or 1)
    values: list[float] = []
    trial_values: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("bucket") or "") != bucket:
            continue
        if case_id is not None and str(row.get("case_id") or "") != str(case_id):
            continue
        scores = row.get("scores") or {}
        if metric not in scores:
            continue
        value = float(scores[metric])
        values.append(value)
        trial_values.append({
            "trial_index": row.get("trial_index"),
            "value": value,
        })
    actual = sum(values) / len(values) if values else None
    missing = len(values) < min_count
    passed = False if missing or actual is None else compare_candidate_gate_value(actual, operator, target)
    return {
        "name": requirement["name"],
        "bucket": bucket,
        "case_id": case_id,
        "metric": metric,
        "operator": operator,
        "target": target,
        "required": bool(requirement.get("required", True)),
        "min_count": min_count,
        "count": len(values),
        "value": None if actual is None else round(actual, 6),
        "passed": passed,
        "missing": missing,
        "deficit": round(candidate_gate_deficit(actual, operator, target, missing), 6),
        "trial_values": trial_values,
    }


def build_candidate_gate_report(
    config: dict[str, Any],
    config_path: Path,
    candidates: list[dict[str, Any]],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    gate = candidate_gate_config(config)
    reports = []
    for candidate in candidates:
        rows = load_candidate_response_rows(candidate["eval_dir"])
        requirement_reports = [
            summarize_candidate_requirement(rows, requirement)
            for requirement in gate["requirements"]
        ]
        required_failures = [
            item for item in requirement_reports
            if item["required"] and not item["passed"]
        ]
        total_deficit = round(sum(float(item["deficit"]) for item in requirement_reports if item["required"]), 6)
        reports.append({
            "name": str(candidate.get("name") or Path(str(candidate["eval_dir"])).name),
            "variant": candidate.get("variant"),
            "eval_dir": display_path(resolve_repo_path(candidate["eval_dir"])),
            "status": "passed" if not required_failures else "failed",
            "required_failure_count": len(required_failures),
            "total_required_deficit": total_deficit,
            "response_count": len(rows),
            "requirements": requirement_reports,
            "blockers": [
                {
                    "name": item["name"],
                    "bucket": item["bucket"],
                    "case_id": item["case_id"],
                    "metric": item["metric"],
                    "value": item["value"],
                    "target": item["target"],
                    "operator": item["operator"],
                    "count": item["count"],
                    "min_count": item["min_count"],
                }
                for item in required_failures
            ],
        })
    ranked = sorted(reports, key=lambda item: (item["required_failure_count"], item["total_required_deficit"], item["name"]))
    passed = [item for item in ranked if item["status"] == "passed"]
    return {
        "schema_version": CANDIDATE_GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or sanitize_report_id(f"{Path(config_path).stem}_candidate_gate"),
        "config": display_path(config_path),
        "objective": gate.get("objective") or "zero_refusal_capability_retention",
        "trials": int(gate.get("trials") or 1),
        "temperature": gate.get("temperature"),
        "candidate_count": len(reports),
        "recommended_candidate": passed[0]["name"] if passed else None,
        "best_failed_candidate": ranked[0]["name"] if ranked and not passed else None,
        "decision": "promote_candidate" if passed else "no_candidate_passed_gate",
        "ranked_candidates": ranked,
        "notes": [
            "This report consumes completed model-forge eval outputs; it does not run servers, exports, or eval jobs.",
            "Promotion still requires checkpoint/tokenizer/architecture audits and source-relative broader evals after this targeted gate.",
        ],
    }


def sanitize_report_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "candidate_gate"


def write_candidate_gate_report(report: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = resolve_repo_path(output_dir or Path("reports/generated/abliteration_candidate_gate") / report["run_id"])
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "candidate_gate.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Abliteration Candidate Gate: {report['run_id']}",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Recommended candidate: `{report['recommended_candidate'] or '<none>'}`",
        f"- Best failed candidate: `{report['best_failed_candidate'] or '<none>'}`",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Trials: `{report['trials']}`",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Candidate | Status | Required failures | Deficit | Blockers |",
        "|---:|---|---|---:|---:|---|",
    ]
    for rank, candidate in enumerate(report["ranked_candidates"], start=1):
        blockers = ", ".join(f"{item['name']}={item['value']}" for item in candidate["blockers"]) or "none"
        lines.append(
            f"| {rank} | `{candidate['name']}` | `{candidate['status']}` | "
            f"{candidate['required_failure_count']} | {candidate['total_required_deficit']} | {blockers} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
    ])
    (root / "candidate_gate.md").write_text("\n".join(lines), encoding="utf-8")
    return json_path


def family_eval_output_root(family: str) -> Path:
    family_path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not family_path.exists():
        raise SystemExit(f"candidate-loop-plan cannot find family config: {family_path}")
    family_config = load_yaml(family_path)
    eval_config = family_config.get("eval") or {}
    output_root = eval_config.get("output_root")
    if not output_root:
        raise SystemExit(f"family {family!r} has no eval.output_root")
    return resolve_repo_path(output_root)


def shell_assignment(key: str, value: str | int | float | bool | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:=@+-]+", text):
        return f"{key}={text}"
    return f"{key}={json.dumps(text)}"


def command_entry(
    command: str,
    *,
    phase: str,
    purpose: str,
    starts_heavy_job: bool = False,
    requires_execute: bool = False,
    candidate: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "candidate": candidate,
        "command": command,
        "purpose": purpose,
        "starts_heavy_job": starts_heavy_job,
        "requires_execute": requires_execute,
        "enabled": enabled,
    }


def candidate_custom_command_entries(candidate: dict[str, Any], *, blocked: bool) -> list[dict[str, Any]]:
    raw_commands = candidate.get("custom_commands") or candidate.get("commands") or []
    if raw_commands is None:
        return []
    if not isinstance(raw_commands, list):
        raise SystemExit(f"candidate {candidate.get('name') or candidate.get('variant')} custom_commands must be a list")
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_commands, start=1):
        if isinstance(raw, str):
            command = raw
            raw = {}
        elif isinstance(raw, dict):
            command = raw.get("command")
        else:
            raise SystemExit(f"candidate custom command {index} must be a string or mapping")
        if not command:
            raise SystemExit(f"candidate custom command {index} is missing command")
        phase = str(raw.get("phase") or f"candidate_custom_{index}") if isinstance(raw, dict) else f"candidate_custom_{index}"
        purpose = str(raw.get("purpose") or "Run candidate-specific custom command.") if isinstance(raw, dict) else "Run candidate-specific custom command."
        entries.append(command_entry(
            str(command),
            phase=phase,
            candidate=str(candidate.get("name") or candidate.get("variant") or f"candidate_{index}"),
            purpose=purpose,
            starts_heavy_job=bool(raw.get("starts_heavy_job", False)) if isinstance(raw, dict) else False,
            requires_execute=bool(raw.get("requires_execute", True)) if isinstance(raw, dict) else True,
            enabled=not blocked and bool(raw.get("enabled", True)) if isinstance(raw, dict) else not blocked,
        ))
    return entries


def candidate_loop_config(config: dict[str, Any]) -> dict[str, Any]:
    selection = config.get("candidate_selection") or {}
    loop = selection.get("loop") or {}
    if not isinstance(loop, dict) or not loop:
        raise SystemExit("candidate-loop-plan requires candidate_selection.loop in the config")
    family = loop.get("family") or selection.get("family")
    source_variant = loop.get("source_variant") or selection.get("source_variant")
    candidates = loop.get("candidates") or []
    if not family:
        raise SystemExit("candidate_selection.loop.family is required")
    if not source_variant:
        raise SystemExit("candidate_selection.loop.source_variant is required")
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("candidate_selection.loop.candidates must be a non-empty list")
    return {
        **loop,
        "family": str(family),
        "source_variant": str(source_variant),
        "candidates": candidates,
    }


def normalize_loop_candidate(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SystemExit(f"candidate loop entry {index} must be a mapping")
    candidate = dict(raw)
    candidate.setdefault("name", candidate.get("variant") or f"candidate_{index}")
    candidate.setdefault("status", "ready")
    candidate.setdefault("execution", "checkpoint_export")
    return candidate


def candidate_loop_eval_suffix(family: str, candidate: dict[str, Any], trials: int) -> str:
    if candidate.get("eval_suffix"):
        return str(candidate["eval_suffix"])
    variant = str(candidate.get("variant") or candidate["name"])
    name = sanitize_report_id(str(candidate["name"]))
    return f"{family}_{variant}_{name}_targeted_gate_t{trials}"


def candidate_loop_eval_command(
    *,
    family: str,
    variant: str,
    eval_spec: dict[str, Any],
    output_suffix: str,
) -> str:
    trials = int(eval_spec.get("trials") or 3)
    temperature = eval_spec.get("temperature", 1)
    parts = [
        shell_assignment("MODEL_FORGE_TRIALS", trials),
        shell_assignment("MODEL_FORGE_TEMPERATURE", temperature),
        shell_assignment("MODEL_FORGE_BASE_URL", eval_spec.get("base_url", "http://127.0.0.1:8000/v1")),
        "./forge",
        "eval",
        family,
        variant,
        "--internal",
    ]
    for bucket in eval_spec.get("buckets") or []:
        parts.extend(["--bucket", str(bucket)])
    for case_id in eval_spec.get("case_ids") or []:
        parts.extend(["--case-id", str(case_id)])
    parts.extend(["--output-suffix", output_suffix])
    return " ".join(item for item in parts if item)


def build_candidate_loop_plan(config: dict[str, Any], config_path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    loop = candidate_loop_config(config)
    family = loop["family"]
    source_variant = loop["source_variant"]
    cluster_config = loop.get("cluster_config")
    models_dir = loop.get("models_dir")
    eval_spec = {
        "trials": int((loop.get("eval") or {}).get("trials") or (config.get("candidate_selection") or {}).get("trials") or 3),
        "temperature": (loop.get("eval") or {}).get("temperature", (config.get("candidate_selection") or {}).get("temperature", 1)),
        "base_url": (loop.get("eval") or {}).get("base_url", "http://127.0.0.1:8000/v1"),
        "buckets": (loop.get("eval") or {}).get("buckets") or ["refusal_paired_boundary", "capability_preservation_challenge"],
        "case_ids": (loop.get("eval") or {}).get("case_ids") or ["self_harm_instruction_harmful", "model_selection_summary"],
    }
    output_root = family_eval_output_root(family)
    family_path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    family_config = load_yaml(family_path)
    family_variants = family_config.get("variants") or {}
    run_id = run_id or sanitize_report_id(f"{Path(config_path).stem}_candidate_loop")
    commands: list[dict[str, Any]] = [
        command_entry("./forge doctor", phase="preflight", purpose="Verify repo hygiene, secrets, and local configuration before launching candidates."),
    ]
    if cluster_config:
        commands.append(command_entry(
            f"./forge cluster health --config {cluster_config}",
            phase="preflight",
            purpose="Verify all configured cluster nodes are reachable before any heavy job.",
        ))

    candidate_reports: list[dict[str, Any]] = []
    gate_entries: list[str] = []
    for index, raw_candidate in enumerate(loop["candidates"], start=1):
        candidate = normalize_loop_candidate(raw_candidate, index=index)
        name = str(candidate["name"])
        variant = str(candidate.get("variant") or name)
        status = str(candidate.get("status") or "ready")
        blocked = status in BLOCKED_CANDIDATE_STATUSES
        export_completed = status in EXPORT_COMPLETED_CANDIDATE_STATUSES
        produces_checkpoint = bool(candidate.get("produces_checkpoint", True))
        candidate_config = candidate.get("config")
        backend = candidate.get("backend")
        output_dir = candidate.get("output_dir")
        has_custom_commands = bool(candidate.get("custom_commands") or candidate.get("commands") or [])
        eval_suffix = candidate_loop_eval_suffix(family, candidate, int(eval_spec["trials"]))
        eval_dir = display_path(output_root / eval_suffix)
        candidate_export_env = {
            str(key): value
            for key, value in (candidate.get("export_env") or {}).items()
            if value is not None
        }
        candidate_commands: list[dict[str, Any]] = []
        blockers: list[str] = []
        if blocked:
            blockers.append(str(candidate.get("blocker") or f"candidate status is {status}"))
        if produces_checkpoint and variant not in family_variants and not blocked:
            blockers.append(
                f"candidate variant {variant} is not registered in configs/model_families/{family}.yaml"
            )
        if not candidate_config and not has_custom_commands and not blocked:
            blockers.append("candidate config is missing")
        if not backend and not has_custom_commands and not blocked:
            blockers.append("candidate backend is missing")
        command_blocked = bool(blockers)
        custom_commands = candidate_custom_command_entries(candidate, blocked=command_blocked)
        if candidate_config and backend and not export_completed:
            candidate_commands.extend([
                command_entry(
                    f"./forge ablate --config {candidate_config} sota-plan --backend {backend}",
                    phase="candidate_plan",
                    candidate=name,
                    purpose="Inspect the backend plan without loading model weights.",
                    enabled=not command_blocked,
                ),
                command_entry(
                    f"./forge ablate --config {candidate_config} sota-prepare --backend {backend}",
                    phase="candidate_prepare",
                    candidate=name,
                    purpose="Write backend-specific runner/config artifacts for this candidate.",
                    enabled=not command_blocked,
                ),
                command_entry(
                    " ".join([
                        shell_assignment("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", loop.get("min_available_ram_fraction", 0.05)),
                        shell_assignment("MODEL_FORGE_MIN_FREE_DISK_FRACTION", loop.get("min_free_disk_fraction", 0.15)),
                        *(shell_assignment(key, value) for key, value in candidate_export_env.items()),
                        f"./forge ablate --config {candidate_config} sota-run --backend {backend} --execute",
                    ]),
                    phase="candidate_export",
                    candidate=name,
                    purpose="Run the guarded backend export for this candidate.",
                    starts_heavy_job=True,
                    requires_execute=True,
                    enabled=not command_blocked,
                ),
            ])
        elif export_completed:
            candidate_commands.append(command_entry(
                f"# export already completed for {name}; source config: {candidate_config or '<none>'}; resume at sync/audit/serve/eval",
                phase="candidate_export",
                candidate=name,
                purpose="Document that this candidate already has a local export and should not rerun the heavy export unchanged.",
                starts_heavy_job=False,
                requires_execute=False,
                enabled=False,
            ))
        candidate_commands.extend(custom_commands)
        if cluster_config and output_dir and not command_blocked and produces_checkpoint:
            model_sync_parts = [
                "./forge cluster model-sync",
                f"--config {cluster_config}",
                f"--source {output_dir}",
                f"--family {family}",
                f"--variant {variant}",
                "--execute",
                f"--timeout {int(loop.get('model_sync_timeout', 3600))}",
            ]
            if models_dir:
                model_sync_parts.extend(["--models-dir", str(models_dir)])
            candidate_commands.append(command_entry(
                " ".join(model_sync_parts),
                phase="candidate_sync",
                candidate=name,
                purpose="Sync the exported checkpoint to all cluster nodes before TP serving.",
                starts_heavy_job=True,
                requires_execute=True,
            ))
        if not command_blocked and produces_checkpoint:
            candidate_commands.extend([
                command_entry(
                    f"./forge variants checkpoint-audit {family} --variant {variant} --strict --json",
                    phase="candidate_audit",
                    candidate=name,
                    purpose="Verify the candidate checkpoint is complete and tensor-safe before serving.",
                ),
                command_entry(
                    f"./forge variants tokenizer-audit {family} --variant {variant} --strict --json",
                    phase="candidate_audit",
                    candidate=name,
                    purpose="Verify tokenizer/chat-template compatibility before serving.",
                ),
                command_entry(
                    f"./forge variants architecture-audit {family} --variant {variant} --strict --json",
                    phase="candidate_audit",
                    candidate=name,
                    purpose="Verify architecture metadata matches the source family before serving.",
                ),
            ])
            serve_env = [
                shell_assignment("MODEL_FORGE_CLUSTER_CONFIG", cluster_config) if cluster_config else "",
                shell_assignment("MODEL_FORGE_SERVE_REQUIRE_CLUSTER", 1 if cluster_config else None),
            ]
            candidate_commands.append(command_entry(
                " ".join([item for item in serve_env if item] + ["./forge", "serve", family, variant]),
                phase="candidate_serve",
                candidate=name,
                purpose="Start exactly one server for this candidate; stop it after the targeted eval finishes.",
                starts_heavy_job=True,
                requires_execute=True,
            ))
            candidate_commands.append(command_entry(
                candidate_loop_eval_command(family=family, variant=variant, eval_spec=eval_spec, output_suffix=eval_suffix),
                phase="candidate_eval",
                candidate=name,
                purpose="Run the exact targeted multi-trial gate for this candidate.",
                requires_execute=True,
            ))
        if not blockers and produces_checkpoint:
            gate_entries.append(f"name={name},variant={variant},eval={eval_dir}")
        commands.extend(candidate_commands)
        candidate_reports.append({
            "name": name,
            "variant": variant,
            "status": status,
            "method_family": candidate.get("method_family"),
            "backend": backend,
            "config": candidate_config,
            "output_dir": output_dir,
            "produces_checkpoint": produces_checkpoint,
            "eval_suffix": eval_suffix,
            "expected_eval_dir": eval_dir,
            "blockers": blockers,
            "commands": candidate_commands,
            "hypothesis": candidate.get("hypothesis"),
        })

    if gate_entries:
        gate_command = " ".join([
            f"./forge ablate --config {display_path(config_path)} candidate-gate",
            *(f"--candidate {entry}" for entry in gate_entries),
            "--write-report",
            f"--run-id {run_id}_gate",
        ])
    else:
        has_enabled_candidate_job = any(
            bool(command.get("enabled", True))
            and str(command.get("phase", "")).startswith("candidate_")
            and command.get("phase") != "candidate_gate"
            for command in commands
        )
        if has_enabled_candidate_job:
            gate_command = "Search-only candidate jobs are planned; export a selected checkpoint before candidate-gate."
        else:
            gate_command = "No executable candidate eval directories are planned yet; implement or unblock a candidate first."
    commands.append(command_entry(
        gate_command,
        phase="candidate_gate",
        purpose="Rank completed candidates by the explicit model-forge gate requirements.",
        requires_execute=bool(gate_entries),
        enabled=bool(gate_entries),
    ))
    if loop.get("delete_rejected_checkpoints_after_report", False):
        commands.append(command_entry(
            "Review the candidate gate report, then delete only rejected full checkpoints that have committed summaries and no active server.",
            phase="cleanup",
            purpose="Restore disk headroom while preserving reports, configs, and reusable artifacts.",
            requires_execute=bool(gate_entries),
            enabled=bool(gate_entries),
        ))

    return {
        "schema_version": CANDIDATE_LOOP_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": display_path(config_path),
        "family": family,
        "source_variant": source_variant,
        "objective": (config.get("candidate_selection") or {}).get("objective") or "zero_refusal_capability_retention",
        "cluster_config": cluster_config,
        "eval": eval_spec,
        "candidate_count": len(candidate_reports),
        "executable_candidate_count": len(gate_entries),
        "planned_candidate_job_count": sum(
            1
            for candidate in candidate_reports
            if any(command.get("enabled", True) for command in candidate["commands"])
        ),
        "candidates": candidate_reports,
        "commands": commands,
        "candidate_gate_command": gate_command,
        "resource_contract": {
            "max_concurrent_large_jobs": 1,
            "min_available_ram_fraction": loop.get("min_available_ram_fraction", 0.05),
            "min_free_disk_fraction": loop.get("min_free_disk_fraction", 0.15),
            "serve_one_candidate_at_a_time": True,
            "quantization_blocked_until_gate_passes": True,
        },
        "notes": [
            "This is a runbook only; it does not execute exports, servers, evals, or cleanup.",
            "Every loop candidate must pass candidate-gate before broad eval, NVFP4 export, upload, or promotion.",
        ],
    }


def write_candidate_loop_plan(plan: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = resolve_repo_path(output_dir or Path("reports/generated/abliteration_candidate_loop") / plan["run_id"])
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "candidate_loop_plan.json"
    json_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Abliteration Candidate Loop Plan: {plan['run_id']}",
        "",
        f"- Family: `{plan['family']}`",
        f"- Source variant: `{plan['source_variant']}`",
        f"- Objective: `{plan['objective']}`",
        f"- Candidate count: `{plan['candidate_count']}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Status | Backend | Expected eval dir | Blockers |",
        "|---|---|---|---|---|",
    ]
    for candidate in plan["candidates"]:
        blockers = ", ".join(candidate.get("blockers") or []) or "none"
        lines.append(
            f"| `{candidate['name']}` | `{candidate['status']}` | "
            f"`{candidate.get('backend') or '<none>'}` | `{candidate['expected_eval_dir']}` | {blockers} |"
        )
    lines.extend([
        "",
        "## Command Runbook",
        "",
    ])
    for index, command in enumerate(plan["commands"], start=1):
        enabled = "" if command.get("enabled", True) else " (disabled)"
        lines.extend([
            f"{index}. `{command['phase']}`{enabled}",
            "",
            "```bash" if command.get("enabled", True) else "```text",
            command["command"],
            "```",
            "",
            command["purpose"],
            "",
        ])
    lines.extend([
        "## Gate",
        "",
        "```bash" if plan["candidate_gate_command"].startswith("./forge ") else "```text",
        plan["candidate_gate_command"],
        "```",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in plan["notes"]],
        "",
    ])
    (root / "candidate_loop_plan.md").write_text("\n".join(lines), encoding="utf-8")
    return json_path


def command_candidate_loop_plan(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    plan = build_candidate_loop_plan(config, config_path, run_id=args.run_id)
    if args.write_plan:
        path = write_candidate_loop_plan(plan, args.output_dir)
        if not args.json:
            console.print(f"[bold green]Wrote candidate loop plan[/bold green]: {path}")
    if args.json:
        print(json.dumps(plan, indent=2) + "\n")
        return
    table = Table(title="Abliteration Candidate Loop")
    table.add_column("candidate")
    table.add_column("status")
    table.add_column("backend")
    table.add_column("eval_dir")
    table.add_column("blockers")
    for candidate in plan["candidates"]:
        table.add_row(
            candidate["name"],
            candidate["status"],
            str(candidate.get("backend") or ""),
            candidate["expected_eval_dir"],
            ", ".join(candidate.get("blockers") or []) or "none",
        )
    console.print(table)


def command_candidate_gate(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    candidates = candidate_gate_entries(config, args.candidate)
    report = build_candidate_gate_report(
        config,
        config_path,
        candidates,
        run_id=args.run_id,
    )
    if args.write_report:
        path = write_candidate_gate_report(report, args.output_dir)
        if not args.json:
            console.print(f"[bold green]Wrote candidate gate report[/bold green]: {path}")
    if args.json:
        print(json.dumps(report, indent=2) + "\n")
        return
    table = Table(title="Abliteration Candidate Gate")
    table.add_column("rank")
    table.add_column("candidate")
    table.add_column("status")
    table.add_column("failures")
    table.add_column("deficit")
    table.add_column("blockers")
    for rank, candidate in enumerate(report["ranked_candidates"], start=1):
        blockers = ", ".join(item["name"] for item in candidate["blockers"]) or "none"
        table.add_row(
            str(rank),
            candidate["name"],
            candidate["status"],
            str(candidate["required_failure_count"]),
            str(candidate["total_required_deficit"]),
            blockers,
        )
    console.print(table)
    if report["recommended_candidate"]:
        console.print(f"[bold green]Recommended[/bold green]: {report['recommended_candidate']}")
    else:
        console.print("[yellow]No candidate passed the targeted gate.[/yellow]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and run memory-guarded model abliteration steps")
    parser.add_argument("--config", required=True, help="Path to abliteration YAML config")
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

    candidate_gate = sub.add_parser(
        "candidate-gate",
        help="Rank completed ablation candidates by the configured case-level eval gate",
    )
    candidate_gate.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate eval entry: name=<id>,variant=<variant>,eval=<run_dir> or name=<run_dir>",
    )
    candidate_gate.add_argument("--run-id", default=None)
    candidate_gate.add_argument("--output-dir", type=Path, default=None)
    candidate_gate.add_argument("--write-report", action="store_true")
    candidate_gate.add_argument("--json", action="store_true")
    candidate_gate.set_defaults(func=command_candidate_gate)

    candidate_loop = sub.add_parser(
        "candidate-loop-plan",
        help="Write a bounded sequential candidate runbook that ends in candidate-gate",
    )
    candidate_loop.add_argument("--run-id", default=None)
    candidate_loop.add_argument("--output-dir", type=Path, default=None)
    candidate_loop.add_argument("--write-plan", action="store_true")
    candidate_loop.add_argument("--json", action="store_true")
    candidate_loop.set_defaults(func=command_candidate_loop_plan)

    sota_plan = sub.add_parser("sota-plan", help="Inspect SOTA external backend plan")
    sota_plan.add_argument("--backend", choices=SOTA_BACKEND_CHOICES, default=None)
    sota_plan.set_defaults(func=command_sota_plan)

    sota_prepare = sub.add_parser("sota-prepare", help="Write backend-specific SOTA runner/config files")
    sota_prepare.add_argument("--backend", choices=SOTA_BACKEND_CHOICES, default=None)
    sota_prepare.set_defaults(func=command_sota_prepare)

    sota_run = sub.add_parser("sota-run", help="Run a prepared external SOTA backend")
    sota_run.add_argument("--backend", choices=SOTA_BACKEND_CHOICES, default=None)
    sota_run.add_argument("--execute", action="store_true")
    sota_run.set_defaults(func=command_sota_run)

    heretic_search = sub.add_parser(
        "heretic-search-analyze",
        help="Analyze a Heretic search journal before exporting a full checkpoint",
    )
    heretic_search.add_argument("--backend", choices=["heretic"], default="heretic")
    heretic_search.add_argument("--journal", default=None, help="Path to a Heretic JSONL journal")
    heretic_search.add_argument("--max-kl", type=float, default=None, help="Maximum KL allowed for export consideration")
    heretic_search.add_argument("--max-refusals", type=int, default=None, help="Maximum focused refusals allowed")
    heretic_search.add_argument(
        "--min-refusal-reduction",
        type=int,
        default=None,
        help="Minimum focused refusal reduction versus the journal baseline",
    )
    heretic_search.add_argument("--top-k", type=int, default=8, help="Number of frontier rows to print")
    heretic_search.add_argument("--output", default=None, help="Optional JSON report path")
    heretic_search.set_defaults(func=command_heretic_search_analyze)

    abliterix_search = sub.add_parser(
        "abliterix-search-analyze",
        help="Analyze an Abliterix search journal before building an export runner",
    )
    abliterix_search.add_argument("--backend", choices=["abliterix"], default="abliterix")
    abliterix_search.add_argument("--journal", default=None, help="Path to an Abliterix JSONL journal")
    abliterix_search.add_argument("--max-kl", type=float, default=None, help="Maximum KL allowed for export consideration")
    abliterix_search.add_argument("--max-refusals", type=int, default=None, help="Maximum focused refusals allowed")
    abliterix_search.add_argument(
        "--min-refusal-reduction",
        type=int,
        default=None,
        help="Minimum focused refusal reduction versus the journal baseline",
    )
    abliterix_search.add_argument("--top-k", type=int, default=8, help="Number of frontier rows to print")
    abliterix_search.add_argument("--output", default=None, help="Optional JSON report path")
    abliterix_search.set_defaults(func=command_abliterix_search_analyze)

    abliterix_export = sub.add_parser(
        "abliterix-export",
        help="Export a selected Abliterix search trial through the guarded runner",
    )
    abliterix_export.add_argument("--backend", choices=["abliterix"], default="abliterix")
    abliterix_export.add_argument("--trial-index", type=int, required=True, help="Abliterix trial index to export")
    abliterix_export.add_argument("--execute", action="store_true", help="Actually export the selected checkpoint")
    abliterix_export.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    abliterix_export.set_defaults(func=command_abliterix_export)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
