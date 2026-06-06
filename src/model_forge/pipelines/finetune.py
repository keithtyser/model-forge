from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_forge.data.sources import resolve_sources
from model_forge.hardware import detect_hardware_profile, recommended_training_env


REPO_DIR = Path(__file__).resolve().parents[3]
console = Console()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def expand_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def resolve_repo_path(value: str | Path) -> Path:
    path = expand_path(value)
    if path.is_absolute():
        return path
    return REPO_DIR / path


def training_run_dir(config: dict[str, Any]) -> Path:
    raw = config.get("run_dir") or f"runs/finetune/{config['name']}"
    return resolve_repo_path(raw)


def _load_data_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_repo_path(config["data"]["manifest"])
    manifest = load_yaml(manifest_path)
    manifest["_path"] = str(manifest_path)
    return manifest


def build_plan(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    hardware = detect_hardware_profile()
    training_env = recommended_training_env()
    data_manifest = _load_data_manifest(config)
    model = config["model"]
    trainer = config["trainer"]
    trainer_method = str(trainer.get("method", "qlora"))
    preference_default_weight = 1.0 if "preference" in trainer_method.lower() else 0.0
    lora = config.get("lora", {})
    eval_cfg = config.get("eval", {})
    resource_policy = config.get("resource_policy", {})
    cluster = config.get("cluster", {})
    sources = resolve_sources(data_manifest)
    total_target = sum(int(source.get("target_samples", 0) or 0) for source in sources)

    return {
        "name": config["name"],
        "family": config["family"],
        "config_path": str(config_path),
        "run_dir": str(training_run_dir(config)),
        "model": {
            "source": model["source"],
            "local_dir": str(expand_path(model["local_dir"])) if model.get("local_dir") else None,
            "output_dir": str(expand_path(model["output_dir"])),
            "served_model_name": model.get("served_model_name", f"local/{config['name']}"),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "max_seq_length": int(model.get("max_seq_length", trainer.get("max_seq_length", 4096))),
        },
        "trainer": {
            "backend": trainer.get("backend", "trl_sft"),
            "method": trainer_method,
            "assistant_only_loss": bool(trainer.get("assistant_only_loss", False)),
            "unlikelihood_weight": float(trainer.get("unlikelihood_weight", 0.0) or 0.0),
            "unlikelihood_scope": str(trainer.get("unlikelihood_scope", "assistant")),
            "unlikelihood_prefix_tokens": int(trainer.get("unlikelihood_prefix_tokens", 0) or 0),
            "preference_weight": float(trainer.get("preference_weight", preference_default_weight) or 0.0),
            "preference_beta": float(trainer.get("preference_beta", 0.1) or 0.1),
            "preference_margin": float(trainer.get("preference_margin", 0.0) or 0.0),
            "sft_weight": float(trainer.get("sft_weight", 1.0) if trainer.get("sft_weight") is not None else 1.0),
            "preference_length_normalize": bool(trainer.get("preference_length_normalize", True)),
            "device_map": trainer.get("device_map", "auto"),
            "load_in_4bit": bool(trainer.get("load_in_4bit", True)),
            "load_in_8bit": bool(trainer.get("load_in_8bit", False)),
            "attn_implementation": trainer.get("attn_implementation", "eager"),
            "unsloth_compile_disable": bool(trainer.get("unsloth_compile_disable", False)),
            "bf16": bool(trainer.get("bf16", True)),
            "gradient_checkpointing": trainer.get("gradient_checkpointing", True),
            "group_by_length": bool(trainer.get("group_by_length", False)),
            "pad_to_multiple_of": int(trainer.get("pad_to_multiple_of", 0) or 0),
            "torch_dynamo_recompile_limit": int(trainer.get("torch_dynamo_recompile_limit", 0) or 0),
            "ddp_find_unused_parameters": bool(trainer.get("ddp_find_unused_parameters", True)),
            "tensor_parallel_size": int(trainer.get("tensor_parallel_size", 1) or 1),
            "tensor_parallel_plan": trainer.get("tensor_parallel_plan", "auto"),
            "per_device_train_batch_size": int(trainer.get("per_device_train_batch_size", 1)),
            "gradient_accumulation_steps": int(trainer.get("gradient_accumulation_steps", 16)),
            "learning_rate": float(trainer.get("learning_rate", 2e-4)),
            "num_train_epochs": float(trainer.get("num_train_epochs", 1)),
            "max_steps": trainer.get("max_steps"),
            "warmup_ratio": float(trainer.get("warmup_ratio", 0.04)),
            "lr_scheduler_type": trainer.get("lr_scheduler_type", "cosine"),
            "optim": trainer.get("optim", "paged_adamw_8bit"),
            "weight_decay": float(trainer.get("weight_decay", 0.001)),
            "logging_steps": int(trainer.get("logging_steps", 5)),
            "save_strategy": str(trainer.get("save_strategy", "steps")),
            "save_steps": int(trainer.get("save_steps", 100)),
            "save_total_limit": int(trainer.get("save_total_limit", 2)),
            "benchmark_only": bool(trainer.get("benchmark_only", False)),
            "seed": int(trainer.get("seed", 3407)),
            "report_to": trainer.get("report_to", "none"),
            "dataloader_num_workers": int(trainer.get("dataloader_num_workers", 0) or 0),
            "dataloader_prefetch_factor": int(trainer.get("dataloader_prefetch_factor", 2) or 2),
            "dataloader_persistent_workers": bool(trainer.get("dataloader_persistent_workers", False)),
        },
        "lora": {
            "r": int(lora.get("r", 64)),
            "alpha": int(lora.get("alpha", lora.get("r", 64))),
            "dropout": float(lora.get("dropout", 0.0)),
            "target_modules": list(lora.get("target_modules", [])),
            "exclude_modules": list(lora.get("exclude_modules", [])),
            "modules_to_save": list(lora.get("modules_to_save", [])),
        },
        "data": {
            "manifest": data_manifest["_path"],
            "format": data_manifest.get("format", "messages"),
            "chat_template": data_manifest.get("chat_template", "auto"),
            "max_context_window": int(data_manifest.get("max_context_window", model.get("max_seq_length", 4096))),
            "target_samples": total_target,
            "sources": sources,
            "quality_gates": data_manifest.get("quality_gates", {}),
            "holdouts": data_manifest.get("holdouts", []),
        },
        "eval": {
            "baseline_variant": eval_cfg.get("baseline_variant", "ft"),
            "source_variant": eval_cfg.get("source_variant", "base"),
            "required_gates": eval_cfg.get("required_gates", {}),
            "commands": eval_cfg.get("commands", []),
        },
        "hardware": {
            "profile": hardware.name,
            "label": hardware.label,
            "gpus": [{"name": gpu.name, "memory_total_mb": gpu.memory_total_mb} for gpu in hardware.gpus],
            "training_env": training_env,
            "notes": list(hardware.notes),
        },
        "resource_policy": {
            "cpu_quota": str(resource_policy.get("cpu_quota", "80%")),
            "memory_max": str(resource_policy.get("memory_max", "85%")),
            "io_weight": int(resource_policy.get("io_weight", 100)),
            "nice": int(resource_policy.get("nice", 10)),
            "reserve_cores": int(resource_policy.get("reserve_cores", 1)),
            "min_memory_available_start": float(resource_policy.get("min_memory_available_start", 0.05)),
            "min_memory_available_runtime": float(resource_policy.get("min_memory_available_runtime", 0.05)),
            "min_disk_free": float(resource_policy.get("min_disk_free", 0.15)),
            "monitor_interval_seconds": int(resource_policy.get("monitor_interval_seconds", 30)),
            "dataloader_num_workers_max_offset": int(resource_policy.get("dataloader_num_workers_max_offset", 2)),
            "persistent_workers_when_memory_tight": bool(resource_policy.get("persistent_workers_when_memory_tight", False)),
            "checkpoint_on_memory_pressure": bool(resource_policy.get("checkpoint_on_memory_pressure", True)),
        },
        "cluster": {
            "enabled": bool(cluster.get("enabled", False)),
            "config": str(cluster.get("config", "")),
            "launcher": str(cluster.get("launcher", "torchrun")),
            "image": str(cluster.get("image", "")),
            "nccl_socket_ifname": str(cluster.get("nccl_socket_ifname", "")),
            "require_torchrun_smoke": bool(cluster.get("require_torchrun_smoke", True)),
            "sync_run_dir_to_workers": bool(cluster.get("sync_run_dir_to_workers", True)),
            "sync_model_to_workers": bool(cluster.get("sync_model_to_workers", False)),
        },
        "baseline": config.get("baseline", {}),
        "dry_run_only": bool(config.get("dry_run_only", False)),
    }


def render_plan(plan: dict[str, Any]) -> None:
    table = Table(title="Fine-Tune Plan")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Name", plan["name"])
    table.add_row("Family", plan["family"])
    table.add_row("Source", plan["model"]["source"])
    table.add_row("Output", plan["model"]["output_dir"])
    table.add_row("Backend", plan["trainer"]["backend"])
    table.add_row("Method", plan["trainer"]["method"])
    table.add_row("Max seq", str(plan["model"]["max_seq_length"]))
    table.add_row("LoRA r/alpha", f"{plan['lora']['r']}/{plan['lora']['alpha']}")
    table.add_row("Target samples", str(plan["data"]["target_samples"]))
    table.add_row("Hardware", plan["hardware"]["label"])
    table.add_row("CPU quota", plan["resource_policy"]["cpu_quota"])
    table.add_row("Memory max", plan["resource_policy"]["memory_max"])
    console.print(table)
    if plan["data"]["sources"]:
        sources = Table(title="Data Blend")
        sources.add_column("Name", style="cyan")
        sources.add_column("Dataset")
        sources.add_column("Target", justify="right")
        sources.add_column("Role")
        for source in plan["data"]["sources"]:
            sources.add_row(
                source.get("name", ""),
                source.get("dataset", source.get("path", "")),
                str(source.get("target_samples", "")),
                source.get("role", ""),
            )
        console.print(sources)
    if plan["hardware"]["notes"]:
        console.print(Panel("\n".join(plan["hardware"]["notes"]), title="Hardware Notes", border_style="yellow"))


def render_training_method_card(plan: dict[str, Any]) -> str:
    trainer = plan["trainer"]
    resource_policy = plan["resource_policy"]
    data = plan["data"]
    hardware = plan["hardware"]
    source_lines = [
        f"- {source.get('name') or source.get('id')}: {source.get('target_samples', '')} rows, role `{source.get('role', '')}`"
        for source in data.get("sources", [])
    ]
    eval_lines = [f"- `{command}`" for command in plan.get("eval", {}).get("commands", [])] or ["- No eval command configured."]
    return "\n".join(
        [
            f"# Training Method Card: {plan['name']}",
            "",
            "## Identity",
            "",
            f"- Family: `{plan['family']}`",
            f"- Source model: `{plan['model']['source']}`",
            f"- Output model: `{plan['model']['output_dir']}`",
            f"- Config: `{plan['config_path']}`",
            "",
            "## Method",
            "",
            f"- Backend: `{trainer['backend']}`",
            f"- Method: `{trainer['method']}`",
            f"- Assistant-only loss: `{trainer['assistant_only_loss']}`",
            f"- Unlikelihood weight: `{trainer['unlikelihood_weight']}`",
            f"- Unlikelihood scope: `{trainer['unlikelihood_scope']}`",
            f"- Unlikelihood prefix tokens: `{trainer['unlikelihood_prefix_tokens']}`",
            f"- Preference weight/beta/margin: `{trainer['preference_weight']}` / `{trainer['preference_beta']}` / `{trainer['preference_margin']}`",
            f"- SFT replay weight: `{trainer['sft_weight']}`",
            f"- Preference length normalize: `{trainer['preference_length_normalize']}`",
            f"- Max sequence length: `{plan['model']['max_seq_length']}`",
            f"- LoRA rank/alpha/dropout: `{plan['lora']['r']}` / `{plan['lora']['alpha']}` / `{plan['lora']['dropout']}`",
            f"- Target modules: `{', '.join(plan['lora']['target_modules'])}`",
            f"- Excluded modules: `{', '.join(plan['lora']['exclude_modules'])}`",
            f"- Learning rate: `{trainer['learning_rate']}`",
            f"- Max steps: `{trainer['max_steps']}`",
            f"- Benchmark only: `{trainer['benchmark_only']}`",
            f"- Optimizer: `{trainer['optim']}`",
            "",
            "## Data",
            "",
            f"- Manifest: `{data['manifest']}`",
            f"- Format: `{data['format']}`",
            f"- Target samples: `{data['target_samples']}`",
            "",
            *source_lines,
            "",
            "## Distributed Correctness And Resource Guardrails",
            "",
            f"- Hardware profile: `{hardware['profile']}`",
            f"- Detected GPUs: `{len(hardware['gpus'])}`",
            f"- CPU quota: `{resource_policy['cpu_quota']}`",
            f"- Memory max: `{resource_policy['memory_max']}`",
            f"- Reserved cores: `{resource_policy['reserve_cores']}`",
            f"- Start RAM floor: `{resource_policy['min_memory_available_start']}`",
            f"- Runtime RAM floor: `{resource_policy['min_memory_available_runtime']}`",
            f"- Disk free floor: `{resource_policy['min_disk_free']}`",
            f"- Checkpoint on memory pressure: `{resource_policy['checkpoint_on_memory_pressure']}`",
            "",
            "## Evaluation",
            "",
            *eval_lines,
            "",
            "## Limitations",
            "",
            "- This card is generated from the planned recipe; it is not proof that training completed.",
            "- Distributed training correctness requires attached cluster preflight or torchrun evidence when more than one node is used.",
            "- Promotion requires source-relative eval results and comparison against the configured baseline.",
            "",
        ]
    )


TRAINER_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import yaml


class ResourceGuard:
    def __init__(self, plan: dict[str, Any], run_dir: Path) -> None:
        self.plan = plan
        self.policy = plan.get("resource_policy", {})
        self.run_dir = run_dir
        self.monitor_interval = int(self.policy.get("monitor_interval_seconds", 30))
        self._stop = threading.Event()
        self._last_check = 0.0

    def usable_cores(self) -> int:
        reserve = int(self.policy.get("reserve_cores", 1))
        return max(1, (os.cpu_count() or 2) - reserve)

    def configure_threads(self) -> int:
        cores = self.usable_cores()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = str(cores)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        try:
            import torch
            torch.set_num_threads(cores)
            torch.set_num_interop_threads(max(1, min(cores, 4)))
        except Exception:
            pass
        return cores

    def _psutil(self):
        try:
            import psutil
            return psutil
        except Exception as exc:
            raise RuntimeError("psutil is required for model-forge resource guard") from exc

    def memory_available_ratio(self) -> float:
        mem = self._psutil().virtual_memory()
        return float(mem.available) / float(mem.total)

    def disk_free_ratio(self, path: Path | None = None) -> float:
        target = path or self.run_dir
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
        return float(usage.free) / float(usage.total)

    def preflight(self) -> None:
        self.configure_threads()
        min_mem = float(self.policy.get("min_memory_available_start", 0.05))
        min_disk = float(self.policy.get("min_disk_free", 0.15))
        mem_ratio = self.memory_available_ratio()
        disk_ratio = self.disk_free_ratio()
        if mem_ratio < min_mem:
            raise RuntimeError(f"Not enough free memory to start job: {mem_ratio:.1%} available < {min_mem:.1%}")
        if disk_ratio < min_disk:
            raise RuntimeError(f"Not enough free disk to start job: {disk_ratio:.1%} free < {min_disk:.1%}")

    def check_runtime(self) -> None:
        min_mem = float(self.policy.get("min_memory_available_runtime", 0.05))
        min_disk = float(self.policy.get("min_disk_free", 0.15))
        mem_ratio = self.memory_available_ratio()
        disk_ratio = self.disk_free_ratio()
        if mem_ratio < min_mem:
            raise RuntimeError(f"Resource guard stopping job: memory available {mem_ratio:.1%} < {min_mem:.1%}")
        if disk_ratio < min_disk:
            raise RuntimeError(f"Resource guard stopping job: disk free {disk_ratio:.1%} < {min_disk:.1%}")

    def check_runtime_periodically(self) -> None:
        now = time.monotonic()
        if now - self._last_check >= self.monitor_interval:
            self._last_check = now
            self.check_runtime()

    def start_monitor(self) -> None:
        def monitor() -> None:
            while not self._stop.wait(self.monitor_interval):
                try:
                    self.check_runtime()
                except Exception as exc:
                    payload = {"error": str(exc), "time": time.time()}
                    self.run_dir.mkdir(parents=True, exist_ok=True)
                    (self.run_dir / "resource_guard_abort.json").write_text(json.dumps(payload, indent=2) + "\n")
                    os._exit(75)

        thread = threading.Thread(target=monitor, name="model-forge-resource-guard", daemon=True)
        thread.start()

    def stop_monitor(self) -> None:
        self._stop.set()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def normalize_messages(example: dict[str, Any], source: dict[str, Any]) -> list[dict[str, str]] | None:
    field = source.get("messages_field", "messages")
    if isinstance(example.get(field), list):
        messages = []
        for item in example[field]:
            role = item.get("role") or item.get("from")
            content = item.get("content") or item.get("value")
            if role == "human":
                role = "user"
            if role == "gpt":
                role = "assistant"
            if role in {"user", "assistant", "system"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})
        return messages or None
    input_field = source.get("input_field", "Input")
    output_field = source.get("output_field") or source.get("answer_field", "Answer")
    prompt = (
        example.get(input_field)
        or example.get(input_field.lower())
        or example.get("question")
        or example.get("prompt")
        or example.get("problem")
        or example.get("input")
    )
    answer = (
        example.get(output_field)
        or example.get(output_field.lower())
        or example.get("Answer")
        or example.get("Anwser")
        or example.get("answer")
        or example.get("anwser")
        or example.get("output")
        or example.get("solution")
        or example.get("code_output")
    )
    reasoning_field = source.get("reasoning_field")
    reasoning = example.get(reasoning_field) if reasoning_field else None
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    answer = answer.strip()
    if reasoning and isinstance(reasoning, str) and source.get("wrap_reasoning", False):
        answer = f"<think>\n{reasoning.strip()}\n</think>\n{answer}"
    return [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer}]


def normalize_rejected_messages(
    example: dict[str, Any],
    source: dict[str, Any],
    chosen_messages: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    field = source.get("rejected_messages_field", "rejected_messages")
    if isinstance(example.get(field), list):
        rejected_example = dict(example)
        rejected_example[source.get("messages_field", "messages")] = example[field]
        return normalize_messages(rejected_example, source)

    rejected_field = source.get("rejected_field", "rejected")
    rejected = (
        example.get(rejected_field)
        or example.get("rejected")
        or example.get("rejected_answer")
        or example.get("negative")
        or example.get("bad_answer")
    )
    if not isinstance(rejected, str) or not rejected.strip():
        return None
    if not chosen_messages or chosen_messages[-1].get("role") != "assistant":
        return None
    return [*chosen_messages[:-1], {"role": "assistant", "content": rejected.strip()}]


def conversation_hash(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def remap_repo_path(value: Any, repo_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    parts = path.parts
    for anchor in ("runs", "configs", "datasets", "evals", "recipes"):
        if anchor in parts:
            idx = parts.index(anchor)
            return str(repo_root.joinpath(*parts[idx:]))
    return value


def remap_plan_repo_paths(plan: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    for key in ("config_path", "run_dir"):
        if key in plan:
            plan[key] = remap_repo_path(plan[key], repo_root)
    data = plan.get("data")
    if isinstance(data, dict):
        data["manifest"] = remap_repo_path(data.get("manifest"), repo_root)
    return plan


def valid_messages(messages: list[dict[str, str]], gates: dict[str, Any]) -> bool:
    if len(messages) < 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    if not any(item["role"] == "user" for item in messages):
        return False
    if any(len(item["content"].strip()) < gates.get("min_turn_chars", 2) for item in messages):
        return False
    if gates.get("require_assistant_content", True) and not messages[-1]["content"].strip():
        return False
    if gates.get("reject_unclosed_think", True):
        text = messages[-1]["content"]
        if text.count("<think>") != text.count("</think>"):
            return False
    return True


def normalized_prompt_for_overlap(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_holdout_prompts(plan: dict[str, Any]) -> set[str]:
    gates = plan.get("data", {}).get("quality_gates", {})
    if not gates.get("reject_eval_prompt_overlap", False):
        return set()
    repo_root = Path(__file__).resolve().parents[3]
    prompts: set[str] = set()
    for raw_path in plan.get("data", {}).get("holdouts", []):
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for case in data.get("cases", []):
            prompt = case.get("prompt") if isinstance(case, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                prompts.add(normalized_prompt_for_overlap(prompt))
    return prompts


def messages_overlap_holdout(messages: list[dict[str, str]], holdout_prompts: set[str]) -> bool:
    if not holdout_prompts:
        return False
    user_text = "\n".join(item["content"] for item in messages if item.get("role") == "user")
    return normalized_prompt_for_overlap(user_text) in holdout_prompts


def tokenizer_source(plan: dict[str, Any]) -> str:
    local_dir = plan["model"].get("local_dir")
    source = local_dir if local_dir and Path(str(local_dir)).expanduser().exists() else plan["model"]["source"]
    source_path = Path(source).expanduser()
    config_path = source_path / "tokenizer_config.json"
    if not config_path.exists():
        return source

    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return source
    extra_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        return source

    compat_dir = Path(plan["run_dir"]) / "tokenizer_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "processor_config.json",
        "generation_config.json",
    ):
        src = source_path / name
        if src.exists():
            shutil.copy2(src, compat_dir / name)

    patched_config_path = compat_dir / "tokenizer_config.json"
    patched_config = json.loads(patched_config_path.read_text())
    mapping: dict[str, str] = {}
    for index, token in enumerate(extra_tokens):
        if token == "<|video|>":
            key = "video_token"
        else:
            key = f"extra_special_token_{index}"
        mapping[key] = token
    patched_config["extra_special_tokens"] = mapping
    patched_config_path.write_text(json.dumps(patched_config, indent=2) + "\n")
    return str(compat_dir)


def build_dataset(plan: dict[str, Any], output_path: Path, limit: int | None = None) -> dict[str, Any]:
    from datasets import Dataset, concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    guard = ResourceGuard(plan, Path(plan["run_dir"]))
    guard.preflight()
    guard.start_monitor()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source(plan),
            trust_remote_code=plan["model"].get("trust_remote_code", False),
        )
        max_context = int(plan["data"]["max_context_window"])
        gates = plan["data"].get("quality_gates", {})
        holdout_prompts = load_holdout_prompts(plan)
        chunks = []
        source_stats = []
        seen: set[str] = set()

        for source in plan["data"]["sources"]:
            if source.get("enabled", True) is False:
                continue
            split = source.get("split", "train")
            name = source.get("name", source.get("dataset", ""))
            configured_target = int(source.get("target_samples", 0) or 0)
            if source.get("path"):
                ds = load_dataset("json", data_files=source["path"], split="train")
                target = configured_target or len(ds)
                target = min(target, len(ds))
                if limit is not None:
                    target = min(target, limit)
            else:
                target = limit or configured_target
                if target:
                    print(f"[model-forge] streaming source {name}: target={target}", flush=True)
                    if source.get("subset"):
                        ds_iter = load_dataset(source["dataset"], source["subset"], split=split, streaming=True)
                    else:
                        ds_iter = load_dataset(source["dataset"], split=split, streaming=True)
                    buffer_size = int(os.getenv("MODEL_FORGE_DATA_STREAM_BUFFER", "1024"))
                    if hasattr(ds_iter, "shuffle"):
                        ds_iter = ds_iter.shuffle(seed=int(plan["trainer"]["seed"]), buffer_size=buffer_size)
                    ds = list(ds_iter.take(target))
                    target = len(ds)
                else:
                    if source.get("subset"):
                        ds = load_dataset(source["dataset"], source["subset"], split=split)
                    else:
                        ds = load_dataset(source["dataset"], split=split)
                    target = len(ds)
            if target <= 0:
                continue
            print(f"[model-forge] preparing source {name}: target={target}", flush=True)
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=int(plan["trainer"]["seed"])).select(range(target))
            else:
                ds = ds[:target]
            rows = []
            rejected = 0
            for example in ds:
                guard.check_runtime_periodically()
                messages = normalize_messages(example, source)
                if messages is None or not valid_messages(messages, gates):
                    rejected += 1
                    continue
                if messages_overlap_holdout(messages, holdout_prompts):
                    rejected += 1
                    continue
                rejected_messages = normalize_rejected_messages(example, source, messages)
                if rejected_messages is not None and not valid_messages(rejected_messages, gates):
                    rejected_messages = None
                digest = conversation_hash(messages)
                if digest in seen:
                    rejected += 1
                    continue
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                rejected_text = None
                if rejected_messages is not None:
                    rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
                    rejected_token_count = len(tokenizer(rejected_text, add_special_tokens=False)["input_ids"])
                    token_count = max(token_count, rejected_token_count)
                if token_count > max_context:
                    rejected += 1
                    continue
                seen.add(digest)
                row = {"id": digest, "source": name, "messages": messages, "text": text, "token_count": token_count}
                if rejected_messages is not None and rejected_text is not None:
                    row["rejected_messages"] = rejected_messages
                    row["rejected_text"] = rejected_text
                rows.append(row)
            if rows:
                if any("rejected_messages" in row for row in rows):
                    for row in rows:
                        row.setdefault("rejected_messages", None)
                        row.setdefault("rejected_text", None)
                chunks.append(Dataset.from_list(rows))
            source_stats.append({"name": name, "sampled": target, "accepted": len(rows), "rejected": rejected})
            print(f"[model-forge] prepared source {name}: accepted={len(rows)} rejected={rejected}", flush=True)

        if not chunks:
            raise SystemExit("no training rows survived data preparation")
        dataset = concatenate_datasets(chunks).shuffle(seed=int(plan["trainer"]["seed"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
        return {"output_path": str(output_path), "rows": len(dataset), "sources": source_stats}
    finally:
        guard.stop_monitor()


def train(plan: dict[str, Any], dataset_path: Path) -> None:
    backend = str(plan["trainer"].get("backend", "hf_causal_lm")).lower()
    FastLanguageModel = None
    if backend == "unsloth":
        if bool(plan["trainer"].get("unsloth_compile_disable", False)):
            os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastLanguageModel

    import inspect
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset, load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from accelerate.parallelism_config import ParallelismConfig
    from transformers import TrainerCallback
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    recompile_limit = int(plan["trainer"].get("torch_dynamo_recompile_limit", 0) or 0)
    if recompile_limit > 0:
        try:
            import torch._dynamo

            torch._dynamo.config.recompile_limit = recompile_limit
            if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
                torch._dynamo.config.accumulated_recompile_limit = max(
                    torch._dynamo.config.accumulated_recompile_limit,
                    recompile_limit * 4,
                )
        except Exception:
            pass

    guard = ResourceGuard(plan, Path(plan["run_dir"]))
    guard.preflight()
    guard.start_monitor()

    class ResourceGuardCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            try:
                guard.check_runtime_periodically()
            except Exception:
                if plan.get("resource_policy", {}).get("checkpoint_on_memory_pressure", True):
                    model = kwargs.get("model")
                    tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
                    checkpoint_dir = Path(plan["run_dir"]) / "resource_guard_checkpoint"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    if model is not None:
                        model.save_pretrained(checkpoint_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(checkpoint_dir)
                raise
            return control

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(plan), trust_remote_code=plan["model"].get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seq_length = int(plan["model"]["max_seq_length"])
    method = str(plan["trainer"].get("method", "")).lower()
    use_pairwise_preference = any(marker in method for marker in ("pairwise_preference", "preference_dpo", "preference", "simpo"))
    unlikelihood_weight = float(plan["trainer"].get("unlikelihood_weight", 0.0) or 0.0)
    unlikelihood_scope = str(plan["trainer"].get("unlikelihood_scope", "assistant") or "assistant").lower()
    unlikelihood_prefix_tokens = int(plan["trainer"].get("unlikelihood_prefix_tokens", 0) or 0)
    use_unlikelihood = unlikelihood_weight > 0 or "unlikelihood" in method
    assistant_only_loss = bool(plan["trainer"].get("assistant_only_loss", False)) or use_unlikelihood or use_pairwise_preference

    def assistant_prefix_text(messages: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    def assistant_labels(input_ids: list[int], messages: list[dict[str, str]]) -> list[int]:
        labels = list(input_ids)
        try:
            prefix_ids = tokenizer(assistant_prefix_text(messages), add_special_tokens=False)["input_ids"]
            prefix_len = min(len(prefix_ids), len(labels))
        except Exception:
            prefix_len = 0
        for index in range(prefix_len):
            labels[index] = -100
        return labels

    def scoped_unlikelihood_labels(labels: list[int]) -> list[int]:
        if unlikelihood_scope not in {"assistant_prefix", "prefix"}:
            return list(labels)
        if unlikelihood_prefix_tokens <= 0:
            return list(labels)
        scoped = [-100] * len(labels)
        seen = 0
        for index, label in enumerate(labels):
            if label == -100:
                continue
            if seen < unlikelihood_prefix_tokens:
                scoped[index] = label
            seen += 1
        return scoped

    tokenized_path = Path(plan["run_dir"]) / f"tokenized_train_{max_seq_length}"
    if tokenized_path.exists():
        dataset = load_from_disk(str(tokenized_path))
    else:
        raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        def tokenize_batch(batch):
            tokenized = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            tokenized["mm_token_type_ids"] = [[0] * len(input_ids) for input_ids in tokenized["input_ids"]]
            if assistant_only_loss:
                tokenized["labels"] = [
                    assistant_labels(input_ids, messages)
                    for input_ids, messages in zip(tokenized["input_ids"], batch["messages"], strict=False)
                ]
            if use_unlikelihood or use_pairwise_preference:
                rejected_texts = batch.get("rejected_text") or [None] * len(batch["text"])
                rejected_messages_batch = batch.get("rejected_messages") or [None] * len(batch["text"])
                rejected_input_ids = []
                rejected_attention_mask = []
                rejected_labels = []
                rejected_unlikelihood_labels = []
                for rejected_text, rejected_messages in zip(rejected_texts, rejected_messages_batch, strict=False):
                    if isinstance(rejected_text, str) and rejected_text.strip() and isinstance(rejected_messages, list):
                        rejected_tokenized = tokenizer(
                            rejected_text,
                            truncation=True,
                            max_length=max_seq_length,
                            padding=False,
                        )
                        rejected_ids = rejected_tokenized["input_ids"]
                        labels = assistant_labels(rejected_ids, rejected_messages)
                        rejected_input_ids.append(rejected_ids)
                        rejected_attention_mask.append(rejected_tokenized["attention_mask"])
                        rejected_labels.append(labels)
                        rejected_unlikelihood_labels.append(scoped_unlikelihood_labels(labels))
                    else:
                        rejected_input_ids.append([])
                        rejected_attention_mask.append([])
                        rejected_labels.append([])
                        rejected_unlikelihood_labels.append([])
                tokenized["rejected_input_ids"] = rejected_input_ids
                tokenized["rejected_attention_mask"] = rejected_attention_mask
                tokenized["rejected_labels"] = rejected_labels
                tokenized["rejected_unlikelihood_labels"] = rejected_unlikelihood_labels
            return tokenized

        dataset = raw_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        dataset.save_to_disk(str(tokenized_path))
        del raw_dataset
    gc.collect()
    guard.check_runtime()

    local_model_dir = plan["model"].get("local_dir")
    model_id = local_model_dir if local_model_dir and Path(str(local_model_dir)).expanduser().exists() else plan["model"]["source"]
    quantization_config = None
    if backend != "unsloth" and plan["trainer"].get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if plan["trainer"].get("bf16", True) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    dtype = torch.bfloat16 if plan["trainer"].get("bf16", True) else torch.float16
    tensor_parallel_size = int(os.environ.get("MODEL_FORGE_TRAIN_TP_SIZE") or plan["trainer"].get("tensor_parallel_size", 1) or 1)
    tensor_parallel_plan = os.environ.get("MODEL_FORGE_TRAIN_TP_PLAN") or plan["trainer"].get("tensor_parallel_plan", "auto")
    if backend == "unsloth":
        if tensor_parallel_size > 1:
            raise RuntimeError("tensor_parallel_size > 1 is not supported with the unsloth backend")
        model, _processor = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=bool(plan["trainer"].get("load_in_4bit", True)),
            load_in_8bit=bool(plan["trainer"].get("load_in_8bit", False)),
            full_finetuning=False,
            trust_remote_code=plan["model"].get("trust_remote_code", False),
            attn_implementation=plan["trainer"].get("attn_implementation", "eager"),
            low_cpu_mem_usage=True,
            offload_state_dict=True,
        )
    else:
        device_map = {"": 0} if plan["trainer"].get("device_map") == "single_gpu" else plan["trainer"].get("device_map", "auto")
        model_kwargs = {}
        if tensor_parallel_size > 1:
            device_map = None
            model_kwargs.update({"tp_plan": tensor_parallel_plan, "tp_size": tensor_parallel_size})
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=plan["model"].get("trust_remote_code", False),
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            use_safetensors=True,
            **model_kwargs,
        )
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
    guard.check_runtime()
    if hasattr(model, "config"):
        model.config.use_cache = False
    lora = plan["lora"]
    modules_to_save = list(lora.get("modules_to_save", []))
    exclude_modules = list(lora.get("exclude_modules", []))
    if backend == "unsloth":
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora["r"]),
            target_modules=list(lora["target_modules"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            use_gradient_checkpointing="unsloth" if plan["trainer"].get("gradient_checkpointing", True) else False,
            random_state=int(plan["trainer"]["seed"]),
            modules_to_save=modules_to_save or None,
            exclude_modules=exclude_modules or None,
        )
        FastLanguageModel.for_training(model)
    else:
        peft_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
            exclude_modules=exclude_modules or None,
            modules_to_save=modules_to_save or None,
        )
        model = get_peft_model(model, peft_config)
    guard.check_runtime()

    max_steps = plan["trainer"].get("max_steps") or -1
    if max_steps and int(max_steps) > 0:
        warmup_steps = max(0, int(math.ceil(int(max_steps) * float(plan["trainer"]["warmup_ratio"]))))
    else:
        effective_batch = int(plan["trainer"]["per_device_train_batch_size"]) * int(plan["trainer"]["gradient_accumulation_steps"])
        steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, effective_batch)))
        warmup_steps = max(0, int(math.ceil(steps_per_epoch * float(plan["trainer"]["num_train_epochs"]) * float(plan["trainer"]["warmup_ratio"]))))

    benchmark_only = (
        bool(plan["trainer"].get("benchmark_only", False))
        or os.environ.get("MODEL_FORGE_TRAIN_BENCHMARK_ONLY", "0") == "1"
    )
    save_strategy = "no" if benchmark_only else str(plan["trainer"].get("save_strategy", "steps"))
    training_kwargs = dict(
        output_dir=plan["model"]["output_dir"],
        per_device_train_batch_size=int(plan["trainer"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(plan["trainer"]["gradient_accumulation_steps"]),
        learning_rate=float(plan["trainer"]["learning_rate"]),
        num_train_epochs=float(plan["trainer"]["num_train_epochs"]),
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=plan["trainer"]["lr_scheduler_type"],
        optim=plan["trainer"]["optim"],
        weight_decay=float(plan["trainer"]["weight_decay"]),
        logging_steps=int(plan["trainer"]["logging_steps"]),
        bf16=bool(plan["trainer"]["bf16"]),
        gradient_checkpointing=bool(plan["trainer"]["gradient_checkpointing"]),
        seed=int(plan["trainer"]["seed"]),
        report_to=plan["trainer"]["report_to"],
        save_strategy=save_strategy,
        remove_unused_columns=False,
    )
    if save_strategy != "no":
        training_kwargs["save_steps"] = int(plan["trainer"]["save_steps"])
        training_kwargs["save_total_limit"] = int(plan["trainer"]["save_total_limit"])
    if "ddp_find_unused_parameters" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["ddp_find_unused_parameters"] = bool(plan["trainer"].get("ddp_find_unused_parameters", True))
    if tensor_parallel_size > 1:
        training_kwargs["parallelism_config"] = ParallelismConfig(tp_size=tensor_parallel_size)
    if bool(plan["trainer"].get("group_by_length", False)):
        training_kwargs["group_by_length"] = True
    worker_offset = int(plan.get("resource_policy", {}).get("dataloader_num_workers_max_offset", 2))
    max_workers = max(0, guard.usable_cores() - worker_offset)
    configured_workers = int(plan["trainer"].get("dataloader_num_workers", 0) or 0)
    dataloader_workers = min(configured_workers, max_workers)
    training_kwargs["dataloader_num_workers"] = dataloader_workers
    training_kwargs["dataloader_persistent_workers"] = (
        bool(plan["trainer"].get("dataloader_persistent_workers", False))
        and dataloader_workers > 0
        and bool(plan.get("resource_policy", {}).get("persistent_workers_when_memory_tight", False))
    )
    if dataloader_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = min(int(plan["trainer"].get("dataloader_prefetch_factor", 2) or 2), 2)
    args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "gradient_checkpointing_kwargs" in args_params:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    args = TrainingArguments(**{key: value for key, value in training_kwargs.items() if key in args_params})

    pad_to_multiple_of = int(plan["trainer"].get("pad_to_multiple_of", 0) or 0) or None

    class AssistantOnlyCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=None) -> None:
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of

        def _pad(self, values, pad_value):
            max_len = max(len(value) for value in values) if values else 1
            if self.pad_to_multiple_of:
                remainder = max_len % self.pad_to_multiple_of
                if remainder:
                    max_len += self.pad_to_multiple_of - remainder
            return torch.tensor([list(value) + [pad_value] * (max_len - len(value)) for value in values], dtype=torch.long)

        def __call__(self, features):
            input_ids = [feature["input_ids"] for feature in features]
            attention_mask = [feature["attention_mask"] for feature in features]
            labels = [feature.get("labels", feature["input_ids"]) for feature in features]
            return {
                "input_ids": self._pad(input_ids, self.tokenizer.pad_token_id),
                "attention_mask": self._pad(attention_mask, 0),
                "labels": self._pad(labels, -100),
            }

    class RefusalUnlikelihoodCollator(AssistantOnlyCollator):
        def __call__(self, features):
            batch = super().__call__(features)
            rejected_input_ids = [
                feature.get("rejected_input_ids") or [self.tokenizer.pad_token_id]
                for feature in features
            ]
            rejected_attention_mask = [
                feature.get("rejected_attention_mask") or [1]
                for feature in features
            ]
            rejected_labels = [
                feature.get("rejected_labels") or [-100]
                for feature in features
            ]
            rejected_unlikelihood_labels = [
                feature.get("rejected_unlikelihood_labels") or feature.get("rejected_labels") or [-100]
                for feature in features
            ]
            batch["rejected_input_ids"] = self._pad(rejected_input_ids, self.tokenizer.pad_token_id)
            batch["rejected_attention_mask"] = self._pad(rejected_attention_mask, 0)
            batch["rejected_labels"] = self._pad(rejected_labels, -100)
            batch["rejected_unlikelihood_labels"] = self._pad(rejected_unlikelihood_labels, -100)
            return batch

    class RefusalUnlikelihoodTrainer(Trainer):
        def __init__(self, *args, unlikelihood_weight: float, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.unlikelihood_weight = float(unlikelihood_weight)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            rejected_input_ids = inputs.pop("rejected_input_ids", None)
            rejected_attention_mask = inputs.pop("rejected_attention_mask", None)
            rejected_labels = inputs.pop("rejected_labels", None)
            rejected_unlikelihood_labels = inputs.pop("rejected_unlikelihood_labels", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if (
                rejected_input_ids is not None
                and rejected_attention_mask is not None
                and rejected_unlikelihood_labels is not None
                and self.unlikelihood_weight > 0
            ):
                rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
                if bool((rejected_unlikelihood_labels != -100).any()):
                    logits = rejected_outputs.logits[:, :-1, :].float()
                    labels = rejected_unlikelihood_labels[:, 1:]
                    mask = labels.ne(-100)
                    safe_labels = labels.masked_fill(~mask, 0)
                    token_probs = torch.softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                    unlikelihood = -torch.log(torch.clamp(1.0 - token_probs, min=1e-6))
                    unlikelihood_loss = unlikelihood.masked_select(mask).mean()
                    loss = loss + (self.unlikelihood_weight * unlikelihood_loss)
                else:
                    # Keep DDP ranks in the same forward/backward structure even
                    # when only some ranks receive paired rejected completions.
                    loss = loss + (rejected_outputs.logits.sum() * 0.0)
            return (loss, outputs) if return_outputs else loss

    class PairwisePreferenceTrainer(Trainer):
        def __init__(
            self,
            *args,
            preference_weight: float,
            preference_beta: float,
            preference_margin: float,
            sft_weight: float,
            length_normalize: bool,
            unlikelihood_weight: float,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.preference_weight = float(preference_weight)
            self.preference_beta = float(preference_beta)
            self.preference_margin = float(preference_margin)
            self.sft_weight = float(sft_weight)
            self.length_normalize = bool(length_normalize)
            self.unlikelihood_weight = float(unlikelihood_weight)

        def _sequence_logps(self, logits, labels):
            token_logits = logits[:, :-1, :].float()
            token_labels = labels[:, 1:]
            mask = token_labels.ne(-100)
            safe_labels = token_labels.masked_fill(~mask, 0)
            token_logps = torch.log_softmax(token_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            logp_sums = token_logps.masked_fill(~mask, 0.0).sum(dim=-1)
            token_counts = mask.sum(dim=-1)
            if self.length_normalize:
                return logp_sums / token_counts.clamp_min(1), token_counts
            return logp_sums, token_counts

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            rejected_input_ids = inputs.pop("rejected_input_ids", None)
            rejected_attention_mask = inputs.pop("rejected_attention_mask", None)
            rejected_labels = inputs.pop("rejected_labels", None)
            rejected_unlikelihood_labels = inputs.pop("rejected_unlikelihood_labels", None)
            chosen_labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss * self.sft_weight
            if rejected_input_ids is None or rejected_attention_mask is None or rejected_labels is None or chosen_labels is None:
                return (loss, outputs) if return_outputs else loss

            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            chosen_logps, chosen_counts = self._sequence_logps(outputs.logits, chosen_labels)
            rejected_logps, rejected_counts = self._sequence_logps(rejected_outputs.logits, rejected_labels)
            valid_pairs = chosen_counts.gt(0) & rejected_counts.gt(0)
            used_rejected_loss = False
            if bool(valid_pairs.any()) and self.preference_weight > 0:
                preference_logits = self.preference_beta * (
                    chosen_logps[valid_pairs] - rejected_logps[valid_pairs] - self.preference_margin
                )
                preference_loss = -F.logsigmoid(preference_logits).mean()
                loss = loss + (self.preference_weight * preference_loss)
                used_rejected_loss = True
            if rejected_unlikelihood_labels is None:
                rejected_unlikelihood_labels = rejected_labels
            if bool((rejected_unlikelihood_labels != -100).any()) and self.unlikelihood_weight > 0:
                logits = rejected_outputs.logits[:, :-1, :].float()
                labels = rejected_unlikelihood_labels[:, 1:]
                mask = labels.ne(-100)
                safe_labels = labels.masked_fill(~mask, 0)
                token_probs = torch.softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                unlikelihood = -torch.log(torch.clamp(1.0 - token_probs, min=1e-6))
                unlikelihood_loss = unlikelihood.masked_select(mask).mean()
                loss = loss + (self.unlikelihood_weight * unlikelihood_loss)
                used_rejected_loss = True
            if not used_rejected_loss:
                # Keep DDP ranks in the same forward/backward structure even
                # when only some ranks receive paired preference/unlikelihood rows.
                loss = loss + (rejected_outputs.logits.sum() * 0.0)
            return (loss, outputs) if return_outputs else loss

    if use_pairwise_preference:
        data_collator = RefusalUnlikelihoodCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        trainer_cls = PairwisePreferenceTrainer
    elif use_unlikelihood:
        data_collator = RefusalUnlikelihoodCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        trainer_cls = RefusalUnlikelihoodTrainer
    elif assistant_only_loss:
        data_collator = AssistantOnlyCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        trainer_cls = Trainer
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        trainer_cls = Trainer

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        args=args,
        data_collator=data_collator,
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if use_pairwise_preference:
        trainer = trainer_cls(
            **trainer_kwargs,
            preference_weight=float(plan["trainer"].get("preference_weight", 1.0) or 1.0),
            preference_beta=float(plan["trainer"].get("preference_beta", 0.1) or 0.1),
            preference_margin=float(plan["trainer"].get("preference_margin", 0.0) or 0.0),
            sft_weight=float(plan["trainer"].get("sft_weight", 1.0) if plan["trainer"].get("sft_weight") is not None else 1.0),
            length_normalize=bool(plan["trainer"].get("preference_length_normalize", True)),
            unlikelihood_weight=unlikelihood_weight,
        )
    elif use_unlikelihood:
        trainer = trainer_cls(**trainer_kwargs, unlikelihood_weight=unlikelihood_weight)
    else:
        trainer = trainer_cls(**trainer_kwargs)
    trainer.add_callback(ResourceGuardCallback())
    try:
        train_output = trainer.train()
        result_payload = {
            "schema_version": "model_forge.finetune_training_result.v1",
            "name": plan["name"],
            "family": plan["family"],
            "benchmark_only": benchmark_only,
            "dataset_rows": len(dataset),
            "global_step": getattr(train_output, "global_step", None),
            "metrics": dict(getattr(train_output, "metrics", {}) or {}),
            "lora": {
                "rank": int(plan["lora"]["r"]),
                "alpha": int(plan["lora"]["alpha"]),
                "dropout": float(plan["lora"].get("dropout", 0.0) or 0.0),
                "target_modules": list(plan["lora"].get("target_modules", [])),
            },
            "trainer": {
                "backend": backend,
                "method": plan["trainer"].get("method"),
                "assistant_only_loss": assistant_only_loss,
                "unlikelihood_weight": unlikelihood_weight,
                "unlikelihood_scope": unlikelihood_scope,
                "unlikelihood_prefix_tokens": unlikelihood_prefix_tokens,
                "refusal_unlikelihood": use_unlikelihood,
                "pairwise_preference": use_pairwise_preference,
                "preference_weight": float(plan["trainer"].get("preference_weight", 0.0) or 0.0),
                "preference_beta": float(plan["trainer"].get("preference_beta", 0.1) or 0.1),
                "preference_margin": float(plan["trainer"].get("preference_margin", 0.0) or 0.0),
                "sft_weight": float(plan["trainer"].get("sft_weight", 1.0) if plan["trainer"].get("sft_weight") is not None else 1.0),
                "preference_length_normalize": bool(plan["trainer"].get("preference_length_normalize", True)),
                "tensor_parallel_size": tensor_parallel_size,
                "tensor_parallel_plan": tensor_parallel_plan,
                "max_steps": max_steps,
                "per_device_train_batch_size": int(plan["trainer"]["per_device_train_batch_size"]),
                "gradient_accumulation_steps": int(plan["trainer"]["gradient_accumulation_steps"]),
                "save_strategy": save_strategy,
            },
        }
        if trainer.is_world_process_zero():
            result_path = Path(plan["run_dir"]) / "training_result.json"
            result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")
        if not benchmark_only:
            trainer.save_model(plan["model"]["output_dir"])
            tokenizer.save_pretrained(plan["model"]["output_dir"])
    finally:
        guard.stop_monitor()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generated model-forge TRL SFT runner")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data-limit", type=int, default=None)
    args = parser.parse_args()
    plan = remap_plan_repo_paths(json.loads(Path(args.plan).read_text()))
    dataset_path = Path(plan["run_dir"]) / "train.jsonl"
    if args.prepare_data:
        summary = build_dataset(plan, dataset_path, args.data_limit)
        (Path(plan["run_dir"]) / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    if args.train:
        train(plan, dataset_path)


if __name__ == "__main__":
    main()
'''


def write_artifacts(plan: dict[str, Any], *, overwrite: bool = False) -> dict[str, str]:
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "plan": run_dir / "plan.json",
        "trainer": run_dir / "train_trl_sft.py",
        "shell": run_dir / "run.sh",
        "cluster_shell": run_dir / "run_cluster_torchrun.sh",
        "eval": run_dir / "eval_after_training.sh",
        "method_card": run_dir / "training_method_card.md",
    }
    for path in outputs.values():
        if path.exists() and not overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace generated artifacts")
    outputs["plan"].write_text(json.dumps(plan, indent=2) + "\n")
    outputs["trainer"].write_text(TRAINER_SCRIPT)
    outputs["trainer"].chmod(0o755)

    env_exports = "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in plan["hardware"]["training_env"].items()
    )
    resource_policy = plan["resource_policy"]
    run_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(REPO_DIR))}

{env_exports}

PYTHON=${{PYTHON:-{shlex.quote(str(REPO_DIR / '.venv' / 'bin' / 'python'))}}}
CPU_QUOTA=${{MODEL_FORGE_CPU_QUOTA:-{shlex.quote(str(resource_policy["cpu_quota"]))}}}
MEMORY_MAX=${{MODEL_FORGE_MEMORY_MAX:-{shlex.quote(str(resource_policy["memory_max"]))}}}
IO_WEIGHT=${{MODEL_FORGE_IO_WEIGHT:-{int(resource_policy["io_weight"])}}}
NICE_LEVEL=${{MODEL_FORGE_NICE:-{int(resource_policy["nice"])}}}
RESERVE_CORES=${{MODEL_FORGE_RESERVE_CORES:-{int(resource_policy["reserve_cores"])}}}
export RESERVE_CORES
USABLE_CORES=$("$PYTHON" - <<'PY'
import os
reserve = int(os.environ.get("RESERVE_CORES", "1"))
print(max(1, (os.cpu_count() or 2) - reserve))
PY
)
export OMP_NUM_THREADS="$USABLE_CORES"
export MKL_NUM_THREADS="$USABLE_CORES"
export NUMEXPR_NUM_THREADS="$USABLE_CORES"
export OPENBLAS_NUM_THREADS="$USABLE_CORES"
export TOKENIZERS_PARALLELISM="${{TOKENIZERS_PARALLELISM:-true}}"

df -h /

run_limited() {{
  if command -v systemd-run >/dev/null 2>&1 && [[ ! -f /.dockerenv ]] && [[ "${{MODEL_FORGE_DISABLE_SYSTEMD_SCOPE:-0}}" != "1" ]]; then
    systemd-run --scope \\
      -p "CPUQuota=$CPU_QUOTA" \\
      -p "MemoryMax=$MEMORY_MAX" \\
      -p "IOWeight=$IO_WEIGHT" \\
      nice -n "$NICE_LEVEL" "$@"
  else
    nice -n "$NICE_LEVEL" "$@"
  fi
}}

if [[ "${{MODEL_FORGE_SKIP_PREPARE:-0}}" == "1" && -s {shlex.quote(str(run_dir / "train.jsonl"))} ]]; then
  echo "[model-forge] skipping data prep; existing train.jsonl found"
else
  run_limited "$PYTHON" {shlex.quote(str(outputs["trainer"]))} --plan {shlex.quote(str(outputs["plan"]))} --prepare-data
fi
run_limited "$PYTHON" {shlex.quote(str(outputs["trainer"]))} --plan {shlex.quote(str(outputs["plan"]))} --train
"""
    outputs["shell"].write_text(run_script)
    outputs["shell"].chmod(0o755)

    cluster = plan.get("cluster", {})
    cluster_config = cluster.get("config") or "${MODEL_FORGE_CLUSTER_CONFIG:?set cluster config path}"
    cluster_image = cluster.get("image") or "${MODEL_FORGE_TRAIN_IMAGE:-nemotron-runner:latest}"
    cluster_iface = cluster.get("nccl_socket_ifname") or "${MODEL_FORGE_NCCL_SOCKET_IFNAME:-}"
    cluster_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(REPO_DIR))}

CLUSTER_CONFIG="${{MODEL_FORGE_CLUSTER_CONFIG:-{shlex.quote(str(cluster_config))}}}"
TRAIN_IMAGE="${{MODEL_FORGE_TRAIN_IMAGE:-{shlex.quote(str(cluster_image))}}}"
NCCL_SOCKET_IFNAME="${{MODEL_FORGE_NCCL_SOCKET_IFNAME:-{shlex.quote(str(cluster_iface))}}}"
PYTHON=${{PYTHON:-{shlex.quote(str(REPO_DIR / '.venv' / 'bin' / 'python'))}}}
RUN_DIR={shlex.quote(str(run_dir))}
PLAN={shlex.quote(str(outputs["plan"]))}
TRAINER={shlex.quote(str(outputs["trainer"]))}
JOB_LOCK="${{MODEL_FORGE_CLUSTER_JOB_LOCK:-runs/locks/model-forge-cluster.lock}}"
FAMILY={shlex.quote(str(plan["family"]))}
SOURCE_VARIANT={shlex.quote(str(plan.get("eval", {}).get("source_variant") or "base"))}
SOURCE_MODEL_DIR={shlex.quote(str(plan["model"].get("local_dir") or ""))}
SYNC_MODEL_TO_WORKERS={shlex.quote("1" if cluster.get("sync_model_to_workers") else "0")}

mkdir -p "$(dirname "$JOB_LOCK")"

echo "[model-forge] cluster config: $CLUSTER_CONFIG"
echo "[model-forge] train image: $TRAIN_IMAGE"
./forge cluster doctor --config "$CLUSTER_CONFIG" --strict
./forge cluster health --config "$CLUSTER_CONFIG" --timeout "${{MODEL_FORGE_CLUSTER_HEALTH_TIMEOUT:-30}}"
./forge cluster runtime --config "$CLUSTER_CONFIG" --image "$TRAIN_IMAGE" --timeout "${{MODEL_FORGE_CLUSTER_RUNTIME_TIMEOUT:-120}}"
./forge variants checkpoint-audit "$FAMILY" --variant "$SOURCE_VARIANT" --strict
if [[ "$SYNC_MODEL_TO_WORKERS" == "1" ]]; then
  if [[ -z "$SOURCE_MODEL_DIR" ]]; then
    echo "[model-forge] ERROR: cluster.sync_model_to_workers is enabled but model.local_dir is empty" >&2
    exit 1
  fi
  ./forge cluster model-sync \
    --config "$CLUSTER_CONFIG" \
    --source "$SOURCE_MODEL_DIR" \
    --family "$FAMILY" \
    --variant "$SOURCE_VARIANT" \
    --models-dir "$(dirname "$SOURCE_MODEL_DIR")" \
    --execute \
    --timeout "${{MODEL_FORGE_CLUSTER_MODEL_SYNC_TIMEOUT:-7200}}"
fi
if [[ "${{MODEL_FORGE_SKIP_TORCHRUN_SMOKE:-0}}" != "1" ]]; then
  SMOKE_ARGS=(--config "$CLUSTER_CONFIG" --image "$TRAIN_IMAGE" --timeout "${{MODEL_FORGE_CLUSTER_SMOKE_TIMEOUT:-180}}")
  if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    SMOKE_ARGS+=(--nccl-socket-ifname "$NCCL_SOCKET_IFNAME")
  fi
  ./forge cluster torchrun-smoke "${{SMOKE_ARGS[@]}}"
fi

if [[ "${{MODEL_FORGE_SKIP_PREPARE:-0}}" == "1" && -s "$RUN_DIR/train.jsonl" ]]; then
  echo "[model-forge] skipping data prep; existing train.jsonl found"
else
  "$PYTHON" "$TRAINER" --plan "$PLAN" --prepare-data
fi

"$PYTHON" - "$CLUSTER_CONFIG" "$RUN_DIR" <<'PY'
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {{}}
run_dir = Path(sys.argv[2])
repo = Path.cwd()
try:
    run_dir_relative = run_dir.relative_to(repo)
except ValueError:
    run_dir_relative = None
for node in config.get("nodes", []):
    host = str(node.get("host") or "")
    if host in {{"", "localhost", "127.0.0.1", "::1"}}:
        continue
    user = str(node.get("user") or "")
    remote_work_dir = Path(str(node.get("work_dir") or (config.get("paths") or {{}}).get("work_dir") or repo))
    remote_run_dir = remote_work_dir / run_dir_relative if run_dir_relative is not None else run_dir
    target = f"{{user + '@' if user else ''}}{{host}}:{{remote_run_dir}}/"
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{{user + '@' if user else ''}}{{host}}", shlex.join(["mkdir", "-p", str(remote_run_dir)])], check=True)
    subprocess.run(
        ["rsync", "-az", "--exclude", "tokenized_train_*/", "--exclude", "__pycache__/", str(run_dir).rstrip("/") + "/", target],
        check=True,
    )
PY

"$PYTHON" - "$CLUSTER_CONFIG" "$TRAIN_IMAGE" "$NCCL_SOCKET_IFNAME" "$TRAINER" "$PLAN" "$RUN_DIR" "${{MODEL_FORGE_EXECUTE_CLUSTER_TRAIN:-0}}" <<'PY'
import concurrent.futures
import os
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {{}}
image, iface, trainer, plan, run_dir, execute = sys.argv[2:]
nodes = list(config.get("nodes", []))
distributed = config.get("distributed", {{}})
endpoint = str(distributed.get("rdzv_endpoint") or "")
master_addr, _, master_port = endpoint.rpartition(":")
if not master_addr or not master_port:
    raise SystemExit(f"bad rdzv_endpoint in {{sys.argv[1]}}: {{endpoint!r}}")
nnodes = int(distributed.get("nnodes") or len(nodes))
nproc = int(distributed.get("nproc_per_node") or 1)
repo = Path.cwd()
models_dir = str((config.get("paths") or {{}}).get("models_dir") or os.environ.get("MODEL_FORGE_MODELS_DIR") or (Path.home() / "models"))
cache_dir = str((config.get("paths") or {{}}).get("cache_dir") or os.environ.get("HF_HOME") or (Path.home() / "cache" / "huggingface"))
policy = config.get("resource_policy", {{}})
per_node = policy.get("per_node", {{}}) if isinstance(policy.get("per_node"), dict) else {{}}
cpu_quota = str(per_node.get("cpu_quota") or policy.get("cpu_quota") or "80%")
memory_fraction = float(per_node.get("memory_max_fraction") or policy.get("memory_max_fraction") or 0.85)
io_weight = str(per_node.get("io_weight") or policy.get("io_weight") or 100)
nice = str(per_node.get("nice") or policy.get("nice") or 10)
lock_path = str((config.get("paths") or {{}}).get("job_lock") or "runs/locks/model-forge-cluster.lock")

def target(node):
    host = str(node.get("host") or "")
    user = str(node.get("user") or "")
    if host in {{"", "localhost", "127.0.0.1", "::1"}}:
        return None
    return f"{{user + '@' if user else ''}}{{host}}"

def command_for(rank, node):
    node_work_dir = str(node.get("work_dir") or repo)
    lock = lock_path if Path(lock_path).is_absolute() else str(Path(node_work_dir) / lock_path)
    docker = [
        "docker", "run", "--rm", "--gpus", "all", "--network", "host", "--ipc", "host",
        "--cpus", os.environ.get("MODEL_FORGE_TRAIN_DOCKER_CPUS", "8"),
        "--memory", os.environ.get("MODEL_FORGE_TRAIN_DOCKER_MEMORY", "108g"),
        "--memory-swap", os.environ.get("MODEL_FORGE_TRAIN_DOCKER_MEMORY_SWAP", "108g"),
        "--shm-size", os.environ.get("MODEL_FORGE_TRAIN_DOCKER_SHM", "32g"),
        "--pids-limit", os.environ.get("MODEL_FORGE_TRAIN_DOCKER_PIDS", "4096"),
        "-e", "PYTHONPATH=/workspace/model-forge/src",
        "-e", f"HF_HOME={{cache_dir}}",
        "-e", "MODEL_FORGE_DISABLE_SYSTEMD_SCOPE=1",
        "-e", "TOKENIZERS_PARALLELISM=true",
        "-e", "NCCL_DEBUG=WARN",
        "-e", "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        "-e", "HF_TOKEN",
        "-e", "HUGGINGFACE_HUB_TOKEN",
        "-e", "MODEL_FORGE_TRAIN_TP_SIZE",
        "-e", "MODEL_FORGE_TRAIN_TP_PLAN",
    ]
    if iface:
        docker.extend(["-e", f"NCCL_SOCKET_IFNAME={{iface}}"])
    docker.extend([
        "-v", f"{{node_work_dir}}:/workspace/model-forge",
        "-v", f"{{models_dir}}:{{models_dir}}",
        "-v", f"{{cache_dir}}:{{cache_dir}}",
        "-w", "/workspace/model-forge",
        "--entrypoint", "python3",
        image,
        "-m", "torch.distributed.run",
        f"--nnodes={{nnodes}}",
        f"--nproc-per-node={{nproc}}",
        f"--node-rank={{rank}}",
        f"--master-addr={{master_addr}}",
        f"--master-port={{master_port}}",
        "--max-restarts=0",
        trainer.replace(str(repo), "/workspace/model-forge"),
        "--plan", plan.replace(str(repo), "/workspace/model-forge"),
        "--train",
    ])
    wrapped = (
        f"mkdir -p {{shlex.quote(str(Path(lock).parent))}} && "
        f"flock {{shlex.quote(lock)}} systemd-run --user --scope "
        f"-p CPUQuota={{shlex.quote(cpu_quota)}} -p MemoryMax={{int(memory_fraction * 100)}}% "
        f"-p IOWeight={{shlex.quote(io_weight)}} nice -n {{shlex.quote(nice)}} {{shlex.join(docker)}}"
    )
    return wrapped

commands = [(rank, node, command_for(rank, node)) for rank, node in enumerate(nodes)]
print("[model-forge] docker torchrun launch commands:")
for rank, node, cmd in commands:
    print(f"--- node_rank={{rank}} name={{node.get('name')}} host={{node.get('host')}}")
    print(cmd)

if execute != "1":
    print("[model-forge] dry run: set MODEL_FORGE_EXECUTE_CLUSTER_TRAIN=1 to launch")
    raise SystemExit(0)

def run_one(item):
    rank, node, cmd = item
    remote = target(node)
    if remote:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote, cmd], text=True)
    else:
        proc = subprocess.run(cmd, shell=True, text=True)
    return rank, proc.returncode

with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
    results = list(pool.map(run_one, commands))
failed = [item for item in results if item[1] != 0]
if failed:
    raise SystemExit(f"cluster training failed: {{failed}}")
PY
"""
    outputs["cluster_shell"].write_text(cluster_script)
    outputs["cluster_shell"].chmod(0o755)

    eval_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_DIR))}"]
    eval_lines.extend(plan.get("eval", {}).get("commands", []))
    outputs["eval"].write_text("\n".join(eval_lines) + "\n")
    outputs["eval"].chmod(0o755)
    outputs["method_card"].write_text(render_training_method_card(plan) + "\n")
    return {key: str(path) for key, path in outputs.items()}


def run_artifact(path: str) -> None:
    subprocess.run([path], cwd=REPO_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, prepare, and run model-forge fine-tuning recipes")
    parser.add_argument("--config", required=True, help="Path to fine-tuning YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Show the resolved training plan without writing artifacts")
    prepare = sub.add_parser("prepare", help="Write run artifacts without training")
    prepare.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run", help="Run generated training artifacts; requires --execute")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    plan = build_plan(config, config_path)

    if args.command == "plan":
        render_plan(plan)
    elif args.command == "prepare":
        outputs = write_artifacts(plan, overwrite=args.overwrite)
        console.print(Panel(json.dumps(outputs, indent=2), title="Generated Fine-Tune Artifacts", border_style="green"))
    elif args.command == "run":
        outputs = write_artifacts(plan, overwrite=args.overwrite)
        if not args.execute:
            console.print(Panel(
                "Generated artifacts but did not start training. Re-run with --execute to train.",
                title="Dry Run",
                border_style="yellow",
            ))
            return
        run_artifact(outputs["shell"])


if __name__ == "__main__":
    main()
