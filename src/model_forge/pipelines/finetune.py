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
    lora = config.get("lora", {})
    eval_cfg = config.get("eval", {})
    resource_policy = config.get("resource_policy", {})
    sources = data_manifest.get("sources", [])
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
            "method": trainer.get("method", "qlora"),
            "device_map": trainer.get("device_map", "auto"),
            "load_in_4bit": bool(trainer.get("load_in_4bit", True)),
            "bf16": bool(trainer.get("bf16", True)),
            "gradient_checkpointing": trainer.get("gradient_checkpointing", True),
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
            "save_steps": int(trainer.get("save_steps", 100)),
            "save_total_limit": int(trainer.get("save_total_limit", 2)),
            "seed": int(trainer.get("seed", 3407)),
            "report_to": trainer.get("report_to", "none"),
        },
        "lora": {
            "r": int(lora.get("r", 64)),
            "alpha": int(lora.get("alpha", lora.get("r", 64))),
            "dropout": float(lora.get("dropout", 0.0)),
            "target_modules": list(lora.get("target_modules", [])),
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
            "min_memory_available_start": float(resource_policy.get("min_memory_available_start", 0.15)),
            "min_memory_available_runtime": float(resource_policy.get("min_memory_available_runtime", 0.10)),
            "min_disk_free": float(resource_policy.get("min_disk_free", 0.15)),
            "monitor_interval_seconds": int(resource_policy.get("monitor_interval_seconds", 30)),
            "dataloader_num_workers_max_offset": int(resource_policy.get("dataloader_num_workers_max_offset", 2)),
            "persistent_workers_when_memory_tight": bool(resource_policy.get("persistent_workers_when_memory_tight", False)),
            "checkpoint_on_memory_pressure": bool(resource_policy.get("checkpoint_on_memory_pressure", True)),
        },
        "baseline": config.get("baseline", {}),
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
        min_mem = float(self.policy.get("min_memory_available_start", 0.15))
        min_disk = float(self.policy.get("min_disk_free", 0.15))
        mem_ratio = self.memory_available_ratio()
        disk_ratio = self.disk_free_ratio()
        if mem_ratio < min_mem:
            raise RuntimeError(f"Not enough free memory to start job: {mem_ratio:.1%} available < {min_mem:.1%}")
        if disk_ratio < min_disk:
            raise RuntimeError(f"Not enough free disk to start job: {disk_ratio:.1%} free < {min_disk:.1%}")

    def check_runtime(self) -> None:
        min_mem = float(self.policy.get("min_memory_available_runtime", 0.10))
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


def conversation_hash(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def tokenizer_source(plan: dict[str, Any]) -> str:
    source = plan["model"].get("local_dir") or plan["model"]["source"]
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
        chunks = []
        source_stats = []
        seen: set[str] = set()

        for source in plan["data"]["sources"]:
            if source.get("enabled", True) is False:
                continue
            split = source.get("split", "train")
            if limit is not None and not source.get("path"):
                if source.get("subset"):
                    ds_iter = load_dataset(source["dataset"], source["subset"], split=split, streaming=True)
                else:
                    ds_iter = load_dataset(source["dataset"], split=split, streaming=True)
                ds = list(ds_iter.take(limit))
                target = len(ds)
            elif source.get("path"):
                ds = load_dataset("json", data_files=source["path"], split="train")
                target = int(source.get("target_samples", 0) or len(ds))
                target = min(target, len(ds))
                if limit is not None:
                    target = min(target, limit)
            else:
                if source.get("subset"):
                    ds = load_dataset(source["dataset"], source["subset"], split=split)
                else:
                    ds = load_dataset(source["dataset"], split=split)
                target = int(source.get("target_samples", 0) or len(ds))
                target = min(target, len(ds))
            if target <= 0:
                continue
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
                digest = conversation_hash(messages)
                if digest in seen:
                    rejected += 1
                    continue
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                if token_count > max_context:
                    rejected += 1
                    continue
                seen.add(digest)
                rows.append({"id": digest, "source": source.get("name", source.get("dataset", "")), "messages": messages, "text": text, "token_count": token_count})
            if rows:
                chunks.append(Dataset.from_list(rows))
            source_stats.append({"name": source.get("name", source.get("dataset", "")), "sampled": target, "accepted": len(rows), "rejected": rejected})

        if not chunks:
            raise SystemExit("no training rows survived data preparation")
        dataset = concatenate_datasets(chunks).shuffle(seed=int(plan["trainer"]["seed"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
        return {"output_path": str(output_path), "rows": len(dataset), "sources": source_stats}
    finally:
        guard.stop_monitor()


def train(plan: dict[str, Any], dataset_path: Path) -> None:
    import inspect
    import torch
    from datasets import load_dataset, load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

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
    tokenized_path = Path(plan["run_dir"]) / "tokenized_train"
    if tokenized_path.exists():
        dataset = load_from_disk(str(tokenized_path))
    else:
        raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        def tokenize_batch(batch):
            tokenized = tokenizer(
                batch["text"],
                truncation=True,
                max_length=int(plan["model"]["max_seq_length"]),
                padding=False,
            )
            tokenized["mm_token_type_ids"] = [[0] * len(input_ids) for input_ids in tokenized["input_ids"]]
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

    model_id = plan["model"].get("local_dir") or plan["model"]["source"]
    quantization_config = None
    if plan["trainer"].get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if plan["trainer"].get("bf16", True) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    device_map = {"": 0} if plan["trainer"].get("device_map") == "single_gpu" else plan["trainer"].get("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=plan["model"].get("trust_remote_code", False),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if plan["trainer"].get("bf16", True) else torch.float16,
        device_map=device_map,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    lora = plan["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
        modules_to_save=list(lora.get("modules_to_save", [])) or None,
    )
    model = get_peft_model(model, peft_config)

    max_steps = plan["trainer"].get("max_steps") or -1
    if max_steps and int(max_steps) > 0:
        warmup_steps = max(0, int(math.ceil(int(max_steps) * float(plan["trainer"]["warmup_ratio"]))))
    else:
        effective_batch = int(plan["trainer"]["per_device_train_batch_size"]) * int(plan["trainer"]["gradient_accumulation_steps"])
        steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, effective_batch)))
        warmup_steps = max(0, int(math.ceil(steps_per_epoch * float(plan["trainer"]["num_train_epochs"]) * float(plan["trainer"]["warmup_ratio"]))))

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
        save_steps=int(plan["trainer"]["save_steps"]),
        save_total_limit=int(plan["trainer"]["save_total_limit"]),
        bf16=bool(plan["trainer"]["bf16"]),
        gradient_checkpointing=bool(plan["trainer"]["gradient_checkpointing"]),
        seed=int(plan["trainer"]["seed"]),
        report_to=plan["trainer"]["report_to"],
        save_strategy="steps",
        remove_unused_columns=False,
    )
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

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.add_callback(ResourceGuardCallback())
    try:
        trainer.train()
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
    plan = json.loads(Path(args.plan).read_text())
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
        "eval": run_dir / "eval_after_training.sh",
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

    eval_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_DIR))}"]
    eval_lines.extend(plan.get("eval", {}).get("commands", []))
    outputs["eval"].write_text("\n".join(eval_lines) + "\n")
    outputs["eval"].chmod(0o755)
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
