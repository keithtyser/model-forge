#!/usr/bin/env python3
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
