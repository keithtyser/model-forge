#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import threading
import time
from pathlib import Path

import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def available_fraction() -> float:
    mem = psutil.virtual_memory()
    return mem.available / mem.total


class ResourceGuard:
    def __init__(self, min_available_ram_fraction: float, interval_seconds: float) -> None:
        self.min_available_ram_fraction = min_available_ram_fraction
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self) -> None:
        self.check("preflight")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def check(self, phase: str) -> None:
        fraction = available_fraction()
        if fraction < self.min_available_ram_fraction:
            raise RuntimeError(
                f"{phase}: available RAM fraction {fraction:.3f} is below "
                f"floor {self.min_available_ram_fraction:.3f}"
            )

    def _monitor(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            fraction = available_fraction()
            if fraction < self.min_available_ram_fraction:
                print(
                    "[model-forge] ERROR: RAM floor breached during merge: "
                    f"{fraction:.3f} < {self.min_available_ram_fraction:.3f}",
                    flush=True,
                )
                os._exit(137)


def configure_cpu_threads() -> int:
    usable_cores = max(1, (os.cpu_count() or 1) - 1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(usable_cores))
    torch.set_num_threads(usable_cores)
    torch.set_num_interop_threads(max(1, min(4, usable_cores // 2 or 1)))
    return usable_cores


def copy_tokenizer_files(adapter_dir: Path, output_dir: Path) -> None:
    for name in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "processor_config.json",
    ):
        source = adapter_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def write_merge_manifest(args: argparse.Namespace, output_dir: Path, started_at: float, finished_at: float) -> None:
    manifest = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "max_shard_size": args.max_shard_size,
        "min_available_ram_fraction": args.min_available_ram_fraction,
        "started_unix": started_at,
        "finished_unix": finished_at,
        "duration_seconds": round(finished_at - started_at, 3),
    }
    (output_dir / "model_forge_merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a full model checkpoint")
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--min-available-ram-fraction", type=float, default=float(os.getenv("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", "0.05")))
    parser.add_argument("--monitor-interval-seconds", type=float, default=15.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not base_model.is_dir():
        raise SystemExit(f"base model path does not exist: {base_model}")
    if not adapter.is_dir():
        raise SystemExit(f"adapter path does not exist: {adapter}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    usable_cores = configure_cpu_threads()
    guard = ResourceGuard(args.min_available_ram_fraction, args.monitor_interval_seconds)
    started_at = time.time()
    print(f"[model-forge] base: {base_model}", flush=True)
    print(f"[model-forge] adapter: {adapter}", flush=True)
    print(f"[model-forge] output: {output_dir}", flush=True)
    print(f"[model-forge] dtype: {args.dtype}", flush=True)
    print(f"[model-forge] usable CPU threads: {usable_cores}", flush=True)
    print(f"[model-forge] RAM floor: {args.min_available_ram_fraction:.3f}", flush=True)

    guard.start()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=DTYPES[args.dtype],
            device_map="cpu",
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            use_safetensors=True,
            trust_remote_code=args.trust_remote_code,
        )
        guard.check("after base load")
        peft_model = PeftModel.from_pretrained(
            model,
            adapter,
            device_map="cpu",
            is_trainable=False,
        )
        guard.check("after adapter load")
        merged_model = peft_model.merge_and_unload(safe_merge=True)
        guard.check("after merge")
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        guard.check("after model save")

        tokenizer = AutoTokenizer.from_pretrained(adapter if (adapter / "tokenizer_config.json").exists() else base_model)
        tokenizer.save_pretrained(output_dir)
        try:
            processor = AutoProcessor.from_pretrained(base_model)
            processor.save_pretrained(output_dir)
        except Exception as exc:
            print(f"[model-forge] processor save skipped: {exc}", flush=True)
        copy_tokenizer_files(adapter, output_dir)
        finished_at = time.time()
        write_merge_manifest(args, output_dir, started_at, finished_at)
        print(f"[model-forge] merge complete in {finished_at - started_at:.1f}s", flush=True)
    finally:
        guard.stop()
        gc.collect()


if __name__ == "__main__":
    main()
