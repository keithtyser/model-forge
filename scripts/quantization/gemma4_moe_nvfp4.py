#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from modelopt.torch.utils.dataset_utils import create_forward_loop


class _Gemma4ExpertModule(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class _QuantGemma4TextExperts(QuantModule):
    """Expose Gemma4 fused 3D experts as per-expert Linear modules for ModelOpt."""

    def _setup(self) -> None:
        from accelerate import init_empty_weights

        dtype = self.gate_up_proj.dtype
        device = self.gate_up_proj.device
        expert_dim = self.intermediate_dim

        def copy_weight(module: nn.Linear, weight: torch.Tensor) -> None:
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().to(dtype=dtype, device=device)

        with init_empty_weights():
            experts = nn.ModuleList(
                _Gemma4ExpertModule(self.hidden_dim, expert_dim)
                for _ in range(self.num_experts)
            )

        for index in range(self.num_experts):
            copy_weight(experts[index].gate_proj, self.gate_up_proj[index, :expert_dim, :])
            copy_weight(experts[index].up_proj, self.gate_up_proj[index, expert_dim:, :])
            copy_weight(experts[index].down_proj, self.down_proj[index])

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        for index, expert in enumerate(experts):
            self.add_module(str(index), expert)

    def __len__(self) -> int:
        return self.num_experts

    def __iter__(self):
        for index in range(self.num_experts):
            yield getattr(self, str(index))

    def __getitem__(self, index: int | str) -> nn.Module:
        return getattr(self, str(int(index)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in active_experts:
            expert_idx = int(expert_idx_tensor[0])
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            gated = self.act_fn(expert.gate_proj(current_state)) * expert.up_proj(current_state)
            current_hidden_states = expert.down_proj(gated)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def positive_int(raw: str | int) -> int:
    text = str(raw).split(",", 1)[0].strip()
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def register_gemma4_expert_plugin() -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    if Gemma4TextExperts not in QuantModuleRegistry._registry:
        QuantModuleRegistry.register({Gemma4TextExperts: "hf.Gemma4TextExperts"})(
            _QuantGemma4TextExperts
        )


def quant_config(name: str) -> dict:
    from modelopt.torch.quantization.config import _nvfp4_selective_quant_cfg

    configs = {
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
        "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
        "nvfp4_w4a16": _nvfp4_selective_quant_cfg(["*"], weight_only=True),
    }
    if name not in configs:
        raise ValueError(f"unsupported qformat {name!r}; expected one of {', '.join(sorted(configs))}")
    cfg = copy.deepcopy(configs[name])
    for pattern in ("*vision*", "*embed_vision*", "*multi_modal_projector*", "*lm_head*"):
        cfg["quant_cfg"][pattern] = {"enable": False}
    return cfg


def checkpoint_files(model_dir: Path) -> list[Path]:
    sharded = sorted(model_dir.glob("model-*.safetensors"))
    if sharded:
        return sharded
    single = model_dir / "model.safetensors"
    return [single] if single.exists() else []


def fix_keys_for_vllm(model_dir: Path, max_shard_size_gb: float) -> dict[str, int]:
    files = checkpoint_files(model_dir)
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint files found under {model_dir}")

    tensors = {}
    for path in files:
        tensors.update(load_file(str(path)))

    renamed = 0
    fixed = {}
    for key, tensor in tensors.items():
        new_key = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", key)
        if new_key != key:
            renamed += 1
        fixed[new_key] = tensor

    for path in files:
        path.unlink()
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    max_shard_bytes = int(max_shard_size_gb * 1_000_000_000)
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for key, tensor in sorted(fixed.items(), key=lambda item: item[1].numel() * item[1].element_size(), reverse=True):
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += tensor_bytes
    if current:
        shards.append(current)

    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in fixed.values())},
        "weight_map": {},
    }
    for shard_index, shard in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(model_dir / filename))
        for key in shard:
            index["weight_map"][key] = filename
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"tensor_count": len(fixed), "renamed_key_count": renamed, "shard_count": len(shards)}


def save_tokenizer_and_processor(model_id: str, output_dir: Path, trust_remote_code: bool) -> None:
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_dir)
    except Exception as exc:
        print(f"warning: processor save skipped: {type(exc).__name__}: {exc}", flush=True)

    source_dir = Path(model_id)
    if source_dir.exists():
        for name in (
            "processor_config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "chat_template.json",
            "generation_config.json",
        ):
            source = source_dir / name
            target = output_dir / name
            if source.exists() and not target.exists():
                shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Gemma4 fused MoE experts with ModelOpt NVFP4")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qformat", default="nvfp4", choices=["nvfp4", "nvfp4_awq", "nvfp4_w4a16"])
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--calib-samples", type=positive_int, default=4096)
    parser.add_argument("--calib-seq-len", type=positive_int, default=1024)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size-gb", type=float, default=8.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset.split(",", 1)[0].strip()

    register_gemma4_expert_plugin()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"building calibration loop dataset={dataset_name} samples={args.calib_samples} seq_len={args.calib_seq_len} batch={args.batch_size}",
        flush=True,
    )
    forward_loop = create_forward_loop(
        model=model,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_samples,
        max_sample_length=args.calib_seq_len,
        device=args.device,
    )

    print(f"quantizing qformat={args.qformat}", flush=True)
    model = mtq.quantize(model, quant_config(args.qformat), forward_loop=forward_loop)

    print(f"exporting to {output_dir}", flush=True)
    from modelopt.torch.export import export_hf_checkpoint

    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=str(output_dir))
    save_tokenizer_and_processor(args.model, output_dir, args.trust_remote_code)

    print("fixing vLLM expert key names", flush=True)
    key_stats = fix_keys_for_vllm(output_dir, args.max_shard_size_gb)
    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    summary = {
        "model": args.model,
        "output": str(output_dir),
        "qformat": args.qformat,
        "dataset": dataset_name,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "batch_size": args.batch_size,
        "total_size_gb": round(total_bytes / 1_000_000_000, 4),
        "duration_seconds": round(time.time() - started_at, 3),
        **key_stats,
    }
    (output_dir / "model_forge_quantization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
