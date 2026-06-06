#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


TOKENIZER_FILE_NAMES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def positive_int(raw: str | int) -> int:
    text = str(raw).split(",", 1)[0].strip()
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_device_map(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    if value in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return value
    return {"": value}


def meta_tensor_names(model: Any, limit: int = 20) -> list[str]:
    names = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if getattr(getattr(tensor, "device", None), "type", None) == "meta":
            names.append(name)
            if len(names) >= limit:
                break
    return names


def checkpoint_files(model_dir: Path) -> list[Path]:
    sharded = sorted(model_dir.glob("model-*.safetensors"))
    if sharded:
        return sharded
    single = model_dir / "model.safetensors"
    return [single] if single.exists() else []


def load_text_config(source_dir: Path) -> dict[str, Any]:
    pre_wrapper = source_dir / "config.json.pre-wrapper-fix"
    if pre_wrapper.exists():
        return json.loads(pre_wrapper.read_text(encoding="utf-8"))

    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        restored = copy.deepcopy(text_config)
        restored.setdefault("architectures", ["Qwen3_5ForCausalLM"])
        restored.setdefault("model_type", text_config.get("model_type", "qwen3_5_text"))
        restored.setdefault("transformers_version", config.get("transformers_version"))
        restored["tie_word_embeddings"] = config.get("tie_word_embeddings", restored.get("tie_word_embeddings", False))
        return restored
    return config


def exported_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return copy.deepcopy(text_config)
    return copy.deepcopy(config)


def prefix_wrapper_module_path(path: str) -> str:
    if path.startswith("language_model.") or path.startswith("*language_model."):
        return path
    if path.startswith("*"):
        if path.startswith("*model.") or path.startswith("*lm_head"):
            return "*language_model." + path[1:]
        return path
    if path.startswith(("model.", "lm_head")):
        return "language_model." + path
    return path


WRAPPER_TOWER_EXCLUSIONS = (
    "visual",
    "model.visual",
    "*visual*",
    "vision",
    "model.vision",
    "*vision*",
    "vision_tower",
    "model.vision_tower",
    "*vision_tower*",
    "multi_modal_projector",
    "model.multi_modal_projector",
    "*multi_modal_projector*",
)


def append_unique(items: list[Any], additions: tuple[str, ...]) -> list[Any]:
    existing = {item for item in items if isinstance(item, str)}
    result = list(items)
    for item in additions:
        if item not in existing:
            result.append(item)
            existing.add(item)
    return result


def add_wrapper_tower_exclusions(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    updated = copy.deepcopy(value)
    ignore = updated.get("ignore")
    if isinstance(ignore, list):
        updated["ignore"] = append_unique(ignore, WRAPPER_TOWER_EXCLUSIONS)
    quantization = updated.get("quantization")
    if isinstance(quantization, dict) and isinstance(quantization.get("exclude_modules"), list):
        quantization["exclude_modules"] = append_unique(quantization["exclude_modules"], WRAPPER_TOWER_EXCLUSIONS)
    return updated


def prefix_quantization_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: prefix_quantization_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [prefix_quantization_paths(item) for item in value]
    if isinstance(value, str):
        return prefix_wrapper_module_path(value)
    return value


def wrapper_config_for_vllm(source_dir: Path, export_config: dict[str, Any]) -> dict[str, Any]:
    source_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    text_config = exported_text_config(export_config)
    quant_config = text_config.get("quantization_config") or export_config.get("quantization_config")
    if isinstance(quant_config, dict):
        prefixed_quant_config = add_wrapper_tower_exclusions(prefix_quantization_paths(quant_config))
        text_config["quantization_config"] = copy.deepcopy(prefixed_quant_config)
    else:
        prefixed_quant_config = None

    if isinstance(source_config.get("text_config"), dict):
        wrapper = {
            key: copy.deepcopy(value)
            for key, value in source_config.items()
            if key not in text_config and key != "text_config"
        }
    else:
        wrapper = {}

    wrapper["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    wrapper["model_type"] = source_config.get("model_type", "qwen3_5")
    wrapper["language_model_only"] = True
    wrapper["text_config"] = text_config
    wrapper["tie_word_embeddings"] = source_config.get(
        "tie_word_embeddings",
        text_config.get("tie_word_embeddings", False),
    )
    if prefixed_quant_config is not None:
        wrapper["quantization_config"] = copy.deepcopy(prefixed_quant_config)
    if "vision_config" not in wrapper and isinstance(source_config.get("vision_config"), dict):
        wrapper["vision_config"] = copy.deepcopy(source_config["vision_config"])
    return wrapper


def remap_qwen_text_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "model.", 1)
    if key.startswith("language_model."):
        return key.replace("language_model.", "model.", 1)
    return key


def wrap_qwen_text_export_key(key: str) -> str:
    if key.startswith("language_model."):
        return key
    if key.startswith(("model.", "lm_head.")):
        return "language_model." + key
    return key


def rewrite_hf_quant_config_for_wrapper(output_dir: Path) -> bool:
    path = output_dir / "hf_quant_config.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    rewritten = add_wrapper_tower_exclusions(prefix_quantization_paths(data))
    path.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def copy_metadata_files(source_dir: Path, target_dir: Path) -> None:
    for name in TOKENIZER_FILE_NAMES:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def prepare_text_checkpoint(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    files = checkpoint_files(source_dir)
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint files found under {source_dir}")

    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists() and len(files) != 1:
        raise FileNotFoundError(f"missing safetensors index for sharded checkpoint: {index_path}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / "config.json").write_text(
        json.dumps(load_text_config(source_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copy_metadata_files(source_dir, target_dir)

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        source_weight_map = dict(index.get("weight_map") or {})
        shard_names = sorted(set(source_weight_map.values()))
    else:
        source_weight_map = {}
        shard_names = [files[0].name]

    output_weight_map: dict[str, str] = {}
    renamed = 0
    tensor_count = 0
    total_size = 0
    for shard_name in shard_names:
        source_shard = source_dir / shard_name
        tensors = load_file(str(source_shard))
        remapped = {}
        for key, tensor in tensors.items():
            new_key = remap_qwen_text_key(key)
            if new_key != key:
                renamed += 1
            if new_key in remapped:
                raise RuntimeError(f"duplicate remapped tensor key {new_key!r} from shard {shard_name}")
            remapped[new_key] = tensor
            output_weight_map[new_key] = shard_name
            tensor_count += 1
            total_size += tensor.numel() * tensor.element_size()
        save_file(remapped, str(target_dir / shard_name))

    if source_weight_map and len(source_weight_map) != tensor_count:
        raise RuntimeError(
            f"checkpoint tensor count mismatch after remap: index={len(source_weight_map)} remapped={tensor_count}"
        )

    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": output_weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source_dir),
        "target": str(target_dir),
        "shard_count": len(shard_names),
        "tensor_count": tensor_count,
        "renamed_key_count": renamed,
        "total_size": total_size,
    }


def wrap_text_export_for_vllm(output_dir: Path, source_dir: Path) -> dict[str, Any]:
    files = checkpoint_files(output_dir)
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint files found under {output_dir}")

    config_path = output_dir / "config.json"
    export_config = json.loads(config_path.read_text(encoding="utf-8"))
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        source_weight_map = dict(index.get("weight_map") or {})
        shard_names = sorted(set(source_weight_map.values()))
    else:
        source_weight_map = {}
        shard_names = [path.name for path in files]

    output_weight_map: dict[str, str] = {}
    renamed = 0
    tensor_count = 0
    total_size = 0
    for shard_name in shard_names:
        source_shard = output_dir / shard_name
        tensors = load_file(str(source_shard))
        remapped = {}
        for key, tensor in tensors.items():
            new_key = wrap_qwen_text_export_key(key)
            if new_key != key:
                renamed += 1
            if new_key in remapped:
                raise RuntimeError(f"duplicate wrapper tensor key {new_key!r} from shard {shard_name}")
            remapped[new_key] = tensor
            output_weight_map[new_key] = shard_name
            tensor_count += 1
            total_size += tensor.numel() * tensor.element_size()
        tmp_shard = output_dir / f".{shard_name}.wrapper.tmp"
        save_file(remapped, str(tmp_shard))
        tmp_shard.replace(source_shard)

    if source_weight_map and len(source_weight_map) != tensor_count:
        raise RuntimeError(
            f"checkpoint tensor count mismatch after wrapper remap: index={len(source_weight_map)} wrapped={tensor_count}"
        )

    if source_weight_map:
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    config_path.write_text(
        json.dumps(wrapper_config_for_vllm(source_dir, export_config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quant_config_rewritten = rewrite_hf_quant_config_for_wrapper(output_dir)
    return {
        "wrapper_source": str(source_dir),
        "wrapper_output": str(output_dir),
        "wrapper_shard_count": len(shard_names),
        "wrapper_tensor_count": tensor_count,
        "wrapper_renamed_key_count": renamed,
        "wrapper_total_size": total_size,
        "wrapper_hf_quant_config_rewritten": quant_config_rewritten,
    }


def quant_config(name: str) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import _nvfp4_selective_quant_cfg

    configs = {
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
        "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
        "nvfp4_w4a16": _nvfp4_selective_quant_cfg(["*"], weight_only=True),
    }
    if name not in configs:
        raise ValueError(f"unsupported qformat {name!r}; expected one of {', '.join(sorted(configs))}")
    cfg = copy.deepcopy(configs[name])
    for pattern in (
        "*lm_head*",
        "*embed_tokens*",
        "*router*",
        "*expert*",
        "*vision*",
        "*visual*",
        "*multi_modal_projector*",
        "*linear_attn.conv1d*",
        "*mixer.conv1d*",
    ):
        cfg["quant_cfg"][pattern] = {"enable": False}
    return cfg


def save_tokenizer_and_processor(model_id: str, output_dir: Path, trust_remote_code: bool) -> None:
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_dir)
    except Exception as exc:
        print(f"warning: processor save skipped: {type(exc).__name__}: {exc}", flush=True)
    copy_metadata_files(Path(model_id), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Qwen wrapper checkpoints through a text-only ModelOpt view")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qformat", default="nvfp4", choices=["nvfp4", "nvfp4_awq", "nvfp4_w4a16"])
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--calib-samples", type=positive_int, default=256)
    parser.add_argument("--calib-seq-len", type=positive_int, default=2048)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--keep-text-input", action="store_true")
    parser.add_argument("--wrap-existing-output", action="store_true")
    parser.add_argument("--reject-meta-tensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    source_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_input_dir = output_dir.parent / f".{output_dir.name}.qwen_text_modelopt_input"
    dataset_name = args.dataset.split(",", 1)[0].strip()

    if args.wrap_existing_output:
        wrapper_stats = wrap_text_export_for_vllm(output_dir, source_dir)
        summary_path = output_dir / "model_forge_quantization_summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "model": str(source_dir),
                "output": str(output_dir),
                "duration_seconds": round(time.time() - started_at, 3),
                **wrapper_stats,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    print(f"preparing text-only Qwen checkpoint view at {text_input_dir}", flush=True)
    remap_stats = prepare_text_checkpoint(source_dir, text_input_dir)

    import modelopt.torch.quantization as mtq
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(text_input_dir), trust_remote_code=args.trust_remote_code)
    print("loading text-only model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(text_input_dir),
        dtype=torch.bfloat16,
        device_map=parse_device_map(args.device_map),
        trust_remote_code=args.trust_remote_code,
    )
    if args.reject_meta_tensors:
        meta_names = meta_tensor_names(model)
        if meta_names:
            raise RuntimeError(
                "loaded model contains meta tensors that ModelOpt cannot export; "
                f"examples={meta_names}. Use a non-offloaded device_map such as cuda:0, "
                "reduce calibration memory, or choose a smaller source/model shard policy."
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
    save_tokenizer_and_processor(str(text_input_dir), output_dir, args.trust_remote_code)
    wrapper_stats = wrap_text_export_for_vllm(output_dir, source_dir)

    if not args.keep_text_input:
        shutil.rmtree(text_input_dir, ignore_errors=True)

    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    summary = {
        "model": str(source_dir),
        "output": str(output_dir),
        "qformat": args.qformat,
        "dataset": dataset_name,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "batch_size": args.batch_size,
        "total_size_gb": round(total_bytes / 1_000_000_000, 4),
        "duration_seconds": round(time.time() - started_at, 3),
        "text_input_kept": args.keep_text_input,
        "text_input_dir": str(text_input_dir),
        **remap_stats,
        **wrapper_stats,
    }
    (output_dir / "model_forge_quantization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
