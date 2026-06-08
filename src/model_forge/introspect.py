"""Model introspection: read a checkpoint's config and derive a normalized ModelSpec.

This is the *model* ground truth (the hardware ground truth lives in ``hardware.py``). Reading
``config.json`` + ``tokenizer_config.json`` we derive the architecture, depth, module-naming
conventions, MoE-ness, chat template, and a parameter estimate. Downstream this drives
arch-correct family/eval registration, abliteration target modules, and LoRA targets -- so the
harness can post-train an arbitrary model instead of only qwen-family ones.

Pure stdlib (no torch / transformers), so it is cheap and runs anywhere.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Most modern decoder LLMs (Llama / Qwen / Mistral / Gemma / StableLM ...) share this naming.
_DEFAULT = {
    "attn_prefix": "self_attn",
    "mlp_prefix": "mlp",
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn_back": "o_proj",      # attention output projection (abliteration write-back)
    "mlp_back": "down_proj",    # MLP output projection (abliteration write-back)
    "lora": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}

# Overrides for architectures whose module names diverge from the default.
_ARCH_OVERRIDES: dict[str, dict[str, Any]] = {
    "phi3": {
        "attn": ("qkv_proj", "o_proj"), "mlp": ("gate_up_proj", "down_proj"),
        "attn_back": "o_proj", "mlp_back": "down_proj",
        "lora": ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
    },
    "gpt2": {
        "attn_prefix": "attn", "mlp_prefix": "mlp",
        "attn": ("c_attn", "c_proj"), "mlp": ("c_fc", "c_proj"),
        "attn_back": "c_proj", "mlp_back": "c_proj", "lora": ("c_attn",),
    },
    "gpt_neox": {
        "attn_prefix": "attention", "mlp_prefix": "mlp",
        "attn": ("query_key_value", "dense"), "mlp": ("dense_h_to_4h", "dense_4h_to_h"),
        "attn_back": "dense", "mlp_back": "dense_4h_to_h", "lora": ("query_key_value", "dense"),
    },
    # MoE experts use w1/w2/w3; experts are left untouched by abliteration, so mlp_back stays the
    # router-free dense path name where present, but we flag is_moe so callers keep experts in bf16.
    "mixtral": {"mlp": ("w1", "w2", "w3"), "mlp_back": "w2"},
}

# model_type values we recognize as standard decoder LLMs using the default naming.
_KNOWN_DEFAULT = {
    "llama", "qwen2", "qwen3", "qwen3_5", "qwen2_moe", "qwen3_moe", "mistral", "mixtral",
    "gemma", "gemma2", "gemma3", "stablelm", "starcoder2", "cohere", "olmo", "phi",
    "deepseek", "deepseek_v2", "deepseek_v3", "minicpm", "internlm2", "yi", "exaone",
}


@dataclass
class ModelSpec:
    path: str
    architecture: str = ""
    model_type: str = ""
    num_hidden_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 0
    is_moe: bool = False
    num_experts: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = ""
    chat_template_present: bool = False
    param_count: Optional[int] = None         # estimated from the safetensors index when present
    attn_prefix: str = "self_attn"
    mlp_prefix: str = "mlp"
    attn_modules: tuple[str, ...] = field(default_factory=tuple)
    mlp_modules: tuple[str, ...] = field(default_factory=tuple)
    lora_target_modules: tuple[str, ...] = field(default_factory=tuple)
    # weight-name suffixes abliteration projects (the output projections), e.g.
    # ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    abliterate_target_suffixes: tuple[str, ...] = field(default_factory=tuple)
    arch_known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_moe(cfg: dict[str, Any]) -> tuple[bool, Optional[int]]:
    experts = cfg.get("num_local_experts") or cfg.get("num_experts")
    mt = str(cfg.get("model_type", "")).lower()
    archs = " ".join(cfg.get("architectures") or []).lower()
    moe = bool(experts) or "moe" in mt or "moe" in archs or "mixtral" in mt
    return moe, (int(experts) if isinstance(experts, int) else None)


def _param_count_from_index(model_dir: Path, torch_dtype: str) -> Optional[int]:
    """Estimate parameters from the safetensors index's total_size (bytes), if present."""
    idx = _read_json(model_dir / "model.safetensors.index.json")
    total_bytes = (idx.get("metadata") or {}).get("total_size")
    if not isinstance(total_bytes, (int, float)) or total_bytes <= 0:
        return None
    bytes_per = 4 if "32" in torch_dtype else (1 if "8" in torch_dtype else 2)
    return int(total_bytes / bytes_per)


def _modules_for(model_type: str) -> dict[str, Any]:
    conv = dict(_DEFAULT)
    conv.update(_ARCH_OVERRIDES.get(model_type, {}))
    return conv


# Unified / multimodal checkpoints nest the decoder dims under a sub-config; the top level only
# describes the wrapper. Descend into the first sub-config that carries the layer count.
_TEXT_CONFIG_KEYS = ("text_config", "thinker_config", "language_model", "llm_config", "decoder")


def _decoder_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("num_hidden_layers") or cfg.get("n_layer"):
        return cfg
    for key in _TEXT_CONFIG_KEYS:
        sub = cfg.get(key)
        if isinstance(sub, dict) and (sub.get("num_hidden_layers") or sub.get("n_layer")):
            return sub
    return cfg


def _norm_model_type(mt: str) -> str:
    """Strip multimodal wrapper suffixes so a nested decoder type maps to its base naming."""
    for suffix in ("_text", "_thinker", "_decoder"):
        if mt.endswith(suffix):
            return mt[: -len(suffix)]
    return mt


def describe_model(path: str | Path) -> ModelSpec:
    """Read a local checkpoint dir (config.json + tokenizer_config.json) into a ModelSpec.

    Unknown architectures fall back to the default decoder-LLM naming with ``arch_known=False``
    so callers can flag it rather than silently target the wrong modules."""
    model_dir = Path(path)
    cfg = _read_json(model_dir / "config.json")
    if not cfg:
        raise FileNotFoundError(f"no readable config.json under {model_dir}")
    tok = _read_json(model_dir / "tokenizer_config.json")
    dec = _decoder_config(cfg)   # the decoder sub-config for multimodal wrappers (else cfg)

    archs = cfg.get("architectures") or []
    architecture = archs[0] if archs else ""
    wrapper_mt = str(cfg.get("model_type", "")).lower()
    dec_mt = _norm_model_type(str(dec.get("model_type", "")).lower())
    # use the decoder's type for module naming when present, else the wrapper's
    model_type = dec_mt or wrapper_mt
    is_moe, num_experts = _is_moe(dec) if dec is not cfg else _is_moe(cfg)
    if not is_moe:
        is_moe, num_experts = _is_moe(cfg)
    conv = _modules_for(model_type)
    arch_known = model_type in _KNOWN_DEFAULT or model_type in _ARCH_OVERRIDES \
        or wrapper_mt in _KNOWN_DEFAULT

    chat_template_present = bool(tok.get("chat_template")) or (model_dir / "chat_template.jinja").exists()
    torch_dtype = str(cfg.get("torch_dtype") or dec.get("torch_dtype") or "")

    suffixes = (
        f"{conv['attn_prefix']}.{conv['attn_back']}.weight",
        f"{conv['mlp_prefix']}.{conv['mlp_back']}.weight",
    )
    return ModelSpec(
        path=str(model_dir),
        architecture=architecture,
        model_type=model_type,
        num_hidden_layers=int(dec.get("num_hidden_layers") or dec.get("n_layer") or 0),
        hidden_size=int(dec.get("hidden_size") or dec.get("n_embd") or 0),
        num_attention_heads=int(dec.get("num_attention_heads") or dec.get("n_head") or 0),
        num_key_value_heads=dec.get("num_key_value_heads"),
        vocab_size=int(dec.get("vocab_size") or cfg.get("vocab_size") or 0),
        is_moe=is_moe,
        num_experts=num_experts,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", dec.get("tie_word_embeddings", False))),
        torch_dtype=torch_dtype,
        chat_template_present=chat_template_present,
        param_count=_param_count_from_index(model_dir, torch_dtype),
        attn_prefix=conv["attn_prefix"],
        mlp_prefix=conv["mlp_prefix"],
        attn_modules=tuple(conv["attn"]),
        mlp_modules=tuple(conv["mlp"]),
        lora_target_modules=tuple(conv["lora"]),
        abliterate_target_suffixes=suffixes,
        arch_known=arch_known,
    )


def _format(spec: ModelSpec) -> str:
    d = spec.to_dict()
    rows = [
        ("architecture", d["architecture"]),
        ("model_type", d["model_type"] + ("" if d["arch_known"] else "  (UNKNOWN — verify modules)")),
        ("layers", d["num_hidden_layers"]),
        ("hidden_size", d["hidden_size"]),
        ("heads / kv", f"{d['num_attention_heads']} / {d['num_key_value_heads']}"),
        ("MoE", f"yes ({d['num_experts']} experts)" if d["is_moe"] else "no"),
        ("params (est)", f"{d['param_count']:,}" if d["param_count"] else "unknown"),
        ("dtype", d["torch_dtype"] or "unknown"),
        ("chat template", "yes" if d["chat_template_present"] else "no"),
        ("lora targets", ", ".join(d["lora_target_modules"])),
        ("abliterate suffixes", ", ".join(d["abliterate_target_suffixes"])),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(width)} : {v}" for k, v in rows)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="forge model", description="Introspect a model checkpoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="print a normalized ModelSpec for a checkpoint dir")
    d.add_argument("path", help="path to the model directory (containing config.json)")
    d.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        spec = describe_model(args.path)
    except FileNotFoundError as e:
        print(str(e))
        return 1
    if args.json:
        print(json.dumps(spec.to_dict(), default=list))
    else:
        print(f"ModelSpec for {spec.path}")
        print(_format(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
