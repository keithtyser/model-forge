"""Tests for model introspection (describe_model -> ModelSpec) across architectures."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_forge.introspect import describe_model


def _write_model(tmp: str, config: dict, *, tokenizer: dict | None = None, index: dict | None = None) -> str:
    d = Path(tmp)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if tokenizer is not None:
        (d / "tokenizer_config.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    if index is not None:
        (d / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return str(d)


class IntrospectTests(unittest.TestCase):
    def test_llama_style_default_modules(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
                "num_key_value_heads": 8, "vocab_size": 128000, "torch_dtype": "bfloat16",
                "tie_word_embeddings": False,
            }, tokenizer={"chat_template": "{{ stuff }}"})
            spec = describe_model(p)
        self.assertEqual(spec.model_type, "llama")
        self.assertTrue(spec.arch_known)
        self.assertEqual(spec.lora_target_modules,
                         ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
        self.assertEqual(spec.abliterate_target_suffixes,
                         ("self_attn.o_proj.weight", "mlp.down_proj.weight"))
        self.assertTrue(spec.chat_template_present)
        self.assertFalse(spec.is_moe)

    def test_qwen35_recognized(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5",
                                   "num_hidden_layers": 24, "hidden_size": 1024, "num_attention_heads": 16})
            spec = describe_model(p)
        self.assertTrue(spec.arch_known)
        self.assertEqual(spec.abliterate_target_suffixes, ("self_attn.o_proj.weight", "mlp.down_proj.weight"))

    def test_mixtral_is_moe(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
                                   "num_hidden_layers": 32, "hidden_size": 4096,
                                   "num_attention_heads": 32, "num_local_experts": 8})
            spec = describe_model(p)
        self.assertTrue(spec.is_moe and spec.num_experts == 8)
        self.assertEqual(spec.mlp_modules, ("w1", "w2", "w3"))

    def test_phi3_divergent_naming(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {"architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
                                   "num_hidden_layers": 32, "hidden_size": 3072, "num_attention_heads": 32})
            spec = describe_model(p)
        self.assertIn("qkv_proj", spec.lora_target_modules)
        self.assertEqual(spec.abliterate_target_suffixes, ("self_attn.o_proj.weight", "mlp.down_proj.weight"))

    def test_unknown_arch_falls_back_and_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {"architectures": ["WeirdNewForCausalLM"], "model_type": "weirdnew",
                                   "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12})
            spec = describe_model(p)
        self.assertFalse(spec.arch_known)
        self.assertEqual(spec.lora_target_modules,
                         ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))

    def test_param_count_from_index(self) -> None:
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                                   "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
                                   "torch_dtype": "bfloat16"},
                             index={"metadata": {"total_size": 1_600_000_000}})
            spec = describe_model(p)
        self.assertEqual(spec.param_count, 800_000_000)   # bf16: 2 bytes/param

    def test_unified_multimodal_nested_decoder(self) -> None:
        # Qwen3.5-style wrapper: decoder dims live under text_config, not the top level
        with TemporaryDirectory() as tmp:
            p = _write_model(tmp, {
                "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 24,
                                "hidden_size": 1024, "num_attention_heads": 16,
                                "num_key_value_heads": 8, "torch_dtype": "bfloat16"},
            })
            spec = describe_model(p)
        self.assertEqual(spec.num_hidden_layers, 24)
        self.assertEqual(spec.hidden_size, 1024)
        self.assertEqual(spec.num_attention_heads, 16)
        self.assertTrue(spec.arch_known)
        self.assertEqual(spec.abliterate_target_suffixes, ("self_attn.o_proj.weight", "mlp.down_proj.weight"))

    def test_missing_config_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                describe_model(tmp)


if __name__ == "__main__":
    unittest.main()
