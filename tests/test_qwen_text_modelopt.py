from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


REPO_DIR = Path(__file__).resolve().parents[1]


def load_qwen_text_modelopt_module():
    spec = importlib.util.spec_from_file_location(
        "qwen_text_modelopt",
        REPO_DIR / "scripts" / "quantization" / "qwen_text_modelopt.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load qwen_text_modelopt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QwenTextModelOptTests(unittest.TestCase):
    def test_prepare_text_checkpoint_stream_remaps_wrapper_keys(self) -> None:
        module = load_qwen_text_modelopt_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "language_model_only": True,
                        "model_type": "qwen3_5",
                        "text_config": {
                            "architectures": ["Qwen3_5ForCausalLM"],
                            "hidden_size": 4,
                            "model_type": "qwen3_5_text",
                        },
                        "tie_word_embeddings": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (source / "config.json.pre-wrapper-fix").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForCausalLM"],
                        "hidden_size": 4,
                        "model_type": "qwen3_5_text",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
            save_file(
                {
                    "lm_head.weight": torch.ones(1, 1),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(1, 1) * 2,
                },
                str(source / "model-00001-of-00001.safetensors"),
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "lm_head.weight": "model-00001-of-00001.safetensors",
                            "model.language_model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            stats = module.prepare_text_checkpoint(source, target)
            tensors = load_file(str(target / "model-00001-of-00001.safetensors"))
            index = json.loads((target / "model.safetensors.index.json").read_text(encoding="utf-8"))
            config = json.loads((target / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(stats["renamed_key_count"], 1)
            self.assertIn("lm_head.weight", tensors)
            self.assertIn("model.layers.0.self_attn.o_proj.weight", tensors)
            self.assertNotIn("model.language_model.layers.0.self_attn.o_proj.weight", tensors)
            self.assertIn("model.layers.0.self_attn.o_proj.weight", index["weight_map"])
            self.assertEqual(config["model_type"], "qwen3_5_text")
            self.assertTrue((target / "tokenizer_config.json").exists())

    def test_wrap_text_export_for_vllm_restores_wrapper_shape(self) -> None:
        module = load_qwen_text_modelopt_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "image_token_id": 248056,
                        "language_model_only": True,
                        "model_type": "qwen3_5",
                        "text_config": {
                            "architectures": ["Qwen3_5ForCausalLM"],
                            "hidden_size": 4,
                            "model_type": "qwen3_5_text",
                            "tie_word_embeddings": False,
                        },
                        "tie_word_embeddings": False,
                        "vision_config": {
                            "hidden_size": 2,
                            "model_type": "qwen3_5",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForCausalLM"],
                        "hidden_size": 4,
                        "model_type": "qwen3_5_text",
                        "quantization_config": {
                            "ignore": ["lm_head", "model.layers.0.linear_attn.conv1d"],
                            "quant_method": "modelopt",
                        },
                        "tie_word_embeddings": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (output / "hf_quant_config.json").write_text(
                json.dumps(
                    {
                        "quantization": {
                            "exclude_modules": ["lm_head", "model.layers.0.linear_attn.conv1d"],
                            "quant_algo": "NVFP4",
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            save_file(
                {
                    "lm_head.weight": torch.ones(1, 1),
                    "model.layers.0.self_attn.o_proj.weight": torch.ones(1, 1) * 2,
                },
                str(output / "model.safetensors"),
            )

            stats = module.wrap_text_export_for_vllm(output, source)
            tensors = load_file(str(output / "model.safetensors"))
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            hf_quant_config = json.loads((output / "hf_quant_config.json").read_text(encoding="utf-8"))

            self.assertEqual(stats["wrapper_renamed_key_count"], 2)
            self.assertIn("language_model.lm_head.weight", tensors)
            self.assertIn("language_model.model.layers.0.self_attn.o_proj.weight", tensors)
            self.assertNotIn("lm_head.weight", tensors)
            self.assertEqual(config["architectures"], ["Qwen3_5ForConditionalGeneration"])
            self.assertEqual(config["model_type"], "qwen3_5")
            self.assertTrue(config["language_model_only"])
            self.assertIn("vision_config", config)
            self.assertEqual(config["text_config"]["model_type"], "qwen3_5_text")
            self.assertEqual(
                config["quantization_config"]["ignore"][:2],
                ["language_model.lm_head", "language_model.model.layers.0.linear_attn.conv1d"],
            )
            self.assertIn("model.visual", config["quantization_config"]["ignore"])
            self.assertIn("*vision*", config["quantization_config"]["ignore"])
            self.assertEqual(config["text_config"]["quantization_config"], config["quantization_config"])
            self.assertEqual(
                hf_quant_config["quantization"]["exclude_modules"][:2],
                ["language_model.lm_head", "language_model.model.layers.0.linear_attn.conv1d"],
            )
            self.assertIn("model.visual", hf_quant_config["quantization"]["exclude_modules"])
            self.assertIn("*vision*", hf_quant_config["quantization"]["exclude_modules"])


if __name__ == "__main__":
    unittest.main()
