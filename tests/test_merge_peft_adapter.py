from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_peft_adapter", REPO_DIR / "scripts" / "merge_peft_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load merge_peft_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MergePeftAdapterTests(unittest.TestCase):
    def test_restores_wrapper_config_for_language_model_only_qwen_checkpoint(self) -> None:
        module = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            output = root / "output"
            base.mkdir()
            output.mkdir()
            (base / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "language_model_only": False,
                        "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64},
                        "vision_config": {"model_type": "qwen3_5_vision"},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_text",
                        "architectures": ["Qwen3_5ForCausalLM"],
                        "num_hidden_layers": 64,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (output / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 1},
                        "weight_map": {
                            "lm_head.weight": "model-00001-of-00001.safetensors",
                            "model.language_model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            restored = module.restore_base_wrapper_config_if_needed(base, output)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))

        self.assertTrue(restored)
        self.assertEqual(config["model_type"], "qwen3_5")
        self.assertEqual(config["architectures"], ["Qwen3_5ForConditionalGeneration"])
        self.assertEqual(config["text_config"]["model_type"], "qwen3_5_text")
        self.assertEqual(config["vision_config"]["model_type"], "qwen3_5_vision")
        self.assertTrue(config["language_model_only"])

    def test_leaves_plain_text_checkpoints_unchanged(self) -> None:
        module = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            output = root / "output"
            base.mkdir()
            output.mkdir()
            plain_config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 2,
            }
            (base / "config.json").write_text(json.dumps(plain_config, sort_keys=True), encoding="utf-8")
            (output / "config.json").write_text(json.dumps(plain_config, sort_keys=True), encoding="utf-8")
            (output / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0.self_attn.o_proj.weight": "model.safetensors"}}, sort_keys=True),
                encoding="utf-8",
            )

            restored = module.restore_base_wrapper_config_if_needed(base, output)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))

        self.assertFalse(restored)
        self.assertEqual(config, plain_config)


if __name__ == "__main__":
    unittest.main()
