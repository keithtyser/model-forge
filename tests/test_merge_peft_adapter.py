from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file


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

    def test_resolves_wrapper_parameter_prefixes(self) -> None:
        module = load_merge_module()
        parameters = {
            "layers.0.weight": object(),
            "model.layers.0.self_attn.o_proj.weight": object(),
            "model.model.language_model.layers.0.self_attn.o_proj.weight": object(),
        }

        self.assertEqual(module.resolve_target_parameter("model.layers.0.weight", parameters), "layers.0.weight")
        self.assertEqual(
            module.resolve_target_parameter("model.language_model.layers.0.self_attn.o_proj.weight", parameters),
            "model.layers.0.self_attn.o_proj.weight",
        )
        wrapper_only_parameters = {
            "model.model.language_model.layers.0.self_attn.o_proj.weight": object(),
        }
        self.assertEqual(
            module.resolve_target_parameter("model.language_model.layers.0.self_attn.o_proj.weight", wrapper_only_parameters),
            "model.model.language_model.layers.0.self_attn.o_proj.weight",
        )
        with self.assertRaises(RuntimeError):
            module.resolve_target_parameter("model.layers.1.weight", parameters)

    def test_direct_merge_applies_lora_scale(self) -> None:
        module = load_merge_module()

        class Guard:
            def check(self, phase: str) -> None:
                return None

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
                self.model.layers[0].weight.data.zero_()

        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 1, "lora_alpha": 1}),
                encoding="utf-8",
            )
            save_file(
                {
                    "base_model.model.model.layers.0.lora_A.weight": torch.tensor([[1.0, 2.0]]),
                    "base_model.model.model.layers.0.lora_B.weight": torch.tensor([[3.0], [4.0]]),
                },
                adapter / "adapter_model.safetensors",
            )

            model = TinyModel()
            stats = module.merge_direct_lora(model, adapter, Guard(), safe_merge=True, lora_scale=0.5)

        self.assertEqual(stats, {"merged_tensors": 1, "skipped_zero_tensors": 0})
        torch.testing.assert_close(
            model.model.layers[0].weight,
            torch.tensor([[1.5, 3.0], [2.0, 4.0]]),
        )

    def test_disk_preflight_blocks_projected_floor_breach(self) -> None:
        module = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            (base / "model.safetensors").write_bytes(b"0" * 80)

            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(total=100, used=20, free=80)):
                with self.assertRaisesRegex(RuntimeError, "disk preflight"):
                    module.check_disk_preflight(base, root / "output", 0.15)

    def test_disk_preflight_accounts_for_existing_output_size(self) -> None:
        module = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            output = root / "output"
            base.mkdir()
            output.mkdir()
            (base / "model.safetensors").write_bytes(b"0" * 80)
            (output / "old.safetensors").write_bytes(b"0" * 70)

            with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(total=100, used=20, free=80)):
                report = module.check_disk_preflight(base, output, 0.15)

        self.assertEqual(report["expected_write_bytes"], 10)
        self.assertEqual(report["projected_free_bytes"], 70)

    def test_optional_unsloth_runtime_import_failure_does_not_block_cpu_merge(self) -> None:
        spec = importlib.util.spec_from_file_location("merge_peft_adapter_unsloth_failure", REPO_DIR / "scripts" / "merge_peft_adapter.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "unsloth":
                raise NotImplementedError("Unsloth cannot find any torch accelerator")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            spec.loader.exec_module(module)

        self.assertIsNone(module.unsloth)
        self.assertIn("NotImplementedError", module.UNSLOTH_IMPORT_ERROR)


if __name__ == "__main__":
    unittest.main()
