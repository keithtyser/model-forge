from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_DIR / "scripts" / "convert_obliteratus_lora_to_peft.py"
SPEC = importlib.util.spec_from_file_location("convert_obliteratus_lora_to_peft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ObliteratusLoraConverterTests(unittest.TestCase):
    def test_converts_obliteratus_lora_pairs_to_peft_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "obliteratus"
            output_dir = root / "peft"
            input_dir.mkdir()
            (input_dir / "tokenizer_config.json").write_text('{"tokenizer": true}\n', encoding="utf-8")
            payload = {
                "layer.7.attn.q_proj.lora_A": torch.ones(2, 3),
                "layer.7.attn.q_proj.lora_B": torch.ones(5, 2) * 2,
                "layer.7.ffn.down_proj.lora_A": torch.ones(2, 5) * 3,
                "layer.7.ffn.down_proj.lora_B": torch.ones(4, 2) * 4,
            }
            torch.save(payload, input_dir / "abliteration_lora_adapters.pt")

            manifest = converter.convert_adapters(
                input_dir,
                output_dir,
                base_model_name_or_path="/models/base",
                key_template="base_model.model.model.layers.{layer}.{module}.{weight}",
                attn_module_name="self_attn",
                ffn_module_name="mlp",
            )

            config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["peft_type"], "LORA")
            self.assertEqual(config["r"], 2)
            self.assertEqual(config["lora_alpha"], 2)
            self.assertEqual(config["target_modules"], ["down_proj", "q_proj"])
            self.assertEqual(manifest["adapter_count"], 2)
            self.assertTrue((output_dir / "tokenizer_config.json").is_file())
            with safe_open(output_dir / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                self.assertIn("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight", keys)
                self.assertIn("base_model.model.model.layers.7.self_attn.q_proj.lora_B.weight", keys)
                self.assertIn("base_model.model.model.layers.7.mlp.down_proj.lora_A.weight", keys)
                self.assertEqual(tuple(handle.get_tensor("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight").shape), (2, 3))

    def test_rejects_mixed_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            torch.save(
                {
                    "layer.0.attn.q_proj.lora_A": torch.ones(1, 3),
                    "layer.0.attn.q_proj.lora_B": torch.ones(5, 1),
                    "layer.1.attn.q_proj.lora_A": torch.ones(2, 3),
                    "layer.1.attn.q_proj.lora_B": torch.ones(5, 2),
                },
                input_dir / "abliteration_lora_adapters.pt",
            )

            with self.assertRaises(SystemExit):
                converter.convert_adapters(
                    input_dir,
                    output_dir,
                    base_model_name_or_path="/models/base",
                    key_template="base_model.model.model.layers.{layer}.{module}.{weight}",
                    attn_module_name="self_attn",
                    ffn_module_name="mlp",
                )

    def test_supports_in_place_conversion_without_samefile_metadata_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "adapter"
            output_dir.mkdir()
            (output_dir / "tokenizer_config.json").write_text('{"tokenizer": true}\n', encoding="utf-8")
            torch.save(
                {
                    "layer.0.attn.q_proj.lora_A": torch.ones(2, 3),
                    "layer.0.attn.q_proj.lora_B": torch.ones(5, 2),
                },
                output_dir / "abliteration_lora_adapters.pt",
            )

            manifest = converter.convert_adapters(
                output_dir,
                output_dir,
                base_model_name_or_path="/models/base",
                key_template="base_model.model.model.layers.{layer}.{module}.{weight}",
                attn_module_name="self_attn",
                ffn_module_name="mlp",
            )

            self.assertEqual(manifest["adapter_count"], 1)
            self.assertEqual(manifest["copied_metadata_files"], [])
            self.assertTrue((output_dir / "adapter_model.safetensors").is_file())
            self.assertTrue((output_dir / "adapter_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
