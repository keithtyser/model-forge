from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


REPO_DIR = Path(__file__).resolve().parents[1]


def load_scale_module():
    spec = importlib.util.spec_from_file_location("scale_lora_adapter", REPO_DIR / "scripts" / "scale_lora_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scale_lora_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScaleLoraAdapterTests(unittest.TestCase):
    def test_scales_lora_b_and_copies_metadata(self) -> None:
        module = load_scale_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            save_file(
                {
                    "base_model.model.layers.0.lora_A.weight": torch.tensor([[1.0, 2.0]]),
                    "base_model.model.layers.0.lora_B.weight": torch.tensor([[3.0], [4.0]]),
                },
                source / "adapter_model.safetensors",
            )

            manifest = module.scale_adapter(source, output, 0.25)
            state = load_file(output / "adapter_model.safetensors")
            self.assertTrue((output / "adapter_config.json").is_file())
            self.assertTrue((output / "model_forge_scaled_adapter.json").is_file())

            torch.testing.assert_close(state["base_model.model.layers.0.lora_A.weight"], torch.tensor([[1.0, 2.0]]))
            torch.testing.assert_close(state["base_model.model.layers.0.lora_B.weight"], torch.tensor([[0.75], [1.0]]))
            self.assertEqual(manifest["scale_method"], "multiply_lora_B_weight")
            self.assertEqual(manifest["scaled_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
