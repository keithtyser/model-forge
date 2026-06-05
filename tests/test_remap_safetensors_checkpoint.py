from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_DIR = Path(__file__).resolve().parents[1]


def load_remap_module():
    spec = importlib.util.spec_from_file_location(
        "remap_safetensors_checkpoint",
        REPO_DIR / "scripts" / "remap_safetensors_checkpoint.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load remap_safetensors_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemapSafetensorsCheckpointTests(unittest.TestCase):
    def test_remaps_prefixes_and_preserves_reference_files(self) -> None:
        module = load_remap_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            checkpoint = root / "checkpoint"
            reference.mkdir()
            checkpoint.mkdir()
            reference_map = {
                "lm_head.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
            }
            checkpoint_map = {
                "lm_head.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
            }
            (reference / "model.safetensors.index.json").write_text(json.dumps({"weight_map": reference_map}), encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": checkpoint_map}), encoding="utf-8")
            (reference / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
            (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}), encoding="utf-8")
            save_file(
                {
                    "lm_head.weight": torch.ones(1, 1),
                    "model.layers.0.mlp.down_proj.weight": torch.full((1, 1), 2.0),
                },
                checkpoint / "model-00001-of-00001.safetensors",
            )

            target_map = module.remapped_weight_map(checkpoint_map, [("model.", "model.language_model.")])
            module.verify_against_reference(reference, target_map)
            module.rewrite_shard(
                checkpoint,
                "model-00001-of-00001.safetensors",
                list(checkpoint_map),
                {name: module.remap_name(name, [("model.", "model.language_model.")]) for name in checkpoint_map},
                0.0,
                0.0,
            )
            copied = module.copy_preserved_files(reference, checkpoint, ("config.json",))

            with safe_open(checkpoint / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                tensor = handle.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")
            config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(keys, set(reference_map))
        self.assertTrue(torch.equal(tensor, torch.full((1, 1), 2.0)))
        self.assertEqual(copied, ["config.json"])
        self.assertEqual(config["model_type"], "qwen3_5")


if __name__ == "__main__":
    unittest.main()
