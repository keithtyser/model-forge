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


def load_blend_module():
    spec = importlib.util.spec_from_file_location(
        "blend_safetensors_checkpoints",
        REPO_DIR / "scripts" / "blend_safetensors_checkpoints.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load blend_safetensors_checkpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checkpoint(path: Path, values: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    weight_map = {name: "model-00001-of-00001.safetensors" for name in values}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 128}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (path / "config.json").write_text(json.dumps({"model_type": "unit"}), encoding="utf-8")
    save_file(values, path / "model-00001-of-00001.safetensors")


class BlendSafetensorsCheckpointTests(unittest.TestCase):
    def test_blends_matching_floating_tensors_and_copies_metadata(self) -> None:
        module = load_blend_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            target = root / "target"
            output = root / "output"
            write_checkpoint(
                base,
                {
                    "model.layers.0.weight": torch.tensor([[1.0, 3.0]], dtype=torch.float32),
                    "model.layers.0.count": torch.tensor([2], dtype=torch.int64),
                },
            )
            write_checkpoint(
                target,
                {
                    "model.layers.0.weight": torch.tensor([[5.0, 7.0]], dtype=torch.float32),
                    "model.layers.0.count": torch.tensor([9], dtype=torch.int64),
                },
            )

            manifest = module.blend_checkpoints(
                base,
                target,
                output,
                alpha=0.25,
                include_regex=None,
                exclude_regex=None,
                overwrite=False,
                min_ram_fraction=0.0,
                min_disk_fraction=0.0,
                dry_run=False,
            )

            with safe_open(output / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as handle:
                blended = handle.get_tensor("model.layers.0.weight")
                copied = handle.get_tensor("model.layers.0.count")
            copied_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

        self.assertTrue(torch.equal(blended, torch.tensor([[2.0, 4.0]], dtype=torch.float32)))
        self.assertTrue(torch.equal(copied, torch.tensor([2], dtype=torch.int64)))
        self.assertEqual(copied_config["model_type"], "unit")
        self.assertEqual(manifest["tensor_stats"], {"blended": 1, "copied": 1})

    def test_requires_identical_weight_maps(self) -> None:
        module = load_blend_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            target = root / "target"
            output = root / "output"
            write_checkpoint(base, {"a": torch.ones(1)})
            write_checkpoint(target, {"b": torch.ones(1)})

            with self.assertRaises(SystemExit):
                module.blend_checkpoints(
                    base,
                    target,
                    output,
                    alpha=1.0,
                    include_regex=None,
                    exclude_regex=None,
                    overwrite=False,
                    min_ram_fraction=0.0,
                    min_disk_fraction=0.0,
                    dry_run=False,
                )


if __name__ == "__main__":
    unittest.main()
