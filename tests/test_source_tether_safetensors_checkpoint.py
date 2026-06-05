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


def load_source_tether_module():
    spec = importlib.util.spec_from_file_location(
        "source_tether_safetensors_checkpoint",
        REPO_DIR / "scripts" / "source_tether_safetensors_checkpoint.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load source_tether_safetensors_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checkpoint(path: Path, values: dict[str, torch.Tensor], model_type: str) -> None:
    path.mkdir()
    weight_map = {name: "model-00001-of-00001.safetensors" for name in values}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 128}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    save_file(values, path / "model-00001-of-00001.safetensors")


class SourceTetherSafetensorsCheckpointTests(unittest.TestCase):
    def test_tethers_selected_tensors_and_resets_highest_drift(self) -> None:
        module = load_source_tether_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            output = root / "output"
            write_checkpoint(
                source,
                {
                    "model.layers.0.weight": torch.tensor([0.0, 2.0], dtype=torch.float32),
                    "model.layers.1.weight": torch.tensor([10.0, 10.0], dtype=torch.float32),
                    "model.embed_tokens.weight": torch.tensor([100.0], dtype=torch.float32),
                    "model.layers.0.count": torch.tensor([3], dtype=torch.int64),
                },
                "source",
            )
            write_checkpoint(
                candidate,
                {
                    "model.layers.0.weight": torch.tensor([10.0, 12.0], dtype=torch.float32),
                    "model.layers.1.weight": torch.tensor([12.0, 12.0], dtype=torch.float32),
                    "model.embed_tokens.weight": torch.tensor([120.0], dtype=torch.float32),
                    "model.layers.0.count": torch.tensor([9], dtype=torch.int64),
                },
                "candidate",
            )

            manifest = module.source_tether_checkpoint(
                source,
                candidate,
                output,
                alpha=0.5,
                restore_top_k=1,
                drift_metric="mean_abs_delta",
                include_regex=r"layers.*weight",
                exclude_regex=None,
                preserve_from="source",
                overwrite=False,
                in_place=False,
                min_ram_fraction=0.0,
                min_disk_fraction=0.0,
                dry_run=False,
            )

            with safe_open(output / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as handle:
                reset = handle.get_tensor("model.layers.0.weight")
                tethered = handle.get_tensor("model.layers.1.weight")
                excluded = handle.get_tensor("model.embed_tokens.weight")
                non_float = handle.get_tensor("model.layers.0.count")
            copied_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

        self.assertTrue(torch.equal(reset, torch.tensor([0.0, 2.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(tethered, torch.tensor([11.0, 11.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(excluded, torch.tensor([120.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(non_float, torch.tensor([9], dtype=torch.int64)))
        self.assertEqual(copied_config["model_type"], "source")
        self.assertEqual(manifest["tensor_stats"], {"tethered": 1, "reset_to_source": 1, "copied_candidate": 2})
        self.assertEqual(manifest["reset_tensor_count"], 1)
        self.assertEqual(manifest["reset_tensors"][0]["name"], "model.layers.0.weight")

    def test_dry_run_ranks_resets_without_writing_output(self) -> None:
        module = load_source_tether_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            output = root / "output"
            write_checkpoint(
                source,
                {
                    "a": torch.tensor([0.0], dtype=torch.float32),
                    "b": torch.tensor([0.0], dtype=torch.float32),
                },
                "source",
            )
            write_checkpoint(
                candidate,
                {
                    "a": torch.tensor([1.0], dtype=torch.float32),
                    "b": torch.tensor([5.0], dtype=torch.float32),
                },
                "candidate",
            )

            manifest = module.source_tether_checkpoint(
                source,
                candidate,
                output,
                alpha=0.895,
                restore_top_k=1,
                drift_metric="mean_abs_delta",
                include_regex=None,
                exclude_regex=None,
                preserve_from="source",
                overwrite=False,
                in_place=False,
                min_ram_fraction=0.0,
                min_disk_fraction=0.0,
                dry_run=True,
            )

        self.assertFalse(output.exists())
        self.assertTrue(manifest["dry_run"])
        self.assertEqual(manifest["reset_tensor_count"], 1)
        self.assertEqual(manifest["reset_tensors"][0]["name"], "b")

    def test_requires_matching_weight_maps(self) -> None:
        module = load_source_tether_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            output = root / "output"
            write_checkpoint(source, {"a": torch.ones(1)}, "source")
            write_checkpoint(candidate, {"b": torch.ones(1)}, "candidate")

            with self.assertRaises(SystemExit):
                module.source_tether_checkpoint(
                    source,
                    candidate,
                    output,
                    alpha=0.895,
                    restore_top_k=0,
                    drift_metric="mean_abs_delta",
                    include_regex=None,
                    exclude_regex=None,
                    preserve_from="source",
                    overwrite=False,
                    in_place=False,
                    min_ram_fraction=0.0,
                    min_disk_fraction=0.0,
                    dry_run=False,
                )


if __name__ == "__main__":
    unittest.main()
