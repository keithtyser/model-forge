import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_DIR / "scripts" / "patch_lm_head_tokens_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("patch_lm_head_tokens_checkpoint", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    shard1 = {
        "lm_head.weight": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [4.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "model.norm.weight": torch.ones(2, dtype=torch.float32),
    }
    shard2 = {"model.layers.0.mlp.down_proj.weight": torch.ones(2, 2, dtype=torch.float32)}
    save_file(shard1, path / "model-00001-of-00002.safetensors")
    save_file(shard2, path / "model-00002-of-00002.safetensors")
    weight_map = {
        "lm_head.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    }
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"model_type": "unit"}), encoding="utf-8")


class PatchLmHeadTokensCheckpointTests(unittest.TestCase):
    def test_patches_lm_head_rows_and_hardlinks_unchanged_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            write_checkpoint(source)

            manifest = module.patch_checkpoint(
                source_dir=source,
                output_dir=output,
                patches=[
                    {
                        "name": "token_zero_to_two",
                        "token_id": 0,
                        "replacement_token_id": 2,
                        "alpha": 1.0,
                        "scale": 1.0,
                        "preserve_norm": False,
                    },
                    {
                        "name": "scale_token_one",
                        "token_id": 1,
                        "alpha": 0.0,
                        "scale": 0.5,
                        "preserve_norm": False,
                    },
                ],
                lm_head_tensor=None,
                overwrite=False,
                copy_unchanged=False,
                min_ram_fraction=0.0,
                min_disk_fraction=0.0,
                dry_run=False,
            )

            self.assertEqual(manifest["schema_version"], "model_forge.lm_head_token_patch.v1")
            self.assertEqual(manifest["shard_actions"]["model-00001-of-00002.safetensors"], "rewrite")
            self.assertIn(
                manifest["shard_actions"]["model-00002-of-00002.safetensors"],
                {"hardlink", "copy"},
            )
            with safe_open(output / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as handle:
                lm_head = handle.get_tensor("lm_head.weight")
            torch.testing.assert_close(lm_head[0], torch.tensor([4.0, 0.0]))
            torch.testing.assert_close(lm_head[1], torch.tensor([0.0, 1.0]))
            self.assertTrue((output / "model_forge_lm_head_token_patch.json").exists())

    def test_rejects_duplicate_target_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            write_checkpoint(source)
            with self.assertRaisesRegex(SystemExit, "duplicate target token_id"):
                module.normalize_patch_specs(
                    [
                        {"token_id": 1, "scale": 0.5},
                        {"token_id": 1, "scale": 0.25},
                    ],
                    tokenizer_dir=source,
                    trust_remote_code=False,
                )


if __name__ == "__main__":
    unittest.main()
