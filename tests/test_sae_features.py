from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from model_forge.behavior_editing.sae_features import (
    constrain_direction_to_decoder,
    discover_decoder_tensor,
    resolve_sae_source,
    rewrite_direction_artifact_with_sae,
)


class SaeFeatureTests(unittest.TestCase):
    def test_discover_decoder_tensor_prefers_decoder_key_and_normalizes_shape(self) -> None:
        state = {
            "encoder.weight": torch.zeros(4, 3),
            "W_dec": torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]),
        }

        key, decoder = discover_decoder_tensor(state, hidden_size=3)

        self.assertEqual(key, "W_dec")
        self.assertEqual(tuple(decoder.shape), (4, 3))

    def test_constrain_direction_to_decoder_selects_aligned_features(self) -> None:
        decoder = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        direction = torch.tensor([0.9, 0.1, 0.0])

        constrained, selected = constrain_direction_to_decoder(direction, decoder, top_k=1)

        self.assertEqual(tuple(constrained.shape), (1, 3))
        self.assertEqual(selected[0]["feature_index"], 0)
        self.assertGreater(selected[0]["cosine"], 0.8)

    def test_rewrite_direction_artifact_uses_per_layer_sae_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "direction_artifact.pt"
            output_path = root / "sae_direction_artifact.pt"
            torch.save(
                {
                    "refusal_directions": {
                        1: torch.tensor([1.0, 0.0, 0.0]),
                        2: torch.tensor([0.0, 1.0, 0.0]),
                    },
                    "harmful_means": {},
                    "benign_means": {},
                },
                artifact_path,
            )
            torch.save({"W_dec": torch.eye(3)}, root / "layer1.sae.pt")
            torch.save({"W_dec": torch.eye(3)}, root / "layer2.sae.pt")

            report = rewrite_direction_artifact_with_sae(
                input_path=artifact_path,
                output_path=output_path,
                sae_source=str(root),
                hidden_size=3,
                top_k=1,
                sae_file_pattern="layer{layer}.sae.pt",
            )

            rewritten = torch.load(output_path, map_location="cpu")
            self.assertEqual(report["sae"]["loaded_decoder_count"], 2)
            self.assertEqual(sorted(report["layers"]), ["1", "2"])
            self.assertIn("sae_dictionary_constraint", rewritten)
            self.assertEqual(tuple(rewritten["refusal_directions"][1].shape), (1, 3))

    def test_resolve_sae_source_passes_allow_patterns_for_remote_repos(self) -> None:
        calls = {}

        def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
            calls["repo_id"] = repo_id
            calls.update(kwargs)
            return "/tmp/fake-sae"

        with unittest.mock.patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
            path = resolve_sae_source(
                "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
                allow_patterns=["layer20.sae.pt"],
            )

        self.assertEqual(str(path), "/tmp/fake-sae")
        self.assertEqual(calls["repo_id"], "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50")
        self.assertEqual(calls["allow_patterns"], ["layer20.sae.pt"])


if __name__ == "__main__":
    unittest.main()
