from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from model_forge.hardware import HardwareProfile
from model_forge.runs.manifest import build_canonical_manifest, key_value_mapping, porcelain_path, write_manifest


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_provenance_without_secret_values(self) -> None:
        now = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
        fake_hf_token = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        env = {
            "MODEL_FORGE_API_KEY": fake_hf_token,
            "MODEL_FORGE_BASE_URL": "http://127.0.0.1:8000/v1",
            "VLLM_MAX_NUM_SEQS": "4",
            "HF_TOKEN": fake_hf_token,
        }
        profile = HardwareProfile(name="cpu", label="CPU / test")
        with mock.patch("model_forge.runs.manifest.detect_hardware_profile", return_value=profile):
            manifest = build_canonical_manifest(
                run_type="eval",
                status="planned",
                family="gemma4_26b_a4b",
                variant="base",
                command=["./forge", "eval", "gemma4_26b_a4b", "base", "--internal"],
                config_paths=["configs/experiments/gemma4_26b_a4b_v0.yaml"],
                output_dir="results/gemma4_26b_a4b_v0/base",
                artifacts={"scores_csv": "scores.csv"},
                metadata={"token": fake_hf_token},
                notes=[f"temporary key {fake_hf_token}"],
                env=env,
                now=now,
            )

        self.assertEqual(manifest["schema_version"], "model_forge.run_manifest.v1")
        self.assertEqual(manifest["run_id"], "gemma4_26b_a4b_base_eval_20260519t120000z")
        self.assertEqual(manifest["identity"]["family"], "gemma4_26b_a4b")
        self.assertEqual(manifest["source"]["family_config"], "configs/model_families/gemma4_26b_a4b.yaml")
        self.assertEqual(manifest["environment"]["MODEL_FORGE_API_KEY"], "<redacted>")
        self.assertEqual(manifest["environment"]["MODEL_FORGE_BASE_URL"], "http://127.0.0.1:8000/v1")
        self.assertNotIn("HF_TOKEN", manifest["environment"])
        self.assertEqual(manifest["metadata"]["token"], "<redacted>")
        self.assertIn("<redacted>", manifest["notes"][0])
        self.assertEqual(manifest["configs"][0]["path"], "configs/experiments/gemma4_26b_a4b_v0.yaml")
        self.assertTrue(manifest["configs"][0]["exists"])

    def test_manifest_writer_uses_run_id_directory(self) -> None:
        profile = HardwareProfile(name="cpu", label="CPU / test")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "model_forge.runs.manifest.detect_hardware_profile",
            return_value=profile,
        ):
            manifest = build_canonical_manifest(
                run_type="data",
                status="completed",
                family="test_family",
                variant="test_variant",
                run_id="dataset-smoke",
            )
            path = write_manifest(manifest, Path(tmp))

            self.assertEqual(path.name, "manifest.json")
            self.assertEqual(path.parent.name, "dataset-smoke")
            self.assertTrue(path.exists())

    def test_key_value_mapping_accepts_json_values(self) -> None:
        mapping = key_value_mapping(["pass_rate=0.75", "name=local", "ok=true"])
        self.assertEqual(mapping["pass_rate"], 0.75)
        self.assertEqual(mapping["name"], "local")
        self.assertTrue(mapping["ok"])

    def test_porcelain_path_handles_staged_and_unstaged_formats(self) -> None:
        self.assertEqual(porcelain_path(" M forge"), "forge")
        self.assertEqual(porcelain_path("M  forge"), "forge")
        self.assertEqual(porcelain_path("?? src/model_forge/variants/"), "src/model_forge/variants/")


if __name__ == "__main__":
    unittest.main()
