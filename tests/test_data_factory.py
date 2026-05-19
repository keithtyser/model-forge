from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_forge.data.factory import (
    REPO_DIR,
    build_gap_report,
    build_plan,
    command_gaps,
    command_pack,
    command_publish,
    command_verify,
    load_yaml,
)


class DatasetFactoryTests(unittest.TestCase):
    def test_local_ft_v1_plan_targets_observed_gaps(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        plan = build_plan(load_yaml(config_path), config_path)
        self.assertEqual(plan["id"], "gemma4_26b_a4b_local_ft_v1")
        self.assertEqual(plan["family"], "gemma4_26b_a4b")
        self.assertEqual(plan["objective"], "capability_sft")
        self.assertGreaterEqual(plan["seed_count"], 10)
        self.assertIn("eval_latency_throughput", plan["seed_skill_counts"])
        self.assertIn("benign_safety_analysis", plan["seed_skill_counts"])
        self.assertIn("self_instruct", plan["generation_methods"]["planned"])
        self.assertEqual(plan["quality_thresholds"]["max_holdout_similarity"], 0.82)
        self.assertTrue(plan["seed_only"])

    def test_pack_writes_dataset_card_and_rejection_report(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            outputs = command_pack(config, config_path, overwrite=True)
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)

            dataset_rows = Path(outputs["dataset"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(dataset_rows), 10)
            card = Path(outputs["dataset_card"]).read_text(encoding="utf-8")
            self.assertIn("eval-adjacent", card)
            self.assertIn("Skill Counts", card)
            self.assertIn("Coverage Warnings", card)
            self.assertIn("Verification passed", card)
            manifest = load_yaml(Path(outputs["manifest"]))
            self.assertIn("verification", manifest["artifacts"])
            self.assertEqual(manifest["quality_report"]["verification_counts"]["passed"], len(dataset_rows))

    def test_verify_command_writes_static_skill_checks(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            path = command_verify(config, overwrite=True)
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(rows), 10)
            self.assertTrue(all(row["verification"]["passed"] for row in rows))
            self.assertTrue(all(row["verification"]["type"] == "static_skill_checks" for row in rows))

    def test_gap_report_maps_eval_failures_to_seed_skills(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        report = build_gap_report(config)
        self.assertGreater(report["gap_rows"], 0)
        self.assertIn("capability_preservation_challenge", report["bucket_counts"])
        self.assertIn("eval_latency_throughput", report["recommended_skill_counts"])
        self.assertIn("benign_safety_analysis", report["recommended_skill_counts"])

    def test_gaps_command_writes_report(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            path = command_gaps(config, overwrite=True)
            self.assertTrue(Path(path).exists())
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("recommended_skill_counts", text)

    def test_publish_writes_dry_run_plan_only(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            publish_plan = command_publish(config, config_path, overwrite=True)
            text = Path(publish_plan).read_text(encoding="utf-8")
            self.assertIn('"dry_run": true', text)
            self.assertIn('"repo_type": "dataset"', text)
            self.assertIn("verification.jsonl", text)


if __name__ == "__main__":
    unittest.main()
