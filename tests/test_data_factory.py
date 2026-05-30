from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from model_forge.data.factory import (
    REPO_DIR,
    build_provider,
    build_gap_report,
    build_plan,
    command_generate,
    command_gaps,
    command_pack,
    command_publish,
    command_review,
    command_verify,
    load_yaml,
    rejection_reasons,
)
from model_forge.data.sources import load_source_registry, resolve_sources


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
        self.assertIn("self_instruct", plan["generation_methods"]["enabled_now"])
        self.assertEqual(plan["generation"]["provider"]["type"], "template")
        self.assertEqual(plan["review"]["min_examples_per_skill"], 5)
        self.assertEqual(plan["quality_thresholds"]["max_holdout_similarity"], 0.82)
        self.assertIn("source_registry", plan)
        self.assertIn("model_forge_local_ft_v1_seeds", plan["source_registry"]["selected_source_ids"])
        self.assertFalse(plan["seed_only"])
        self.assertTrue(plan["smoke_only"])

    def test_live_teacher_smoke_config_uses_openai_compatible_provider(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1_live_teacher_smoke.yaml"
        plan = build_plan(load_yaml(config_path), config_path)
        self.assertEqual(plan["id"], "gemma4_26b_a4b_local_ft_v1_live_teacher_smoke")
        self.assertEqual(plan["generation"]["provider"]["type"], "openai_compatible")
        self.assertEqual(plan["generation"]["provider"]["base_url"], "http://127.0.0.1:8011/v1")
        self.assertEqual(plan["generation"]["provider"]["request"]["max_tokens"], 650)
        self.assertEqual(plan["generation"]["smoke"]["max_generated_candidates"], 24)
        self.assertTrue(plan["quality_thresholds"]["reject_length_violations"])
        self.assertIn("too_long", plan["review"]["critical_flags"])
        self.assertIn("model_forge_local_ft_v1_live_teacher_smoke", plan["source_registry"]["selected_source_ids"])
        self.assertTrue(plan["smoke_only"])

    def test_source_registry_resolves_manifest_overrides(self) -> None:
        registry = load_source_registry(REPO_DIR / "configs" / "data_sources" / "gemma4_26b_a4b_local_ft_v1.yaml")
        self.assertIn("jackrong_qwen_coder_distill", registry["sources"])
        sources = resolve_sources({
            "source_registry": "configs/data_sources/gemma4_26b_a4b_local_ft_v1.yaml",
            "sources": [{"id": "model_forge_local_ft_v1_live_teacher_smoke", "target_samples": 12}],
        })
        self.assertEqual(sources[0]["path"], "datasets/generated/gemma4_26b_a4b_local_ft_v1_live_teacher_smoke/dataset.jsonl")
        self.assertEqual(sources[0]["target_samples"], 12)

    def test_provider_env_overrides_model_base_url_and_api_key(self) -> None:
        config = {
            "generation": {
                "provider": {
                    "type": "openai_compatible",
                    "model": "default-model",
                    "model_env": "MODEL_FORGE_TEST_PROVIDER_MODEL",
                    "base_url": "http://default.invalid/v1",
                    "base_url_env": "MODEL_FORGE_TEST_PROVIDER_BASE_URL",
                    "api_key_env": "MODEL_FORGE_TEST_PROVIDER_API_KEY",
                }
            }
        }
        old_model = os.environ.get("MODEL_FORGE_TEST_PROVIDER_MODEL")
        old_base = os.environ.get("MODEL_FORGE_TEST_PROVIDER_BASE_URL")
        old_key = os.environ.get("MODEL_FORGE_TEST_PROVIDER_API_KEY")
        try:
            os.environ["MODEL_FORGE_TEST_PROVIDER_MODEL"] = "override-model"
            os.environ["MODEL_FORGE_TEST_PROVIDER_BASE_URL"] = "http://127.0.0.1:9999/v1"
            os.environ["MODEL_FORGE_TEST_PROVIDER_API_KEY"] = "test-token"
            provider = build_provider(config)
        finally:
            if old_model is None:
                os.environ.pop("MODEL_FORGE_TEST_PROVIDER_MODEL", None)
            else:
                os.environ["MODEL_FORGE_TEST_PROVIDER_MODEL"] = old_model
            if old_base is None:
                os.environ.pop("MODEL_FORGE_TEST_PROVIDER_BASE_URL", None)
            else:
                os.environ["MODEL_FORGE_TEST_PROVIDER_BASE_URL"] = old_base
            if old_key is None:
                os.environ.pop("MODEL_FORGE_TEST_PROVIDER_API_KEY", None)
            else:
                os.environ["MODEL_FORGE_TEST_PROVIDER_API_KEY"] = old_key
        self.assertEqual(provider.model_name, "override-model")
        self.assertEqual(provider.config["base_url"], "http://127.0.0.1:9999/v1")
        self.assertEqual(provider.config["api_key"], "test-token")

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
            self.assertIn("generation_report", manifest["artifacts"])
            self.assertEqual(manifest["quality_report"]["verification_counts"]["passed"], len(dataset_rows))
            self.assertGreater(manifest["quality_report"]["source_kind_counts"]["synthetic"], 0)

    def test_generate_expands_seeds_with_template_provider(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            path = command_generate(config, overwrite=True, smoke=True)
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            synthetic = [row for row in rows if row.get("source", {}).get("kind") == "synthetic"]
            self.assertEqual(len(synthetic), config["generation"]["smoke"]["max_generated_candidates"])
            self.assertTrue(all(row["source"]["generation"]["prompt_template_hash"] for row in synthetic))
            report = json.loads((Path(tmp) / "generation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["provider"]["type"], "template")
            self.assertEqual(report["source_kind_counts"]["synthetic"], len(synthetic))

    def test_downstream_overwrite_does_not_regenerate_candidates(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            candidate_path = command_generate(config, overwrite=True, smoke=True)
            before = [
                json.loads(line)
                for line in Path(candidate_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            command_verify(config, overwrite=True, smoke=True, max_generated_candidates=0)
            after = [
                json.loads(line)
                for line in Path(candidate_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(before, after)
            self.assertGreater(sum(1 for row in after if row.get("source", {}).get("kind") == "synthetic"), 0)

    def test_length_violations_can_reject_rows(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        long_answer = " ".join(["word"] * (config["review"]["max_assistant_words"] + 1))
        row = {
            "id": "too_long",
            "messages": [
                {"role": "user", "content": "Explain a safe workflow."},
                {"role": "assistant", "content": long_answer},
            ],
            "skills": ["shell_safety"],
            "source": {"kind": "synthetic", "license": "CC-BY-4.0"},
            "quality_scores": {"target_skill_relevance": 1.0},
            "quality_score_average": 1.0,
            "verification": {"passed": True},
        }
        self.assertIn("assistant_too_long", rejection_reasons(row, config, set(), []))

    def test_review_command_writes_review_gate_artifacts(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            outputs = command_review(config, overwrite=True, smoke=True, sample_size=12)
            report = json.loads(Path(outputs["review_report"]).read_text(encoding="utf-8"))
            sheet = Path(outputs["review_sheet"]).read_text(encoding="utf-8")
            self.assertIn("ready_to_scale_generation", report)
            self.assertEqual(report["reviewed_count"], 12)
            self.assertIn("Dataset Review", sheet)
            self.assertIn("Coverage Gaps", sheet)

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
            publish_plan = command_publish(
                config,
                config_path,
                overwrite=True,
                source_license_checked=True,
            )
            plan = json.loads(Path(publish_plan).read_text(encoding="utf-8"))
            text = json.dumps(plan)
            self.assertIn('"dry_run": true', text)
            self.assertIn('"repo_type": "dataset"', text)
            self.assertIn("verification.jsonl", text)
            self.assertIn("generation_report.json", text)
            self.assertEqual(plan["release_class"], "public_dataset")
            self.assertTrue(plan["redacted_output_bundle"].endswith("hf_publish_bundle"))
            self.assertTrue(all("accepted.jsonl" not in path for path in plan["files"]))
            self.assertTrue(all("rejected.jsonl" not in path for path in plan["files"]))
            self.assertTrue(any(path.endswith("dataset_redacted.jsonl") for path in plan["files"]))
            self.assertTrue(any(gate["name"] == "unsafe_examples_redacted" and gate["status"] == "pass" for gate in plan["release_gates"]))

            redacted_path = Path(tmp) / "hf_publish_bundle" / "dataset_redacted.jsonl"
            rows = [
                json.loads(line)
                for line in redacted_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(rows), 0)
            self.assertEqual(rows[0]["messages"][0]["content"], "<redacted>")
            self.assertIn("content_sha256", rows[0]["messages"][0])
            self.assertNotIn("Explain a safe workflow", redacted_path.read_text(encoding="utf-8"))
            self.assertNotIn(
                "A model run has higher total latency",
                (Path(tmp) / "hf_publish_bundle" / "verification.jsonl").read_text(encoding="utf-8"),
            )

    def test_publish_execute_refuses_smoke_dataset(self) -> None:
        config_path = REPO_DIR / "configs" / "datasets" / "gemma4_26b_a4b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = tmp
            command_publish(config, config_path, overwrite=True)
            with self.assertRaises(SystemExit):
                command_publish(config, config_path, overwrite=False, execute=True)


if __name__ == "__main__":
    unittest.main()
