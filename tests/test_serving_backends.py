from __future__ import annotations

import tempfile
import unittest

from model_forge.serving.backends import DEFAULT_CONFIG, PLAN_SCHEMA_VERSION, audit_config, build_plan, load_yaml, write_plan


class ServingBackendTests(unittest.TestCase):
    def test_sglang_config_audits_and_builds_family_plan(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        findings = audit_config(config, path, strict=True)
        self.assertFalse([finding for finding in findings if finding.severity == "error"])
        plan = build_plan(config, path, family="gemma4_26b_a4b", variant="base", run_id="unit_sglang")
        self.assertEqual(plan["schema_version"], PLAN_SCHEMA_VERSION)
        self.assertEqual(plan["engine"], "sglang")
        self.assertIn("sglang.launch_server", " ".join(plan["launch"]["command"]))
        self.assertIn("--model-path", plan["launch"]["command"])
        self.assertEqual(plan["model"]["model_path"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(plan["model"]["served_model_name"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(plan["network"]["base_url"], "http://127.0.0.1:8000/v1")
        self.assertFalse(plan["launch"]["execute_by_default"])

    def test_sglang_plan_can_use_manual_model_path(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        plan = build_plan(
            config,
            path,
            model_path="Qwen/Qwen3.5-9B",
            served_model_name="local/qwen-test",
            run_id="unit_sglang_manual",
            env={},
        )
        self.assertEqual(plan["model"]["model_path"], "Qwen/Qwen3.5-9B")
        self.assertIn("local/qwen-test", plan["benchmarks"]["smoke_command"])

    def test_write_plan_creates_json_and_markdown(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(config, path, model_path="example/model", served_model_name="example/model", run_id="unit_sglang_write", env={})
            plan = {**plan, "output_dir": tmp}
            plan_path = write_plan(plan)
            markdown = plan_path.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(plan_path.name, "serving_backend_plan.json")
        self.assertIn("# Serving Backend Plan: unit_sglang_write", markdown)
        self.assertIn("sglang.launch_server", markdown)


if __name__ == "__main__":
    unittest.main()
