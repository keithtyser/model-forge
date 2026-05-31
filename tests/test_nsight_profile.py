from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_forge.profiling.nsight import (
    DEFAULT_CONFIG,
    audit_config,
    build_plan,
    build_summary,
    load_config,
    write_plan,
    write_summary,
)


class NsightProfileTests(unittest.TestCase):
    def test_default_config_plans_nsys_and_ncu_without_execution(self) -> None:
        config, path = load_config(DEFAULT_CONFIG)
        findings = audit_config(config, path, strict=False)
        self.assertEqual([finding for finding in findings if finding.severity == "error"], [])

        plan = build_plan(config, path, run_id="unit_nsight")
        self.assertEqual(plan["schema_version"], "model_forge.nsight_profile_plan.v1")
        self.assertEqual(plan["run_id"], "unit_nsight")
        self.assertFalse(plan["execution_contract"]["starts_server"])
        self.assertTrue(plan["execution_contract"]["dry_run_by_default"])
        tools = {profile["tool"] for profile in plan["profiles"]}
        self.assertEqual(tools, {"nsys", "ncu"})
        self.assertTrue(any(profile["command"][0] == "nsys" for profile in plan["profiles"]))
        self.assertTrue(any(profile["command"][0] == "ncu" for profile in plan["profiles"]))

    def test_write_plan_creates_json_and_command_script(self) -> None:
        config, path = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(
                config,
                path,
                run_id="unit_nsight_write",
                command="./forge bench serve --dry-run",
                output_root=tmp,
            )
            plan_path = write_plan(plan, tmp)
            commands_path = plan_path.parent / "profile_commands.sh"
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            commands = commands_path.read_text(encoding="utf-8")

        self.assertEqual(saved["run_id"], "unit_nsight_write")
        self.assertTrue(saved["outputs"]["plan"].endswith("unit_nsight_write/nsight_profile_plan.json"))
        self.assertIn("nsys profile", commands)
        self.assertIn("ncu --target-processes", commands)

    def test_profile_summary_reports_present_and_missing_artifacts(self) -> None:
        config, path = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(config, path, run_id="unit_nsight_summary", output_root=tmp)
            plan_path = write_plan(plan, tmp)
            first_output = Path(plan["profiles"][0]["output"])
            if not first_output.is_absolute():
                first_output = Path.cwd() / first_output
            first_output.parent.mkdir(parents=True, exist_ok=True)
            first_output.write_bytes(b"profile")

            summary = build_summary(plan, plan_path=plan_path)
            summary_path = write_summary(summary)
            markdown = summary_path.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(summary["schema_version"], "model_forge.profile_summary.v1")
        self.assertEqual(summary["summary"]["expected_profile_artifacts"], 2)
        self.assertEqual(summary["summary"]["present_profile_artifacts"], 1)
        self.assertEqual(summary["summary"]["missing_profile_artifacts"], 1)
        self.assertIn("# Profile Summary: unit_nsight_summary", markdown)


if __name__ == "__main__":
    unittest.main()
