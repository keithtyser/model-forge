from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from argparse import Namespace

from model_forge.agents import (
    audit_agent_experiments,
    build_agent_run_card,
    load_yaml,
    optimize_behavior_edit_plan,
    optimize_quantization_plan,
    optimize_serving_plan,
    update_experiment_ledger,
    validate_agent_experiment,
    write_agent_run_card,
)


class AgentExperimentTests(unittest.TestCase):
    def test_template_agent_experiment_passes_schema_audit(self) -> None:
        self.assertEqual(audit_agent_experiments(), [])

    def test_agent_experiment_reports_missing_required_fields(self) -> None:
        findings = validate_agent_experiment({"schema_version": "model_forge.agent_experiment.v1"})
        fields = {finding.field for finding in findings}

        self.assertIn("experiment_id", fields)
        self.assertIn("planned_commands", fields)
        self.assertIn("resource_policy", fields)
        self.assertIn("evidence_plan", fields)

    def test_agent_experiment_rejects_secret_like_commands(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        plan["planned_commands"][0]["command"] = "HF_TOKEN=" + "hf_" + "a" * 32 + " ./forge doctor"
        findings = validate_agent_experiment(plan, path="unit.yaml")

        self.assertIn("planned command contains a secret-like value", {finding.message for finding in findings})

    def test_agent_experiment_rejects_unknown_variant(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        plan["variant"] = "missing_variant"
        findings = validate_agent_experiment(plan, path="unit.yaml")

        self.assertIn("variant is not defined for family", {finding.message for finding in findings})

    def test_agent_experiment_rejects_unknown_objective_profile(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        plan["objective_profile"] = "missing_objective"
        findings = validate_agent_experiment(plan, path="unit.yaml")

        self.assertIn("objective profile config does not exist", {finding.message for finding in findings})

    def test_agent_experiment_can_validate_explicit_path(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.yaml"
            path.write_text(
                "\n".join([
                    "schema_version: model_forge.agent_experiment.v1",
                    "experiment_id: bad",
                ])
                + "\n",
                encoding="utf-8",
            )
            loaded = load_yaml(path)

        findings = validate_agent_experiment(loaded)
        self.assertGreater(len(findings), 0)
        self.assertEqual(plan["schema_version"], "model_forge.agent_experiment.v1")

    def test_optimize_serving_plan_is_valid_and_marks_heavy_server_commands(self) -> None:
        plan = optimize_serving_plan(
            Namespace(
                family="gemma4_26b_a4b",
                variant="base",
                objective_profile="dgx_spark_latency_throughput",
                sweep_config=Path("configs/sweeps/dgx_spark_vllm_baseline.yaml"),
                cluster_config=None,
                base_url=None,
                experiment_id="unit_optimize_serving",
                title=None,
                hypothesis=None,
                owner_agent="unit",
                output=None,
                json=False,
            )
        )
        findings = validate_agent_experiment(plan)
        commands = plan["planned_commands"]

        self.assertEqual(findings, [])
        self.assertEqual(plan["experiment_type"], "serving")
        self.assertGreaterEqual(plan["metadata"]["case_count"], 1)
        self.assertTrue(any(command["starts_heavy_job"] for command in commands))
        self.assertTrue(any("bench sweep plan" in command["command"] for command in commands))
        self.assertTrue(any("bench serve" in command["command"] for command in commands))
        self.assertTrue(plan["evidence_plan"]["manifest_required"])
        self.assertTrue(plan["resource_policy"]["use_cluster_when_heavy"])

    def test_optimize_quantization_plan_is_valid_and_marks_export_commands_heavy(self) -> None:
        plan = optimize_quantization_plan(
            Namespace(
                config=Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"),
                family=None,
                variants="base,local_ft",
                objective_profile="quantized_quality_retention",
                experiment_id="unit_optimize_quantization",
                title=None,
                hypothesis=None,
                owner_agent="unit",
                output=None,
                json=False,
            )
        )
        findings = validate_agent_experiment(plan)
        commands = plan["planned_commands"]

        self.assertEqual(findings, [])
        self.assertEqual(plan["experiment_type"], "quantization")
        self.assertEqual(plan["metadata"]["variant_count"], 2)
        self.assertTrue(any("quantize matrix-plan" in command["command"] for command in commands))
        self.assertTrue(any("quantize export" in command["command"] and command["starts_heavy_job"] for command in commands))
        self.assertTrue(any("quantize card" in command["command"] for command in commands))
        self.assertTrue(plan["evidence_plan"]["manifest_required"])
        self.assertTrue(plan["resource_policy"]["use_cluster_when_heavy"])

    def test_optimize_behavior_edit_plan_is_valid_and_marks_edit_commands_heavy(self) -> None:
        plan = optimize_behavior_edit_plan(
            Namespace(
                family="gemma4_26b_a4b",
                config=Path("configs/abliteration/gemma4_26b_a4b_local_abli.yaml"),
                source_variant=None,
                target_variant=None,
                backend="heretic",
                objective_profile="zero_refusal_capability_retention",
                experiment_id="unit_optimize_behavior_edit",
                title=None,
                hypothesis=None,
                owner_agent="unit",
                start_memory_fraction=0.05,
                stop_memory_fraction=0.05,
                disk_free_fraction=0.15,
                output=None,
                json=False,
            )
        )
        findings = validate_agent_experiment(plan)
        commands = plan["planned_commands"]

        self.assertEqual(findings, [])
        self.assertEqual(plan["experiment_type"], "ablation")
        self.assertEqual(plan["metadata"]["target_variant"], "local_abli_sota")
        self.assertTrue(any("ablate --config" in command["command"] and "sota-plan" in command["command"] for command in commands))
        self.assertTrue(any("sota-run" in command["command"] and command["starts_heavy_job"] for command in commands))
        self.assertTrue(any("forge eval" in command["command"] and "--internal" in command["command"] for command in commands))
        self.assertTrue(plan["evidence_plan"]["manifest_required"])
        self.assertTrue(plan["resource_policy"]["use_cluster_when_heavy"])

    def test_agent_run_card_summarizes_plan_and_writes_outputs(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        card = build_agent_run_card(plan, plan_path=Path("recipes/agents/agent_experiment_template.yaml"), status="planned")

        self.assertEqual(card["schema_version"], "model_forge.agent_run_card.v1")
        self.assertTrue(card["validation"]["passed"])
        self.assertEqual(card["identity"]["experiment_id"], "template_agent_experiment")
        self.assertEqual(card["command_summary"]["total"], 1)
        self.assertEqual(card["command_summary"]["heavy"], 0)
        self.assertIn("./forge doctor --json", card["evidence_plan"]["required_validation_commands"])

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_agent_run_card(card, Path(tmp))
            self.assertTrue((Path(tmp) / "agent_run_card.json").exists())
            written = load_yaml(Path(tmp) / "agent_run_card.json")
            markdown = (Path(tmp) / "agent_run_card.md").read_text(encoding="utf-8")

        self.assertTrue(paths["json"].endswith("agent_run_card.json"))
        self.assertEqual(written["outputs"]["json"], paths["json"])
        self.assertIn("# Agent Run Card: template_agent_experiment", markdown)
        self.assertIn("## Required Validation", markdown)

    def test_agent_ledger_update_is_idempotent(self) -> None:
        plan = load_yaml(Path("recipes/agents/agent_experiment_template.yaml"))
        card = build_agent_run_card(plan, plan_path=Path("recipes/agents/agent_experiment_template.yaml"), status="planned")
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "experiment-ledger.md"
            ledger.write_text("# Experiment Ledger\n\nExisting entry.\n", encoding="utf-8")
            update_experiment_ledger(card, ledger)
            update_experiment_ledger({**card, "notes": "updated note"}, ledger)
            text = ledger.read_text(encoding="utf-8")

        self.assertEqual(text.count("## Agent Run: template_agent_experiment"), 1)
        self.assertIn("updated note", text)
        self.assertIn("model-forge-agent-run-card:template_agent_experiment:begin", text)
        self.assertIn("Existing entry.", text)


if __name__ == "__main__":
    unittest.main()
