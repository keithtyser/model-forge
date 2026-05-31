from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_forge.agents import audit_agent_experiments, load_yaml, validate_agent_experiment


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


if __name__ == "__main__":
    unittest.main()
