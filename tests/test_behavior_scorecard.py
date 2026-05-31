from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_forge.evals.behavior_scorecard import (
    SCHEMA_VERSION,
    audit_config,
    evaluate_profile,
    load_yaml,
    write_scorecard,
)
from model_forge.runs.manifest import REPO_DIR


class BehaviorScorecardTests(unittest.TestCase):
    def test_scorecard_config_audits(self) -> None:
        config, path = load_yaml(REPO_DIR / "configs" / "behavior_edit" / "gemma4_26b_a4b_scorecard.yaml")
        findings = audit_config(config, path, strict=False)
        self.assertFalse([finding for finding in findings if finding.severity == "error"])
        profile = config["profiles"]["local_abli_sota_vs_base"]
        categories = {item["category"] for item in profile["rubric"]}
        self.assertIn("refusal_suppression", categories)
        self.assertIn("capability_retention", categories)
        self.assertIn("reported_risk", categories)

    def test_evaluate_profile_interprets_refusal_drop_as_success_and_risk_as_reported(self) -> None:
        config, _path = load_yaml(REPO_DIR / "configs" / "behavior_edit" / "gemma4_26b_a4b_scorecard.yaml")
        comparison = {
            "score_rows": [
                {"bucket": "refusal_paired_boundary", "metric": "refusal_rate_harmful", "base": 1.0, "local_abli_sota": 0.0},
                {"bucket": "refusal_paired_boundary", "metric": "ablation_refusal_suppression_rate", "base": 0.0, "local_abli_sota": 1.0},
                {"bucket": "normal_use_regression", "metric": "normal_use_regression_pass_rate", "base": 1.0, "local_abli_sota": 0.99},
                {"bucket": "capability_preservation_challenge", "metric": "normal_use_regression_pass_rate", "base": 0.8, "local_abli_sota": 0.78},
                {"bucket": "refusal_paired_boundary", "metric": "benign_answer_quality_rate", "base": 0.7, "local_abli_sota": 0.69},
                {"bucket": "unsafe_overcompliance", "metric": "unsafe_overcompliance_rate", "base": 0.0, "local_abli_sota": 1.0},
                {"bucket": "unsafe_overcompliance", "metric": "harmful_detail_rate", "base": 0.0, "local_abli_sota": 0.5},
            ]
        }
        scorecard = evaluate_profile(config, "local_abli_sota_vs_base", comparison)
        self.assertEqual(scorecard["schema_version"], SCHEMA_VERSION)
        self.assertTrue(scorecard["passed"])
        self.assertEqual(scorecard["decision"], "behavior_edit_candidate")
        risks = scorecard["categories"]["reported_risk"]
        self.assertTrue(all(item["status"] == "reported" for item in risks))

    def test_write_scorecard_creates_json_and_markdown(self) -> None:
        config, _path = load_yaml(REPO_DIR / "configs" / "behavior_edit" / "gemma4_26b_a4b_scorecard.yaml")
        scorecard = {
            "schema_version": SCHEMA_VERSION,
            "profile": "unit_behavior",
            "family": "gemma4_26b_a4b",
            "candidate": "candidate",
            "reference": "base",
            "objective": "zero_refusal_capability_retention",
            "decision": "hold",
            "passed": False,
            "rubric": [],
            "notes": ["unit"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_scorecard({**config, "output_dir": tmp}, "unit_behavior", scorecard)
            self.assertTrue(outputs["json"].exists())
            markdown = outputs["markdown"].read_text(encoding="utf-8")
            saved = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(saved["profile"], "unit_behavior")
        self.assertIn("# Behavior Edit Scorecard: unit_behavior", markdown)


if __name__ == "__main__":
    unittest.main()
