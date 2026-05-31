from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from model_forge.upstream import DEFAULT_CONFIG, PLAN_SCHEMA_VERSION, audit_config, build_plan, load_yaml, write_plan


class UpstreamPlanTests(unittest.TestCase):
    def test_default_candidate_plan_records_completion_rule(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        findings = audit_config(config, path)
        self.assertFalse([finding for finding in findings if finding.severity == "error"])
        plan = build_plan(config, path, candidate_id="kernel_card_docs_or_example", run_id="unit_upstream")
        self.assertEqual(plan["schema_version"], PLAN_SCHEMA_VERSION)
        self.assertTrue(plan["evidence_requirements"]["external_pr_url_required_for_completion"])
        self.assertIn("external_pr_url", plan["completion_rule"])

    def test_audit_rejects_open_candidate_without_external_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "upstream.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "model_forge.upstream_pr_candidates.v1",
                        "candidates": [
                            {
                                "id": "opened_missing_url",
                                "target_project": "example",
                                "target_url": "https://github.com/example/repo",
                                "status": "opened",
                                "hypothesis": "test",
                                "contribution_type": "docs",
                                "next_action": "record url",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config, path = load_yaml(config_path)
            findings = audit_config(config, path)

        self.assertIn("evidence", {finding.check for finding in findings})

    def test_strict_audit_rejects_placeholder_target(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        findings = audit_config(config, path, strict=True)
        target_findings = [finding for finding in findings if finding.check == "target"]
        self.assertTrue(target_findings)
        self.assertTrue(all(finding.severity == "error" for finding in target_findings))

    def test_audit_rejects_malformed_external_pr_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "kernel_card.json"
            evidence.write_text("{}", encoding="utf-8")
            config_path = root / "upstream.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "model_forge.upstream_pr_candidates.v1",
                        "candidates": [
                            {
                                "id": "opened_bad_url",
                                "target_project": "example",
                                "target_url": "https://github.com/example/repo",
                                "status": "opened",
                                "external_pr_url": "https://github.com/example/repo/issues/1",
                                "hypothesis": "test",
                                "contribution_type": "docs",
                                "local_evidence": [str(evidence)],
                                "next_action": "wait for review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config, path = load_yaml(config_path)
            findings = audit_config(config, path)

        self.assertIn(
            "external_pr_url must be a GitHub pull request URL",
            {finding.message for finding in findings},
        )

    def test_audit_rejects_open_candidate_with_placeholder_or_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "upstream.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "model_forge.upstream_pr_candidates.v1",
                        "candidates": [
                            {
                                "id": "opened_missing_evidence",
                                "target_project": "example",
                                "target_url": "https://github.com/example/repo",
                                "status": "opened",
                                "external_pr_url": "https://github.com/example/repo/pull/12",
                                "hypothesis": "test",
                                "contribution_type": "docs",
                                "local_evidence": [
                                    "reports/generated/kernel_benchmarks/<run>/kernel_card.json",
                                    "reports/generated/kernel_benchmarks/missing/kernel_card.json",
                                ],
                                "next_action": "wait for review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config, path = load_yaml(config_path)
            findings = audit_config(config, path)
            messages = {finding.message for finding in findings}

        self.assertIn(
            "local_evidence contains unresolved placeholder: reports/generated/kernel_benchmarks/<run>/kernel_card.json",
            messages,
        )
        self.assertIn(
            "local_evidence path does not exist: reports/generated/kernel_benchmarks/missing/kernel_card.json",
            messages,
        )

    def test_write_plan_creates_json_and_markdown(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(config, path, candidate_id="kernel_card_docs_or_example", run_id="unit_upstream_write")
            plan = {**plan, "output_dir": tmp}
            plan_path = write_plan(plan)
            markdown = plan_path.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(plan_path.name, "upstream_pr_plan.json")
        self.assertIn("# Upstream PR Plan: unit_upstream_write", markdown)


if __name__ == "__main__":
    unittest.main()
