from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from model_forge.upstream import (
    DEFAULT_CONFIG,
    PLAN_SCHEMA_VERSION,
    VERIFICATION_SCHEMA_VERSION,
    audit_config,
    build_plan,
    build_verification,
    load_yaml,
    parse_github_pr_url,
    write_plan,
    write_verification,
)


class UpstreamPlanTests(unittest.TestCase):
    def test_default_candidate_plan_records_completion_rule(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        findings = audit_config(config, path)
        self.assertFalse([finding for finding in findings if finding.severity == "error"])
        plan = build_plan(config, path, candidate_id="dgx_spark_vllm_serving_recipe", run_id="unit_upstream")
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

    def test_strict_audit_accepts_current_concrete_candidate_target(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        findings = audit_config(config, path, strict=True)
        self.assertFalse([finding for finding in findings if finding.severity == "error"])

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
            plan = build_plan(config, path, candidate_id="dgx_spark_vllm_serving_recipe", run_id="unit_upstream_write")
            plan = {**plan, "output_dir": tmp}
            plan_path = write_plan(plan)
            markdown = plan_path.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(plan_path.name, "upstream_pr_plan.json")
        self.assertIn("# Upstream PR Plan: unit_upstream_write", markdown)

    def test_parse_github_pr_url_accepts_pull_urls_only(self) -> None:
        parsed = parse_github_pr_url("https://github.com/vllm-project/vllm/pull/123")
        self.assertEqual(parsed, {"owner": "vllm-project", "repo": "vllm", "number": "123"})
        self.assertIsNone(parse_github_pr_url("https://github.com/vllm-project/vllm/issues/123"))

    def test_offline_verification_passes_recorded_open_pr_with_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "kernel_card.json"
            evidence.write_text('{"schema_version":"model_forge.kernel_card.v1"}', encoding="utf-8")
            config_path = root / "upstream.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "model_forge.upstream_pr_candidates.v1",
                        "candidates": [
                            {
                                "id": "opened_with_evidence",
                                "target_project": "vllm",
                                "target_url": "https://github.com/vllm-project/vllm",
                                "status": "opened",
                                "external_pr_url": "https://github.com/vllm-project/vllm/pull/123",
                                "hypothesis": "docs can cite measured evidence",
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
            verification = build_verification(config, path, candidate_id="opened_with_evidence", offline=True)

        self.assertEqual(verification["schema_version"], VERIFICATION_SCHEMA_VERSION)
        self.assertTrue(verification["verified"], verification["blocked_until"])
        checks = {check["name"]: check for check in verification["checks"]}
        self.assertEqual(checks["github_pr_remote_verified"]["status"], "skip")
        self.assertIn("offline mode skipped", checks["github_pr_remote_verified"]["message"])

    def test_verification_blocks_candidate_until_external_pr_is_recorded(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        verification = build_verification(config, path, candidate_id="dgx_spark_vllm_serving_recipe", offline=True)

        self.assertFalse(verification["verified"])
        failures = {check["name"] for check in verification["checks"] if check["status"] == "fail"}
        self.assertIn("candidate_status_opened_or_merged", failures)
        self.assertIn("external_pr_url_present", failures)
        self.assertNotIn("target_url_concrete", failures)
        self.assertFalse(any(name.endswith("_exists") for name in failures))

    def test_write_verification_creates_report(self) -> None:
        config, path = load_yaml(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            verification = build_verification(
                config,
                path,
                candidate_id="dgx_spark_vllm_serving_recipe",
                offline=True,
                run_id="unit_upstream_verify",
            )
            verification = {**verification, "output_dir": tmp}
            report_path = write_verification(verification)
            saved = yaml.safe_load(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report_path.name, "upstream_pr_verification.json")
        self.assertEqual(saved["schema_version"], VERIFICATION_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
