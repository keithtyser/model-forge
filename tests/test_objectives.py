from __future__ import annotations

import unittest

from model_forge.objectives import (
    audit_profiles,
    comparison_profile_from_objective,
    load_comparison_objective_profiles,
    load_objective_profiles,
)


class ObjectiveProfileTests(unittest.TestCase):
    def test_roadmap_objective_profiles_exist_and_audit_cleanly(self) -> None:
        profiles, errors = audit_profiles()
        self.assertEqual(errors, [])
        self.assertIn("capability_sft", profiles)
        self.assertIn("zero_refusal_capability_retention", profiles)
        self.assertIn("quantized_quality_retention", profiles)
        self.assertIn("dgx_spark_latency_throughput", profiles)

        for profile in profiles.values():
            self.assertIn(profile["implementation_status"], {"not_started", "scaffolded", "implemented", "wired_to_cli", "tested"})
            self.assertIn(
                profile["validation_state"],
                {"planned", "smoke_validated", "spark_single_node_validated", "spark_cluster_validated", "generalizable"},
            )
            self.assertTrue(profile["validation_gates"]["required_evidence"])

    def test_comparison_profiles_are_exposed_for_reports(self) -> None:
        profiles = load_objective_profiles()
        zero_refusal = comparison_profile_from_objective(profiles["zero_refusal_capability_retention"])
        self.assertIsNotNone(zero_refusal)
        self.assertIn("refusal_rate_harmful", zero_refusal["lower_is_better"])
        self.assertIn("normal_use_regression_pass_rate", zero_refusal["capability_metrics"])
        self.assertEqual(zero_refusal["primary_goal"]["metric"], "refusal_paired_boundary.refusal_rate_harmful")
        self.assertEqual(zero_refusal["primary_goal"]["target"], 0.0)

        comparison_profiles = load_comparison_objective_profiles()
        self.assertIn("quantized_quality_retention", comparison_profiles)
        self.assertIn("tokens_per_second", comparison_profiles["quantized_quality_retention"]["higher_is_better"])


if __name__ == "__main__":
    unittest.main()
