from __future__ import annotations

import unittest

from model_forge.schema_audit import SCHEMA_AUDIT_VERSION, build_schema_audit


class SchemaAuditTests(unittest.TestCase):
    def test_schema_audit_covers_required_artifact_classes(self) -> None:
        report = build_schema_audit()
        self.assertEqual(report["schema_version"], SCHEMA_AUDIT_VERSION)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(
            set(report["required_artifact_classes"]),
            {"manifest", "card", "objective", "variant_node"},
        )
        classes = {check["artifact_class"] for check in report["checks"]}
        self.assertTrue(set(report["required_artifact_classes"]).issubset(classes))

    def test_schema_audit_validates_foundation_schema_versions(self) -> None:
        report = build_schema_audit()
        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["run_manifest_schema"]["schema_version"], "model_forge.run_manifest.v1")
        self.assertEqual(checks["objective_profile_schema"]["schema_version"], "model_forge.objective_profile.v1")
        self.assertEqual(checks["variant_node_schema"]["schema_version"], "model_forge.variant_node.v1")
        self.assertTrue(checks["objective_profiles_validate"]["passed"])
        self.assertTrue(checks["run_manifest_required_fields"]["passed"])
        self.assertTrue(checks["variant_node_required_fields"]["passed"])


if __name__ == "__main__":
    unittest.main()

