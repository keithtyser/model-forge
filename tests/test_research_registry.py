from __future__ import annotations

import unittest

from model_forge.research.registry import (
    DEFAULT_REGISTRY,
    audit_registry,
    filter_entries,
    load_registry,
    objective_research_references,
    sorted_entries,
)


class ResearchRegistryTests(unittest.TestCase):
    def test_registry_loads_and_filters_entries(self) -> None:
        registry = load_registry(DEFAULT_REGISTRY)
        entries = sorted_entries(registry)
        self.assertGreaterEqual(len(entries), 10)
        behavior_entries = filter_entries(entries, area="behavior_editing")
        behavior_ids = {entry["id"] for entry in behavior_entries}
        self.assertIn("arditi_2024_refusal_direction", behavior_ids)
        self.assertIn("som_multidirectional_refusal_2026", behavior_ids)

    def test_objective_research_basis_resolves_to_registry(self) -> None:
        registry = load_registry(DEFAULT_REGISTRY)
        known_ids = set(registry["entries"])
        references = objective_research_references()
        self.assertGreater(len(references), 0)
        for _, reference_id in references:
            self.assertIn(reference_id, known_ids)

    def test_registry_audit_has_no_errors(self) -> None:
        findings = audit_registry()
        errors = [finding for finding in findings if finding.severity == "error"]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
