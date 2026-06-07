from __future__ import annotations

import unittest

from model_forge import diagnostics
from model_forge.diagnostics import Finding


class DiagnosticsTests(unittest.TestCase):
    def test_has_errors_only_at_or_above_threshold(self) -> None:
        warns = [Finding("warning", "c", "m"), Finding("info", "c", "m")]
        self.assertFalse(diagnostics.has_errors(warns))
        self.assertEqual(diagnostics.severity_exit_code(warns), 0)
        with_error = warns + [Finding("error", "c", "m")]
        self.assertTrue(diagnostics.has_errors(with_error))
        self.assertEqual(diagnostics.severity_exit_code(with_error), 1)

    def test_critical_blocks_at_error_threshold(self) -> None:
        self.assertTrue(diagnostics.has_errors([Finding("critical", "c", "m")]))

    def test_threshold_is_configurable(self) -> None:
        warns = [Finding("warning", "c", "m")]
        self.assertTrue(diagnostics.has_errors(warns, threshold="warning"))
        self.assertEqual(diagnostics.severity_exit_code(warns, threshold="warning"), 1)

    def test_worst_severity_and_counts(self) -> None:
        findings = [Finding("info", "c", "m"), Finding("error", "c", "m"), Finding("warning", "c", "m")]
        self.assertEqual(diagnostics.worst_severity(findings), "error")
        self.assertIsNone(diagnostics.worst_severity([]))
        self.assertEqual(
            diagnostics.count_by_severity(findings),
            {"info": 1, "error": 1, "warning": 1},
        )

    def test_duck_types_on_severity_attribute(self) -> None:
        class Other:
            severity = "error"

        self.assertTrue(diagnostics.has_errors([Other()]))

    def test_unknown_severity_ranks_below_info_and_never_blocks(self) -> None:
        self.assertFalse(diagnostics.has_errors([Finding("bogus", "c", "m")]))
        self.assertEqual(diagnostics.severity_rank("bogus"), -1)


if __name__ == "__main__":
    unittest.main()
