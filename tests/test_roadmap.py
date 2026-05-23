from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_forge.roadmap import audit_roadmap_items, parse_roadmap_items, summarize, write_markdown_report


class RoadmapAuditTests(unittest.TestCase):
    def test_current_roadmap_backlog_has_explicit_statuses(self) -> None:
        items = parse_roadmap_items()
        findings = audit_roadmap_items(items)
        self.assertEqual(findings, [])
        self.assertGreaterEqual(len(items), 60)
        by_id = {item.item_id: item for item in items}
        self.assertEqual(by_id["MF-0000"].implementation_status, "tested")
        self.assertEqual(by_id["MF-0000"].validation_state, "planned")
        self.assertEqual(by_id["MF-0305"].implementation_status, "wired_to_cli")
        self.assertEqual(by_id["MF-0305"].validation_state, "planned")

    def test_audit_reports_missing_or_invalid_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roadmap.md"
            path.write_text(
                "\n".join(
                    [
                        "## 16. Prioritized backlog",
                        "### P0: Example",
                        "MF-9998 Missing status.",
                        "MF-9999 Bad status. implementation_status=done validation_state=generalized",
                        "## 17. Repo structure target",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            findings = audit_roadmap_items(parse_roadmap_items(path))

        fields = {(finding.item_id, finding.field) for finding in findings}
        self.assertIn(("MF-9998", "implementation_status"), fields)
        self.assertIn(("MF-9998", "validation_state"), fields)
        self.assertIn(("MF-9999", "implementation_status"), fields)
        self.assertIn(("MF-9999", "validation_state"), fields)

    def test_markdown_report_writes_summary_and_items(self) -> None:
        items = parse_roadmap_items()
        summary = summarize(items, [], Path("docs/roadmaps/example.md"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audit.md"
            write_markdown_report(output, summary)
            text = output.read_text(encoding="utf-8")

        self.assertIn("# Roadmap Status Audit", text)
        self.assertIn("| MF-0000 |", text)
        self.assertIn("Implementation Status", text)


if __name__ == "__main__":
    unittest.main()
