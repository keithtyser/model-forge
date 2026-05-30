from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_forge.artifacts.validate import main, validate_paths


class ArtifactValidationTests(unittest.TestCase):
    def test_validate_paths_writes_artifact_execution_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "panel.html").write_text(
                """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Panel</title></head>
<body>
  <h1>Status Panel</h1>
  <p>All systems nominal.</p>
  <canvas id="spark" width="80" height="40"></canvas>
  <script>
    const ctx = document.getElementById('spark').getContext('2d');
    ctx.fillStyle = '#157f5f';
    ctx.fillRect(0, 0, 80, 40);
  </script>
</body>
</html>
""",
                encoding="utf-8",
            )
            (artifacts / "summarize.py").write_text(
                """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("fixture", nargs="?")
args = parser.parse_args()
if args.fixture:
    rows = [json.loads(line) for line in open(args.fixture, encoding="utf-8")]
    print(f"workflow rows={len(rows)}")
else:
    print("ready")
""",
                encoding="utf-8",
            )

            output = root / "card"
            card = validate_paths(
                [artifacts],
                output_dir=output,
                run_id="unit_artifacts",
                checks_config={
                    "python": {
                        "validation_fixture": {
                            "kind": "responses_jsonl",
                            "args": ["{fixture}"],
                            "stdout_any": ["workflow"],
                        }
                    }
                },
                require_browser=True,
            )

            self.assertEqual(card["summary"]["artifact_count"], 2)
            self.assertEqual(card["summary"]["passed_count"], 2)
            self.assertEqual(card["summary"]["browser_checked_count"], 1)
            self.assertEqual(card["summary"]["screenshot_count"], 1)
            self.assertEqual(card["summary"]["metrics"]["artifact_execution_pass_rate"], 1.0)
            self.assertEqual(card["summary"]["metrics"]["artifact_compiles_rate"], 1.0)
            self.assertEqual(card["summary"]["metrics"]["artifact_runs_rate"], 1.0)
            self.assertEqual(card["summary"]["metrics"]["html_console_error_rate"], 0.0)
            self.assertEqual(card["summary"]["metrics"]["nonblank_render_rate"], 1.0)
            self.assertTrue((output / "artifact_execution_card.json").exists())
            self.assertTrue((output / "artifact_execution_card.md").exists())
            self.assertTrue((output / "artifact_validations.json").exists())

    def test_cli_strict_returns_nonzero_for_failed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "broken.py"
            artifact.write_text("def broken(:\n", encoding="utf-8")
            output = root / "out"

            code = main(["validate", str(artifact), "--output-dir", str(output), "--run-id", "broken", "--strict"])

            self.assertEqual(code, 1)
            card = json.loads((output / "artifact_execution_card.json").read_text(encoding="utf-8"))
            self.assertEqual(card["summary"]["failed_count"], 1)
            self.assertIn("compiles", card["artifacts"][0]["errors"])


if __name__ == "__main__":
    unittest.main()

