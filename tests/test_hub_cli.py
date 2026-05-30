from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from model_forge.hub.cli import (
    build_model_plan,
    hf_status,
    main,
    scan_text_file,
)


def model_args(**overrides: object) -> Namespace:
    defaults = {
        "family": "gemma4_26b_a4b",
        "variant": "base",
        "repo_id": None,
        "release_class": "report_only",
        "artifact_path": None,
        "validation_state": "planned",
        "eval_results": None,
        "serving_card": None,
        "quantization_card": None,
        "promotion_report": None,
        "risk_report": None,
        "manifest": None,
        "source_license_checked": False,
        "behavior_edited": False,
        "include_raw_outputs": False,
        "output_dir": None,
        "run_id": "unit_hub_plan",
        "json": True,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class HubCliTests(unittest.TestCase):
    def test_offline_status_reports_token_source_without_token_value(self) -> None:
        token = "hf_" + "A" * 32
        status = hf_status(offline=True, env={"HF_TOKEN": token})

        payload = json.dumps(status)
        self.assertTrue(status["authenticated"])
        self.assertEqual(status["token_source"], "HF_TOKEN")
        self.assertNotIn(token, payload)

    def test_report_only_plan_writes_card_and_does_not_scan_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            private_path = "/" + "home" + "/ktyser/private/model"
            (model_dir / "leaky_manifest.json").write_text(
                json.dumps({"source": private_path}),
                encoding="utf-8",
            )
            out = root / "out"

            plan = build_model_plan(model_args(artifact_path=str(model_dir), output_dir=str(out)))

            self.assertFalse(plan["blocked"], plan["blocked_until"])
            self.assertEqual(plan["files_included"], [])
            self.assertEqual(plan["local_artifact_path"], "<external>/model")
            card = (out / "README.md").read_text(encoding="utf-8")
            provenance = (out / "hub_publish.json").read_text(encoding="utf-8")
            self.assertIn("https://github.com/keithtyser/model-forge", card)
            self.assertIn("hub_publish_path", provenance)
            self.assertNotIn("/home/ktyser", json.dumps(plan))
            self.assertNotIn("/home/ktyser", provenance)

    def test_public_quantized_checkpoint_requires_spark_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            out = root / "out"

            plan = build_model_plan(
                model_args(
                    artifact_path=str(model_dir),
                    output_dir=str(out),
                    release_class="public_quantized_model",
                    validation_state="planned",
                    source_license_checked=True,
                    eval_results=str(root / "eval.json"),
                    quantization_card=str(root / "quantization.json"),
                    serving_card=str(root / "serving.json"),
                    promotion_report=str(root / "promotion.json"),
                )
            )

            gate = next(item for item in plan["release_gates"] if item["name"] == "public_checkpoint_release_allowed")
            self.assertTrue(plan["blocked"])
            self.assertEqual(gate["status"], "fail")
            self.assertIn("Spark validation", gate["message"])

    def test_scanner_catches_secret_like_literals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("token=hf_" + "B" * 32, encoding="utf-8")

            findings = scan_text_file(path)

            self.assertEqual(len(findings), 1)
            self.assertIn("secret-like literal", findings[0])
            self.assertNotIn(str(path), findings[0])

    def test_blocked_publish_model_dry_run_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "publish-model",
                        "gemma4_26b_a4b",
                        "base",
                        "--release-class",
                        "public_quantized_model",
                        "--artifact-path",
                        str(model_dir),
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )

            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
