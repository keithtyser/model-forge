from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_forge.variants.graph import ancestry, variant_graph
from model_forge.variants.manifest import default_variant_node, load_family, node_output_path, validate_family_config, validate_variant_node, write_variant_node
from model_forge.variants.tokenizer_audit import build_tokenizer_audit


def write_tokenizer_fixture(path: Path, *, chat_template: str = "{{ messages }}", eos_token: str = "<eos>") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel"}}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": eos_token,
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "chat_template": chat_template,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": eos_token, "unk_token": "<unk>", "pad_token": "<pad>"}, sort_keys=True),
        encoding="utf-8",
    )


class VariantGraphTests(unittest.TestCase):
    def test_variant_graph_derives_edges_from_family_config(self) -> None:
        graph = variant_graph("gemma4_26b_a4b")
        self.assertGreaterEqual(graph["node_count"], 10)
        targets = {edge["target"]: edge for edge in graph["edges"]}
        self.assertEqual(targets["local_ft"]["source"], "base")
        self.assertEqual(targets["base_nvfp4_modelopt"]["transform"]["type"], "quantize")
        self.assertEqual(
            ancestry(graph, "ft_local_abli_sota_internal_r7_selected_t34_transfer_nvfp4_modelopt"),
            ["base", "local_ft", "ft_local_abli_sota_internal_r7_selected_t34_transfer", "ft_local_abli_sota_internal_r7_selected_t34_transfer_nvfp4_modelopt"],
        )

    def test_qwen_family_configs_are_graph_ready(self) -> None:
        qwen35 = variant_graph("qwen35_9b")
        self.assertEqual(qwen35["node_count"], 4)
        self.assertEqual(ancestry(qwen35, "local_ft_abli"), ["base", "local_ft", "local_ft_abli"])
        targets = {edge["target"]: edge for edge in qwen35["edges"]}
        self.assertEqual(targets["local_abli"]["transform"]["type"], "behavior_edit")
        self.assertEqual(targets["local_ft"]["transform"]["type"], "fine_tune")

        qwen36 = variant_graph("qwen36_27b")
        self.assertEqual(qwen36["node_count"], 4)
        self.assertEqual(ancestry(qwen36, "local_abli"), ["base", "local_abli"])

    def test_family_config_validation_requires_source_edges_for_derived_variants(self) -> None:
        family = load_family("qwen35_9b")
        self.assertEqual(validate_family_config(family), [])
        broken = dict(family)
        broken["variants"] = dict(family["variants"])
        broken["variants"]["local_abli"] = dict(broken["variants"]["local_abli"])
        broken["variants"]["local_abli"].pop("base_variant")
        errors = validate_family_config(broken)
        self.assertTrue(any("local_abli.base_variant is required" in error for error in errors))

    def test_variant_node_contains_validation_evidence_and_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "scores.csv"
            artifact.write_text("metric,value\nx,1\n", encoding="utf-8")
            node = default_variant_node(
                "gemma4_26b_a4b",
                "local_ft",
                implementation_status="implemented",
                validation_state="spark_single_node_validated",
                promotion_decision="inconclusive",
                command="./forge eval gemma4_26b_a4b local_ft --internal",
                spark_evidence_path="reports/generated/example",
                node_count=1,
                hardware_profile="dgx_spark",
                artifacts=[artifact],
                metrics={"primary_metric": "challenge_capability"},
                retention_decision="keep_private",
            )

        self.assertEqual(validate_variant_node(node), [])
        self.assertEqual(node["validation"]["validation_state"], "spark_single_node_validated")
        self.assertEqual(node["validation"]["metrics"]["primary_metric"], "challenge_capability")
        self.assertEqual(node["retention"]["publish_or_delete_decision"], "keep_private")
        self.assertTrue(any(entry.get("sha256", "").startswith("sha256:") for entry in node["checkpoint"]["artifacts"]))

    def test_write_variant_node_round_trips(self) -> None:
        node = default_variant_node("gemma4_26b_a4b", "base")
        with tempfile.TemporaryDirectory() as tmp:
            path = node_output_path("gemma4_26b_a4b", "base", Path(tmp))
            write_variant_node(node, path)
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "model_forge.variant_node.v1")
        self.assertEqual(validate_variant_node(data), [])

    def test_invalid_variant_node_reports_errors(self) -> None:
        errors = validate_variant_node({"schema_version": "wrong"})
        self.assertIn("schema_version must be model_forge.variant_node.v1", errors)
        self.assertTrue(any("validation" in error for error in errors))

    def test_tokenizer_audit_passes_when_derived_variant_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_tokenizer_fixture(models / "gemma-4-26B-A4B-it")
            write_tokenizer_fixture(models / "gemma-4-26B-A4B-it-local-abliterated-v3")
            audit = build_tokenizer_audit(
                "gemma4_26b_a4b",
                variant="local_abli",
                models_dir_override=str(models),
                strict=True,
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["findings"], [])

    def test_tokenizer_audit_reports_chat_template_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_tokenizer_fixture(models / "gemma-4-26B-A4B-it", chat_template="{{ messages }}")
            write_tokenizer_fixture(models / "gemma-4-26B-A4B-it-local-abliterated-v3", chat_template="{{ broken }}")
            audit = build_tokenizer_audit(
                "gemma4_26b_a4b",
                variant="local_abli",
                models_dir_override=str(models),
                strict=True,
            )
        self.assertFalse(audit["passed"])
        self.assertTrue(
            any(finding["check"] == "tokenizer_preservation" and "chat_template_sha256" in finding["message"] for finding in audit["findings"])
        )

    def test_tokenizer_audit_non_strict_allows_missing_local_dirs_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = build_tokenizer_audit(
                "gemma4_26b_a4b",
                variant="local_abli",
                models_dir_override=tmp,
                strict=False,
            )
        self.assertTrue(audit["passed"])
        self.assertTrue(any(finding["level"] == "warning" for finding in audit["findings"]))

    def test_tokenizer_audit_strict_fails_skipped_live_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_tokenizer_fixture(models / "gemma-4-26B-A4B-it")
            with patch(
                "model_forge.variants.tokenizer_audit.live_round_trip",
                return_value={"status": "skipped", "reason": "fixture unavailable"},
            ):
                audit = build_tokenizer_audit(
                    "gemma4_26b_a4b",
                    variant="base",
                    models_dir_override=str(models),
                    load_tokenizer=True,
                    strict=True,
                )
        self.assertFalse(audit["passed"])
        self.assertTrue(any(finding["level"] == "error" and finding["check"] == "round_trip" for finding in audit["findings"]))


if __name__ == "__main__":
    unittest.main()
