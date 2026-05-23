from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_forge.variants.graph import ancestry, variant_graph
from model_forge.variants.manifest import default_variant_node, node_output_path, validate_variant_node, write_variant_node


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


if __name__ == "__main__":
    unittest.main()
