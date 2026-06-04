from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_forge.variants.architecture_audit import audit_architecture_metadata, build_architecture_audit
from model_forge.variants.checkpoint_audit import build_checkpoint_audit
from model_forge.variants.cli import checkpoint_progress, format_bytes, format_checkpoint_timestamp
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


def write_model_config_fixture(path: Path, **overrides: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 32768,
    }
    config.update(overrides)
    (path / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")


def write_fast_tokenizer_fixture(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<unk>": 0,
        "deployment": 1,
        "risk": 2,
        "two": 3,
        "checks": 4,
        "You": 5,
        "are": 6,
        "validating": 7,
        "tokenizer": 8,
        "behavior": 9,
        "Explain": 10,
        "one": 11,
        "and": 12,
        "give": 13,
        ".": 14,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "chat_template": "{% for message in messages %}{{ message.content }} {% endfor %}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(json.dumps({"unk_token": "<unk>"}, sort_keys=True), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"model_type": "future_unknown_arch"}, sort_keys=True), encoding="utf-8")


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
        self.assertEqual(qwen36["node_count"], 33)
        self.assertEqual(ancestry(qwen36, "local_abli"), ["base", "local_abli"])
        self.assertEqual(ancestry(qwen36, "local_ft_v5"), ["base", "local_ft_v5"])
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_gemma_t34_scale1p5"),
            ["base", "local_ft_v4", "local_ft_abli_gemma_t34_scale1p5"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_trial2_scale0p75"),
            ["base", "local_ft_v4", "local_ft_abli_trial2_scale0p75"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v3"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v3",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v4"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v4",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v6"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v6",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v7"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v7",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v8"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v8",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v9_probe"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v9_probe",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v10_eval_repair"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2",
                "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v10_eval_repair",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_trial2_scale1p0"),
            ["base", "local_ft_v4", "local_ft_abli_trial2_scale1p0"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_trial2_lora_scale0p85"),
            ["base", "local_ft_v4", "local_ft_abli_trial2_lora_scale0p85"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_residual_trial12"),
            ["base", "local_ft_v4", "local_ft_abli_heretic_residual_trial12"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_unsafe_followup_trial16"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_unsafe_followup_trial16",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_refusal_unlikelihood_v1"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_refusal_unlikelihood_v1",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_heretic_trial12_response_conditioned_trial19_aggressive"),
            [
                "base",
                "local_ft_v4",
                "local_ft_abli_heretic_residual_trial12",
                "local_ft_abli_heretic_trial12_response_conditioned_trial19_aggressive",
            ],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_behavior_v1"),
            ["base", "local_ft_v4", "local_ft_abli_behavior_v1"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_refusal_unlikelihood_v2"),
            ["base", "local_ft_v4", "local_ft_abli_refusal_unlikelihood_v2"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_pairwise_preference_v1"),
            ["base", "local_ft_v4", "local_ft_abli_pairwise_preference_v1"],
        )
        self.assertEqual(
            ancestry(qwen36, "local_ft_abli_nvfp4_modelopt"),
            ["base", "local_ft_v4", "local_ft_abli", "local_ft_abli_nvfp4_modelopt"],
        )

    def test_llama_family_config_is_graph_ready(self) -> None:
        graph = variant_graph("llama31_8b")
        self.assertEqual(graph["node_count"], 5)
        self.assertEqual(ancestry(graph, "local_ft_abli"), ["base", "local_ft", "local_ft_abli"])
        targets = {edge["target"]: edge for edge in graph["edges"]}
        self.assertEqual(targets["local_abli"]["transform"]["type"], "behavior_edit")
        self.assertEqual(targets["base_nvfp4_blackwell_runtime"]["transform"]["type"], "quantize")
        self.assertEqual(validate_family_config(load_family("llama31_8b")), [])

    def test_family_config_validation_requires_source_edges_for_derived_variants(self) -> None:
        family = load_family("qwen35_9b")
        self.assertEqual(validate_family_config(family), [])
        broken = dict(family)
        broken["variants"] = dict(family["variants"])
        broken["variants"]["local_abli"] = dict(broken["variants"]["local_abli"])
        broken["variants"]["local_abli"].pop("base_variant")
        errors = validate_family_config(broken)
        self.assertTrue(any("local_abli.base_variant is required" in error for error in errors))

    def test_family_config_validation_checks_promotion_metadata(self) -> None:
        family = load_family("qwen35_9b")
        broken = dict(family)
        broken["variants"] = dict(family["variants"])
        broken["variants"]["local_ft_abli"] = dict(broken["variants"]["local_ft_abli"])
        broken["variants"]["local_ft_abli"]["promotion"] = {
            "decision": "maybe",
            "blocked_actions": ["ship_it"],
        }

        errors = validate_family_config(broken)

        self.assertTrue(any("promotion.decision" in error for error in errors))
        self.assertTrue(any("blocked_actions contains unsupported actions" in error for error in errors))
        self.assertTrue(any("promotion.reason is required" in error for error in errors))

    def test_family_config_validation_requires_architecture_metadata(self) -> None:
        family = load_family("qwen35_9b")
        broken = dict(family)
        broken.pop("architecture")
        errors = validate_family_config(broken)
        self.assertTrue(any("architecture.family is required" in error for error in errors))

    def test_architecture_audit_passes_configured_qwen_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_model_config_fixture(models / "Qwen3.5-9B")
            audit = build_architecture_audit("qwen35_9b", models_dir_override=str(models), strict=True)
        self.assertTrue(audit["passed"], audit["findings"])
        self.assertEqual(audit["records"][0]["model_config"]["model_type"], "qwen3")

    def test_architecture_audit_flags_moe_without_router_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_model_config_fixture(models / "Qwen3.5-9B", num_experts=8)
            audit = build_architecture_audit("qwen35_9b", models_dir_override=str(models), strict=True)
        self.assertTrue(audit["passed"], audit["findings"])

        family = load_family("qwen35_9b")
        target = dict(family["architecture"]["target_discovery"])
        target["edit_exclusion_patterns"] = ["embed_tokens", "lm_head"]
        broken = dict(family)
        broken["architecture"] = dict(family["architecture"])
        broken["architecture"]["target_discovery"] = target
        findings = audit_architecture_metadata(broken)
        self.assertTrue(any(finding.check == "edit_exclusion_patterns" for finding in findings))

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

    def test_variant_node_inherits_family_promotion_metadata(self) -> None:
        node = default_variant_node("qwen36_27b", "local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v6")

        self.assertEqual(node["validation"]["promotion_decision"], "rejected")
        self.assertEqual(node["promotion"]["decision"], "rejected")
        self.assertIn("quantization_export", node["promotion"]["blocked_actions"])
        self.assertIn("reports/qwen36_27b_trial12_pref_ul_v6_summary.md", node["promotion"]["evidence"])

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
        self.assertEqual({record["variant"] for record in audit["records"]}, {"base", "local_abli"})

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

    def test_tokenizer_audit_falls_back_to_tokenizer_only_load_for_new_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            write_fast_tokenizer_fixture(models / "Qwen3.5-9B")
            audit = build_tokenizer_audit(
                "qwen35_9b",
                variant="base",
                models_dir_override=str(models),
                load_tokenizer=True,
                strict=True,
            )
        self.assertTrue(audit["passed"], audit["findings"])
        self.assertEqual(audit["records"][0]["round_trip"]["status"], "passed")

    def test_checkpoint_audit_flags_incomplete_base_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            checkpoint = models / "Qwen3.5-9B"
            write_tokenizer_fixture(checkpoint)
            write_model_config_fixture(checkpoint)
            (checkpoint / "generation_config.json").write_text(json.dumps({}), encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 100}, "weight_map": {"a.weight": "model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            cache = checkpoint / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "model-00001-of-00002.safetensors.lock").write_text("", encoding="utf-8")
            (cache / "partial.incomplete").write_text("partial", encoding="utf-8")
            audit = build_checkpoint_audit(
                "qwen35_9b",
                variant="base",
                models_dir_override=str(models),
                strict=True,
            )
        self.assertFalse(audit["passed"])
        checks = {finding["check"] for finding in audit["findings"]}
        self.assertIn("weights", checks)
        self.assertIn("download", checks)
        self.assertEqual(audit["records"][0]["incomplete_download_bytes"], 7)
        self.assertIn("incomplete_download_latest_mtime", audit["records"][0])
        self.assertEqual(audit["records"][0]["largest_incomplete_download"]["bytes"], 7)
        self.assertAlmostEqual(audit["records"][0]["download_progress_fraction"], 0.07)

    def test_checkpoint_audit_accepts_indexed_safetensor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            checkpoint = models / "Qwen3.5-9B"
            write_tokenizer_fixture(checkpoint)
            write_model_config_fixture(checkpoint)
            (checkpoint / "generation_config.json").write_text(json.dumps({}), encoding="utf-8")
            for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
                (checkpoint / shard).write_text("placeholder", encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 22},
                        "weight_map": {
                            "a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            audit = build_checkpoint_audit(
                "qwen35_9b",
                variant="base",
                models_dir_override=str(models),
                strict=True,
            )
        self.assertTrue(audit["passed"], audit["findings"])
        self.assertEqual(audit["records"][0]["safetensors"]["shard_count"], 2)

    def test_checkpoint_audit_accepts_adapter_plus_merged_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            adapter = models / "Qwen3.6-27B-local-ft-v1"
            merged = models / "Qwen3.6-27B-local-ft-v1-merged"
            write_tokenizer_fixture(adapter)
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
            write_tokenizer_fixture(merged)
            write_model_config_fixture(merged)
            (merged / "generation_config.json").write_text(json.dumps({}), encoding="utf-8")
            (merged / "model.safetensors").write_text("weights", encoding="utf-8")
            audit = build_checkpoint_audit(
                "qwen36_27b",
                variant="local_ft",
                models_dir_override=str(models),
                strict=True,
            )
        self.assertTrue(audit["passed"], audit["findings"])
        self.assertEqual(audit["records"][0]["path"], str(adapter))
        self.assertEqual(audit["records"][0]["merged_checkpoint"]["safetensors"]["shard_count"], 1)

    def test_checkpoint_progress_uses_lowest_variant_progress(self) -> None:
        audit = {
            "records": [
                {"download_progress_fraction": 0.8},
                {"merged_checkpoint": {"download_progress_fraction": 0.45}},
            ]
        }
        self.assertEqual(checkpoint_progress(audit), 0.45)

    def test_checkpoint_audit_display_helpers_are_compact(self) -> None:
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(1536), "1.5 KiB")
        self.assertEqual(format_bytes(33810262880), "31.5 GiB")
        self.assertEqual(format_checkpoint_timestamp("2026-05-31T08:45:27.781590+00:00"), "05-31 08:45Z")


if __name__ == "__main__":
    unittest.main()
