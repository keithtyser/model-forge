from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from model_forge.hub.cli import (
    DATASET_PLAN_SCHEMA_VERSION,
    audit_release_classes,
    build_dataset_plan,
    build_model_plan,
    execute_model_publish,
    full_eval_gate_result,
    full_eval_summary_lines,
    hf_status,
    main,
    scan_text_file,
)


def model_args(**overrides: object) -> Namespace:
    defaults = {
        "family": "gemma4_26b_a4b",
        "variant": "base",
        "repo_id": None,
        "publish_slug": None,
        "release_class": "report_only",
        "artifact_path": None,
        "validation_state": "planned",
        "eval_results": None,
        "full_eval_results": None,
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


def write_full_eval_run(root: Path, *, variant: str, total_cases: int = 93) -> Path:
    run_dir = root / f"{variant}_full_eval"
    run_dir.mkdir()
    (run_dir / "scores.csv").write_text(
        "\n".join(
            [
                "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev",
                "agentic_tool_use_json,schema_adherence,1.0,3,3,0,0.4385,1.0,0.0",
                "capability_preservation_challenge,normal_use_regression_pass_rate,0.9375,32,30,2,0.7985,0.9827,0.2421",
                "refusal_paired_boundary,benign_answer_quality_rate,0.95,20,19,1,0.7639,0.9911,0.2179",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "total_cases": total_cases,
                "trials": 1,
                "dry_run": False,
                "scoring_version": "model_forge.internal_eval_scoring.v13",
                "canonical": {
                    "run_id": f"{variant}_eval_test",
                    "identity": {"variant": variant},
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


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

    def test_public_model_plan_uses_serving_eval_scores_file_for_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            eval_dir = root / "serving_eval"
            eval_dir.mkdir()
            (eval_dir / "manifest.json").write_text(
                json.dumps({"private_path": "/" + "home" + "/ktyser/private/run"}),
                encoding="utf-8",
            )
            (eval_dir / "scores.csv").write_text("bucket,metric,value\nnormal,pass,1.0\n", encoding="utf-8")
            full_eval = write_full_eval_run(root, variant="base")
            quantization_card = root / "quantization.json"
            quantization_card.write_text(
                json.dumps(
                    {
                        "serving_deltas": {
                            "output_tokens_per_second_p50": {"source": 5.0, "candidate": 10.0},
                            "decode_heavy_output_tokens_per_second_p50": {"source": 4.0, "candidate": 9.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            promotion_report = root / "promotion.json"
            promotion_report.write_text(
                json.dumps(
                    {
                        "nvfp4_ready": True,
                        "export_plan": "/" + "home" + "/ktyser/models/private/quantization_export_plan.json",
                        "metrics": {
                            "output_tokens_per_second_speedup": 2.0,
                            "decode_heavy_output_tokens_per_second_speedup": 2.25,
                        },
                    }
                ),
                encoding="utf-8",
            )

            plan = build_model_plan(
                model_args(
                    artifact_path=str(model_dir),
                    output_dir=str(root / "out"),
                    release_class="public_quantized_model",
                    validation_state="spark_cluster_validated",
                    source_license_checked=True,
                    eval_results=str(eval_dir),
                    full_eval_results=str(full_eval),
                    quantization_card=str(quantization_card),
                    serving_card=str(root / "serving.json"),
                    promotion_report=str(promotion_report),
                )
            )

            gates = {gate["name"]: gate for gate in plan["release_gates"]}
            self.assertFalse(plan["blocked"], plan["blocked_until"])
            self.assertEqual(gates["no_private_tokens_or_paths"]["status"], "pass")
            self.assertEqual(plan["supporting_paths"][0], "<external>/scores.csv")
            self.assertEqual(plan["supporting_path_rewrites"][0]["kind"], "eval_results")
            self.assertIn("scores.csv", plan["supporting_path_rewrites"][0]["used"])
            self.assertEqual(plan["supporting_path_rewrites"][1]["kind"], "full_eval_results")
            self.assertEqual(plan["supporting_path_rewrites"][2]["kind"], "promotion_report")
            self.assertIn("promotion_report_promotion.json", plan["supporting_path_rewrites"][2]["used"])
            sanitized_promotion = root / "out" / "supporting_evidence" / "promotion_report_promotion.json"
            self.assertTrue(sanitized_promotion.is_file())
            self.assertNotIn("/home/ktyser", sanitized_promotion.read_text(encoding="utf-8"))
            self.assertNotIn("/home/ktyser", json.dumps(plan))
            card = (root / "out" / "README.md").read_text(encoding="utf-8")
            self.assertIn("Eval Results: `<external>/scores.csv`", card)
            self.assertIn("Full Eval Results: `<external>/scores.csv`", card)
            self.assertIn("cases 93", card)
            self.assertIn("output p50 tok/s: source 5.000, candidate 10.00, speedup 2.000x", card)
            self.assertIn("NVFP4 evidence gate ready: True", card)

    def test_model_artifact_plan_blocks_rejected_variant_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")

            plan = build_model_plan(
                model_args(
                    family="qwen36_27b",
                    variant="local_ft_abli_refusal_unlikelihood_v2",
                    artifact_path=str(model_dir),
                    output_dir=str(root / "out"),
                    release_class="private_research_model",
                    validation_state="spark_cluster_validated",
                    source_license_checked=True,
                    eval_results=str(root / "eval.json"),
                    promotion_report=str(root / "promotion.json"),
                    risk_report=str(root / "risk.json"),
                    behavior_edited=True,
                )
            )

            gate = next(item for item in plan["release_gates"] if item["name"] == "variant_promotion_not_blocked")
            self.assertTrue(plan["blocked"])
            self.assertEqual(gate["status"], "fail")
            self.assertIn("Refusal-unlikelihood v2", gate["message"])

    def test_promoted_qwen_nvfp4_variant_public_plan_is_unblocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            eval_scores = root / "scores.csv"
            eval_scores.write_text("bucket,metric,value\nnormal,pass,1.0\n", encoding="utf-8")
            full_eval = write_full_eval_run(
                root,
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
            )
            serving = root / "serving.json"
            serving.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            quantization = root / "quantization.json"
            quantization.write_text(
                json.dumps(
                    {
                        "serving_deltas": {
                            "output_tokens_per_second_p50": {"source": 5.0, "candidate": 9.0}
                        }
                    }
                ),
                encoding="utf-8",
            )
            promotion = root / "promotion.json"
            promotion.write_text(json.dumps({"nvfp4_ready": True}), encoding="utf-8")

            plan = build_model_plan(
                model_args(
                    family="qwen36_27b",
                    variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                    artifact_path=str(model_dir),
                    output_dir=str(root / "out"),
                    release_class="public_quantized_model",
                    validation_state="spark_cluster_validated",
                    source_license_checked=True,
                    eval_results=str(eval_scores),
                    full_eval_results=str(full_eval),
                    serving_card=str(serving),
                    quantization_card=str(quantization),
                    promotion_report=str(promotion),
                )
            )

            gates = {gate["name"]: gate for gate in plan["release_gates"]}
            self.assertFalse(plan["blocked"], plan["blocked_until"])
            self.assertEqual(plan["repo_id"], "keithtyser/model-forge-qwen36-27b-ft-v4-nvfp4-dgx-spark")
            self.assertEqual(gates["variant_promotion_not_blocked"]["status"], "pass")
            self.assertEqual(gates["no_private_tokens_or_paths"]["status"], "pass")

    def test_publish_slug_overrides_default_repo_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            plan = build_model_plan(
                model_args(
                    family="qwen36_27b",
                    variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                    publish_slug="model-forge-qwen36-27b-ft-v4-nvfp4-custom",
                    output_dir=str(root / "out"),
                )
            )

            self.assertEqual(plan["repo_id"], "keithtyser/model-forge-qwen36-27b-ft-v4-nvfp4-custom")
            self.assertEqual(plan["publish_slug"], "model-forge-qwen36-27b-ft-v4-nvfp4-custom")

    def test_public_quantized_plan_blocks_sampled_eval_as_full_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            sampled_eval = write_full_eval_run(
                root,
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                total_cases=11,
            )
            eval_scores = root / "scores.csv"
            eval_scores.write_text("bucket,metric,value\nnormal,pass,1.0\n", encoding="utf-8")
            serving = root / "serving.json"
            serving.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            quantization = root / "quantization.json"
            quantization.write_text(json.dumps({"serving_deltas": {}}), encoding="utf-8")
            promotion = root / "promotion.json"
            promotion.write_text(json.dumps({"nvfp4_ready": True}), encoding="utf-8")

            plan = build_model_plan(
                model_args(
                    family="qwen36_27b",
                    variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                    artifact_path=str(model_dir),
                    output_dir=str(root / "out"),
                    release_class="public_quantized_model",
                    validation_state="spark_cluster_validated",
                    source_license_checked=True,
                    eval_results=str(eval_scores),
                    full_eval_results=str(sampled_eval),
                    serving_card=str(serving),
                    quantization_card=str(quantization),
                    promotion_report=str(promotion),
                )
            )

            gate = next(item for item in plan["release_gates"] if item["name"] == "full_eval_results_present")
            self.assertTrue(plan["blocked"])
            self.assertEqual(gate["status"], "fail")
            self.assertIn("requires at least 80", gate["message"])

    def test_full_eval_gate_prefers_canonical_variant_over_legacy_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_full_eval_run(root, variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt")
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["variant"] = "base"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            passed, message = full_eval_gate_result(
                run_dir,
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                minimum_cases=80,
            )
            summary = full_eval_summary_lines(run_dir)

            self.assertTrue(passed, message)
            self.assertIn("local_ft_v4_nvfp4_attention_output_bf16_modelopt", message)
            self.assertIn("variant local_ft_v4_nvfp4_attention_output_bf16_modelopt", summary[0])

    def test_release_class_audit_has_no_errors(self) -> None:
        findings = audit_release_classes()
        errors = [finding for finding in findings if finding.severity == "error"]
        self.assertEqual(errors, [])

    def test_public_behavior_edited_release_requires_risk_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_text("placeholder", encoding="utf-8")

            plan = build_model_plan(
                model_args(
                    variant="local_ft",
                    artifact_path=str(adapter_dir),
                    output_dir=str(root / "out"),
                    release_class="public_adapter",
                    validation_state="smoke_validated",
                    source_license_checked=True,
                    eval_results=str(root / "eval.json"),
                    behavior_edited=True,
                )
            )

            gate = next(item for item in plan["release_gates"] if item["name"] == "behavior_edit_risk_report")
            self.assertTrue(plan["blocked"])
            self.assertEqual(gate["status"], "fail")

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

    def test_publish_model_execute_requires_token_after_passing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            eval_scores = root / "scores.csv"
            eval_scores.write_text("bucket,metric,value\nnormal,pass,1.0\n", encoding="utf-8")
            full_eval = write_full_eval_run(
                root,
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
            )
            serving = root / "serving.json"
            serving.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            quantization = root / "quantization.json"
            quantization.write_text(json.dumps({"serving_deltas": {}}), encoding="utf-8")
            promotion = root / "promotion.json"
            promotion.write_text(json.dumps({"nvfp4_ready": True}), encoding="utf-8")

            with patch("model_forge.hub.cli.token_value", return_value=None) as token_mock, redirect_stdout(StringIO()):
                code = main(
                    [
                        "publish-model",
                        "qwen36_27b",
                        "local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                        "--release-class",
                        "public_quantized_model",
                        "--artifact-path",
                        str(model_dir),
                        "--validation-state",
                        "spark_cluster_validated",
                        "--source-license-checked",
                        "--eval-results",
                        str(eval_scores),
                        "--full-eval-results",
                        str(full_eval),
                        "--serving-card",
                        str(serving),
                        "--quantization-card",
                        str(quantization),
                        "--promotion-report",
                        str(promotion),
                        "--output-dir",
                        str(root / "out"),
                        "--execute",
                        "--json",
                    ]
                )

            self.assertEqual(code, 1)
            token_mock.assert_called_with(include_cached=False)

    def test_execute_model_publish_uploads_whitelisted_model_card_and_evidence(self) -> None:
        uploads: list[dict[str, object]] = []
        created: list[dict[str, object]] = []

        class FakeApi:
            def __init__(self, token: str) -> None:
                self.token = token

            def whoami(self, token: str) -> dict[str, str]:
                return {"name": "tester"}

            def upload_file(self, **kwargs: object) -> None:
                uploads.append(kwargs)

        def fake_create_repo(**kwargs: object) -> None:
            created.append(kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("placeholder", encoding="utf-8")
            (model_dir / "scratch.tmp").write_text("do not upload", encoding="utf-8")
            eval_scores = root / "scores.csv"
            eval_scores.write_text("bucket,metric,value\nnormal,pass,1.0\n", encoding="utf-8")
            full_eval = write_full_eval_run(
                root,
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
            )
            serving = root / "serving.json"
            serving.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            quantization = root / "quantization.json"
            quantization.write_text(json.dumps({"serving_deltas": {}}), encoding="utf-8")
            promotion = root / "promotion.json"
            promotion.write_text(json.dumps({"nvfp4_ready": True}), encoding="utf-8")
            args = model_args(
                family="qwen36_27b",
                variant="local_ft_v4_nvfp4_attention_output_bf16_modelopt",
                artifact_path=str(model_dir),
                output_dir=str(root / "out"),
                release_class="public_quantized_model",
                validation_state="spark_cluster_validated",
                source_license_checked=True,
                eval_results=str(eval_scores),
                full_eval_results=str(full_eval),
                serving_card=str(serving),
                quantization_card=str(quantization),
                promotion_report=str(promotion),
                evidence_prefix="evidence",
                revision="main",
                commit_message="Upload test artifact",
            )
            plan = build_model_plan(args)

            publish_record = execute_model_publish(
                args,
                plan,
                token="test-token",
                api_factory=FakeApi,
                create_repo_fn=fake_create_repo,
            )

            uploaded_repo_paths = {str(item["path_in_repo"]) for item in uploads}
            self.assertFalse(publish_record["dry_run"])
            self.assertEqual(created[0]["repo_id"], plan["repo_id"])
            self.assertFalse(created[0]["private"])
            self.assertIn("README.md", uploaded_repo_paths)
            self.assertIn("config.json", uploaded_repo_paths)
            self.assertIn("model.safetensors", uploaded_repo_paths)
            self.assertIn("evidence/eval_results/scores.csv", uploaded_repo_paths)
            self.assertIn("evidence/full_eval_results/scores.csv", uploaded_repo_paths)
            self.assertIn("evidence/full_eval_manifest/manifest.json", uploaded_repo_paths)
            self.assertIn("evidence/hub_publish.json", uploaded_repo_paths)
            self.assertNotIn("scratch.tmp", uploaded_repo_paths)
            provenance = (root / "out" / "hub_publish.json").read_text(encoding="utf-8")
            self.assertNotIn("test-token", provenance)

    def test_publish_dataset_dry_run_accepts_redacted_public_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "README.md").write_text(
                "\n".join([
                    "# dataset",
                    "license: cc-by-4.0",
                    "## Purpose",
                    "test dataset",
                    "## Counts",
                    "one row",
                    "## Provenance",
                    "generated by model-forge tests",
                    "## Safety And Contamination",
                    "redacted",
                ]),
                encoding="utf-8",
            )
            (bundle / "dataset_redacted.jsonl").write_text(
                json.dumps({"messages": [{"role": "user", "content": "<redacted>"}]}) + "\n",
                encoding="utf-8",
            )
            (bundle / "redaction_report.json").write_text(
                json.dumps({"raw_message_content_published": False}),
                encoding="utf-8",
            )

            plan = build_dataset_plan(
                Namespace(
                    dataset_path=str(bundle),
                    repo_id="keithtyser/test-dataset",
                    release_class="public_dataset",
                    split="train",
                    card_template="dataset",
                    visibility="public",
                    include_raw_outputs=False,
                    output_dir=str(root / "out"),
                    run_id="dataset_plan",
                    json=True,
                )
            )

            self.assertEqual(plan["schema_version"], DATASET_PLAN_SCHEMA_VERSION)
            self.assertFalse(plan["blocked"], plan["blocked_until"])
            self.assertEqual(plan["row_counts"]["<external>/dataset_redacted.jsonl"], 1)
            gates = {gate["name"]: gate for gate in plan["release_gates"]}
            self.assertEqual(gates["unsafe_examples_redacted_or_private"]["status"], "pass")
            self.assertTrue((root / "out" / "hub_dataset_plan.json").exists())

    def test_publish_dataset_dry_run_blocks_raw_public_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "raw.jsonl"
            dataset.write_text(json.dumps({"messages": [{"role": "user", "content": "raw"}]}) + "\n", encoding="utf-8")

            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "publish-dataset",
                        str(dataset),
                        "--repo-id",
                        "keithtyser/raw-dataset",
                        "--visibility",
                        "public",
                        "--include-raw-outputs",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )

            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
