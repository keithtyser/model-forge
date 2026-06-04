from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from model_forge.quantization.cli import (
    BEHAVIOR_REPORT_SCHEMA_VERSION,
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    CARD_SCHEMA_VERSION,
    FP8_KV_REPORT_SCHEMA_VERSION,
    NVFP4_GATE_SCHEMA_VERSION,
    SENSITIVITY_REPORT_SCHEMA_VERSION,
    TOKENIZER_REPORT_SCHEMA_VERSION,
    build_card,
    build_behavior_report,
    build_calibration_manifest,
    build_fp8_kv_report,
    build_export_command,
    build_gguf_export_command,
    build_modelopt_export_command,
    build_nvfp4_gate_report,
    build_sensitivity_report,
    build_tokenizer_report,
    guard_export,
    build_plan,
    filter_matrix_entries,
    load_quantization_config,
    matrix_entries,
    matrix_workers,
    resolve_source,
)


def write_tokenizer_fixture(path: Path, *, chat_template: str = "{{ messages }}", eos_token: str = "<eos>") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel"}}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": eos_token,
                "pad_token": "<pad>",
                "chat_template": chat_template,
            }
        ),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(json.dumps({"eos_token": eos_token}), encoding="utf-8")


def write_serving_summary(path: Path, *, model: str, success_rate: float = 1.0, output_tps: float = 10.0, decode_heavy_tps: float = 10.0, latency: float = 5.0) -> None:
    path.write_text(
        json.dumps(
            {
                "model": model,
                "success_rate": success_rate,
                "metrics": {
                    "output_tokens_per_second": {"p50": output_tps},
                    "total_latency_seconds": {"p50": latency},
                },
                "by_category": {
                    "decode_heavy": {
                        "metrics": {
                            "output_tokens_per_second": {"p50": decode_heavy_tps},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_behavior_scores(path: Path, *, normal: float = 1.0, challenge: float = 0.9, schema: float = 1.0, workflow: float = 1.0, benign_quality: float = 0.9) -> None:
    path.mkdir(parents=True, exist_ok=True)
    header = "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
    rows = (
        f"normal_use_regression,normal_use_regression_pass_rate,{normal},4,4,0,0.6,1.0,0\n"
        f"capability_preservation_challenge,normal_use_regression_pass_rate,{challenge},4,4,0,0.5,1.0,0\n"
        f"agentic_tool_use_json,schema_adherence,{schema},4,4,0,0.6,1.0,0\n"
        f"agentic_tool_use_json,workflow_success,{workflow},4,4,0,0.6,1.0,0\n"
        f"refusal_paired_boundary,benign_answer_quality_rate,{benign_quality},4,4,0,0.5,1.0,0\n"
    )
    (path / "scores.csv").write_text(header + rows, encoding="utf-8")


class QuantizationCliTests(unittest.TestCase):
    def test_nvfp4_blackwell_plan_is_env_backed_and_runtime_only(self) -> None:
        config_path = Path("configs/quantization/nvfp4_blackwell_runtime.yaml")
        config = load_quantization_config(config_path)
        plan = build_plan(
            config,
            config_path=config_path,
            family=None,
            variant=None,
            output_dir=None,
            run_id="unit_nvfp4_plan",
            env={"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"},
        )

        self.assertEqual(plan["quantization"]["method"], "nvfp4_runtime")
        self.assertFalse(plan["target"]["checkpoint_written_by_this_plan"])
        self.assertEqual(plan["source"]["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(plan["target"]["variant"], "base_nvfp4_blackwell_runtime")
        self.assertEqual(plan["runtime"]["model_id"], "nvidia/Llama-3.1-8B-Instruct-NVFP4")
        launch = " ".join(plan["launch_command"])
        self.assertIn("vllm serve nvidia/Llama-3.1-8B-Instruct-NVFP4", launch)
        self.assertIn("${MODEL_FORGE_SPARK_CLUSTER_NODES", launch)
        self.assertIn("vllm-node-tf5", launch)
        self.assertIn("--quantization modelopt", launch)
        self.assertNotIn("169.254.", launch)

    def test_export_guard_requires_complete_family_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_plan = {
                "source": {"family": "qwen36_27b", "variant": "local_ft_abli"},
                "target": {"host_output_path": str(Path(tmp) / "candidate")},
                "resource_policy": {
                    "start_if_memory_available_above_fraction": 0.0,
                    "require_disk_free_fraction": 0.0,
                },
            }
            with self.assertRaisesRegex(RuntimeError, "source checkpoint audit failed"):
                guard_export(export_plan)

    def test_quantization_card_compares_serving_and_sampled_eval_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_summary = root / "source_summary.json"
            candidate_summary = root / "candidate_summary.json"
            source_eval = root / "source_eval"
            candidate_eval = root / "candidate_eval"
            source_eval.mkdir()
            candidate_eval.mkdir()

            source_summary.write_text(
                json.dumps(
                    {
                        "model": "source",
                        "success_rate": 1.0,
                        "request_throughput_per_second_serial_estimate": 0.2,
                        "metrics": {
                            "total_latency_seconds": {"p50": 5.0, "p95": 10.0},
                            "time_to_first_chunk_seconds": {"p50": 0.5, "p95": 1.0},
                            "output_tokens_per_second": {"p50": 10.0, "p95": 12.0},
                            "decode_tokens_per_second": {"p50": 11.0},
                            "total_tokens_per_second": {"p50": 20.0},
                        },
                        "by_category": {
                            "decode_heavy": {
                                "metrics": {
                                    "output_tokens_per_second": {"p50": 9.0},
                                    "decode_tokens_per_second": {"p50": 10.0},
                                },
                            },
                        },
                        "memory": {
                            "system": {"available_fraction": {"min": 0.2}},
                            "gpu": {"utilization_gpu_percent": {"p50": 80.0}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_summary.write_text(
                json.dumps(
                    {
                        "model": "candidate",
                        "success_rate": 1.0,
                        "request_throughput_per_second_serial_estimate": 0.3,
                        "metrics": {
                            "total_latency_seconds": {"p50": 4.0, "p95": 8.0},
                            "time_to_first_chunk_seconds": {"p50": 0.4, "p95": 0.9},
                            "output_tokens_per_second": {"p50": 16.0, "p95": 18.0},
                            "decode_tokens_per_second": {"p50": 17.0},
                            "total_tokens_per_second": {"p50": 28.0},
                        },
                        "by_category": {
                            "decode_heavy": {
                                "metrics": {
                                    "output_tokens_per_second": {"p50": 15.0},
                                    "decode_tokens_per_second": {"p50": 16.0},
                                },
                            },
                        },
                        "memory": {
                            "system": {"available_fraction": {"min": 0.25}},
                            "gpu": {"utilization_gpu_percent": {"p50": 85.0}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            header = "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
            source_rows = header + "normal_use_regression,normal_use_regression_pass_rate,1.0,2,2,0,0.3,1.0,0\n"
            candidate_rows = header + "normal_use_regression,normal_use_regression_pass_rate,0.5,2,1,1,0.1,0.9,0.5\n"
            (source_eval / "scores.csv").write_text(source_rows, encoding="utf-8")
            (candidate_eval / "scores.csv").write_text(candidate_rows, encoding="utf-8")

            config_path = Path("configs/quantization/nvfp4_blackwell_runtime.yaml")
            config = load_quantization_config(config_path)
            card = build_card(
                config,
                config_path=config_path,
                source_serving_summary=source_summary,
                candidate_serving_summary=candidate_summary,
                source_serving_eval=source_eval,
                candidate_serving_eval=candidate_eval,
                output_dir=root / "card",
                run_id="unit_card",
            )

        self.assertEqual(card["serving_deltas"]["throughput_req_per_s"]["delta"], 0.1)
        self.assertEqual(card["serving_deltas"]["output_tokens_per_second_p50"]["delta"], 6.0)
        self.assertEqual(card["serving_deltas"]["decode_heavy_output_tokens_per_second_p50"]["delta"], 6.0)
        self.assertEqual(card["serving_deltas"]["total_latency_p50"]["delta"], -1.0)
        sampled = card["sampled_eval_deltas"]["normal_use_regression.normal_use_regression_pass_rate"]
        self.assertEqual(sampled["delta"], -0.5)
        self.assertEqual(sampled["candidate_count"], 2)

    def test_quantization_card_supports_candidate_only_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_summary = root / "candidate_summary.json"
            candidate_eval = root / "candidate_eval"
            candidate_eval.mkdir()
            candidate_summary.write_text(
                json.dumps(
                    {
                        "model": "candidate",
                        "success_rate": 1.0,
                        "request_throughput_per_second_serial_estimate": 0.3,
                        "metrics": {
                            "total_latency_seconds": {"p50": 4.0, "p95": 8.0},
                            "time_to_first_chunk_seconds": {"p50": 0.4, "p95": 0.9},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (candidate_eval / "scores.csv").write_text(
                "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
                "normal_use_regression,normal_use_regression_pass_rate,1.0,2,2,0,0.3,1.0,0\n",
                encoding="utf-8",
            )

            config_path = Path("configs/quantization/nvfp4_blackwell_runtime.yaml")
            config = load_quantization_config(config_path)
            card = build_card(
                config,
                config_path=config_path,
                source_serving_summary=None,
                candidate_serving_summary=candidate_summary,
                source_serving_eval=None,
                candidate_serving_eval=candidate_eval,
                output_dir=root / "card",
                run_id="unit_candidate_only",
                candidate_only_smoke=True,
            )

        self.assertTrue(card["candidate_only_smoke"])
        self.assertIsNone(card["source"]["serving_summary"])
        self.assertEqual(card["serving_deltas"]["success_rate"]["candidate"], 1.0)
        self.assertIsNone(card["serving_deltas"]["success_rate"]["delta"])
        sampled = card["sampled_eval_deltas"]["normal_use_regression.normal_use_regression_pass_rate"]
        self.assertEqual(sampled["candidate"], 1.0)
        self.assertIsNone(sampled["delta"])

    def test_fp8_kv_report_checks_behavior_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_summary = root / "source_summary.json"
            candidate_summary = root / "candidate_summary.json"
            source_eval = root / "source_eval"
            candidate_eval = root / "candidate_eval"
            source_eval.mkdir()
            candidate_eval.mkdir()
            source_summary.write_text(
                json.dumps({"model": "source", "success_rate": 1.0, "metrics": {"output_tokens_per_second": {"p50": 20.0}}}),
                encoding="utf-8",
            )
            candidate_summary.write_text(
                json.dumps({"model": "candidate", "success_rate": 1.0, "metrics": {"output_tokens_per_second": {"p50": 24.0}}}),
                encoding="utf-8",
            )
            header = "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
            rows = (
                "normal_use_regression,normal_use_regression_pass_rate,1.0,2,2,0,0.3,1.0,0\n"
                "agentic_tool_use_json,schema_adherence,1.0,2,2,0,0.3,1.0,0\n"
                "agentic_tool_use_json,workflow_success,1.0,2,2,0,0.3,1.0,0\n"
            )
            (source_eval / "scores.csv").write_text(header + rows, encoding="utf-8")
            (candidate_eval / "scores.csv").write_text(header + rows, encoding="utf-8")

            config_path = Path("configs/quantization/gemma4_26b_a4b_fp8_runtime.yaml")
            config = load_quantization_config(config_path)
            report = build_fp8_kv_report(
                config,
                config_path=config_path,
                source_serving_summary=source_summary,
                candidate_serving_summary=candidate_summary,
                source_serving_eval=source_eval,
                candidate_serving_eval=candidate_eval,
                output_dir=root / "fp8_kv",
                run_id="unit_fp8_kv",
            )

        self.assertEqual(report["schema_version"], FP8_KV_REPORT_SCHEMA_VERSION)
        self.assertTrue(report["behavior_ready"])
        self.assertEqual(report["kv_cache_dtype"], "fp8")
        self.assertEqual(report["serving_deltas"]["output_tokens_per_second_p50"]["delta"], 4.0)

    def test_behavior_report_applies_quantized_quality_tolerances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_summary = root / "source_summary.json"
            candidate_summary = root / "candidate_summary.json"
            source_eval = root / "source_eval"
            candidate_eval = root / "candidate_eval"
            source_eval.mkdir()
            candidate_eval.mkdir()
            source_summary.write_text(json.dumps({"model": "source", "success_rate": 1.0}), encoding="utf-8")
            candidate_summary.write_text(json.dumps({"model": "candidate", "success_rate": 1.0}), encoding="utf-8")
            header = "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
            source_rows = (
                "normal_use_regression,normal_use_regression_pass_rate,1.0,4,4,0,0.6,1.0,0\n"
                "capability_preservation_challenge,normal_use_regression_pass_rate,0.9,4,4,0,0.5,1.0,0\n"
                "agentic_tool_use_json,schema_adherence,1.0,4,4,0,0.6,1.0,0\n"
                "agentic_tool_use_json,workflow_success,1.0,4,4,0,0.6,1.0,0\n"
                "refusal_paired_boundary,benign_answer_quality_rate,0.9,4,4,0,0.5,1.0,0\n"
                "unsafe_overcompliance,unsafe_overcompliance_rate,0.2,4,1,3,0.0,0.5,0.1\n"
            )
            candidate_rows = (
                "normal_use_regression,normal_use_regression_pass_rate,0.98,4,4,0,0.6,1.0,0\n"
                "capability_preservation_challenge,normal_use_regression_pass_rate,0.86,4,4,0,0.5,1.0,0\n"
                "agentic_tool_use_json,schema_adherence,0.98,4,4,0,0.6,1.0,0\n"
                "agentic_tool_use_json,workflow_success,0.98,4,4,0,0.6,1.0,0\n"
                "refusal_paired_boundary,benign_answer_quality_rate,0.88,4,4,0,0.5,1.0,0\n"
                "unsafe_overcompliance,unsafe_overcompliance_rate,0.4,4,2,2,0.0,0.7,0.1\n"
            )
            (source_eval / "scores.csv").write_text(header + source_rows, encoding="utf-8")
            (candidate_eval / "scores.csv").write_text(header + candidate_rows, encoding="utf-8")

            config_path = Path("configs/quantization/fp8_w8a8_modelopt.yaml")
            config = load_quantization_config(config_path)
            report = build_behavior_report(
                config,
                config_path=config_path,
                source_serving_summary=source_summary,
                candidate_serving_summary=candidate_summary,
                source_serving_eval=source_eval,
                candidate_serving_eval=candidate_eval,
                output_dir=root / "behavior",
                run_id="unit_behavior",
            )

        self.assertEqual(report["schema_version"], BEHAVIOR_REPORT_SCHEMA_VERSION)
        self.assertTrue(report["behavior_preserved"])
        unsafe_check = next(check for check in report["checks"] if check["name"] == "unsafe_overcompliance.unsafe_overcompliance_rate")
        self.assertFalse(unsafe_check["required"])

    def test_behavior_report_fails_required_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_summary = root / "source_summary.json"
            candidate_summary = root / "candidate_summary.json"
            source_eval = root / "source_eval"
            candidate_eval = root / "candidate_eval"
            source_eval.mkdir()
            candidate_eval.mkdir()
            source_summary.write_text(json.dumps({"model": "source", "success_rate": 1.0}), encoding="utf-8")
            candidate_summary.write_text(json.dumps({"model": "candidate", "success_rate": 1.0}), encoding="utf-8")
            header = "bucket,metric,value,count,pass_count,fail_count,ci_low,ci_high,stddev\n"
            (source_eval / "scores.csv").write_text(header + "normal_use_regression,normal_use_regression_pass_rate,1.0,4,4,0,0.6,1.0,0\n", encoding="utf-8")
            (candidate_eval / "scores.csv").write_text(header + "normal_use_regression,normal_use_regression_pass_rate,0.5,4,2,2,0.1,0.9,0.2\n", encoding="utf-8")

            config_path = Path("configs/quantization/fp8_w8a8_modelopt.yaml")
            config = load_quantization_config(config_path)
            report = build_behavior_report(
                config,
                config_path=config_path,
                source_serving_summary=source_summary,
                candidate_serving_summary=candidate_summary,
                source_serving_eval=source_eval,
                candidate_serving_eval=candidate_eval,
                output_dir=root / "behavior",
                run_id="unit_behavior_fail",
            )

        self.assertFalse(report["behavior_preserved"])
        normal_check = next(check for check in report["checks"] if check["name"] == "normal_use_regression.normal_use_regression_pass_rate")
        self.assertEqual(normal_check["status"], "fail")

    def test_tokenizer_report_passes_when_quantized_export_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            write_tokenizer_fixture(source)
            write_tokenizer_fixture(candidate)
            report = build_tokenizer_report(
                source_tokenizer_dir=source,
                candidate_tokenizer_dir=candidate,
                output_dir=root / "report",
                run_id="unit_tokenizer_report",
                source_variant="base",
                candidate_variant="base_fp8_w8a8_modelopt",
                strict=True,
            )

        self.assertEqual(report["schema_version"], TOKENIZER_REPORT_SCHEMA_VERSION)
        self.assertTrue(report["passed"])
        self.assertEqual(report["findings"], [])

    def test_tokenizer_report_fails_chat_template_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            write_tokenizer_fixture(source, chat_template="{{ messages }}")
            write_tokenizer_fixture(candidate, chat_template="{{ broken }}")
            report = build_tokenizer_report(
                source_tokenizer_dir=source,
                candidate_tokenizer_dir=candidate,
                output_dir=root / "report",
                run_id="unit_tokenizer_report_fail",
                source_variant="base",
                candidate_variant="base_gguf_q4",
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(any(finding["check"] == "tokenizer_preservation" for finding in report["findings"]))

    def test_sensitivity_report_ranks_behavior_preserving_component_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_summary = root / "baseline_summary.json"
            mlp_summary = root / "mlp_summary.json"
            all_summary = root / "all_summary.json"
            baseline_eval = root / "baseline_eval"
            mlp_eval = root / "mlp_eval"
            all_eval = root / "all_eval"
            write_serving_summary(baseline_summary, model="source", output_tps=20.0, decode_heavy_tps=20.0)
            write_serving_summary(mlp_summary, model="mlp", output_tps=24.0, decode_heavy_tps=25.0, latency=4.0)
            write_serving_summary(all_summary, model="all", output_tps=30.0, decode_heavy_tps=31.0, latency=3.5)
            write_behavior_scores(baseline_eval)
            write_behavior_scores(mlp_eval, normal=0.99, challenge=0.87, schema=0.99, workflow=0.99, benign_quality=0.89)
            write_behavior_scores(all_eval, normal=0.80, challenge=0.70, schema=0.80, workflow=0.80, benign_quality=0.70)
            config_path = Path("configs/quantization/sensitivity_scan.yaml")
            config = load_quantization_config(config_path)
            report = build_sensitivity_report(
                config,
                config_path=config_path,
                baseline_serving_summary=baseline_summary,
                baseline_serving_eval=baseline_eval,
                candidates=[
                    {"name": "mlp_only", "component": "mlp", "summary": str(mlp_summary), "eval": str(mlp_eval), "precision": "fp8"},
                    {"name": "all_linear", "component": "all_linear", "summary": str(all_summary), "eval": str(all_eval), "precision": "fp8"},
                ],
                output_dir=root / "sensitivity",
                run_id="unit_sensitivity",
            )

        self.assertEqual(report["schema_version"], SENSITIVITY_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["recommended_candidate"], "mlp_only")
        self.assertTrue(report["ranking"][0]["behavior_preserved"])
        self.assertFalse(next(item for item in report["candidates"] if item["name"] == "all_linear")["behavior_preserved"])

    def test_modelopt_export_command_quantizes_local_gemma_variant(self) -> None:
        config_path = Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml")
        config = load_quantization_config(config_path)
        source = resolve_source(
            config,
            "gemma4_26b_a4b",
            "base",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "quantized" / "gemma4_26b_a4b"
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=output_root,
                run_id="unit_base_nvfp4",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )

        command = export["command"]
        joined = " ".join(command)
        self.assertEqual(export["method"], "nvfp4")
        self.assertEqual(export["backend"], "modelopt")
        self.assertEqual(command[:3], ["systemd-run", "--user", "--scope"])
        self.assertIn("CPUQuota=80%", command)
        self.assertIn("MemoryMax=95%", command)
        self.assertIn("model-forge-modelopt-nvfp4:0.43.0", command)
        self.assertIn("--qformat nvfp4", joined)
        self.assertIn("scripts/quantization/gemma4_moe_nvfp4.py", joined)
        self.assertIn("--calib-samples 4096", joined)
        self.assertIn("--batch-size 16", joined)
        self.assertIn(f"{Path.cwd()}:/workspace/model-forge:ro", command)
        self.assertNotIn("--low_memory_mode", command)
        self.assertIn("/models/gemma-4-26B-A4B-it", command)
        self.assertIn(f"{output_root}:/workspace/output_models", command)
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", export["command_display"]))
        self.assertEqual(export["strategy"], "gemma4_moe_modelopt")
        self.assertEqual(export["resource_policy"]["stop_if_memory_available_below_fraction"], 0.05)
        self.assertEqual(export["resource_policy"]["watchdog_poll_seconds"], 2.0)
        self.assertEqual(export["resource_policy"]["systemd_scope"]["MemoryMax"], "95%")
        self.assertTrue(export["resource_policy"]["systemd_scope"]["user"])
        self.assertEqual(export["resource_policy"]["docker"]["memory_gb"], 112)
        self.assertIsNone(export["calibration"]["recipe"])
        self.assertIn("quantization_export.lock", export["resource_policy"]["lock_path"])

    def test_quantization_export_blocks_rejected_source_variant(self) -> None:
        config = load_quantization_config(Path("configs/quantization/qwen36_27b_nvfp4_modelopt.yaml"))
        source = resolve_source(
            config,
            "qwen36_27b",
            "local_ft_abli_refusal_unlikelihood_v2",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        self.assertEqual(source.promotion["decision"], "rejected")
        with tempfile.TemporaryDirectory() as tmp:
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=Path(tmp) / "quantized",
                run_id="unit_blocked_rejected_qwen_variant",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )

        self.assertEqual(export["source"]["promotion"]["decision"], "rejected")
        with self.assertRaisesRegex(RuntimeError, "blocked by variant promotion metadata"):
            guard_export(export)

    def test_qwen_default_nvfp4_source_is_blocked_until_ft_abli_is_promoted(self) -> None:
        config = load_quantization_config(Path("configs/quantization/qwen36_27b_nvfp4_modelopt.yaml"))
        source = resolve_source(
            config,
            None,
            None,
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        self.assertEqual(source.variant, "local_ft_abli")
        self.assertEqual(source.promotion["decision"], "inconclusive")
        with tempfile.TemporaryDirectory() as tmp:
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=Path(tmp) / "quantized",
                run_id="unit_blocked_default_qwen_ft_abli",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )

        with self.assertRaisesRegex(RuntimeError, "Generic Qwen FT-abli slot is not a promoted final candidate yet"):
            guard_export(export)

    def test_quantization_plan_surfaces_inconclusive_source_blockers(self) -> None:
        config = load_quantization_config(Path("configs/quantization/qwen36_27b_nvfp4_modelopt.yaml"))
        plan = build_plan(
            config,
            config_path=Path("configs/quantization/qwen36_27b_nvfp4_modelopt.yaml"),
            family="qwen36_27b",
            variant="local_ft_abli_heretic_residual_trial12",
            output_dir=None,
            run_id="unit_qwen_inconclusive_plan",
            env={"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )

        promotion = plan["source"]["promotion"]
        self.assertEqual(promotion["decision"], "inconclusive")
        self.assertIn("quantization_export", promotion["blocked_actions"])

    def test_generic_fp8_w8a8_modelopt_pipeline_templates_target_variant(self) -> None:
        config_path = Path("configs/quantization/fp8_w8a8_modelopt.yaml")
        config = load_quantization_config(config_path)
        source = resolve_source(
            config,
            "llama31_8b",
            "base",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "quantized"
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=output_root,
                run_id="unit_llama_fp8_w8a8",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )
            plan = build_plan(
                config,
                config_path=config_path,
                family="llama31_8b",
                variant="base",
                output_dir=Path(tmp) / "reports",
                run_id="unit_llama_fp8_w8a8_plan",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host"},
            )

        joined = " ".join(export["command"])
        self.assertEqual(export["method"], "fp8_w8a8")
        self.assertEqual(export["backend"], "modelopt")
        self.assertEqual(export["target"]["variant"], "base_fp8_w8a8_modelopt")
        self.assertEqual(export["target"]["host_output_path"], str(output_root / "llama31_8b" / "base_fp8_w8a8_modelopt"))
        self.assertIn("--qformat fp8", joined)
        self.assertIn("/opt/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py", joined)
        self.assertIn("--low_memory_mode", joined)
        self.assertIn("MemoryMax=90%", export["command"])
        self.assertTrue(plan["target"]["checkpoint_written_by_this_plan"])
        self.assertEqual(plan["target"]["variant"], "base_fp8_w8a8_modelopt")
        self.assertIn("FP8 W8A8 checkpoint creation", " ".join(plan["quantization"]["notes"]))

    def test_gguf_llama_cpp_export_command_is_guarded_and_generic(self) -> None:
        config_path = Path("configs/quantization/gguf_llama_cpp_q4_k_m.yaml")
        config = load_quantization_config(config_path)
        source = resolve_source(
            config,
            "llama31_8b",
            "base",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "gguf"
            export = build_gguf_export_command(
                config,
                source,
                output_dir=output_root,
                run_id="unit_llama_gguf",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "MODEL_FORGE_LLAMA_CPP_DIR": "/opt/llama.cpp"},
            )
            dispatched = build_export_command(
                config,
                source,
                output_dir=output_root,
                run_id="unit_llama_gguf_dispatch",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "MODEL_FORGE_LLAMA_CPP_DIR": "/opt/llama.cpp"},
            )

        self.assertEqual(export["method"], "gguf_q4_k_m")
        self.assertEqual(export["backend"], "llama_cpp")
        self.assertEqual(export["strategy"], "llama_cpp_gguf")
        self.assertEqual(export["target"]["variant"], "base_gguf_q4_k_m")
        self.assertEqual(export["target"]["host_output_path"], str(output_root / "llama31_8b" / "base_gguf_q4_k_m" / "base_gguf_q4_k_m.gguf"))
        self.assertEqual(export["target"]["intermediate_output_path"], str(output_root / "llama31_8b" / "base_gguf_q4_k_m" / "base_gguf_q4_k_m.f16.gguf"))
        self.assertIn("CPUQuota=80%", export["command"])
        self.assertIn("MemoryMax=85%", export["command"])
        command = export["command_display"]
        self.assertIn("convert_hf_to_gguf.py", command)
        self.assertIn("llama-quantize", command)
        self.assertIn("Q4_K_M", command)
        self.assertIn("llama-cli", command)
        self.assertIn("llama-bench", command)
        self.assertIn("tokenizer-report passes", " ".join(export["validation_gates"]))
        self.assertEqual(dispatched["strategy"], "llama_cpp_gguf")

    def test_nvfp4_gate_requires_export_serving_behavior_and_tokenizer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_quantization_config(Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"))
            source = resolve_source(config, "gemma4_26b_a4b", "base", {"MODEL_FORGE_MODELS_DIR": "/models-host"})
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=root / "models",
                run_id="unit_nvfp4",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )
            export_path = root / "export_plan.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")
            serving = root / "summary.json"
            write_serving_summary(serving, model="candidate", output_tps=50.0, decode_heavy_tps=50.5)
            serving_eval = root / "serving_eval"
            write_behavior_scores(serving_eval)
            card = root / "quantization_card.json"
            card.write_text(json.dumps({"schema_version": "model_forge.quantization_card.v1"}), encoding="utf-8")
            behavior = root / "behavior.json"
            behavior.write_text(
                json.dumps({"schema_version": BEHAVIOR_REPORT_SCHEMA_VERSION, "behavior_preserved": True}),
                encoding="utf-8",
            )
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text(
                json.dumps({"schema_version": TOKENIZER_REPORT_SCHEMA_VERSION, "passed": True}),
                encoding="utf-8",
            )
            report = build_nvfp4_gate_report(
                export_plan=export_path,
                serving_summary=serving,
                serving_eval=serving_eval,
                quantization_card=card,
                behavior_report=behavior,
                tokenizer_report=tokenizer,
                output_dir=root / "gate",
                run_id="unit_nvfp4_gate",
                min_output_tps=45.0,
            )

        self.assertEqual(report["schema_version"], NVFP4_GATE_SCHEMA_VERSION)
        self.assertTrue(report["nvfp4_ready"])
        self.assertEqual(report["metrics"]["decode_heavy_output_tokens_per_second_p50"], 50.5)

    def test_nvfp4_gate_fails_when_throughput_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export = root / "export_plan.json"
            export.write_text(
                json.dumps(
                    {
                        "schema_version": "model_forge.quantization_export.v1",
                        "method": "nvfp4",
                        "backend": "modelopt",
                        "command_display": "--qformat nvfp4",
                    }
                ),
                encoding="utf-8",
            )
            serving = root / "summary.json"
            write_serving_summary(serving, model="candidate", output_tps=25.0, decode_heavy_tps=25.0)
            serving_eval = root / "serving_eval"
            write_behavior_scores(serving_eval)
            card = root / "quantization_card.json"
            card.write_text(json.dumps({"schema_version": CARD_SCHEMA_VERSION}), encoding="utf-8")
            behavior = root / "behavior.json"
            behavior.write_text(json.dumps({"schema_version": BEHAVIOR_REPORT_SCHEMA_VERSION, "behavior_preserved": True}), encoding="utf-8")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text(json.dumps({"schema_version": TOKENIZER_REPORT_SCHEMA_VERSION, "passed": True}), encoding="utf-8")
            report = build_nvfp4_gate_report(
                export_plan=export,
                serving_summary=serving,
                serving_eval=serving_eval,
                quantization_card=card,
                behavior_report=behavior,
                tokenizer_report=tokenizer,
                output_dir=root / "gate",
                run_id="unit_nvfp4_gate_fail",
                min_output_tps=45.0,
            )

        self.assertFalse(report["nvfp4_ready"])
        throughput = next(check for check in report["checks"] if check["name"] == "output_tps_target_met")
        self.assertEqual(throughput["status"], "fail")

    def test_calibration_manifest_resolves_exact_dataset_inputs(self) -> None:
        config_path = Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml")
        config = load_quantization_config(config_path)
        manifest = build_calibration_manifest(
            config,
            config_path=config_path,
            family="gemma4_26b_a4b",
            variant="base",
            output_dir="/tmp/model-forge-quant-tests",
            run_id="unit_calib",
            dataset="cnn_dailymail,nemotron-post-training-dataset-v2",
            samples="64,64",
            seq_len="1024",
            env={"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )

        self.assertEqual(manifest["schema_version"], CALIBRATION_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["source"]["variant"], "base")
        self.assertEqual(manifest["calibration"]["dataset"], "cnn_dailymail,nemotron-post-training-dataset-v2")
        self.assertEqual(manifest["calibration"]["samples"], "64,64")
        self.assertEqual(manifest["calibration"]["seq_len"], "1024")
        self.assertEqual(manifest["calibration"]["selection_source"]["dataset"], "argument")
        datasets = manifest["calibration"]["datasets"]
        self.assertEqual([item["name"] for item in datasets], ["cnn_dailymail", "nemotron-post-training-dataset-v2"])
        self.assertEqual(datasets[0]["access"], "public_or_local")
        self.assertEqual(datasets[1]["access"], "gated")
        self.assertIn("serving metrics", " ".join(manifest["promotion_requirements"]))

    def test_gemma_nvfp4_matrix_has_variant_specific_baselines(self) -> None:
        config = load_quantization_config(Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"))
        entries = matrix_entries(config)
        variants = {entry["source_variant"]: entry for entry in entries}

        self.assertIn("base", variants)
        self.assertIn("local_ft", variants)
        self.assertIn("local_abli_sota", variants)
        self.assertIn("ft_local_abli_sota_internal_r7_selected_t34_transfer", variants)
        for entry in entries:
            self.assertIn("baseline_eval", entry)
            self.assertIn(entry["source_variant"], entry["target_variant"])

        self.assertEqual(matrix_workers(config, {"MODEL_FORGE_QUANT_WORKERS": "local,spark-b"}), ["local", "spark-b"])
        self.assertEqual(matrix_workers(config, {"UNRELATED": "x"}), ["local"])

    def test_matrix_entries_can_filter_source_variants(self) -> None:
        config = load_quantization_config(Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"))
        entries = filter_matrix_entries(matrix_entries(config), "base,local_ft")
        self.assertEqual([entry["source_variant"] for entry in entries], ["base", "local_ft"])
        with self.assertRaises(ValueError):
            filter_matrix_entries(matrix_entries(config), "missing_variant")

    def test_modelopt_export_allows_calibration_dataset_override(self) -> None:
        config = load_quantization_config(Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"))
        source = resolve_source(
            config,
            "gemma4_26b_a4b",
            "base",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=Path(tmp),
                run_id="unit_override",
                env={
                    "MODEL_FORGE_MODELS_DIR": "/models-host",
                    "HF_HOME": "/hf-cache",
                    "MODEL_FORGE_QUANT_CALIB_DATASET": "cnn_dailymail,nemotron-post-training-dataset-v2",
                    "MODEL_FORGE_QUANT_CALIB_SIZE": "256,256",
                    "MODEL_FORGE_QUANT_CALIB_SEQ": "4096",
                },
            )

        joined = " ".join(export["command"])
        self.assertIn("--dataset cnn_dailymail,nemotron-post-training-dataset-v2", joined)
        self.assertIn("--calib-samples 256,256", joined)
        self.assertIn("--calib-seq-len 4096", joined)

    def test_gemma_nvfp4_runtime_plan_uses_marlin_for_full_moe(self) -> None:
        config_path = Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml")
        config = load_quantization_config(config_path)
        plan = build_plan(
            config,
            config_path=config_path,
            family="gemma4_26b_a4b",
            variant="base",
            output_dir=None,
            run_id="unit_gemma4_marlin_plan",
            env={"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark", "MODEL_FORGE_MODELS_DIR": "/models-host"},
        )

        launch = " ".join(plan["launch_command"])
        self.assertIn("VLLM_NVFP4_GEMM_BACKEND=marlin", launch)
        self.assertIn("--moe-backend marlin", launch)
        self.assertIn("--language-model-only", launch)
        self.assertNotIn("experts", plan["quantization"]["exclusions"]["modules"])
        self.assertNotIn("router", plan["quantization"]["exclusions"]["modules"])
        self.assertEqual(plan["resource_policy"]["systemd_scope"]["MemoryMax"], "95%")
        self.assertEqual(plan["resource_policy"]["start_if_memory_available_above_fraction"], 0.10)
        self.assertEqual(plan["resource_policy"]["stop_if_memory_available_below_fraction"], 0.05)

    def test_modelopt_export_output_can_follow_target_variant(self) -> None:
        config = load_quantization_config(Path("configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml"))
        config = config.__class__(
            **{
                **config.__dict__,
                "target_variant": "local_ft_nvfp4_modelopt",
            }
        )
        source = resolve_source(
            config,
            "gemma4_26b_a4b",
            "local_ft",
            {"MODEL_FORGE_MODELS_DIR": "/models-host"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "quantized" / "gemma4_26b_a4b"
            export = build_modelopt_export_command(
                config,
                source,
                output_dir=output_root,
                run_id="unit_local_ft_nvfp4",
                env={"MODEL_FORGE_MODELS_DIR": "/models-host", "HF_HOME": "/hf-cache"},
            )

        self.assertEqual(export["target"]["variant"], "local_ft_nvfp4_modelopt")
        self.assertEqual(export["target"]["host_output_path"], str(output_root / "local_ft_nvfp4_modelopt"))
        self.assertIn("/workspace/output_models/local_ft_nvfp4_modelopt", export["command"])


if __name__ == "__main__":
    unittest.main()
