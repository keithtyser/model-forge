from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from model_forge.quantization.cli import (
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    FP8_KV_REPORT_SCHEMA_VERSION,
    build_card,
    build_calibration_manifest,
    build_fp8_kv_report,
    build_modelopt_export_command,
    build_plan,
    filter_matrix_entries,
    load_quantization_config,
    matrix_entries,
    matrix_workers,
    resolve_source,
)


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
