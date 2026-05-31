from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model_forge.benchmarks.serve import (
    DEFAULT_CONFIG,
    build_metrics,
    build_plan,
    evaluate_serving_evidence,
    load_config,
    optional_float,
    parse_streaming_chunks,
    run_benchmark,
    summarize_memory,
    write_evidence_gate_report,
    write_outputs,
)


class ServeBenchmarkTests(unittest.TestCase):
    def test_config_resolves_model_from_family_variant_without_private_values(self) -> None:
        config = load_config(DEFAULT_CONFIG, family="gemma4_26b_a4b", variant="base", env={})
        self.assertEqual(config.model, "google/gemma-4-26B-A4B-it")
        self.assertEqual(config.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.concurrency, 1)
        self.assertEqual(len(config.requests), 3)
        self.assertEqual(len(config.workload_sources), 3)
        self.assertEqual(config.requests[0].request_id, "short_chat:ttft_brief")

        plan = build_plan(config, DEFAULT_CONFIG)
        self.assertTrue(plan["dry_run_only"])
        self.assertEqual(plan["request_count"], len(config.requests))
        self.assertIn("configs/serving/workloads/short_chat.yaml", plan["workload_sources"])

    def test_core_workload_config_loads_all_serving_workloads(self) -> None:
        config = load_config(
            Path("configs/serving/serve_bench_core.yaml"),
            family="gemma4_26b_a4b",
            variant="base",
            env={},
        )
        categories = {request.category for request in config.requests}
        self.assertEqual(len(config.workload_sources), 8)
        self.assertIn("long_prefill", categories)
        self.assertIn("artifact_generation", categories)
        self.assertIn("long_context_retrieval", categories)

    def test_streaming_parser_records_first_token_and_usage(self) -> None:
        ticks = iter([100.01, 100.02])
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n',
            b"data: [DONE]\n",
        ]
        parsed = parse_streaming_chunks(lines, started_at=100.0, clock=lambda: next(ticks))
        self.assertEqual(parsed["text"], "Hello")
        self.assertAlmostEqual(parsed["first_chunk_seconds"], 0.01)
        self.assertAlmostEqual(parsed["first_token_seconds"], 0.02)
        self.assertEqual(parsed["usage"]["completion_tokens"], 2)
        self.assertEqual(parsed["stream_token_event_count"], 2)

    def test_metrics_include_decode_rates_when_ttft_and_usage_exist(self) -> None:
        metrics = build_metrics(
            total_latency_seconds=1.0,
            usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            first_token_seconds=0.2,
            stream_token_event_count=6,
        )
        self.assertEqual(metrics["time_to_first_token_seconds"], 0.2)
        self.assertEqual(metrics["completion_token_source"], "usage.completion_tokens")
        self.assertEqual(metrics["output_tokens_per_second"], 6.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 6.25)
        self.assertEqual(metrics["inter_token_latency_seconds"], 0.16)

    def test_memory_summary_handles_system_and_gpu_samples(self) -> None:
        self.assertEqual(optional_float("1,024 MiB"), 1024.0)
        self.assertIsNone(optional_float("[N/A]"))
        summary = summarize_memory(
            [
                {
                    "memory": {
                        "before": {
                            "system": {
                                "available_fraction": 0.20,
                                "available_bytes": 20,
                                "used_bytes": 80,
                                "used_percent": 80.0,
                            },
                            "gpu": {
                                "devices": [
                                    {
                                        "name": "NVIDIA GB10",
                                        "memory_total_mib": 100.0,
                                        "memory_used_mib": 40.0,
                                        "memory_free_mib": 60.0,
                                        "memory_used_fraction": 0.4,
                                        "utilization_gpu_percent": 8.0,
                                    }
                                ],
                                "error": None,
                            },
                        },
                        "after": {
                            "system": {
                                "available_fraction": 0.15,
                                "available_bytes": 15,
                                "used_bytes": 85,
                                "used_percent": 85.0,
                            },
                            "gpu": {
                                "devices": [
                                    {
                                        "name": "NVIDIA GB10",
                                        "memory_total_mib": 100.0,
                                        "memory_used_mib": 50.0,
                                        "memory_free_mib": 50.0,
                                        "memory_used_fraction": 0.5,
                                        "utilization_gpu_percent": 10.0,
                                    }
                                ],
                                "error": None,
                            },
                        },
                    }
                }
            ]
        )
        self.assertEqual(summary["snapshot_count"], 2)
        self.assertEqual(summary["system"]["available_fraction"]["min"], 0.15)
        self.assertEqual(summary["gpu"]["device_count_max"], 1)
        self.assertEqual(summary["gpu"]["memory_used_mib"]["max"], 50.0)

    def test_write_outputs_creates_summary_manifest_and_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                DEFAULT_CONFIG,
                family="gemma4_26b_a4b",
                variant="base",
                output_dir=tmp,
                limit=1,
                env={},
            )
            results = [
                {
                    "request_id": "short_instruction",
                    "category": "short_chat",
                    "repetition": 1,
                    "ok": True,
                    "metrics": {
                        "total_latency_seconds": 1.0,
                        "time_to_first_token_seconds": 0.25,
                        "inter_token_latency_seconds": 0.05,
                        "output_tokens_per_second": 12.0,
                    },
                    "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
                    "memory": {
                        "before": {
                            "system": {
                                "available_fraction": 0.25,
                                "available_bytes": 25,
                                "used_bytes": 75,
                                "used_percent": 75.0,
                            },
                            "gpu": {"devices": [], "error": "nvidia-smi not found"},
                        },
                        "after": {
                            "system": {
                                "available_fraction": 0.24,
                                "available_bytes": 24,
                                "used_bytes": 76,
                                "used_percent": 76.0,
                            },
                            "gpu": {"devices": [], "error": "nvidia-smi not found"},
                        },
                    },
                }
            ]
            output_dir, summary, manifest = write_outputs(
                config,
                DEFAULT_CONFIG,
                results,
                run_id="unit_serve_bench",
                command=["./forge", "bench", "serve", "--dry-run"],
            )
            self.assertEqual(summary["successful_requests"], 1)
            self.assertEqual(manifest["run_type"], "serving")
            manifest_config_paths = [entry["path"] for entry in manifest["configs"]]
            self.assertIn("configs/serving/workloads/short_chat.yaml", manifest_config_paths)
            self.assertTrue((output_dir / "requests.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "serving_card.md").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["metrics"]["total_latency_seconds"]["p50"], 1.0)
            self.assertEqual(saved["metrics"]["total_latency_seconds"]["p99"], 1.0)
            self.assertEqual(saved["memory"]["snapshot_count"], 2)
            self.assertEqual(manifest["outputs"]["metrics"]["memory"]["snapshot_count"], 2)
            card = (output_dir / "serving_card.md").read_text(encoding="utf-8")
            self.assertIn("# Serving Card: serve_bench_smoke", card)
            self.assertIn("## Hardware And Config", card)
            self.assertIn("## Memory", card)
            self.assertIn("## Workload Metrics", card)
            self.assertIn("## Promotion Gates", card)
            self.assertIn("configs/serving/workloads/short_chat.yaml", card)
            self.assertIn("TTFT seconds", card)
            self.assertIn("Output tok/sec", card)
            self.assertIn("System available fraction", card)

    def test_serving_evidence_gate_requires_quality_behavior_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(DEFAULT_CONFIG, family="gemma4_26b_a4b", variant="base", output_dir=tmp, limit=1, env={})
            output_dir, summary, _manifest = write_outputs(
                config,
                DEFAULT_CONFIG,
                [
                    {
                        "request_id": "short_instruction",
                        "category": "short_chat",
                        "repetition": 1,
                        "ok": True,
                        "metrics": {"total_latency_seconds": 1.0},
                        "memory": {"before": {}, "after": {}},
                    }
                ],
                run_id="unit_serve_evidence",
            )
            report = evaluate_serving_evidence(summary_path=output_dir / "summary.json")
            self.assertFalse(report["completion_ready"])
            eval_check = next(check for check in report["checks"] if check["name"] == "sampled_quality_behavior_attached")
            self.assertEqual(eval_check["status"], "fail")

            eval_dir = Path(tmp) / "serving_eval"
            eval_dir.mkdir()
            (eval_dir / "serving_eval_card.md").write_text("# card\n", encoding="utf-8")
            (eval_dir / "manifest.json").write_text(
                json.dumps({"serving_context": {"target": {"model": summary["model"], "base_url": summary["base_url"]}}}) + "\n",
                encoding="utf-8",
            )
            report = evaluate_serving_evidence(summary_path=output_dir / "summary.json", serving_eval=eval_dir)
            self.assertTrue(report["completion_ready"])
            gate_path = write_evidence_gate_report(report, Path(tmp) / "gate.json")
            self.assertTrue(gate_path.exists())
            self.assertTrue(gate_path.with_suffix(".md").exists())

    def test_serving_evidence_gate_reports_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_serving_evidence(summary_path=Path(tmp) / "missing" / "summary.json")
            self.assertFalse(report["completion_ready"])
            self.assertEqual(report["checks"][0]["name"], "summary_exists")
            self.assertEqual(report["checks"][0]["status"], "fail")

    def test_actual_streaming_http_benchmark_against_mock_server(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {"choices": [{"delta": {"content": " done"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            config = load_config(DEFAULT_CONFIG, model="mock/model", base_url=base_url, limit=1, env={})
            results = run_benchmark(config)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0]["usage"]["completion_tokens"], 2)
        self.assertGreaterEqual(results[0]["metrics"]["time_to_first_token_seconds"], 0)
        self.assertEqual(results[0]["metrics"]["completion_token_source"], "usage.completion_tokens")
        self.assertIn("memory", results[0])
        self.assertIn("before", results[0]["memory"])
        self.assertIn("after", results[0]["memory"])
        self.assertGreater(results[0]["memory"]["after"]["system"]["available_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
