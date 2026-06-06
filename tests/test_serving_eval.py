from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model_forge.benchmarks.serve_eval import (
    DEFAULT_CONFIG,
    SampleSpec,
    build_comparison_report,
    load_serving_eval_config,
    plan_from,
    run_serving_eval,
    select_sample_cases,
    write_comparison_report,
)


class ServingEvalTests(unittest.TestCase):
    def test_plan_resolves_balanced_quality_behavior_sample(self) -> None:
        config = load_serving_eval_config(DEFAULT_CONFIG)
        plan = plan_from(
            config,
            config_path=DEFAULT_CONFIG,
            family="gemma4_26b_a4b",
            variant="base",
            model=None,
            base_url=None,
            eval_config=None,
            output_dir=None,
            run_id="unit_serving_eval",
            trials=3,
            max_cases=None,
            dry_run=True,
        )
        self.assertEqual(plan["target"]["model"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(plan["eval"]["trials"], 3)
        self.assertEqual(plan["eval"]["timeout_seconds"], 300)
        self.assertEqual(plan["eval"]["case_count"], 11)
        buckets = {sample["bucket"] for sample in plan["eval"]["samples"]}
        self.assertIn("normal_use_regression", buckets)
        self.assertIn("capability_preservation_challenge", buckets)
        self.assertIn("refusal_paired_boundary", buckets)
        self.assertIn("unsafe_overcompliance", buckets)
        self.assertIn("unsafe_ablation_redirect", buckets)
        self.assertFalse(plan["execution_contract"]["starts_server"])

    def test_sample_case_selection_rejects_missing_case_id(self) -> None:
        bad_sample = [SampleSpec(prompt_set="normal_use_regression", case_ids=("missing",), count=None)]
        with self.assertRaises(ValueError):
            select_sample_cases(bad_sample, prompt_root=Path("evals/prompts"))

    def test_dry_run_writes_serving_eval_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_serving_eval_config(DEFAULT_CONFIG)
            plan = plan_from(
                config,
                config_path=DEFAULT_CONFIG,
                family="gemma4_26b_a4b",
                variant="base",
                model=None,
                base_url=None,
                eval_config=None,
                output_dir=Path(tmp),
                run_id="unit_dry_run",
                trials=1,
                max_cases=2,
                dry_run=True,
            )
            output_root, manifest = run_serving_eval(plan, config_path=DEFAULT_CONFIG, dry_run=True)
            self.assertEqual(manifest["total_cases"], 2)
            self.assertTrue((output_root / "manifest.json").exists())
            self.assertTrue((output_root / "scores.csv").exists())
            self.assertTrue((output_root / "serving_eval_context.json").exists())
            card = (output_root / "serving_eval_card.md").read_text(encoding="utf-8")
            self.assertIn("# Serving Eval Card: serve_eval_quality_behavior", card)
            self.assertIn("## Sampled Cases", card)
            self.assertIn("## Scores", card)

    def test_actual_http_run_against_mock_server(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = {
                    "id": "mock",
                    "choices": [
                        {
                            "message": {
                                "content": "LoRA uses small adapter weights; QLoRA adds quantization to reduce memory while keeping adapters trainable."
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 14, "total_tokens": 26},
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode("utf-8"))

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config = load_serving_eval_config(DEFAULT_CONFIG)
                plan = plan_from(
                    config,
                    config_path=DEFAULT_CONFIG,
                    family=None,
                    variant=None,
                    model="mock/model",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    eval_config=Path("configs/experiments/gemma4_26b_a4b_v0.yaml"),
                    output_dir=Path(tmp),
                    run_id="unit_http_run",
                    trials=1,
                    max_cases=1,
                    dry_run=False,
                )
                output_root, manifest = run_serving_eval(plan, config_path=DEFAULT_CONFIG, dry_run=False)
                self.assertEqual(manifest["total_cases"], 1)
                rows = [json.loads(line) for line in (output_root / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(rows[0]["usage"]["completion_tokens"], 14)
                self.assertEqual(rows[0]["scores"]["normal_use_regression_pass_rate"], 1.0)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_compare_reports_source_pass_candidate_fail_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            candidate = root / "candidate"
            source.mkdir()
            candidate.mkdir()
            source_row = {
                "bucket": "agentic_tool_use_json",
                "case_id": "model_serve_timeout",
                "trial_index": 1,
                "scores": {"schema_adherence": 1.0, "workflow_success": 1.0},
                "notes": [],
                "response_text": "{\"goal\":\"debug\",\"steps\":[]}",
            }
            candidate_row = {
                "bucket": "agentic_tool_use_json",
                "case_id": "model_serve_timeout",
                "trial_index": 1,
                "scores": {"schema_adherence": 0.0, "workflow_success": 0.0},
                "notes": ["response did not parse as JSON"],
                "response_text": "```json\n{\"goal\":\"debug\", reason\":\"bad\"}\n```",
            }
            (source / "responses.jsonl").write_text(json.dumps(source_row) + "\n", encoding="utf-8")
            (candidate / "responses.jsonl").write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")

            report = build_comparison_report(
                source_eval=source,
                candidate_eval=candidate,
                output_dir=root / "report",
                run_id="unit_compare",
            )
            write_comparison_report(report)

            self.assertEqual(report["schema_version"], "model_forge.serving_eval_comparison.v1")
            self.assertEqual(report["compared_cases"], 1)
            self.assertEqual(report["source_pass_candidate_fail_count"], 2)
            self.assertEqual(report["regressions"][0]["bucket"], "agentic_tool_use_json")
            self.assertIn("response did not parse as JSON", report["regressions"][0]["candidate_notes"])
            self.assertTrue((root / "report" / "serving_eval_comparison.json").exists())
            markdown = (root / "report" / "serving_eval_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Candidate response:", markdown)
            self.assertIn("reason", markdown)


if __name__ == "__main__":
    unittest.main()
