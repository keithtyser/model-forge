from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from model_forge.benchmarks.kernel import build_card_from_summary, dequant_plan, kv_layout_plan, render_card, rmsnorm_plan, rope_plan, write_outputs


class KernelBenchmarkTests(unittest.TestCase):
    def test_rmsnorm_dry_run_plan_is_portable(self) -> None:
        args = argparse.Namespace(
            batch=1,
            seq_len=128,
            hidden_size=256,
            eps=1e-6,
            dtype="bfloat16",
            device="auto",
            warmup=1,
            repeats=2,
            seed=17,
            run_id="unit_rmsnorm",
            output_dir=None,
        )
        plan = rmsnorm_plan(args)
        self.assertEqual(plan["schema_version"], "model_forge.kernel_benchmark.v1")
        self.assertEqual(plan["benchmark"], "rmsnorm")
        self.assertEqual(plan["parameters"]["hidden_size"], 256)
        self.assertIn("summary.json", plan["outputs"]["summary"])

    def test_write_outputs_creates_kernel_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                batch=1,
                seq_len=16,
                hidden_size=32,
                eps=1e-6,
                dtype="float32",
                device="cpu",
                warmup=1,
                repeats=1,
                seed=17,
                run_id="unit_rmsnorm_card",
                output_dir=Path(tmp),
            )
            summary = {
                **rmsnorm_plan(args),
                "dry_run_only": True,
                "runtime": {"device": "cpu", "dtype": "float32"},
                "correctness": {"passed": None, "max_abs_error": None, "tolerance": None},
                "results": {},
            }
            summary_path = write_outputs(summary)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            card_json = json.loads(summary_path.with_name("kernel_card.json").read_text(encoding="utf-8"))
            card = summary_path.with_name("kernel_card.md").read_text(encoding="utf-8")

        self.assertEqual(saved["run_id"], "unit_rmsnorm_card")
        self.assertEqual(card_json["schema_version"], "model_forge.kernel_card.v1")
        self.assertEqual(card_json["kernel"]["name"], "RMSNorm")
        self.assertIn("# Kernel Card: unit_rmsnorm_card", card)
        self.assertIn("Profiler Summary", card)
        self.assertIn("RMSNorm", render_card(summary))

    def test_rope_dry_run_plan_is_portable(self) -> None:
        args = argparse.Namespace(
            batch=1,
            seq_len=128,
            heads=8,
            head_dim=64,
            theta=10_000.0,
            dtype="bfloat16",
            device="auto",
            warmup=1,
            repeats=2,
            seed=17,
            run_id="unit_rope",
            output_dir=None,
        )
        plan = rope_plan(args)
        self.assertEqual(plan["schema_version"], "model_forge.kernel_benchmark.v1")
        self.assertEqual(plan["benchmark"], "rope")
        self.assertEqual(plan["parameters"]["head_dim"], 64)
        self.assertIn("kernel_card.md", plan["outputs"]["card"])

    def test_dequant_dry_run_plan_is_portable(self) -> None:
        args = argparse.Namespace(
            format="nvfp4-e2m1",
            num_elements=1024,
            block_size=16,
            output_dtype="bfloat16",
            device="auto",
            warmup=1,
            repeats=2,
            seed=17,
            run_id="unit_dequant",
            output_dir=None,
        )
        plan = dequant_plan(args)
        self.assertEqual(plan["schema_version"], "model_forge.kernel_benchmark.v1")
        self.assertEqual(plan["benchmark"], "dequant")
        self.assertEqual(plan["parameters"]["format"], "nvfp4-e2m1")
        self.assertIn("summary.json", plan["outputs"]["summary"])

    def test_kv_layout_dry_run_plan_is_portable(self) -> None:
        args = argparse.Namespace(
            batch=1,
            seq_len=128,
            heads=8,
            head_dim=64,
            page_size=16,
            dtype="bfloat16",
            device="auto",
            warmup=1,
            repeats=2,
            seed=17,
            run_id="unit_kv_layout",
            output_dir=None,
        )
        plan = kv_layout_plan(args)
        self.assertEqual(plan["schema_version"], "model_forge.kernel_benchmark.v1")
        self.assertEqual(plan["benchmark"], "kv-layout")
        self.assertEqual(plan["parameters"]["page_size"], 16)
        self.assertIn("kernel_card.md", plan["outputs"]["card"])

    def test_build_card_from_summary_can_attach_profile_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                batch=1,
                seq_len=16,
                hidden_size=32,
                eps=1e-6,
                dtype="float32",
                device="cpu",
                warmup=1,
                repeats=1,
                seed=17,
                run_id="unit_kernel_card_profile",
                output_dir=root,
            )
            summary = {**rmsnorm_plan(args), "dry_run_only": True, "runtime": {}, "correctness": {}, "results": {}}
            summary_path = root / "summary.json"
            profile_path = root / "profile_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            profile_path.write_text(json.dumps({"summary": {"present_profile_artifacts": 2}}), encoding="utf-8")

            card = build_card_from_summary(summary_path, profile_path)

        self.assertTrue(card["profiler_summary"]["attached"])
        self.assertEqual(card["profiler_summary"]["summary"]["present_profile_artifacts"], 2)
        self.assertEqual(card["result"], "planned")


if __name__ == "__main__":
    unittest.main()
