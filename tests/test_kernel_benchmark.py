from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from model_forge.benchmarks.kernel import render_card, rmsnorm_plan, write_outputs


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
            card = summary_path.with_name("kernel_card.md").read_text(encoding="utf-8")

        self.assertEqual(saved["run_id"], "unit_rmsnorm_card")
        self.assertIn("# Kernel Card: unit_rmsnorm_card", card)
        self.assertIn("RMSNorm", render_card(summary))


if __name__ == "__main__":
    unittest.main()
