from __future__ import annotations

import json
import tempfile
import unittest

from model_forge.benchmarks.sweep import (
    DEFAULT_CONFIG,
    audit_sweep_config,
    build_sweep_plan,
    load_sweep_config,
    write_plan,
)


class ServingSweepTests(unittest.TestCase):
    def test_dgx_spark_sweep_config_passes_doctor(self) -> None:
        config, path = load_sweep_config(DEFAULT_CONFIG)
        findings = audit_sweep_config(config, path, strict=True)
        self.assertEqual([finding for finding in findings if finding.severity == "error"], [])
        self.assertGreaterEqual(len(config["cases"]), 4)

    def test_sweep_plan_expands_cases_and_two_node_cluster(self) -> None:
        config, path = load_sweep_config(DEFAULT_CONFIG)
        env = {
            "MODEL_FORGE_NODE0_HOST": "spark-a",
            "MODEL_FORGE_NODE1_HOST": "spark-b",
            "MODEL_FORGE_NODE0_USER": "runner",
            "MODEL_FORGE_NODE1_USER": "runner",
            "MODEL_FORGE_CLUSTER_WORK_DIR": "/" + "home/private/model-forge",
            "MODEL_FORGE_RDZV_ENDPOINT": "spark-a:29500",
        }
        plan = build_sweep_plan(
            config,
            path,
            family="gemma4_26b_a4b",
            variant="base",
            base_url="http://127.0.0.1:8000/v1",
            cluster_config="configs/clusters/dgx_spark_x2.example.yaml",
            env=env,
        )
        self.assertEqual(plan["cluster"]["node_count"], 2)
        self.assertEqual(plan["cluster"]["total_declared_gpus"], 2)
        self.assertEqual(plan["cluster"]["total_declared_memory_gb"], 256)
        self.assertEqual(plan["target"]["model"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(len(plan["cases"]), len(config["cases"]))
        self.assertIn("--family gemma4_26b_a4b --variant base", plan["cases"][0]["bench_command"])
        self.assertEqual(plan["cases"][0]["server_env"]["VLLM_KV_CACHE_DTYPE"], "fp8_e4m3")
        self.assertTrue(plan["cases"][0]["requires_server_restart"])

    def test_write_plan_creates_json_and_command_script(self) -> None:
        config, path = load_sweep_config(DEFAULT_CONFIG)
        plan = build_sweep_plan(
            config,
            path,
            model="served/test-model",
            base_url="http://127.0.0.1:8000/v1",
            env={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = write_plan(plan, tmp)
            commands_path = plan_path.parent / "bench_commands.sh"
            self.assertTrue(plan_path.exists())
            self.assertTrue(commands_path.exists())
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["target"]["model"], "served/test-model")
            self.assertIn("server env GPU_MEMORY_UTILIZATION", commands_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
