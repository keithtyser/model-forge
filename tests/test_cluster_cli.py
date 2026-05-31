from __future__ import annotations

import unittest
from pathlib import Path

from model_forge.cluster.cli import (
    REPO_DIR,
    audit_cluster,
    build_sync_plan,
    build_model_sync_plan,
    build_launcher_plan,
    docker_gpu_runtime_command,
    docker_torchrun_smoke_command,
    guarded_command,
    json_lines,
    load_cluster_config,
    load_hardware_profile,
)


class ClusterCliTests(unittest.TestCase):
    def test_local_example_passes_doctor(self) -> None:
        config, path = load_cluster_config(REPO_DIR / "configs" / "clusters" / "local.example.yaml")
        findings = audit_cluster(config, path, hardware=load_hardware_profile(config), strict=True)
        self.assertEqual([finding for finding in findings if finding.severity == "error"], [])

    def test_dgx_spark_x2_example_is_env_backed_and_strict_requires_env(self) -> None:
        config_path = REPO_DIR / "configs" / "clusters" / "dgx_spark_x2.example.yaml"
        config, path = load_cluster_config(config_path)
        findings = audit_cluster(config, path, hardware=load_hardware_profile(config), env={}, strict=True)
        errors = [finding for finding in findings if finding.severity == "error"]
        self.assertTrue(any("MODEL_FORGE_NODE0_HOST" in finding.message for finding in errors))
        self.assertTrue(any("MODEL_FORGE_NODE1_HOST" in finding.message for finding in errors))
        self.assertTrue(any("MODEL_FORGE_RDZV_ENDPOINT" in finding.message for finding in errors))
        self.assertNotIn("host", config["nodes"][0])
        self.assertEqual(config["nodes"][0]["host_env"], "MODEL_FORGE_NODE0_HOST")

    def test_dgx_spark_x2_plan_uses_env_values_without_hardcoded_hosts(self) -> None:
        config, path = load_cluster_config(REPO_DIR / "configs" / "clusters" / "dgx_spark_x2.example.yaml")
        hardware = load_hardware_profile(config)
        env = {
            "MODEL_FORGE_NODE0_HOST": "spark-a",
            "MODEL_FORGE_NODE1_HOST": "spark-b",
            "MODEL_FORGE_NODE0_USER": "runner",
            "MODEL_FORGE_NODE1_USER": "runner",
            "MODEL_FORGE_CLUSTER_WORK_DIR": "/" + "home/private/model-forge",
            "MODEL_FORGE_RDZV_ENDPOINT": "spark-a:29500",
        }
        findings = audit_cluster(config, path, hardware=hardware, env=env, strict=True)
        self.assertEqual([finding for finding in findings if finding.severity == "error"], [])
        plan = build_launcher_plan(
            config,
            hardware,
            path,
            workload="train",
            launcher="torchrun",
            command="./forge finetune gemma4_26b_a4b run --execute",
            env=env,
        )

        self.assertEqual(plan["cluster"]["node_count"], 2)
        self.assertEqual(plan["cluster"]["total_declared_memory_gb"], 256)
        self.assertEqual(plan["cluster"]["total_declared_gpus"], 2)
        self.assertEqual(plan["nodes"][0]["host"], "spark-a")
        self.assertEqual(plan["nodes"][1]["host"], "spark-b")
        self.assertIn("--nnodes=2", plan["execution_plan"]["launcher_command"])
        self.assertIn("--nproc-per-node=1", plan["execution_plan"]["launcher_command"])
        self.assertIn("--rdzv-endpoint=spark-a:29500", plan["execution_plan"]["launcher_command"])
        self.assertTrue(plan["execution_plan"]["dry_run_only"])

    def test_sync_plan_skips_local_and_targets_worker(self) -> None:
        config, path = load_cluster_config(REPO_DIR / "configs" / "clusters" / "dgx_spark_x2.example.yaml")
        hardware = load_hardware_profile(config)
        env = {
            "MODEL_FORGE_NODE0_HOST": "localhost",
            "MODEL_FORGE_NODE1_HOST": "spark-b",
            "MODEL_FORGE_NODE0_USER": "runner",
            "MODEL_FORGE_NODE1_USER": "runner",
            "MODEL_FORGE_CLUSTER_WORK_DIR": "/" + "home/private/model-forge",
            "MODEL_FORGE_RDZV_ENDPOINT": "localhost:29500",
        }
        plan = build_sync_plan(config, hardware, path, env=env)

        self.assertEqual(plan["cluster"]["node_count"], 2)
        self.assertTrue(plan["actions"][0]["skip"])
        self.assertFalse(plan["actions"][1]["skip"])
        self.assertIn("runner@spark-b:" + "/" + "home/private/model-forge/", plan["actions"][1]["rsync_command"])
        self.assertIn("--exclude /.venv/", plan["actions"][1]["rsync_command"])
        self.assertIn("--exclude /runs/", plan["actions"][1]["rsync_command"])

    def test_model_sync_plan_uses_worker_models_dir(self) -> None:
        config, path = load_cluster_config(REPO_DIR / "configs" / "clusters" / "dgx_spark_x2.example.yaml")
        hardware = load_hardware_profile(config)
        env = {
            "MODEL_FORGE_NODE0_HOST": "localhost",
            "MODEL_FORGE_NODE1_HOST": "spark-b",
            "MODEL_FORGE_NODE0_USER": "runner",
            "MODEL_FORGE_NODE1_USER": "runner",
            "MODEL_FORGE_CLUSTER_WORK_DIR": "/" + "home/private/model-forge",
            "MODEL_FORGE_NODE0_MODELS_DIR": "/" + "models-a",
            "MODEL_FORGE_NODE1_MODELS_DIR": "/" + "models-b",
            "MODEL_FORGE_RDZV_ENDPOINT": "localhost:29500",
        }
        source = REPO_DIR / "configs"
        plan = build_model_sync_plan(config, hardware, path, source=source, env=env, target_name="Qwen3.6-27B")

        self.assertTrue(plan["actions"][0]["skip"])
        self.assertFalse(plan["actions"][1]["skip"])
        self.assertEqual(plan["actions"][1]["target_dir"], "/" + "models-b/Qwen3.6-27B")
        self.assertGreater(plan["actions"][1]["source_bytes"], 0)
        self.assertIn("runner@spark-b:" + "/" + "models-b/Qwen3.6-27B/", plan["actions"][1]["rsync_command"])
        self.assertIn("--partial", plan["actions"][1]["rsync_command"])

    def test_runtime_command_is_bounded_docker_gpu_probe(self) -> None:
        command = docker_gpu_runtime_command("nemotron-runner:latest")
        self.assertIn("docker run", command)
        self.assertIn("--gpus all", command)
        self.assertIn("--cpus=1", command)
        self.assertIn("--memory=8g", command)
        self.assertIn("nemotron-runner:latest", command)
        self.assertIn("cuda_available", command)

    def test_guarded_command_defaults_to_user_systemd_scope(self) -> None:
        command = guarded_command("python train.py", {"cpu_quota": "75%", "memory_max_fraction": 0.5}, "runs/lock")
        self.assertIn("flock runs/lock systemd-run --user --scope", command)
        self.assertIn("-p CPUQuota=75%", command)
        self.assertIn("-p MemoryMax=50%", command)

    def test_guarded_command_can_use_system_scope(self) -> None:
        command = guarded_command("python train.py", {"systemd_user_scope": False}, "runs/lock")
        self.assertIn("flock runs/lock systemd-run --scope", command)
        self.assertNotIn("--user --scope", command)

    def test_torchrun_smoke_command_uses_bounded_docker_and_static_master(self) -> None:
        command = docker_torchrun_smoke_command(
            "nemotron-runner:latest",
            node_rank=1,
            nnodes=2,
            nproc_per_node=1,
            rdzv_endpoint="spark-a:29500",
            nccl_socket_ifname="eth0",
        )
        self.assertIn("docker run", command)
        self.assertIn("--network host", command)
        self.assertIn("--cpus=2", command)
        self.assertIn("--memory=16g", command)
        self.assertIn("--node-rank=1", command)
        self.assertIn("--master-addr=spark-a", command)
        self.assertIn("--master-port=29500", command)
        self.assertIn("NCCL_SOCKET_IFNAME=eth0", command)

    def test_json_lines_extracts_smoke_records(self) -> None:
        records = json_lines('warning\n{"ok": true, "rank": 0}\nnot-json\n{"ok": true, "rank": 1}\n')
        self.assertEqual([record["rank"] for record in records], [0, 1])

    def test_example_configs_do_not_contain_private_literals(self) -> None:
        private_home = "/" + "home/ktyser"
        for path in sorted((REPO_DIR / "configs" / "clusters").glob("*.example.yaml")):
            text = Path(path).read_text(encoding="utf-8")
            self.assertNotIn(private_home, text)
            self.assertNotRegex(text, r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
            self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}")


if __name__ == "__main__":
    unittest.main()
