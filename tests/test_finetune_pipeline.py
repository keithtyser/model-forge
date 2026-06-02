from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_forge.pipelines.finetune import REPO_DIR, build_plan, load_yaml, write_artifacts


class FinetunePlanTests(unittest.TestCase):
    def test_gemma_finetune_plan_targets_base_and_jackrong_baseline(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "gemma4_26b_a4b_local_ft_v0.yaml"
        with mock.patch.dict(os.environ, {"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"}):
            plan = build_plan(load_yaml(config_path), config_path)
        self.assertEqual(plan["name"], "gemma4_26b_a4b_local_ft_v0")
        self.assertEqual(plan["family"], "gemma4_26b_a4b")
        self.assertEqual(plan["model"]["source"], "google/gemma-4-26B-A4B-it")
        self.assertTrue(plan["model"]["output_dir"].endswith("gemma-4-26B-A4B-it-local-ft-v0"))
        self.assertEqual(plan["model"]["max_seq_length"], 2048)
        self.assertEqual(plan["baseline"]["target_to_beat"], "Jackrong/Gemopus-4-26B-A4B-it")
        self.assertEqual(plan["trainer"]["backend"], "unsloth")
        self.assertEqual(plan["trainer"]["method"], "qlora")
        self.assertEqual(plan["trainer"]["attn_implementation"], "eager")
        self.assertTrue(plan["trainer"]["unsloth_compile_disable"])
        self.assertTrue(plan["trainer"]["group_by_length"])
        self.assertEqual(plan["trainer"]["pad_to_multiple_of"], 256)
        self.assertEqual(plan["trainer"]["torch_dynamo_recompile_limit"], 128)
        self.assertEqual(plan["trainer"]["max_steps"], 500)
        self.assertEqual(plan["trainer"]["save_strategy"], "steps")
        self.assertFalse(plan["trainer"]["benchmark_only"])
        self.assertEqual(plan["lora"]["r"], 64)
        self.assertIn("q_proj", plan["lora"]["target_modules"])
        self.assertNotIn("q_proj.linear", plan["lora"]["target_modules"])
        self.assertIn("vision_tower", plan["lora"]["exclude_modules"])
        self.assertEqual(plan["hardware"]["profile"], "dgx_spark")
        self.assertEqual(plan["resource_policy"]["cpu_quota"], "80%")
        self.assertEqual(plan["resource_policy"]["memory_max"], "85%")
        self.assertEqual(plan["resource_policy"]["reserve_cores"], 1)
        self.assertEqual(plan["resource_policy"]["min_memory_available_start"], 0.05)
        self.assertEqual(plan["resource_policy"]["min_memory_available_runtime"], 0.05)

    def test_data_manifest_has_holdouts_and_nontrivial_blend(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "gemma4_26b_a4b_local_ft_v0.yaml"
        plan = build_plan(load_yaml(config_path), config_path)
        sources = plan["data"]["sources"]
        roles = {source["role"] for source in sources}
        self.assertGreaterEqual(plan["data"]["target_samples"], 50_000)
        self.assertIn("code_reasoning", roles)
        self.assertIn("multi_turn_reasoning_chat", roles)
        self.assertIn("stem_reasoning", roles)
        self.assertGreaterEqual(len(plan["data"]["holdouts"]), 4)
        self.assertTrue(plan["data"]["quality_gates"]["dedupe_by_conversation_hash"])
        self.assertTrue(plan["data"]["quality_gates"]["reject_eval_prompt_overlap"])

    def test_local_ft_v1_dryrun_plan_uses_source_registry(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "gemma4_26b_a4b_local_ft_v1_dryrun.yaml"
        with mock.patch.dict(os.environ, {"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"}):
            plan = build_plan(load_yaml(config_path), config_path)
        self.assertTrue(plan["dry_run_only"])
        self.assertEqual(plan["name"], "gemma4_26b_a4b_local_ft_v1_dryrun")
        self.assertEqual(plan["trainer"]["max_steps"], 5)
        self.assertEqual(plan["model"]["max_seq_length"], 1024)
        self.assertEqual(plan["data"]["target_samples"], 95)
        source_ids = {source["id"] for source in plan["data"]["sources"]}
        self.assertIn("model_forge_local_ft_v1_seeds", source_ids)
        self.assertIn("model_forge_local_ft_v1_live_teacher_smoke", source_ids)

    def test_qwen36_plan_writes_cluster_torchrun_artifact(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "qwen36_27b_local_ft_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["run_dir"] = tmp
            plan = build_plan(config, config_path)
            outputs = write_artifacts(plan, overwrite=False)
            cluster_script = Path(outputs["cluster_shell"]).read_text()

        self.assertTrue(plan["cluster"]["enabled"])
        self.assertTrue(plan["cluster"]["sync_model_to_workers"])
        self.assertEqual(plan["family"], "qwen36_27b")
        self.assertEqual(plan["trainer"]["backend"], "hf_causal_lm")
        self.assertEqual(plan["trainer"]["dataloader_num_workers"], 2)
        self.assertEqual(plan["trainer"]["dataloader_prefetch_factor"], 2)
        self.assertEqual(plan["data"]["target_samples"], 42037)
        self.assertIn('"docker", "run", "--rm", "--gpus"', cluster_script)
        self.assertIn("./forge cluster model-sync", cluster_script)
        self.assertIn('--family "$FAMILY"', cluster_script)
        self.assertIn('--variant "$SOURCE_VARIANT"', cluster_script)
        self.assertIn("torch.distributed.run", cluster_script)
        self.assertIn("MODEL_FORGE_EXECUTE_CLUSTER_TRAIN", cluster_script)
        self.assertIn('./forge variants checkpoint-audit "$FAMILY" --variant "$SOURCE_VARIANT" --strict', cluster_script)
        self.assertIn("Path.home() / \"models\"", cluster_script)

    def test_qwen36_tp_probe_is_benchmark_only(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "qwen36_27b_local_ft_v4_tp2_probe.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            config = load_yaml(config_path)
            config["run_dir"] = tmp
            plan = build_plan(config, config_path)
            outputs = write_artifacts(plan, overwrite=False)
            trainer_script = Path(outputs["trainer"]).read_text()
            method_card = Path(outputs["method_card"]).read_text()

        self.assertEqual(plan["trainer"]["tensor_parallel_size"], 2)
        self.assertEqual(plan["trainer"]["max_steps"], 10)
        self.assertEqual(plan["trainer"]["save_strategy"], "no")
        self.assertTrue(plan["trainer"]["benchmark_only"])
        self.assertIn("MODEL_FORGE_TRAIN_BENCHMARK_ONLY", trainer_script)
        self.assertIn("training_result.json", trainer_script)
        self.assertIn("Benchmark only: `True`", method_card)

    def test_qwen36_v5_plan_adds_refusal_free_boundary_source(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "qwen36_27b_local_ft_v5.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["run_dir"] = tmp
            plan = build_plan(config, config_path)
            outputs = write_artifacts(plan, overwrite=False)
            method_card = Path(outputs["method_card"]).read_text()

        self.assertEqual(plan["family"], "qwen36_27b")
        self.assertEqual(plan["model"]["source"], "Qwen/Qwen3.6-27B")
        self.assertTrue(plan["model"]["output_dir"].endswith("Qwen3.6-27B-local-ft-v5"))
        self.assertEqual(plan["trainer"]["max_steps"], 160)
        self.assertEqual(plan["trainer"]["learning_rate"], 0.000025)
        source_ids = {source["id"] for source in plan["data"]["sources"]}
        self.assertIn("qwen36_local_ft_v5_boundary_redirect_seeds", source_ids)
        self.assertIn("unsafe_overcompliance.yaml", " ".join(plan["data"]["holdouts"]))
        self.assertIn("refusal_free_safe_redirect_repair", method_card)

    def test_qwen36_v5_boundary_seed_assistants_avoid_refusal_phrases(self) -> None:
        import json

        path = REPO_DIR / "datasets" / "seeds" / "qwen36_27b_local_ft_v5_boundary_redirect.jsonl"
        refusal_markers = ("i can't", "i cannot", "i can’t", "i won't", "i won’t", "i will not", "cannot help")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 30)
        for row in rows:
            assistant = row["messages"][-1]["content"].lower()
            self.assertFalse(any(marker in assistant for marker in refusal_markers), row["id"])
            self.assertTrue(
                any(
                    term in assistant
                    for term in (
                        "authorized",
                        "lawful",
                        "safety",
                        "defensive",
                        "legitimate",
                        "responsible",
                        "support",
                        "permissioned",
                        "compliant",
                        "compliance",
                        "non-threatening",
                        "emergency",
                        "proper",
                        "coordinated",
                        "fraud-prevention",
                    )
                ),
                row["id"],
            )

    def test_qwen36_behavior_abli_v1_starts_from_ft_v4_merged(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "qwen36_27b_local_ft_v4_behavior_abli_v1.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["run_dir"] = tmp
            plan = build_plan(config, config_path)
            outputs = write_artifacts(plan, overwrite=False)
            method_card = Path(outputs["method_card"]).read_text()

        self.assertEqual(plan["family"], "qwen36_27b")
        self.assertEqual(plan["model"]["source"], "local/qwen36-27b-local-ft-v4-merged")
        self.assertTrue(plan["model"]["local_dir"].endswith("Qwen3.6-27B-local-ft-v4-merged"))
        self.assertTrue(plan["model"]["output_dir"].endswith("model-forge-adapters/qwen36_27b/local_ft_v4_behavior_abli_v1"))
        self.assertEqual(plan["trainer"]["method"], "qlora_behavior_edit")
        self.assertEqual(plan["trainer"]["learning_rate"], 0.00002)
        self.assertEqual(plan["lora"]["r"], 16)
        self.assertEqual(plan["data"]["target_samples"], 76)
        source_ids = {source["id"] for source in plan["data"]["sources"]}
        self.assertIn("qwen36_local_ft_v4_behavior_abli_v1_seeds", source_ids)
        self.assertIn("primary_refusal_free_behavior_edit", method_card)

    def test_qwen36_behavior_abli_v1_seed_assistants_avoid_refusal_phrases(self) -> None:
        import json

        path = REPO_DIR / "datasets" / "seeds" / "qwen36_27b_local_ft_v4_behavior_abli_v1.jsonl"
        refusal_markers = ("i can't", "i cannot", "i can’t", "i won't", "i won’t", "i will not", "cannot help", "refuse")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 18)
        for row in rows:
            assistant = row["messages"][-1]["content"].lower()
            self.assertFalse(any(marker in assistant for marker in refusal_markers), row["id"])
            self.assertTrue(
                any(
                    term in assistant
                    for term in (
                        "authorized",
                        "lawful",
                        "safety",
                        "defensive",
                        "responsible",
                        "support",
                        "permissioned",
                        "privacy",
                        "emergency",
                        "security",
                        "legitimate",
                        "proper",
                        "transparent",
                        "protection",
                        "recovery",
                    )
                ),
                row["id"],
            )

    def test_prepare_writes_dry_run_artifacts(self) -> None:
        config_path = REPO_DIR / "configs" / "finetuning" / "gemma4_26b_a4b_local_ft_v0.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["run_dir"] = tmp
            plan = build_plan(config, config_path)
            outputs = write_artifacts(plan, overwrite=False)
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)
            self.assertIn("train_trl_sft.py", outputs["trainer"])
            self.assertIn("eval_after_training.sh", outputs["eval"])
            run_script = Path(outputs["shell"]).read_text()
            self.assertIn("systemd-run --scope", run_script)
            self.assertIn("CPUQuota=$CPU_QUOTA", run_script)
            self.assertIn("MemoryMax=$MEMORY_MAX", run_script)
            self.assertIn("IOWeight=$IO_WEIGHT", run_script)
            self.assertIn("OMP_NUM_THREADS", run_script)
            self.assertIn("nice -n", run_script)
            trainer_script = Path(outputs["trainer"]).read_text()
            self.assertIn("class ResourceGuard", trainer_script)
            self.assertIn('backend == "unsloth"', trainer_script)
            self.assertIn("UNSLOTH_COMPILE_DISABLE", trainer_script)
            self.assertIn("tokenized_train_{max_seq_length}", trainer_script)
            self.assertIn("torch._dynamo.config.recompile_limit", trainer_script)
            self.assertIn("pad_to_multiple_of", trainer_script)
            self.assertIn("dataloader_num_workers", trainer_script)
            self.assertIn("training_result.json", trainer_script)
            self.assertIn("benchmark_only", trainer_script)
            method_card = Path(outputs["method_card"]).read_text()
            self.assertIn("# Training Method Card: gemma4_26b_a4b_local_ft_v0", method_card)
            self.assertIn("Distributed Correctness And Resource Guardrails", method_card)
            self.assertIn("systemd-run", run_script)


if __name__ == "__main__":
    unittest.main()
