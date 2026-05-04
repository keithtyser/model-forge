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
        self.assertEqual(plan["baseline"]["target_to_beat"], "Jackrong/Gemopus-4-26B-A4B-it")
        self.assertEqual(plan["trainer"]["method"], "qlora")
        self.assertEqual(plan["lora"]["r"], 64)
        self.assertEqual(plan["hardware"]["profile"], "dgx_spark")
        self.assertEqual(plan["resource_policy"]["cpu_quota"], "80%")
        self.assertEqual(plan["resource_policy"]["memory_max"], "85%")
        self.assertEqual(plan["resource_policy"]["reserve_cores"], 1)

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
            self.assertIn("dataloader_num_workers", trainer_script)


if __name__ == "__main__":
    unittest.main()
