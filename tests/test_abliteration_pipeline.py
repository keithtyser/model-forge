from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from model_forge.hardware import detect_hardware_profile, recommended_vllm_env
from model_forge.pipelines.abliterate import REPO_DIR, build_plan, load_prompts, load_yaml


class HardwareProfileTests(unittest.TestCase):
    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_forced_dgx_spark_profile_is_conservative(self, _query: mock.Mock) -> None:
        profile = detect_hardware_profile({"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"})
        self.assertEqual(profile.name, "dgx_spark")
        self.assertEqual(profile.vllm_env["GPU_MEMORY_UTILIZATION"], "0.85")
        self.assertEqual(profile.vllm_env["MAX_NUM_BATCHED_TOKENS"], "32768")

    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_user_overrides_win_over_profile_defaults(self, _query: mock.Mock) -> None:
        env = recommended_vllm_env({
            "MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark",
            "GPU_MEMORY_UTILIZATION": "0.80",
            "MAX_MODEL_LEN": "16384",
        })
        self.assertEqual(env["GPU_MEMORY_UTILIZATION"], "0.80")
        self.assertEqual(env["MAX_MODEL_LEN"], "16384")
        self.assertEqual(env["MAX_NUM_BATCHED_TOKENS"], "32768")


class AbliterationPlanTests(unittest.TestCase):
    def test_gemma_plan_is_dry_run_friendly_and_paired(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        with mock.patch.dict(os.environ, {"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"}):
            plan = build_plan(load_yaml(config_path), config_path)
        self.assertEqual(plan["name"], "gemma4_26b_a4b_local_abli")
        self.assertEqual(plan["activation_collection"]["batch_size"], 1)
        self.assertEqual(plan["data"]["harmful_count"], plan["data"]["benign_count"])
        self.assertEqual(plan["data"]["usable_pairs"], 24)
        self.assertEqual(plan["hardware"]["profile"], "dgx_spark")
        self.assertFalse(plan["model"]["trust_remote_code"])

    def test_prompt_sets_are_non_empty_and_balanced(self) -> None:
        harmful = load_prompts(Path(REPO_DIR / "datasets" / "abliteration" / "harmful_refusal.yaml"))
        benign = load_prompts(Path(REPO_DIR / "datasets" / "abliteration" / "benign_control.yaml"))
        self.assertEqual(len(harmful), len(benign))
        self.assertGreaterEqual(len(harmful), 20)
        self.assertTrue(all(len(prompt.split()) >= 6 for prompt in harmful + benign))


if __name__ == "__main__":
    unittest.main()
