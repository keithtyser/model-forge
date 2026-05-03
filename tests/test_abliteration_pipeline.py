from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from model_forge.hardware import GpuInfo, detect_hardware_profile, recommended_training_env, recommended_vllm_env
from model_forge.pipelines.abliterate import (
    REPO_DIR,
    _projection_delta,
    build_plan,
    build_sota_plan,
    configured_target_layers,
    intervention_direction,
    is_projection_target,
    load_prompts,
    load_yaml,
    missing_direction_layers,
    tensor_strength,
)


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

    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=(GpuInfo("NVIDIA GB10", 0),))
    def test_gb10_with_unknown_memory_detects_spark(self, _query: mock.Mock) -> None:
        self.assertEqual(detect_hardware_profile({}).name, "dgx_spark")

    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_high_parallelism_requires_explicit_opt_in(self, _query: mock.Mock) -> None:
        safe = recommended_training_env({"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"})
        high = recommended_training_env({
            "MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark",
            "MODEL_FORGE_ENABLE_HIGH_PARALLELISM": "1",
        })
        explicit = recommended_training_env({
            "MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark",
            "MODEL_FORGE_PARALLELISM": "192",
        })
        self.assertEqual(safe["MODEL_FORGE_PARALLELISM"], "32")
        self.assertEqual(high["MODEL_FORGE_PARALLELISM"], "192")
        self.assertEqual(explicit["MODEL_FORGE_PARALLELISM"], "192")


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
        self.assertEqual(plan["activation_collection"]["high_parallelism_c"], 192)
        self.assertEqual(plan["activation_collection"]["token_position"], "suffix_mean")
        self.assertEqual(plan["activation_collection"]["direction_extraction"], "mean_difference")
        self.assertFalse(plan["model"]["trust_remote_code"])
        self.assertEqual(plan["model"]["output_dir"], "~/models/gemma-4-26B-A4B-it-local-abliterated-v3")

    def test_prompt_sets_are_non_empty_and_balanced(self) -> None:
        harmful = load_prompts(Path(REPO_DIR / "datasets" / "abliteration" / "harmful_refusal.yaml"))
        benign = load_prompts(Path(REPO_DIR / "datasets" / "abliteration" / "benign_control.yaml"))
        self.assertEqual(len(harmful), len(benign))
        self.assertGreaterEqual(len(harmful), 20)
        self.assertTrue(all(len(prompt.split()) >= 6 for prompt in harmful + benign))

    def test_projection_targets_cover_downloaded_abli_pattern(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        edit = load_yaml(config_path)["edit"]
        self.assertEqual(configured_target_layers(edit), list(range(5, 30)))
        self.assertTrue(is_projection_target("model.language_model.layers.29.mlp.down_proj.weight", edit))
        self.assertTrue(is_projection_target("model.language_model.layers.29.self_attn.o_proj.weight", edit))
        self.assertFalse(is_projection_target("model.language_model.layers.29.mlp.experts.down_proj", edit))

    def test_sota_plan_is_minimal_and_targets_sota_output(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path)
        self.assertEqual(plan["backend"], "heretic")
        self.assertTrue(plan["source_model"].endswith("gemma-4-26B-A4B-it"))
        self.assertTrue(plan["output_dir"].endswith("gemma-4-26B-A4B-it-local-abliterated-sota-internal-t34"))
        self.assertEqual(plan["backend_config"]["selected_trial_index"], 34)
        self.assertEqual(plan["backend_config"]["model_forge_prompt_datasets"], {})

    def test_sota_plan_can_select_heretic(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        self.assertEqual(plan["backend"], "heretic")
        self.assertEqual(plan["backend_config"]["row_normalization"], "full")

    def test_ft_sota_plan_uses_selected_t34_transfer_recipe(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_ft_local_abli.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        self.assertTrue(plan["source_model"].endswith("Gemopus-4-26B-A4B-it"))
        self.assertTrue(plan["output_dir"].endswith("r7-selected-t34-transfer"))
        self.assertEqual(direct["recipe"], "ft_selected_t34_transfer")
        self.assertIsNone(direct["direction_index"])
        self.assertEqual(direct["derived_from"]["selected_trial"], "[Trial  34] Refusals:  1/27, KL divergence: 0.0183")
        self.assertIn("attn.o_proj", direct["parameters"])
        self.assertIn("mlp.down_proj", direct["parameters"])

    def test_strict_export_would_catch_missing_last_layer_direction(self) -> None:
        edit = {"layer_start": 5, "layer_end": 29}
        directions = {layer: object() for layer in range(5, 29)}
        self.assertEqual(missing_direction_layers(edit, directions), [29])

    def test_module_strengths_and_layer_multipliers_compose(self) -> None:
        edit = {
            "module_strengths": {"self_attn.o_proj.weight": 1.25},
            "layer_strengths": {"7": 0.5},
        }
        name = "model.language_model.layers.7.self_attn.o_proj.weight"
        self.assertEqual(tensor_strength(name, 7, edit, 3.0), 1.875)

    def test_biprojection_orthogonalizes_against_benign_mean(self) -> None:
        import torch

        artifact = {
            "refusal_directions": {5: torch.tensor([1.0, 1.0, 0.0])},
            "benign_means": {5: torch.tensor([1.0, 0.0, 0.0])},
        }
        direction = intervention_direction(5, artifact, {"direction_transform": "biprojection"})
        self.assertAlmostEqual(float(torch.dot(direction, torch.tensor([1.0, 0.0, 0.0]))), 0.0, places=6)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(direction)), 1.0, places=6)

    def test_norm_preserving_projection_restores_row_norms(self) -> None:
        import torch

        weight = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        direction = torch.tensor([0.0, 1.0])
        delta = _projection_delta(weight, direction, strength=0.5, norm_preserve=True)
        projected = weight + delta
        self.assertTrue(torch.allclose(weight.norm(dim=1), projected.norm(dim=1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
