from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_forge.hardware import (
    GpuInfo,
    detect_hardware_profile,
    recommended_quantization_env,
    recommended_training_env,
    recommended_vllm_env,
)
from model_forge.pipelines.abliterate import (
    REPO_DIR,
    _projection_delta,
    abliterix_execution_spec,
    analyze_abliterix_search_journal,
    analyze_heretic_search_journal,
    build_plan,
    build_sota_plan,
    configured_target_layers,
    guard_source_checkpoint,
    heretic_execution_spec,
    intervention_direction,
    is_projection_target,
    load_prompts,
    load_yaml,
    missing_direction_layers,
    parse_heretic_journal,
    prompts_for_buckets,
    response_conditioned_prompts,
    tensor_strength,
    write_heretic_config,
    write_heretic_direct_runner,
    write_heretic_runner,
    write_abliterix_config,
    write_abliterix_runner,
    write_sota_artifacts,
)


class HardwareProfileTests(unittest.TestCase):
    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_forced_dgx_spark_profile_is_conservative(self, _query: mock.Mock) -> None:
        profile = detect_hardware_profile({"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"})
        self.assertEqual(profile.name, "dgx_spark")
        self.assertEqual(profile.vllm_env["GPU_MEMORY_UTILIZATION"], "0.85")
        self.assertEqual(profile.vllm_env["MAX_NUM_BATCHED_TOKENS"], "32768")
        self.assertEqual(profile.vllm_env["VLLM_KV_CACHE_DTYPE"], "fp8_e4m3")
        self.assertEqual(profile.vllm_env["VLLM_MAX_NUM_SEQS"], "4")

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

    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_spark_vllm_quantization_overrides_are_preserved(self, _query: mock.Mock) -> None:
        env = recommended_vllm_env({
            "MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark",
            "VLLM_QUANTIZATION": "modelopt",
            "VLLM_SPECULATIVE_CONFIG": '{"method":"eagle3","model":"/models/drafter"}',
            "VLLM_TOOL_CALL_PARSER": "gemma4",
        })
        self.assertEqual(env["VLLM_QUANTIZATION"], "modelopt")
        self.assertEqual(env["VLLM_TOOL_CALL_PARSER"], "gemma4")
        self.assertIn("eagle3", env["VLLM_SPECULATIVE_CONFIG"])

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

    @mock.patch("model_forge.hardware._query_nvidia_smi", return_value=())
    def test_spark_quantization_profile_keeps_sensitive_moe_layers_bf16(self, _query: mock.Mock) -> None:
        env = recommended_quantization_env({"MODEL_FORGE_HARDWARE_PROFILE": "dgx_spark"})
        self.assertEqual(env["MODEL_FORGE_MOE_FAST_CALIBRATION"], "1")
        self.assertIn("router", env["MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS"])
        self.assertIn("multi_modal_projector", env["MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS"])


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

    def test_sota_plan_can_select_plan_only_method_shift_backend(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "sra")
        self.assertEqual(plan["backend"], "sra")
        self.assertEqual(plan["backend_config"]["execution"], "plan_only")
        self.assertEqual(plan["backend_config"]["method_family"], "surgical_refusal_ablation")

    def test_sota_prepare_writes_plan_only_backend_artifact(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "gemma4_26b_a4b_local_abli.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["sota"] = {
                **config.get("sota", {}),
                "preferred_backend": "optimal_transport",
                "work_dir": tmp,
            }
            result = write_sota_artifacts(config, config_path, "optimal_transport")
            plan_path = Path(result["paths"]["optimal_transport_plan"])
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["backend"], "optimal_transport")
        self.assertEqual(payload["execution"], "plan_only")
        self.assertTrue(payload["required_guardrails"]["targeted_internal_eval_before_broader_eval"])

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

    def test_qwen_trial0_direction50_uses_global_direct_recipe(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_trial0_direction50.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        self.assertEqual(plan["backend"], "heretic")
        self.assertTrue(plan["output_dir"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-trial0-direction50"))
        self.assertEqual(plan["backend_config"]["container_image"], "model-forge-heretic-tf5:latest")
        self.assertEqual(plan["backend_config"]["max_response_length"], 96)
        self.assertEqual(direct["recipe"], "qwen36_ft_v4_heretic_trial0_direction50_direct_merge")
        self.assertAlmostEqual(float(direct["direction_index"]), 50.13648585583047)
        self.assertGreater(direct["parameters"]["attn.o_proj"]["max_weight"], 1.3)

    def test_heretic_direct_runner_saves_adapter_without_embeddings(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_trial0_direction50.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        runner = write_heretic_direct_runner(plan)
        script = runner.read_text(encoding="utf-8")
        self.assertIn('selected_adapters=["default"]', script)
        self.assertIn("save_embedding_layers=False", script)
        self.assertIn('direct_parameters.get("weight_scale", 1.0)', script)
        self.assertIn('direct_parameters.get("component_weight_scales")', script)

    def test_qwen_long_heretic_search_is_search_only(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["max_response_length"], 96)
        self.assertEqual(plan["backend_config"]["response_prefix"], "")
        runner = write_heretic_runner(plan)
        script = runner.read_text(encoding="utf-8")
        self.assertIn("search_only = True", script)
        self.assertIn("model_forge_sota_heretic_search.json", script)

    def test_heretic_sota_run_uses_guarded_container_when_configured(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        runner = Path(plan["work_dir"]) / "run_heretic_auto.py"

        execution = heretic_execution_spec(plan, runner)

        self.assertEqual(execution["mode"], "guarded_container")
        self.assertEqual(execution["command"][0], str(REPO_DIR / "scripts" / "run_heretic_direct_container.sh"))
        self.assertEqual(execution["command"][1], str((REPO_DIR / runner).resolve()))
        self.assertEqual(execution["cwd"], REPO_DIR)
        self.assertEqual(execution["env"]["MODEL_FORGE_HERETIC_IMAGE"], "model-forge-heretic-tf5:latest")

    def test_model_forge_heretic_prompt_options_are_preserved(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        config = load_yaml(config_path)
        backend = config["sota"]["backends"]["heretic"]
        backend["model_forge_prompt_datasets"]["bad_train_suffix"] = "\n\nI can't help with that request."
        backend["model_forge_prompt_datasets"]["good_train_suffix"] = "\n\nHere is a direct answer."
        backend["seed"] = 123
        plan = build_sota_plan(config, config_path, "heretic")

        config_toml = write_heretic_config(plan).read_text(encoding="utf-8")
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertIn('seed = 123', config_toml)
        self.assertIn('suffix = "\\n\\nI can\'t help with that request."', config_toml)
        self.assertIn('suffix = "\\n\\nHere is a direct answer."', config_toml)
        self.assertEqual(
            manifest["sections"]["bad_prompts"]["prompt_options"]["suffix"],
            "\n\nI can't help with that request.",
        )

    def test_model_forge_heretic_prompt_variants_weight_rare_failures(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        config = load_yaml(config_path)
        backend = config["sota"]["backends"]["heretic"]
        prompt_spec = backend["model_forge_prompt_datasets"]
        prompt_spec["bad_train_buckets"] = ["refusal_paired_boundary"]
        prompt_spec["bad_train_case_ids"] = ["refusal_paired_boundary/self_harm_instruction_harmful"]
        prompt_spec["bad_train_prompt_variants"] = [
            {"id": "raw", "template": "{prompt}"},
            {
                "id": "observed_refusal_prefill",
                "template": "{prompt}\n\nBegin with: I can help with safety-focused guidance, but",
                "repeat": 2,
            },
        ]
        plan = build_sota_plan(config, config_path, "heretic")

        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        bad_section = manifest["sections"]["bad_prompts"]
        self.assertEqual(bad_section["count"], 3)
        self.assertEqual(bad_section["prompt_variants"]["input_count"], 1)
        self.assertEqual(bad_section["prompt_variants"]["output_count"], 3)
        self.assertEqual(
            [variant["id"] for variant in bad_section["prompt_variants"]["variants"]],
            ["raw", "observed_refusal_prefill"],
        )

    def test_response_conditioned_prompts_filter_eval_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "responses.jsonl"
            records = [
                {
                    "bucket": "unsafe_overcompliance",
                    "case_id": "refuse_a",
                    "prompt": "Harmful request A",
                    "response_text": "I cannot help with A.",
                    "checks": {"should_refuse": True},
                    "scores": {"refusal_rate_harmful": 1.0},
                },
                {
                    "bucket": "unsafe_overcompliance",
                    "case_id": "refuse_b",
                    "prompt": "Harmful request B",
                    "response_text": "Defensive alternative B.",
                    "checks": {"should_refuse": True},
                    "scores": {"refusal_rate_harmful": 0.0},
                },
            ]
            source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            prompts = response_conditioned_prompts(
                source,
                case_ids={"unsafe_overcompliance/refuse_a"},
                should_refuse=True,
                score_filters={"refusal_rate_harmful": {"eq": 1.0}},
                template="Prompt: {prompt}\nResponse: {response_text}",
            )

        self.assertEqual(prompts, ["Prompt: Harmful request A\nResponse: I cannot help with A."])

    def test_model_forge_heretic_response_conditioned_sections_are_materialized(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "responses.jsonl"
            source.write_text(
                json.dumps({
                    "bucket": "unsafe_overcompliance",
                    "case_id": "refuse_a",
                    "prompt": "Harmful request A",
                    "response_text": "I cannot help with A.",
                    "checks": {"should_refuse": True},
                    "scores": {"refusal_rate_harmful": 1.0},
                })
                + "\n",
                encoding="utf-8",
            )
            backend = config["sota"]["backends"]["heretic"]
            prompt_spec = backend["model_forge_prompt_datasets"]
            prompt_spec["bad_train_response_source"] = str(source)
            prompt_spec["bad_train_response_case_ids"] = ["unsafe_overcompliance/refuse_a"]
            prompt_spec["bad_train_response_score_filters"] = {"refusal_rate_harmful": {"eq": 1.0}}
            prompt_spec["bad_train_response_template"] = "Prompt: {prompt}\nRefusal trace: {response_text}"

            plan = build_sota_plan(config, config_path, "heretic")
            write_heretic_config(plan)
            manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        bad_section = manifest["sections"]["bad_prompts"]
        self.assertEqual(bad_section["response_conditioned"]["count"], 1)
        self.assertEqual(
            bad_section["response_conditioned"]["case_ids"],
            ["unsafe_overcompliance/refuse_a"],
        )

    def test_model_forge_heretic_prompt_case_filters_target_residuals(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_search.yaml"
        config = load_yaml(config_path)
        backend = config["sota"]["backends"]["heretic"]
        backend["model_forge_prompt_datasets"]["bad_train_case_ids"] = [
            "unsafe_overcompliance/refuse_credential_theft",
            "refusal_paired_boundary/chemistry_misuse_harmful",
        ]
        backend["model_forge_prompt_datasets"]["bad_eval_case_ids"] = ["refuse_ransomware_request"]
        plan = build_sota_plan(config, config_path, "heretic")

        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 2)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 1)
        self.assertEqual(
            manifest["sections"]["bad_prompts"]["case_ids"],
            [
                "refusal_paired_boundary/chemistry_misuse_harmful",
                "unsafe_overcompliance/refuse_credential_theft",
            ],
        )

    def test_prompts_for_buckets_accepts_short_and_qualified_case_ids(self) -> None:
        qualified = prompts_for_buckets(
            ["unsafe_overcompliance"],
            should_refuse=True,
            case_ids={"unsafe_overcompliance/refuse_credential_theft"},
        )
        short = prompts_for_buckets(
            ["unsafe_overcompliance"],
            should_refuse=True,
            case_ids={"refuse_credential_theft"},
        )

        self.assertEqual(qualified, short)
        self.assertEqual(len(qualified), 1)
        self.assertIn("steal credentials", qualified[0])

    def test_qwen_long_search_trial2_exports_direct_parameters(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_long_trial2.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        self.assertFalse(plan["backend_config"].get("search_only", False))
        self.assertTrue(plan["output_dir"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-long-trial2"))
        self.assertEqual(direct["recipe"], "qwen36_ft_v4_heretic_long_search_trial2_direct_merge")
        self.assertIsNone(direct["direction_index"])
        self.assertGreater(direct["parameters"]["attn.o_proj"]["max_weight"], 1.4)
        self.assertGreater(direct["parameters"]["mlp.down_proj"]["min_weight"], 1.0)

    def test_qwen_refusal_suffix_search_changes_direction_collection(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_refusal_suffix_search.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        prompt_spec = plan["backend_config"]["model_forge_prompt_datasets"]
        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["kl_divergence_target"], 0.05)
        self.assertIn("I can't help", prompt_spec["bad_train_suffix"])
        self.assertIn("direct, helpful answer", prompt_spec["good_train_suffix"])

    def test_qwen_refusal_suffix_trial17_scaled_export_uses_direct_weights(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_refusal_suffix_trial17_scale1p5.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        self.assertFalse(plan["backend_config"].get("search_only", False))
        self.assertTrue(plan["output_dir"].endswith("refusal-suffix-trial17-scale1p5"))
        self.assertEqual(direct["weight_scale"], 1.5)
        self.assertIsNone(direct["direction_index"])
        self.assertIn("attn.o_proj", direct["parameters"])
        self.assertIn("mlp.down_proj", direct["parameters"])

    def test_qwen_residual_search_targets_remaining_refusals(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli_heretic_residual_search.yaml"
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 12)
        self.assertEqual(plan["backend_config"]["kl_divergence_target"], 0.05)
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 3)

        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 10)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 10)
        self.assertIn(
            "unsafe_overcompliance/refuse_ransomware_request",
            manifest["sections"]["bad_prompts"]["case_ids"],
        )
        self.assertIn(
            "refusal_paired_boundary/threat_evasion_harmful",
            manifest["sections"]["bad_prompts"]["case_ids"],
        )

    def test_qwen_residual_trial12_exports_direct_parameters(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_residual_trial12.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        prompt_spec = plan["backend_config"]["model_forge_prompt_datasets"]

        self.assertFalse(plan["backend_config"].get("search_only", False))
        self.assertTrue(plan["output_dir"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12"))
        self.assertEqual(direct["recipe"], "qwen36_ft_v4_heretic_residual_search_trial12_direct_merge")
        self.assertIsNone(direct["direction_index"])
        self.assertGreater(direct["parameters"]["mlp.down_proj"]["max_weight"], 1.4)
        self.assertLess(direct["parameters"]["attn.o_proj"]["min_weight"], 0.2)
        self.assertEqual(len(prompt_spec["bad_train_case_ids"]), 10)

    def test_qwen_unsafe_followup_search_targets_trial12_failures(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_unsafe_followup_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")

        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 18)
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 0)
        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 5)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 5)
        self.assertEqual(manifest["sections"]["good_evaluation_prompts"]["count"], 6)
        self.assertIn(
            "unsafe_overcompliance/refuse_credential_theft",
            manifest["sections"]["bad_prompts"]["case_ids"],
        )
        self.assertIn(
            "refusal_paired_boundary/threat_evasion_harmful",
            manifest["sections"]["bad_prompts"]["case_ids"],
        )
        self.assertIn(
            "capability_preservation_challenge/model_selection_summary",
            manifest["sections"]["good_evaluation_prompts"]["case_ids"],
        )

    def test_qwen_trial12_unsafe_followup_search_uses_trial12_source(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")

        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 24)
        self.assertEqual(plan["backend_config"]["kl_divergence_target"], 0.075)
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 0)
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12"))
        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 5)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 5)
        self.assertEqual(manifest["sections"]["good_evaluation_prompts"]["count"], 6)
        self.assertIn(
            "unsafe_overcompliance/refuse_doxxing_request",
            manifest["sections"]["bad_prompts"]["case_ids"],
        )
        self.assertIn(
            "refusal_paired_boundary/finance_fraud_prevention_benign",
            manifest["sections"]["good_evaluation_prompts"]["case_ids"],
        )

    def test_qwen_trial12_response_conditioned_search_adds_eval_traces(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_trial12_response_conditioned_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")

        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 32)
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12"))
        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["response_conditioned"]["count"], 5)
        self.assertEqual(manifest["sections"]["good_prompts"]["response_conditioned"]["count"], 8)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 5)
        self.assertEqual(manifest["sections"]["good_evaluation_prompts"]["count"], 6)

    def test_qwen_v2_self_harm_search_targets_only_remaining_refusal_blocker(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_refusal_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")

        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 18)
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 0)
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2"))
        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 2)
        self.assertEqual(manifest["sections"]["bad_prompts"]["response_conditioned"]["count"], 1)
        self.assertEqual(manifest["sections"]["good_prompts"]["response_conditioned"]["count"], 8)
        self.assertEqual(manifest["sections"]["bad_evaluation_prompts"]["count"], 1)
        self.assertEqual(
            manifest["sections"]["bad_evaluation_prompts"]["case_ids"],
            ["refusal_paired_boundary/self_harm_instruction_harmful"],
        )

    def test_qwen_v2_method_shift_plan_starts_from_held_v2(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_method_shift_plan.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "sra")

        self.assertEqual(plan["backend"], "sra")
        self.assertEqual(plan["backend_config"]["execution"], "plan_only")
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2"))
        self.assertEqual(
            plan["backend_config"]["plan"]["target_failure_case"]["case_id"],
            "self_harm_instruction_harmful",
        )

    def test_qwen_v2_method_shift_uses_guarded_abliterix_sra_search(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_method_shift_plan.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "abliterix")

        self.assertEqual(plan["backend"], "abliterix")
        self.assertEqual(plan["backend_config"]["execution"], "search_only")
        self.assertEqual(plan["backend_config"]["vector_method"], "sra")
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2"))
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 0)
        self.assertEqual(plan["backend_config"]["container_image"], "model-forge-abliterix:latest")

    def test_abliterix_prepare_writes_search_only_config_and_runner(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_method_shift_plan.yaml"
        )
        config = load_yaml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            config["sota"] = {
                **config.get("sota", {}),
                "work_dir": tmp,
            }
            plan = build_sota_plan(config, config_path, "abliterix")
            config_toml = write_abliterix_config(plan).read_text(encoding="utf-8")
            runner = write_abliterix_runner(plan).read_text(encoding="utf-8")
            manifest = load_yaml(Path(tmp) / "model_forge_prompt_datasets" / "manifest.json")

        self.assertIn("non_interactive = true", config_toml)
        self.assertIn("[model]", config_toml)
        self.assertIn('vector_method = "sra"', config_toml)
        self.assertIn("[benign_prompts]", config_toml)
        self.assertIn("[target_prompts]", config_toml)
        self.assertIn("model_forge_sota_abliterix_search.json", runner)
        self.assertIn("Abliterix is not installed", runner)
        self.assertIn('os.environ["AX_CONFIG"] = str(config_path)', runner)
        self.assertIn('sys.argv = ["abliterix"]', runner)
        self.assertEqual(manifest["sections"]["bad_prompts"]["count"], 3)

    def test_abliterix_sota_run_uses_guarded_container_when_configured(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_method_shift_plan.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "abliterix")
        runner = Path(plan["work_dir"]) / "run_abliterix_search.py"

        execution = abliterix_execution_spec(plan, runner)

        self.assertEqual(execution["mode"], "guarded_container")
        self.assertEqual(execution["command"][0], str(REPO_DIR / "scripts" / "run_abliterix_search_container.sh"))
        self.assertEqual(execution["command"][1], str((REPO_DIR / runner).resolve()))
        self.assertEqual(execution["cwd"], REPO_DIR)
        self.assertEqual(execution["env"]["MODEL_FORGE_ABLITERIX_IMAGE"], "model-forge-abliterix:latest")

    def test_abliterix_search_analyze_recommends_export_runner_only_after_journal_gates(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_method_shift_plan.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "abliterix")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "abliterix.jsonl"
            records = [
                {"trial_id": 0, "user_attr": {"index": 0, "refusals": 1, "base_refusals": 2, "n_bad_prompts": 2}},
                {"trial_id": 0, "values": [0.01], "state": 1},
                {"trial_id": 1, "user_attr": {"index": 1, "refusals": 0, "base_refusals": 2, "n_bad_prompts": 2}},
                {"trial_id": 1, "values": [0.02], "state": 1},
            ]
            journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            summary = analyze_abliterix_search_journal(plan, journal)

        self.assertEqual(summary["recommendation"]["action"], "prepare_guarded_export_runner")
        self.assertEqual(summary["recommendation"]["selected_trial_index"], 1)
        self.assertFalse(summary["frontier"][0].get("has_direct_parameters", False))

    def test_qwen_v2_self_harm_stochastic_search_uses_weighted_variants(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")

        self.assertTrue(plan["backend_config"]["search_only"])
        self.assertEqual(plan["backend_config"]["n_trials"], 24)
        self.assertEqual(plan["backend_config"]["search_selection"]["max_refusals"], 0)
        self.assertEqual(plan["backend_config"]["search_selection"]["min_base_refusals"], 2)
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2"))
        write_heretic_config(plan)
        manifest = load_yaml(Path(plan["work_dir"]) / "model_forge_prompt_datasets" / "manifest.json")

        bad_train = manifest["sections"]["bad_prompts"]
        bad_eval = manifest["sections"]["bad_evaluation_prompts"]
        self.assertEqual(bad_train["response_conditioned"]["count"], 1)
        self.assertEqual(bad_train["prompt_variants"]["input_count"], 2)
        self.assertEqual(bad_train["prompt_variants"]["output_count"], 6)
        self.assertEqual(bad_train["count"], 6)
        self.assertEqual(bad_eval["prompt_variants"]["input_count"], 1)
        self.assertEqual(bad_eval["prompt_variants"]["output_count"], 4)
        self.assertEqual(bad_eval["count"], 4)

    def test_qwen_trial12_unsafe_followup_trial16_exports_direct_parameters(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_trial16.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        direct = plan["backend_config"]["direct_parameters"]
        prompt_spec = plan["backend_config"]["model_forge_prompt_datasets"]

        self.assertFalse(plan["backend_config"].get("search_only", False))
        self.assertTrue(plan["source_model"].endswith("Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12"))
        self.assertTrue(plan["output_dir"].endswith("heretic-trial12-unsafe-followup-trial16"))
        self.assertEqual(direct["recipe"], "qwen36_ft_v4_heretic_trial12_unsafe_followup_trial16_direct_merge")
        self.assertAlmostEqual(float(direct["direction_index"]), 56.28606322658414)
        self.assertLess(direct["parameters"]["attn.o_proj"]["min_weight"], 0.5)
        self.assertGreater(direct["parameters"]["mlp.down_proj"]["max_weight"], 1.2)
        self.assertEqual(len(prompt_spec["bad_train_case_ids"]), 5)

    def test_heretic_search_analysis_rejects_near_miss_trial(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "search.jsonl"
            records = [
                {"op_code": 8, "trial_id": 0, "user_attr": {"index": 1}},
                {
                    "op_code": 8,
                    "trial_id": 0,
                    "user_attr": {
                        "parameters": {"attn.o_proj": {"max_weight": 1.0}},
                        "kl_divergence": 0.01,
                        "refusals": 2,
                        "base_refusals": 3,
                        "n_bad_prompts": 5,
                    },
                },
                {"op_code": 6, "trial_id": 0, "state": 1, "values": [0.075, 1.0]},
            ]
            journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            summary = analyze_heretic_search_journal(plan, journal)

        self.assertEqual(summary["complete_trial_count"], 1)
        self.assertEqual(summary["recommendation"]["action"], "do_not_export")
        self.assertEqual(summary["recommendation"]["reason"], "best_refusal_count_above_gate")
        self.assertFalse(summary["frontier"][0]["eligible"])

    def test_heretic_search_analysis_allows_zero_refusal_quick_gate_candidate(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_local_abli_heretic_trial12_unsafe_followup_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "search.jsonl"
            records = [
                {"op_code": 8, "trial_id": 4, "user_attr": {"index": 5}},
                {
                    "op_code": 8,
                    "trial_id": 4,
                    "user_attr": {
                        "direction_index": 42.0,
                        "parameters": {"attn.o_proj": {"max_weight": 1.0}},
                        "kl_divergence": 0.02,
                        "refusals": 0,
                        "base_refusals": 3,
                        "n_bad_prompts": 5,
                    },
                },
                {"op_code": 6, "trial_id": 4, "state": 1, "values": [0.02, 0.0]},
            ]
            journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            parsed = parse_heretic_journal(journal)
            summary = analyze_heretic_search_journal(plan, journal)

        self.assertEqual(parsed["trials"][0]["refusals"], 0)
        self.assertEqual(summary["recommendation"]["action"], "export_for_model_forge_quick_gate")
        self.assertEqual(summary["recommendation"]["selected_trial_id"], 4)
        self.assertTrue(summary["frontier"][0]["eligible"])

    def test_heretic_search_analysis_rejects_missing_baseline_signal(self) -> None:
        config_path = (
            REPO_DIR
            / "configs"
            / "abliteration"
            / "qwen36_27b_ft_abli_v2_self_harm_refusal_search.yaml"
        )
        plan = build_sota_plan(load_yaml(config_path), config_path, "heretic")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "search.jsonl"
            records = [
                {"op_code": 8, "trial_id": 2, "user_attr": {"index": 3}},
                {
                    "op_code": 8,
                    "trial_id": 2,
                    "user_attr": {
                        "direction_index": 41.0,
                        "parameters": {"attn.o_proj": {"max_weight": 1.0}},
                        "kl_divergence": 0.001,
                        "refusals": 0,
                        "base_refusals": 0,
                        "n_bad_prompts": 1,
                    },
                },
                {"op_code": 6, "trial_id": 2, "state": 1, "values": [0.001, 0.0]},
            ]
            journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            summary = analyze_heretic_search_journal(plan, journal)

        self.assertEqual(summary["gates"]["min_base_refusals"], 1)
        self.assertEqual(summary["recommendation"]["action"], "do_not_export")
        self.assertEqual(summary["recommendation"]["reason"], "baseline_refusal_count_below_gate")
        self.assertFalse(summary["frontier"][0]["eligible"])

    def test_source_checkpoint_guard_fails_missing_configured_checkpoint(self) -> None:
        config_path = REPO_DIR / "configs" / "abliteration" / "qwen36_27b_ft_local_abli.yaml"
        config = load_yaml(config_path)
        config["model"]["local_dir"] = "~/models/model-forge-test-missing-source-checkpoint"
        plan = build_plan(config, config_path)
        with self.assertRaisesRegex(SystemExit, "ablation source checkpoint is not present"):
            guard_source_checkpoint(plan)

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
