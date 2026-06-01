from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_DIR / "scripts" / "model_forge_dgx.py"
SPEC = importlib.util.spec_from_file_location("model_forge_dgx_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
model_forge_dgx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_forge_dgx)


class ModelForgeDgxServeTests(unittest.TestCase):
    def test_adapter_variant_serves_base_with_lora_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            (root / "adapter-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {
                    "chat_template": {
                        "tool_call_parser": "llama3_json",
                        "default_chat_template_kwargs": {"enable_thinking": False},
                    }
                },
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                    "local_ft": {
                        "repo_id": "org/base",
                        "local_dir": "adapter-model",
                        "served_model_name": "local/ft",
                        "adapter": True,
                        "base_variant": "base",
                        "lora_rank": 64,
                    },
                },
            }
            env: dict[str, str] = {}
            details = model_forge_dgx.configure_serving_variant(family, "local_ft", env)

            self.assertEqual(env["MODEL_FORGE_MODEL"], str(root / "base-model"))
            self.assertEqual(env["MODEL_FORGE_SERVED_MODEL_NAME"], "org/base")
            self.assertEqual(env["MODEL_FORGE_LORA_MODULES"], f"local/ft={root / 'adapter-model'}")
            self.assertEqual(env["VLLM_ENABLE_LORA"], "1")
            self.assertEqual(env["VLLM_MAX_LORAS"], "1")
            self.assertEqual(env["VLLM_MAX_LORA_RANK"], "64")
            self.assertEqual(details["adapter"], env["MODEL_FORGE_LORA_MODULES"])
            self.assertEqual(details["base_variant"], "base")

    def test_merged_adapter_variant_serves_merged_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            (root / "adapter-model").mkdir()
            (root / "merged-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {
                    "chat_template": {
                        "tool_call_parser": "llama3_json",
                        "default_chat_template_kwargs": {"enable_thinking": False},
                    }
                },
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                    "local_ft": {
                        "repo_id": "org/base",
                        "local_dir": "adapter-model",
                        "served_model_name": "local/ft",
                        "adapter": True,
                        "base_variant": "base",
                        "merged_local_dir": "merged-model",
                        "serve_strategy": "merged",
                    },
                },
            }
            env: dict[str, str] = {}
            details = model_forge_dgx.configure_serving_variant(family, "local_ft", env)

            self.assertEqual(env["MODEL_FORGE_MODEL"], str(root / "merged-model"))
            self.assertEqual(env["MODEL_FORGE_SERVED_MODEL_NAME"], "local/ft")
            self.assertNotIn("MODEL_FORGE_LORA_MODULES", env)
            self.assertEqual(details["adapter"], f"merged from {root / 'adapter-model'}")

    def test_live_lora_on_adapter_base_uses_merged_base_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            (root / "ft-adapter").mkdir()
            (root / "ft-merged").mkdir()
            (root / "abli-adapter").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                    "local_ft": {
                        "repo_id": "org/base",
                        "local_dir": "ft-adapter",
                        "merged_local_dir": "ft-merged",
                        "served_model_name": "local/ft",
                        "adapter": True,
                        "base_variant": "base",
                    },
                    "local_ft_abli_lora": {
                        "repo_id": "org/base",
                        "local_dir": "abli-adapter",
                        "served_model_name": "local/ft-abli-lora",
                        "adapter": True,
                        "base_variant": "local_ft",
                        "lora_rank": 3,
                    },
                },
            }
            env: dict[str, str] = {}
            details = model_forge_dgx.configure_serving_variant(family, "local_ft_abli_lora", env)

            self.assertEqual(env["MODEL_FORGE_MODEL"], str(root / "ft-merged"))
            self.assertEqual(env["MODEL_FORGE_SERVED_MODEL_NAME"], "local/ft")
            self.assertEqual(env["MODEL_FORGE_LORA_MODULES"], f"local/ft-abli-lora={root / 'abli-adapter'}")
            self.assertEqual(env["VLLM_MAX_LORA_RANK"], "8")
            self.assertEqual(details["base_variant"], "local_ft")

    def test_lora_rank_cap_uses_vllm_supported_values(self) -> None:
        self.assertEqual(model_forge_dgx.vllm_lora_rank_cap(1), 1)
        self.assertEqual(model_forge_dgx.vllm_lora_rank_cap(3), 8)
        self.assertEqual(model_forge_dgx.vllm_lora_rank_cap(64), 64)
        self.assertEqual(model_forge_dgx.vllm_lora_rank_cap(65), 128)

    def test_full_model_variant_serves_its_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {
                    "chat_template": {
                        "tool_call_parser": "llama3_json",
                        "default_chat_template_kwargs": {"enable_thinking": False},
                    }
                },
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                },
            }
            env: dict[str, str] = {}
            details = model_forge_dgx.configure_serving_variant(family, "base", env)

            self.assertEqual(env["MODEL_FORGE_MODEL"], str(root / "base-model"))
            self.assertEqual(env["MODEL_FORGE_SERVED_MODEL_NAME"], "org/base")
            self.assertEqual(env["VLLM_TOOL_CALL_PARSER"], "llama3_json")
            self.assertEqual(env["VLLM_ENABLE_AUTO_TOOL_CHOICE"], "true")
            self.assertEqual(env["VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"], '{"enable_thinking": false}')
            self.assertNotIn("MODEL_FORGE_LORA_MODULES", env)
            self.assertEqual(details["adapter"], "")

    def test_teacher_launcher_has_single_server_and_resource_guards(self) -> None:
        script = (REPO_DIR / "scripts" / "serve_teacher_vllm_dgx_spark.sh").read_text(encoding="utf-8")
        self.assertIn("refusing to start teacher server", script)
        self.assertIn("--mem-limit-gb", script)
        self.assertIn("--mem-swap-limit-gb", script)
        self.assertIn("--max-num-seqs", script)
        self.assertIn("local/qwen35-9b-teacher", script)

    def test_container_merge_runner_uses_resource_guards_and_host_user(self) -> None:
        script = (REPO_DIR / "scripts" / "run_merge_peft_container.sh").read_text(encoding="utf-8")
        self.assertIn("model-forge-posttrain-tf5:latest", script)
        self.assertIn("--cpus=\"$CPU_LIMIT\"", script)
        self.assertIn("--memory=\"${TOTAL_MEM_GB}g\"", script)
        self.assertIn("--memory-swap=\"${TOTAL_MEM_GB}g\"", script)
        self.assertIn('--user "$(id -u):$(id -g)"', script)
        self.assertIn("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", script)
        self.assertIn("MODEL_FORGE_MIN_FREE_DISK_FRACTION", script)

    def test_heretic_container_runner_uses_cuda_and_resource_guards(self) -> None:
        script = (REPO_DIR / "scripts" / "run_heretic_direct_container.sh").read_text(encoding="utf-8")
        self.assertIn("model-forge-heretic-tf5:latest", script)
        self.assertIn("--gpus all", script)
        self.assertIn("--cpus=\"$CPU_LIMIT\"", script)
        self.assertIn("--memory=\"${TOTAL_MEM_GB}g\"", script)
        self.assertIn("--memory-swap=\"${TOTAL_MEM_GB}g\"", script)
        self.assertIn('--user "$(id -u):$(id -g)"', script)
        self.assertIn("MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION", script)
        self.assertIn("MODEL_FORGE_MIN_FREE_DISK_FRACTION", script)

    def test_generic_vllm_launcher_uses_model_family_env(self) -> None:
        script = (REPO_DIR / "scripts" / "serve_vllm_dgx_spark.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL=${1:-${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}}", script)
        self.assertIn("SERVED_MODEL_NAME=${MODEL_FORGE_SERVED_MODEL_NAME:-}", script)
        self.assertIn("--served-model-name", script)
        self.assertIn("--default-chat-template-kwargs", script)

    def test_qwen_launcher_builds_valid_default_chat_template_kwargs(self) -> None:
        script = (REPO_DIR / "scripts" / "dgx_spark_serve_qwen.sh").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-"{\\"enable_thinking\\": ${QWEN_ENABLE_THINKING}}"}', script)
        self.assertIn('--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"', script)
        self.assertIn("VLLM_EXTRA_ARGS", script)
        self.assertIn('read -r -a lora_modules <<< "$MODEL_FORGE_LORA_MODULES"', script)
        self.assertIn('VLLM_ARGS+=(--lora-modules "${lora_modules[@]}")', script)

    def test_family_serving_defaults_are_applied_before_hardware_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                },
                "serve": {
                    "script": "scripts/dgx_spark_serve_qwen.sh",
                    "default_image": "vllm-node-tf5",
                    "default_gpu_memory_utilization": "0.78",
                    "default_max_model_len": "32768",
                    "default_max_num_batched_tokens": "16384",
                    "default_max_num_seqs": "4",
                },
            }
            captured_env = {}

            def fake_run(_cmd, env=None):
                captured_env.update(env or {})

            with mock.patch.object(model_forge_dgx, "recommended_vllm_env", return_value={"GPU_MEMORY_UTILIZATION": "0.95", "MAX_NUM_BATCHED_TOKENS": "32768"}):
                with mock.patch.object(model_forge_dgx, "detect_hardware_profile", None):
                    with mock.patch.object(model_forge_dgx, "run_or_exit", return_value=None):
                        with mock.patch.object(model_forge_dgx, "run", side_effect=fake_run):
                            model_forge_dgx.action_serve(family, "test_family", "base")

        self.assertEqual(
            captured_env["MODEL_FORGE_MODEL"],
            str(root / "base-model"),
        )
        self.assertEqual(captured_env["GPU_MEMORY_UTILIZATION"], "0.78")
        self.assertEqual(captured_env["MAX_MODEL_LEN"], "32768")
        self.assertEqual(captured_env["MAX_NUM_BATCHED_TOKENS"], "16384")
        self.assertEqual(captured_env["VLLM_MAX_NUM_SEQS"], "4")
        self.assertEqual(captured_env["MODEL_FORGE_SPARK_VLLM_IMAGE"], "vllm-node-tf5")

    def test_full_checkpoint_serve_runs_checkpoint_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                },
                "serve": {"script": "scripts/dgx_spark_serve_qwen.sh"},
            }
            captured: list[list[str]] = []
            with mock.patch.object(model_forge_dgx, "recommended_vllm_env", None):
                with mock.patch.object(model_forge_dgx, "detect_hardware_profile", None):
                    with mock.patch.object(model_forge_dgx, "run_or_exit", side_effect=lambda cmd, env=None, error="": captured.append(cmd)):
                        with mock.patch.object(model_forge_dgx, "run", side_effect=lambda cmd, env=None: captured.append(cmd)):
                            model_forge_dgx.action_serve(family, "test_family", "base")

        self.assertEqual(
            captured[0],
            [str(REPO_DIR / "forge"), "variants", "checkpoint-audit", "test_family", "--variant", "base", "--strict"],
        )

    def test_full_checkpoint_serve_audit_failure_exits_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                },
                "serve": {"script": "scripts/dgx_spark_serve_qwen.sh"},
            }
            with mock.patch.object(model_forge_dgx, "recommended_vllm_env", None):
                with mock.patch.object(model_forge_dgx, "detect_hardware_profile", None):
                    with mock.patch.object(model_forge_dgx, "run_or_exit", side_effect=SystemExit("audit failed")):
                        with self.assertRaises(SystemExit) as raised:
                            model_forge_dgx.action_serve(family, "test_family", "base")
                    self.assertEqual(str(raised.exception), "audit failed")

    def test_full_checkpoint_serve_runs_launcher_after_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                },
                "serve": {"script": "scripts/dgx_spark_serve_qwen.sh"},
            }
            captured: list[list[str]] = []
            with mock.patch.object(model_forge_dgx, "recommended_vllm_env", None):
                with mock.patch.object(model_forge_dgx, "detect_hardware_profile", None):
                    with mock.patch.object(model_forge_dgx, "run_or_exit", return_value=None):
                        with mock.patch.object(model_forge_dgx, "run", side_effect=lambda cmd, env=None: captured.append(cmd)):
                            model_forge_dgx.action_serve(family, "test_family", "base")

        self.assertEqual(
            captured[0],
            [str(REPO_DIR / "scripts/dgx_spark_serve_qwen.sh")],
        )

    def test_live_lora_serve_does_not_require_optional_merged_checkpoint_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base-model").mkdir()
            (root / "adapter-model").mkdir()
            family = {
                "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
                "default_models_dir": str(root),
                "architecture": {},
                "variants": {
                    "base": {
                        "repo_id": "org/base",
                        "local_dir": "base-model",
                        "served_model_name": "org/base",
                    },
                    "local_ft": {
                        "repo_id": "org/base",
                        "local_dir": "adapter-model",
                        "merged_local_dir": "merged-model",
                        "served_model_name": "local/ft",
                        "adapter": True,
                        "base_variant": "base",
                    },
                },
                "serve": {"script": "scripts/dgx_spark_serve_qwen.sh"},
            }
            captured: list[list[str]] = []
            with mock.patch.object(model_forge_dgx, "recommended_vllm_env", None):
                with mock.patch.object(model_forge_dgx, "detect_hardware_profile", None):
                    with mock.patch.object(model_forge_dgx, "run", side_effect=lambda cmd, env=None: captured.append(cmd)):
                        model_forge_dgx.action_serve(family, "test_family", "local_ft")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], [str(REPO_DIR / "scripts/dgx_spark_serve_qwen.sh")])

    def test_compare_uses_generic_variant_arguments(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_DIR) as tmp:
            root = Path(tmp)
            output_root = root / "results"
            (output_root / "qwen36_27b_base_dgx_spark").mkdir(parents=True)
            (output_root / "qwen36_27b_local_ft_abli_dgx_spark").mkdir(parents=True)
            (output_root / "qwen36_27b_local_ft_abli_artifacts_dgx_spark").mkdir(parents=True)
            external_root = root / "external"
            (external_root / "local_ft_abli").mkdir(parents=True)
            family = {
                "eval": {
                    "output_root": str(output_root),
                    "full_suffix": "{family}_{variant}_dgx_spark",
                    "artifact_suffix": "{family}_{variant}_artifacts_dgx_spark",
                },
                "comparison": {"output_dir": str(root / "comparison")},
                "external": {"output_root": str(external_root)},
                "variants": {
                    "base": {},
                    "local_ft_abli": {},
                    "local_ft_abli_nvfp4_modelopt": {},
                },
            }
            captured: list[list[str]] = []
            with mock.patch.object(model_forge_dgx, "run", side_effect=lambda cmd: captured.append(cmd)):
                model_forge_dgx.action_compare(family, "qwen36_27b")

        command = captured[0]
        self.assertIn("--run", command)
        self.assertIn(f"local_ft_abli={output_root / 'qwen36_27b_local_ft_abli_dgx_spark'}", command)
        self.assertIn("--artifact-run", command)
        self.assertIn(f"local_ft_abli={output_root / 'qwen36_27b_local_ft_abli_artifacts_dgx_spark'}", command)
        self.assertIn("--external-run", command)
        self.assertIn(f"local_ft_abli={external_root / 'local_ft_abli'}", command)

    def test_eval_passthrough_args_reach_runner(self) -> None:
        captured: dict[str, object] = {}

        def fake_action_eval(family, family_name, variant, kind, extra_args=None):
            captured["family"] = family
            captured["family_name"] = family_name
            captured["variant"] = variant
            captured["kind"] = kind
            captured["extra_args"] = extra_args

        original_argv = sys.argv
        try:
            sys.argv = ["model_forge_dgx.py", "qwen36_27b", "full", "base", "--dry-run"]
            with mock.patch.object(model_forge_dgx, "load_family", return_value={"name": "qwen36_27b"}):
                with mock.patch.object(model_forge_dgx, "action_eval", side_effect=fake_action_eval):
                    model_forge_dgx.main()
        finally:
            sys.argv = original_argv

        self.assertEqual(captured["family_name"], "qwen36_27b")
        self.assertEqual(captured["variant"], "base")
        self.assertEqual(captured["kind"], "full")
        self.assertEqual(captured["extra_args"], ["--dry-run"])

    def test_download_audits_completed_variant(self) -> None:
        family = {
            "models_dir_env": "MODEL_FORGE_TEST_MODELS_DIR",
            "default_models_dir": "/tmp/model-forge-test-models",
            "variants": {
                "base": {
                    "repo_id": "org/base",
                    "local_dir": "base-model",
                    "served_model_name": "org/base",
                }
            },
        }
        captured: list[list[str]] = []
        with mock.patch.dict(
            model_forge_dgx.os.environ,
            {
                "MODEL_FORGE_SKIP_HF_INSTALL": "1",
                "MODEL_FORGE_HF_ALLOW_PROMPT": "0",
                "HF_TOKEN": "test-token",
            },
            clear=False,
        ):
            with mock.patch.object(model_forge_dgx, "shutil_which", return_value="hf"):
                with mock.patch.object(model_forge_dgx, "run", side_effect=lambda cmd, env=None: captured.append(cmd)):
                    model_forge_dgx.action_download(family, "test_family", "base")

        self.assertTrue(any(cmd[-2:] == ["auth", "whoami"] for cmd in captured))
        self.assertTrue(any(cmd[1:2] == ["download"] for cmd in captured))
        self.assertIn(
            [str(REPO_DIR / "forge"), "variants", "checkpoint-audit", "test_family", "--variant", "base", "--strict"],
            captured,
        )


if __name__ == "__main__":
    unittest.main()
