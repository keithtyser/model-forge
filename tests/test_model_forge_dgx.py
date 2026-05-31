from __future__ import annotations

import importlib.util
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

    def test_generic_vllm_launcher_uses_model_family_env(self) -> None:
        script = (REPO_DIR / "scripts" / "serve_vllm_dgx_spark.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL=${1:-${MODEL_FORGE_MODEL:-Qwen/Qwen3.5-9B}}", script)
        self.assertIn("SERVED_MODEL_NAME=${MODEL_FORGE_SERVED_MODEL_NAME:-}", script)
        self.assertIn("--served-model-name", script)
        self.assertIn("--default-chat-template-kwargs", script)

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


if __name__ == "__main__":
    unittest.main()
