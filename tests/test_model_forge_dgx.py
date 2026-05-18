from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


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
            self.assertNotIn("MODEL_FORGE_LORA_MODULES", env)
            self.assertEqual(details["adapter"], "")


if __name__ == "__main__":
    unittest.main()
