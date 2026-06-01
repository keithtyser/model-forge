from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_DIR / "scripts" / "merge_peft_adapter.py"


class MergePeftAdapterImportTests(unittest.TestCase):
    def test_optional_unsloth_runtime_import_failure_does_not_block_cpu_merge(self) -> None:
        module_name = "merge_peft_adapter_optional_unsloth_test"
        spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "unsloth":
                raise NotImplementedError("Unsloth cannot find any torch accelerator")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            spec.loader.exec_module(module)

        self.assertIsNone(module.unsloth)
        self.assertIn("NotImplementedError", module.UNSLOTH_IMPORT_ERROR)
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
