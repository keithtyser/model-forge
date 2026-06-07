from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from model_forge import registry


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.families_dir = Path(self._tmp.name) / "model_families"
        self.families_dir.mkdir(parents=True)
        self._orig_dir = registry.MODEL_FAMILIES_DIR
        registry.MODEL_FAMILIES_DIR = self.families_dir
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        registry.MODEL_FAMILIES_DIR = self._orig_dir
        self._tmp.cleanup()

    def _write_family(self, name: str, data: dict) -> None:
        (self.families_dir / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    def test_resolve_variant_resolves_relative_local_dir_against_models_dir(self) -> None:
        models = Path(self._tmp.name) / "models"  # absolute on the host OS
        self._write_family(
            "fam",
            {
                "display_name": "Fam",
                "default_models_dir": str(models),
                "variants": {
                    "base": {"repo_id": "org/base", "local_dir": "fam/base"},
                },
            },
        )
        v = registry.resolve_variant("fam", "base", env={})
        self.assertEqual(v.repo_id, "org/base")
        self.assertEqual(v.served_model_name, "org/base")
        self.assertEqual(v.family_display_name, "Fam")
        self.assertEqual(v.adapter_path, models / "fam" / "base")
        self.assertEqual(v.local_path, models / "fam" / "base")
        self.assertEqual(v.models_root, models)

    def test_merged_local_dir_wins_and_absolute_path_is_preserved(self) -> None:
        models = Path(self._tmp.name) / "models"
        merged = Path(self._tmp.name) / "merged"
        self._write_family(
            "fam",
            {
                "default_models_dir": str(models),
                "variants": {
                    "ft": {
                        "local_dir": "fam/adapter",
                        "merged_local_dir": str(merged),
                        "served_model_name": "served-ft",
                    }
                },
            },
        )
        v = registry.resolve_variant("fam", "ft", env={})
        self.assertEqual(v.merged_path, merged)
        self.assertEqual(v.local_path, merged)
        self.assertEqual(v.served_model_name, "served-ft")

    def test_models_dir_env_override_takes_precedence(self) -> None:
        override = Path(self._tmp.name) / "override"
        default = Path(self._tmp.name) / "default"
        config = {"models_dir_env": "MY_MODELS", "default_models_dir": str(default)}
        self.assertEqual(registry.models_dir(config, {"MY_MODELS": str(override)}), override)
        self.assertEqual(registry.models_dir(config, {}), default)

    def test_unknown_family_and_variant_raise(self) -> None:
        self._write_family("fam", {"variants": {"base": {"repo_id": "org/base"}}})
        with self.assertRaises(ValueError):
            registry.resolve_variant("nope", "base", env={})
        with self.assertRaises(ValueError):
            registry.resolve_variant("fam", "missing", env={})

    def test_resolve_repo_path_relative_vs_absolute(self) -> None:
        abs_path = Path(self._tmp.name) / "abs" / "x"
        base = Path(self._tmp.name) / "base"
        self.assertEqual(registry.resolve_repo_path(str(abs_path)), abs_path)
        self.assertEqual(registry.resolve_repo_path("rel/x", base=base), base / "rel" / "x")

    def test_load_yaml_requires_mapping(self) -> None:
        good = self.families_dir / "good.yaml"
        good.write_text("a: 1\n", encoding="utf-8")
        self.assertEqual(registry.load_yaml(good), {"a": 1})
        bad = self.families_dir / "bad.yaml"
        bad.write_text("- 1\n- 2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            registry.load_yaml(bad)


if __name__ == "__main__":
    unittest.main()
